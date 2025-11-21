from itertools import count, product
from os.path import dirname, join

import numpy as np  # For vectorised computations (performed in C).
import pytest
from hypothesis import given, strategies as st, settings
from hypothesis.extra.numpy import arrays
from numpy.random import PCG64DXSM, Generator  # New random generator.
from scipy.spatial.distance import cdist
from shapely import Polygon

import src.adsorpy.molecule_lib as mol  # Homebrew lib of molecules.
import src.adsorpy.randomsequentialadsorption as rsarun
from src.adsorpy.rsa_calculator import squared_cdist
from src.adsorpy.rsa_config import RsaConfig  # Config of the simulation.



class ExampleSimulation:
    """Test class for the simulation. Stops step by step for verification of inialisation steps."""

    __test__ = False

    def __init__(self, seed: int):
        self.seed: int = seed
        self.configname: str | None = None
        self.rsa_config: RsaConfig | None = None
        self.surf: rsarun.Surface | None = None
        self.molecules: list[rsarun.MoleculeGroup] | None = None
        self.sim: rsarun.Simulator | None = None
        self.rng: Generator | None = None

    def set_configname(self, configname: str) -> None:
        """Sets the name of the config file for this run.

        :param configname: The name of the config file.
        """
        self.configname = configname


configfiles = [
    "config_test_soft.json",
    "config_test_hard.json",
    "config_test_periodic.json",
]
molgr_count = [1, 2, 3]
surf_type = ["triangular", "square", "honeycomb"]

parameters: list[tuple[str, int, str]] = list(product(configfiles, molgr_count, surf_type))
"""A Cartesian product of the three parameter lists. Example: [A, B] x [1, 2] --> [(A, 1), (B, 2), (A, 2), (B, 2)]"""


# A parametrised test runs over the provided parameters. The parameters are split between the comma-delimited names.
@pytest.mark.parametrize("configname,molgr_count,surf_type", parameters, scope="class")
class TestWithParameters:
    """Tests the RSA simulation using various parameters. Can be expanded by making new test configs."""

    # A fixture sets up the testing environment/provides test data in a test suite.
    @pytest.fixture(scope="class")
    def simulator(self):
        """Fixture for the test simulation class."""
        return ExampleSimulation(123123)

    def test_load_config(self, simulator, configname, molgr_count, surf_type):
        """Is the config loaded correctly?"""
        simulator.set_configname(configname)
        simulator.rsa_config = RsaConfig(
            join(dirname(__file__), "test_data", simulator.configname)
        )
        assert simulator.rsa_config is not None

    def test_create_surface(self, simulator, configname, molgr_count, surf_type):
        """Is the sirface created correctly?"""
        simulator.rng = Generator(PCG64DXSM(simulator.seed))

        simulator.surf = rsarun.Surface(simulator.rsa_config, lattice_type=surf_type)
        simulator.surf.generate_grid(simulator.rng)
        assert (
                simulator.surf.x_max > 0.0 and simulator.surf.y_max > 0.0
        )  # 0. by default, larger if successful.

    def test_create_molecules(self, simulator, configname, molgr_count, surf_type):
        """Are the molecules created correctly? Divergent branches for the parametrised test."""
        simulator.molecules = []
        if molgr_count == 1:
            rot_syms = [2]
            rot_cnts = [360]
            mirror_syms = [True]
            mol_list = [mol.discorectangle([4.1, 3.1])]
        elif molgr_count == 2:
            rot_syms = [2, 1]
            rot_cnts = [360, 360]
            mirror_syms = [True, False]
            mol_list = [mol.discorectangle([4.1, 3.1]), mol.discorectangle([3.1, 2.1], 1)]
        elif molgr_count == 3:
            rot_syms = [2, 1, 0]
            rot_cnts = [360, 360, 360]
            mirror_syms = [True, False, True]
            mol_list = [mol.discorectangle([4.1, 3.1]), mol.discorectangle([3.1, 2.1], 1), mol.circulium(2.05)]
        mgc = count()
        for idx, pp in enumerate(mol_list):
            simulator.molecules.append(
                rsarun.MoleculeGroup(
                    simulator.rsa_config,
                    pp,
                    rot_syms[idx],
                    mirror_syms[idx],
                    simulator.surf.all_site_count,
                    mgc,
                    rot_cnts[idx],
                )
            )

        assert len(simulator.molecules) == molgr_count and all(
            [cule is not None for cule in simulator.molecules]
        )

    def test_boundaryparameters_creation(
            self, simulator, configname, molgr_count, surf_type
    ):
        """Do the boundary parameters initialise correctly?"""
        dbl_max_rad = 2.0 * max([ii.max_radius for ii in simulator.molecules])
        simulator.surf.bp.biggest_radius = dbl_max_rad
        simulator.surf.bp.generate_boundary_conditions(simulator.surf)
        for mdx, molec in enumerate(simulator.molecules):
            molec.bp.biggest_radius = dbl_max_rad
            molec.bp.generate_boundary_conditions(simulator.surf, molec)
            molec.generate_rotated_molecules(molec.bp, simulator.molecules)

        sb = simulator.surf.bp
        assert np.count_nonzero([sb.periodic_flag, sb.hard_flag, sb.soft_flag]) == 1

    def test_generate_simulation(self, simulator, configname, molgr_count, surf_type):
        """Does the simulation gneerate correctly?"""
        simulator.sim = rsarun.Simulator(
            simulator.rsa_config,
            False,
            simulator.surf,
            simulator.molecules,
            simulator.rng,
        )
        assert simulator.sim is not None

    def test_place_fist_molecule(self, simulator, configname, molgr_count, surf_type):
        """Is the first moelcule placed correctly?"""
        succ, *_ = simulator.sim.attempt_place_molecule(
            simulator.surf, simulator.molecules[0]
        )

        if (
                succ or not simulator.sim.bp.hard_flag
        ):  # Adsorption cannot be guaranteed on the first step for a hard surface.
            assert (
                    np.count_nonzero(simulator.sim.mol_data.stored_data["exists"]) == 1
            )  # One molecule exists on the surf!

        if simulator.sim.bp.periodic_flag:
            assert (
                    np.count_nonzero(simulator.sim.mol_data.stored_mirr_data["exists"]) >= 1
            )
            assert (
                    np.count_nonzero(simulator.sim.mol_data.stored_mirr_data["exists"])
                    == simulator.sim.molgroups[0].bp.mirror_counter
            )

    def test_buffer_trimming(self, simulator, configname, molgr_count, surf_type):
        """Are multiple sites removed during the buffer trimming (r neighbourhood clearance) step?"""
        if simulator.sim.total_molecule_counter > 0:
            assert (
                    simulator.sim.surf.all_site_count
                    - np.count_nonzero(simulator.sim.molgroups[0].vacant)
                    > 1
            )

    def test_random_placement(self, simulator, configname, molgr_count, surf_type):
        """Does placement work for the codosing scheme?"""
        simulator.sim.attempt_random_placement(simulator.surf, *simulator.molecules)

    def test_try_placement(self, simulator, configname, molgr_count, surf_type):
        """Does cascade placement work?"""
        simulator.sim.attempt_cascading_placement(simulator.surf, *simulator.molecules)

    def test_run_simulation(self, simulator, configname, molgr_count, surf_type):
        """Does the simulation run correctly until saturated? Are no sites vacant?"""
        while np.any(simulator.sim.molgroups[0].vacant):
            simulator.sim.attempt_place_molecule(simulator.surf, simulator.molecules[0])

        assert not np.any(simulator.sim.molgroups[0].vacant)

    def test_no_overlap(self, simulator, configname, molgr_count, surf_type):
        """Tests whether none of the molecules overlap. Tests periodic molecules as well for periodic boundary conditions."""
        polygons: np.ndarray[Polygon]
        if simulator.sim.bp.periodic_flag:
            existing = simulator.sim.mol_data.stored_mirr_data["exists"]
            polygons = simulator.sim.mol_data.stored_mirr_data["polygon"][existing]
        else:
            existing = simulator.sim.mol_data.stored_data["exists"]
            polygons = simulator.sim.mol_data.stored_data["polygon"][existing]

        ii: Polygon
        jj: Polygon
        overlap = False
        for idx, ii in enumerate(polygons):  # TODO: Add STRtree to speed up?
            for jj in polygons[idx + 1:]:
                overlap = ii.intersects(jj)
                if overlap:
                    break
            if overlap:
                break

        assert not overlap

    def test_no_large_gaps(self, simulator, configname, molgr_count, surf_type):
        """Tests whether there are sites with gaps larger than the molecule's circumradius.
        In that case, a molecule should be able to fit, and the simulation terminated before saturation.
        """
        gaps = simulator.sim.analyse_gap_size(simulator.sim.surf)
        circumradius = simulator.sim.molgroups[0].max_radius
        # raise NotImplementedError(simulator.sim.mol_data.orig_to_mirrors['mirr_ids'])
        assert np.all(gaps <= circumradius)

    @pytest.fixture(scope="class")
    def alt_simulator(self):
        """Fixture for the test simulation class, seed differs by 1."""
        return ExampleSimulation(123124)

    def test_run_alt(
            self, simulator, alt_simulator, configname, molgr_count, surf_type
    ):
        """Run the aforementioned tests for the alternative simulation as well."""
        arguments = (configname, molgr_count, surf_type)
        self.test_load_config(alt_simulator, *arguments)
        self.test_create_surface(alt_simulator, *arguments)
        self.test_create_molecules(alt_simulator, *arguments)
        self.test_boundaryparameters_creation(alt_simulator, *arguments)
        self.test_generate_simulation(alt_simulator, *arguments)
        self.test_place_fist_molecule(alt_simulator, *arguments)
        self.test_buffer_trimming(alt_simulator, *arguments)
        self.test_random_placement(alt_simulator, *arguments)
        self.test_try_placement(alt_simulator, *arguments)
        self.test_run_simulation(alt_simulator, *arguments)

    def test_randomness_comparison(
            self, simulator, alt_simulator, configname, molgr_count, surf_type
    ):
        """Test whether the two simulations are different."""

        existing_sim_mols = simulator.sim.mol_data.stored_data[
            simulator.sim.mol_data.stored_data["exists"]
        ]
        existing_altsim_mols = alt_simulator.sim.mol_data.stored_data[
            alt_simulator.sim.mol_data.stored_data["exists"]
        ]

        if (
                existing_sim_mols.size == existing_altsim_mols.size
        ):  # If unequal, results are guaranteed to differ.
            assert not np.all(existing_sim_mols == existing_altsim_mols)

    def test_alt_nooverlap(self, alt_simulator, configname, molgr_count, surf_type):
        """Is there no overlap for the alternative simulation as well?"""
        self.test_no_overlap(alt_simulator, configname, molgr_count, surf_type)

    def test_alt_gaps(self, alt_simulator, configname, molgr_count, surf_type):
        """Test whether the gap size analysis succeeds for the alternative simulation as well."""
        self.test_no_large_gaps(alt_simulator, configname, molgr_count, surf_type)


def test_surfacetype_invalid_input():
    """Is an error raised when the lattice type is incorrect?"""
    with pytest.raises(ValueError):
        rsarun.Surface(
            rsarun.RsaConfig(
                join(dirname(__file__), "test_data", "config_test_soft.json")
            ),
            "Dogbonium",
        ).generate_grid()


def test_boundarytype_invalid_input():
    """Is an error raised for invalid bounary types?"""
    with pytest.raises(TypeError):
        rsarun.BoundaryParameters(10)

    with pytest.raises(ValueError):
        rsarun.BoundaryParameters("Dogbonium")

def test_molecules_invalid_input():
    """Is an error raised for invalid input of the molecules?"""
    with pytest.raises(ValueError):
        sim = ExampleSimulation(13579)
        sim.set_configname("config_test_soft.json")
        sim.rsa_config = RsaConfig(join(dirname(__file__), "test_data", sim.configname))

        rsarun.Simulator(sim.rsa_config, None, None, [], None)

@pytest.fixture(scope="class")
def simple_simulator() -> ExampleSimulation:
    """A class for some simple simulation tests."""
    sim = ExampleSimulation(123321)
    sim.set_configname("config_test_soft.json")
    sim.rsa_config = RsaConfig(join(dirname(__file__), "test_data", sim.configname))
    sim.rng = Generator(PCG64DXSM(sim.seed))
    sim.surf = rsarun.Surface(sim.rsa_config)
    sim.surf.generate_grid(sim.rng)
    sim.molecules = []
    rot_syms = [2]
    rot_cnts = [360]
    mirror_syms = [True]
    mol_list = [mol.discorectangle([4.1, 3.1])]
    mgc = count()
    for idx, pp in enumerate(mol_list):
        sim.molecules.append(
            rsarun.MoleculeGroup(
                sim.rsa_config,
                pp,
                rot_syms[idx],
                mirror_syms[idx],
                sim.surf.all_site_count,
                mgc,
                rot_cnts[idx],
            )
        )

    dbl_max_rad = 2.0 * max([ii.max_radius for ii in sim.molecules])
    sim.surf.bp.biggest_radius = dbl_max_rad
    sim.surf.bp.generate_boundary_conditions(sim.surf)
    for mdx, molec in enumerate(sim.molecules):
        molec.bp.biggest_radius = dbl_max_rad
        molec.bp.generate_boundary_conditions(sim.surf, molec)
        molec.generate_rotated_molecules(molec.bp, sim.molecules)

    sim.sim = rsarun.Simulator(
        sim.rsa_config,
        False,
        sim.surf,
        sim.molecules,
        sim.rng,
    )
    sim.sim.mol_data.max_array_length = 10
    sim.sim.mol_data.__post_init__()

    return sim


class TestMiscSettings:
    """For the testing of miscellaneous settings."""
    def test_adsorption_chance(self, simple_simulator):
        """Does the adsorption fail when the sticking probability is 0?
        """
        simple_simulator.molecules[0].sticking_probability = 0
        success: bool
        success, *_ = simple_simulator.sim.attempt_place_molecule(
            simple_simulator.surf, simple_simulator.molecules[0]
        )
        simple_simulator.molecules[0].sticking_probability = 1

        assert not success

    def test_placement_with_first_angle_and_grid_index(self, simple_simulator):
        """Does the simulation place a molecule when the grid index and rotation index have been provided?
        """
        _, _, _, pos, rot, _ = simple_simulator.sim.attempt_place_molecule(
            simple_simulator.surf, simple_simulator.molecules[0], 0, 0
        )
        assert pos == 0 and rot == 0

    def test_run_simulation_with_first_angle(self, simple_simulator):
        """Does the simulation place a molecule when the first rotation index has been provided?
        """
        while np.any(simple_simulator.sim.molgroups[0].vacant):
            simple_simulator.sim.attempt_place_molecule(
                simple_simulator.surf, simple_simulator.molecules[0], None, 0
            )
        simple_simulator.sim.attempt_place_molecule(
            simple_simulator.surf, simple_simulator.molecules[0], None, 0
        )

        assert not np.any(simple_simulator.sim.molgroups[0].vacant)

    def test_no_overlap(self, simple_simulator):
        """Tests whether none of the molecules overlap. Tests periodic molecules as well for periodic boundary conditions."""
        polygons: np.ndarray[Polygon]
        if simple_simulator.sim.bp.periodic_flag:
            existing = simple_simulator.sim.mol_data.stored_mirr_data["exists"]
            polygons = simple_simulator.sim.mol_data.stored_mirr_data["polygon"][
                existing
            ]
        else:
            existing = simple_simulator.sim.mol_data.stored_data["exists"]
            polygons = simple_simulator.sim.mol_data.stored_data["polygon"][existing]

        ii: Polygon
        jj: Polygon
        overlap = False
        for idx, ii in enumerate(polygons):
            for jj in polygons[idx + 1:]:
                overlap = ii.intersects(jj)
                if (
                        overlap
                ):  # If it fails, do not check any other molecules. Just break.
                    break
            if overlap:
                break

        assert not overlap

    def test_no_large_gaps(self, simple_simulator):
        """Tests whether there are sites with gaps larger than the molecule's circumradius.
        In that case, a molecule should be able to fit, and the simulation terminated before saturation.
        """
        gaps = simple_simulator.sim.analyse_gap_size(simple_simulator.sim.surf)
        circumradius = simple_simulator.sim.molgroups[0].max_radius
        assert np.all(gaps <= circumradius)


@settings(deadline=None)
@given(arr1 =
    arrays(
        dtype=np.float64,
        shape=st.tuples(st.just(2), st.integers(min_value=1, max_value=2000)),
        elements=st.floats(allow_infinity=False, allow_nan=False),
    ),
    arr2 =
    arrays(
        dtype=np.float64,
        shape=st.tuples(st.just(2), st.integers(min_value=1, max_value=2000)),
        elements=st.floats(allow_infinity=False, allow_nan=False),
    ),
)
def test_numba_vs_cdist(arr1, arr2):
    """Compute the squared Euclidean distances using the faster homemade method and the preprogrammed method.
    """
    numba_result = squared_cdist(arr1, arr2)
    cdist_result = cdist(arr1.T, arr2.T, metric='sqeuclidean')

    np.testing.assert_allclose(numba_result, cdist_result)


@pytest.fixture(scope="class")
def gapsim() -> ExampleSimulation:
    """An simulation class for the gap size distribution test.
    """
    sim = ExampleSimulation(123321)
    sim.set_configname("config_test_periodic.json")
    sim.rsa_config = RsaConfig(join(dirname(__file__), "test_data", sim.configname))
    sim.rng = Generator(PCG64DXSM(sim.seed))
    sim.surf = rsarun.Surface(sim.rsa_config)
    sim.surf.generate_grid(sim.rng)

    return sim


def idx_radius_strategy():
    """Define an index on the surface and a radius for the molecule.
    Max idx is sites * sites * 2 - 1
    Max radius is sites * lattice_a

    :return: The strategy used to test the hypothesis.
    """
    sitecount: int = 10
    lattice_a: float = 4.0
    return st.tuples(
        st.integers(min_value=0, max_value=sitecount * sitecount * 2 - 1),
        st.floats(min_value=0.1 * lattice_a, max_value=sitecount * lattice_a),
    )


@settings(max_examples=200)
@given(idx_radius_strategy())
def test_gapsize_analysis(gapsim: ExampleSimulation, idx_radius: tuple[int, float]):
    """Prove that the gap size distribution yields the correct values for circular molecules.

    The distance from the edge of a disk to a point can be derived analytically. This makes it a suitable test case.
    Because the gap size distribution is an integral part of the script and is used to test correctness of output,
    proving that the gap size distribution is correct is important. The test is performed for disks of random sizes.

    :param gapsim: Test class for the gap size distribution.
    :param idx_radius: Index and radius of the test molecule, randomly generated.
    """
    tolerance: float = 1e-12
    "A very small tolerance for the distance function test."
    simrad = gapsim
    index: int
    radius: float
    index, radius = idx_radius
    simrad.molecules = []
    rot_syms = [0]
    rot_cnts = [1]
    mirror_syms = [True]
    mol_list = [mol.circulium(radius)]
    mgc = count()
    for idx, pp in enumerate(mol_list):
        simrad.molecules.append(
            rsarun.MoleculeGroup(
                simrad.rsa_config,
                pp,
                rot_syms[idx],
                mirror_syms[idx],
                simrad.surf.all_site_count,
                mgc,
                rot_cnts[idx],
            )
        )

    dbl_max_rad = 2.0 * max([ii.max_radius for ii in simrad.molecules])
    simrad.surf.bp.biggest_radius = dbl_max_rad
    simrad.surf.bp.generate_boundary_conditions(simrad.surf)
    for mdx, molec in enumerate(simrad.molecules):
        molec.bp.biggest_radius = dbl_max_rad
        molec.bp.generate_boundary_conditions(simrad.surf, molec)
        molec.generate_rotated_molecules(molec.bp, simrad.molecules)

    simrad.sim = rsarun.Simulator(
        simrad.rsa_config,
        False,
        simrad.surf,
        simrad.molecules,
        simrad.rng,
    )

    simrad.sim.attempt_place_molecule(simrad.surf, simrad.molecules[0], grid_idx=index)
    gaps = simrad.sim.analyse_gap_size(simrad.surf, keepzero=True)

    my_gaps = cdist(
        simrad.sim.mol_data.mirror_coords[
        :, simrad.sim.mol_data.stored_mirr_data["exists"]
        ].T,
        simrad.surf.grid_coordinates.T,
    )
    my_gaps = np.min(my_gaps, axis=0)

    in_gaps = my_gaps - simrad.sim.molgroups[0].min_radius
    in_gaps[in_gaps < 0.0] = 0.0

    out_gaps = my_gaps - simrad.sim.molgroups[0].max_radius
    out_gaps[out_gaps < 0.0] = 0.0

    assert simrad.sim.total_molecule_counter == 1

    assert np.all(out_gaps <= gaps + tolerance)
    assert np.all(gaps <= in_gaps + tolerance)


