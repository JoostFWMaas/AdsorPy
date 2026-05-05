"""Test the `run_simulation` function."""
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.strategies import SearchStrategy
from shapely import Polygon, unary_union
from shapely.ops import orient
from shapely.prepared import prep

from src.adsorpy.randomsequentialadsorption import Simulator
from src.adsorpy.rsa_config import RsaConfig
from src.adsorpy.run_simulation import _select_and_run, run_simulation

if TYPE_CHECKING:

    from src.adsorpy.randomsequentialadsorption import GeoArray

SEED = 123654789


@pytest.fixture
def rsa_config() -> RsaConfig:
    """Fixture for a basic RsaConfig object."""
    config_path = Path(__file__).parent / "test_data" / "config_test_periodic.json"
    return RsaConfig(str(config_path))


@pytest.fixture
def default_polygon() -> Polygon:
    """Fixture for a default polygon."""
    return Polygon([(-5, -5), (5, -5), (5, 5), (-5, 5)])


def overlap_tester(simulator: Simulator) -> Literal[0]:
    """Test whether none of the molecules overlap. Solely for periodic boundary conditions."""
    polygons: GeoArray
    existing = simulator.mol_data.stored_mirr_data["exists"]
    polygons = simulator.mol_data.stored_mirr_data["polygon"][existing]

    ii: Polygon
    overlap = False
    for idx, ii in enumerate(polygons):
        prepared_multipolygon = prep(unary_union(polygons[idx + 1 :]))
        overlap = prepared_multipolygon.intersects(ii)  # If there is any overlap, this test fails.
        if overlap:
            break

    assert not overlap

    return 0


def gapsize_tester(simulator: Simulator) -> Literal[0]:
    """Test whether there are sites with gaps larger than the largest molecule's circumradius.

    If gaps are larger, a molecule should be able to fit, and the simulation terminated before saturation.

    :param simulator: Simulator object.
    :returns: 0.
    """
    gaps = simulator.analyse_gap_size(simulator.surf)
    circumradius = max(molgr.max_radius for molgr in simulator.molgroups)
    assert np.all(gaps <= circumradius)

    return 0


def simulation_end_tester(simulator: Simulator, subtests: pytest.Subtests) -> Literal[0]:
    """Test the overlap and gap size after a simulation is done.

    :param simulator: Simulator object.
    :param subtests: Subtests object used to distinguish between overlap and gap size test results.

    :returns: 0.
    """
    test_functions = (overlap_tester, gapsize_tester)
    for t_func in test_functions:
        with subtests.test(f"{t_func.__name__}"):
            t_func(simulator)

    return 0


def test_run_simulation_default_params(
    rsa_config: RsaConfig,
    default_polygon: Polygon,
    subtests: pytest.Subtests,
) -> None:
    """Test the simulation with default parameter return types."""
    results_len: int = 6
    results = run_simulation(rsa_config)
    assert isinstance(results, tuple)
    assert len(results) == results_len
    assert isinstance(results[0], list)
    assert isinstance(results[1], np.ndarray)
    assert isinstance(results[2], int)
    assert isinstance(results[3], np.ndarray)
    assert isinstance(results[4], np.ndarray)
    assert isinstance(results[5], Simulator)
    simulation_end_tester(results[5], subtests)


def test_run_simulation_with_molecule(
    rsa_config: RsaConfig,
    default_polygon: Polygon,
    subtests: pytest.Subtests,
) -> None:
    """Test the simulation with a provided molecule."""
    results = run_simulation(rsa_config, seed=SEED, molecules_list=default_polygon)
    assert len(results[0]) == 1
    simulation_end_tester(results[5], subtests)


def test_run_simulation_invalid_simulation_type(rsa_config: RsaConfig, default_polygon: Polygon) -> None:
    """Test that an invalid simulation type raises a ValueError."""
    with pytest.raises(ValueError, match="The simulation type must be either"):
        run_simulation(rsa_config, simulation_type="invalid_type")


def test_run_simulation_invalid_boundary_condition(rsa_config: RsaConfig, default_polygon: Polygon) -> None:
    """Test that an invalid boundary condition raises a ValueError."""
    with pytest.raises(ValueError, match="Boundary condition must be either"):
        run_simulation(rsa_config, boundary_condition="invalid_condition")


def test_run_simulation_mismatch_lengths(rsa_config: RsaConfig, default_polygon: Polygon) -> None:
    """Test that mismatched list lengths raise a ValueError."""
    with pytest.raises(ValueError, match="Number of molecules, symmetries, and/or rotation counts do not match"):
        run_simulation(
            rsa_config,
            molecules_list=[default_polygon],
            rotation_symmetries=[1, 2],
            reflection_symmetries=[True],
            rotation_counts=[360],
        )


def test_run_simulation_codosing_mismatch_distribution(rsa_config: RsaConfig, default_polygon: Polygon) -> None:
    """Test that mismatched dosing distribution raises a ValueError."""
    with pytest.raises(ValueError, match="Dosing distribution must be the same length as the molecule list"):
        run_simulation(
            rsa_config,
            molecules_list=[default_polygon, default_polygon],
            simulation_type="codosing",
            dosing_distribution=[0.5],
        )


def test_run_simulation_sequential(rsa_config: RsaConfig, default_polygon: Polygon, subtests: pytest.Subtests) -> None:
    """Test the sequential simulation."""
    results = run_simulation(rsa_config, seed=SEED, molecules_list=[default_polygon], simulation_type="sequential")
    assert len(results[0]) == 1
    simulation_end_tester(results[5], subtests)


def test_run_simulation_codosing(rsa_config: RsaConfig, default_polygon: Polygon, subtests: pytest.Subtests) -> None:
    """Test the codosing simulation."""
    new_mol_count: int = 2
    results = run_simulation(
        rsa_config,
        seed=SEED,
        molecules_list=[default_polygon, default_polygon],
        simulation_type="codosing",
        dosing_distribution=[0.5, 0.5],
    )
    assert len(results[0]) == new_mol_count
    simulation_end_tester(results[5], subtests)


def test_run_simulation_cascade(rsa_config: RsaConfig, default_polygon: Polygon, subtests: pytest.Subtests) -> None:
    """Test the cascade simulation."""
    results = run_simulation(rsa_config, seed=SEED, molecules_list=[default_polygon], simulation_type="cascade")
    assert len(results[0]) == 1
    simulation_end_tester(results[5], subtests)


def test_run_simulation_boundary_conditions(rsa_config: RsaConfig, default_polygon: Polygon) -> None:
    """Test different boundary conditions."""
    for boundary in ["soft", "hard", "periodic"]:
        results = run_simulation(rsa_config, seed=SEED, molecules_list=[default_polygon], boundary_condition=boundary)
        assert len(results[0]) == 1


def test_run_simulation_sticking_probability(
    rsa_config: RsaConfig,
    default_polygon: Polygon,
    subtests: pytest.Subtests,
) -> None:
    """Test different sticking probability types."""
    for stck in (0.5, [0.5], np.array([0.5])):
        with subtests.test(f"Sticking type {type(stck).__name__}"):
            results = run_simulation(rsa_config, seed=SEED, molecules_list=[default_polygon], sticking_probability=stck)
            assert len(results[0]) == 1


def test_run_simulation_randomness(
    rsa_config: RsaConfig,
    default_polygon: Polygon,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test the randomness based on datetime seed."""
    fixed_datetime = datetime(2023, 1, 1, 0, 0, 0, 1, tzinfo=timezone.utc)
    monkeypatch.setattr("run_simulation.datetime", fixed_datetime)

    results_1 = run_simulation(rsa_config, include_rejected_flux=True, molecules_list=[default_polygon])

    # Move time forward by 1 second to ensure a different seed.
    monkeypatch.setattr("run_simulation.datetime", fixed_datetime + timedelta(seconds=1))

    results_2 = run_simulation(rsa_config, include_rejected_flux=True, molecules_list=[default_polygon])

    assert results_1[2] != results_2[2]  # Seeds should be different
    assert not np.array_equal(results_1[3], results_2[3])  # The dose timestamps are virtually guaranteed to be unequal.


def test_run_simulation_determinism(rsa_config: RsaConfig, default_polygon: Polygon, subtests: pytest.Subtests) -> None:
    """Test determinism of the pseudorandomness by repeating with the same seed."""
    kwargs = {
        "calculate_gap_size": True,
        "include_rejected_flux": True,
        "molecules_list": [default_polygon],
        "seed": 42,
    }
    results_1 = run_simulation(rsa_config, **kwargs)[:-1]  # type: ignore[arg-type]
    results_2 = run_simulation(rsa_config, **kwargs)[:-1]  # type: ignore[arg-type]
    comparison_test_names = ("Mol count", "Gap size", "Seed", "Flux/dose", "ASF")
    for comparison_test_name, output_1, output_2 in zip(comparison_test_names, results_1, results_2, strict=True):
        with subtests.test(f"{comparison_test_name} equivalence"):
            assert np.array_equal(output_1, output_2)  # type: ignore[arg-type]


@st.composite
def non_degenerate_polygon(
    draw: Callable[[SearchStrategy[list[tuple[float, float]]]], list[tuple[float, float]]],
    min_area: float = 0.01,
) -> Polygon:
    """Generate a non-degenerate polygon with an area greater than min_area."""
    # Generate a list of points ensuring unique coordinates and valid minimum area
    vertices: int = 3
    points = draw(
        st.lists(
            st.tuples(st.floats(-2, 2), st.floats(-2, 2)),
            min_size=vertices,
            max_size=vertices,
            unique_by=lambda x: (round(x[0], 5), round(x[1], 5)),
        ).filter(lambda pts: len(pts) == vertices and Polygon(pts).area > min_area),
    )
    # Create the polygon and ensure it has a valid orientation (clockwise)
    polygon = Polygon(points)
    return orient(polygon)  # Ensure the polygon is oriented correctly


@settings(deadline=None, max_examples=100)
@given(
    site_count=st.integers(1, 20) | st.none(),
    lattice_a=st.floats(0.1, 10) | st.none(),
    boundary_condition=st.sampled_from(["soft", "hard", "periodic"]) | st.none(),
    molecules_list=st.lists(non_degenerate_polygon(min_area=0.01), min_size=1, max_size=4),
    rotation_symmetries=st.lists(st.integers(0, 10), min_size=1, max_size=4) | st.none(),
    reflection_symmetries=st.lists(st.booleans(), min_size=1, max_size=4) | st.none(),
    rotation_counts=st.lists(st.integers(1, 360), min_size=1, max_size=4) | st.none(),
)
def test_run_simulation_property_test(
    site_count: int | None,
    lattice_a: float | None,
    boundary_condition: str | None,
    molecules_list: list[Polygon],
    rotation_symmetries: list[int] | None,
    reflection_symmetries: list[bool] | None,
    rotation_counts: list[int] | None,
) -> None:
    """Test that run_simulation handles random valid inputs correctly.

    In this test, all inputs are varied, and even the molecules are randomly shaped. This is a property test.
    If the length of the lists is not equivalent (or equal to 1), a ValueError test is expected.
    Otherwise, the output is tested.
    """
    rsa_config = RsaConfig(str(Path(__file__).parent / "test_data" / "config_test_periodic.json"))
    seed = 123123

    num_mols = len(molecules_list)

    if (
        (rotation_symmetries is None or len(rotation_symmetries) in {1, num_mols})
        and (reflection_symmetries is None or len(reflection_symmetries) in {1, num_mols})
        and (rotation_counts is None or len(rotation_counts) in {1, num_mols})
    ):
        results = run_simulation(
            rsa_config,
            molecules_list=molecules_list,
            rotation_symmetries=rotation_symmetries,
            reflection_symmetries=reflection_symmetries,
            rotation_counts=rotation_counts,
            site_count=site_count,
            lattice_a=lattice_a,
            boundary_condition=boundary_condition,
            seed=seed,
        )
        molecule_counters, gap_sizes, seed_out, *_ = results

        assert len(molecule_counters) == len(molecules_list)
        assert all(count >= 0 for count in molecule_counters)
        assert isinstance(gap_sizes, np.ndarray)
        assert not gap_sizes.size or gap_sizes.min()
        assert seed == seed_out
    else:
        with pytest.raises(ValueError, match="Number of molecules, symmetries, and/or rotation counts do not match"):
            run_simulation(
                rsa_config,
                molecules_list=molecules_list,
                rotation_symmetries=rotation_symmetries,
                reflection_symmetries=reflection_symmetries,
                rotation_counts=rotation_counts,
                site_count=site_count,
                lattice_a=lattice_a,
                boundary_condition=boundary_condition,
            )


class TestCustomGrid:
    """Class to test errors related to custom grid generation."""

    def test_bad_custom_grid(self) -> None:
        """Custom grid raises an error when either x or y (not both) is missing."""
        rsa_config = RsaConfig(str(Path(__file__).parent / "test_data" / "config_test_periodic.json"))
        with pytest.raises(
            ValueError,
            match=r"A custom grid will only be generated if 'site_x_coords', 'site_y_coords',*",
        ):
            run_simulation(rsa_config=rsa_config, site_x_coords=[1.1])  # type: ignore[arg-type]

    def test_bad_xcoords(self) -> None:
        """Raise an error when the x coordinates are negative. Same effect as the y coordinates."""
        rsa_config = RsaConfig(str(Path(__file__).parent / "test_data" / "config_test_periodic.json"))
        with pytest.raises(ValueError, match=r"Site x coordinates must be positive.*"):
            run_simulation(
                rsa_config=rsa_config,
                site_x_coords=[-1.1],  # type: ignore[arg-type]
                site_y_coords=[-2.0, 1.1],  # type: ignore[arg-type]
                bounding_x_coord=-10,
                bounding_y_coord=-10,
            )

    def test_good_custom_grid(self) -> None:
        """Custom grid generates correctly without errors."""
        sim: Simulator
        rsa_config = RsaConfig(str(Path(__file__).parent / "test_data" / "config_test_periodic.json"))
        *_, sim = run_simulation(
            rsa_config=rsa_config,
            seed=SEED,
            site_x_coords=0.9 * np.repeat(np.arange(10, dtype=np.float64), 10),
            site_y_coords=0.9 * np.tile(np.arange(10, dtype=np.float64), 10),
            bounding_x_coord=10,
            bounding_y_coord=10,
        )

        overlap_tester(sim)
        gapsize_tester(sim)


def test_sequential_fluxreject() -> None:
    """Handle flux-based simulation correctly."""
    rsa_config = RsaConfig(str(Path(__file__).parent / "test_data" / "config_test_periodic.json"))
    flux = run_simulation(rsa_config=rsa_config, seed=SEED, simulation_type="sequential", include_rejected_flux=True)[3]
    assert flux is not None  # The flux return value has to exist.
    assert isinstance(flux, np.ndarray)  # It should be a numpy array.
    assert flux.size  # The numpy array should not be empty.


def test_select_and_run() -> None:
    """When a bad simulation type/dosing scheme is selected, an error should be raised."""
    bad_simulation_type = "potato"
    flux = True
    with pytest.raises(
        ValueError,
        match=f"Simulation type {bad_simulation_type} with rejected_flux = {flux} is not supported.",
    ):
        _select_and_run(None, None, None, bad_simulation_type, flux, None, None)  # type: ignore[arg-type]


def test_wrong_stickingprobability() -> None:
    """When the wrong sticking probability type is used, an error should be raised."""
    rsa_config = RsaConfig(str(Path(__file__).parent / "test_data" / "config_test_periodic.json"))
    with pytest.raises(TypeError, match=r"sticking_probability must be a float, list, or np.ndarray"):
        run_simulation(rsa_config=rsa_config, simulation_type="sequential", sticking_probability="one")  # type: ignore[arg-type]
