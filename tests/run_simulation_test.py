import pytest
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path
from hypothesis import given, strategies as st, settings
from shapely import Polygon
from shapely.ops import orient
from src.adsorpy.run_simulation import run_simulation, RsaConfig, _select_and_run


@pytest.fixture
def rsa_config():
    """Fixture for a basic RsaConfig object."""
    config_path = Path(__file__).parent / "test_data" / "config_test_periodic.json"
    return RsaConfig(str(config_path))


@pytest.fixture
def default_polygon():
    """Fixture for a default polygon."""
    return Polygon([(-5, -5), (5, -5), (5, 5), (-5, 5)])


def test_run_simulation_default_params(rsa_config, default_polygon):
    """Test the simulation with default parameters."""
    results = run_simulation(rsa_config)
    assert isinstance(results, tuple)
    assert len(results) == 5
    assert isinstance(results[0], list)
    assert isinstance(results[1], np.ndarray)
    assert isinstance(results[2], int)
    assert isinstance(results[3], np.ndarray)
    assert isinstance(results[4], np.ndarray)


def test_run_simulation_with_molecule(rsa_config, default_polygon):
    """Test the simulation with a provided molecule."""
    results = run_simulation(rsa_config, molecules_list=default_polygon)
    assert len(results[0]) == 1


def test_run_simulation_invalid_simulation_type(rsa_config, default_polygon):
    """Test that an invalid simulation type raises a ValueError."""
    with pytest.raises(ValueError, match="The simulation type must be either"):
        run_simulation(rsa_config, simulation_type="invalid_type")


def test_run_simulation_invalid_boundary_condition(rsa_config, default_polygon):
    """Test that an invalid boundary condition raises a ValueError."""
    with pytest.raises(ValueError, match="Boundary condition must be either"):
        run_simulation(rsa_config, boundary_condition="invalid_condition")


def test_run_simulation_mismatch_lengths(rsa_config, default_polygon):
    """Test that mismatched list lengths raise a ValueError."""
    with pytest.raises(ValueError, match="Number of molecules, symmetries, and/or rotation counts do not match"):
        run_simulation(
            rsa_config,
            molecules_list=[default_polygon],
            rotation_symmetries=[1, 2],
            reflection_symmetries=[True],
            rotation_counts=[360]
        )


def test_run_simulation_codosing_mismatch_distribution(rsa_config, default_polygon):
    """Test that mismatched dosing distribution raises a ValueError."""
    with pytest.raises(ValueError, match="Dosing distribution must be the same length as the molecule list"):
        run_simulation(
            rsa_config,
            molecules_list=[default_polygon, default_polygon],
            simulation_type="codosing",
            dosing_distribution=[0.5]
        )


def test_run_simulation_sequential(rsa_config, default_polygon):
    """Test the sequential simulation."""
    results = run_simulation(rsa_config, molecules_list=[default_polygon], simulation_type="sequential")
    assert len(results[0]) == 1


def test_run_simulation_codosing(rsa_config, default_polygon):
    """Test the codosing simulation."""
    results = run_simulation(
        rsa_config,
        molecules_list=[default_polygon, default_polygon],
        simulation_type="codosing",
        dosing_distribution=[0.5, 0.5]
    )
    assert len(results[0]) == 2


def test_run_simulation_cascade(rsa_config, default_polygon):
    """Test the cascade simulation."""
    results = run_simulation(rsa_config, molecules_list=[default_polygon], simulation_type="cascade")
    assert len(results[0]) == 1


def test_run_simulation_boundary_conditions(rsa_config, default_polygon):
    """Test different boundary conditions."""
    for boundary in ["soft", "hard", "periodic"]:
        results = run_simulation(rsa_config, molecules_list=[default_polygon], boundary_condition=boundary)
        assert len(results[0]) == 1


def test_run_simulation_randomness(rsa_config, default_polygon, monkeypatch):
    """Test the randomness based on datetime seed."""
    fixed_datetime = datetime(2023, 1, 1, 0, 0, 0, 1)
    monkeypatch.setattr("run_simulation.datetime", fixed_datetime)

    results_1 = run_simulation(rsa_config, molecules_list=[default_polygon])

    # Move time forward by 1 second to ensure a different seed.
    monkeypatch.setattr("run_simulation.datetime", fixed_datetime + timedelta(seconds=1))

    results_2 = run_simulation(rsa_config, molecules_list=[default_polygon])

    assert results_1[2] != results_2[2]  # Seeds should be different


@st.composite
def non_degenerate_polygon(draw, min_area=0.01) -> Polygon:
    """Generate a non-degenerate polygon with an area greater than min_area."""
    # Generate a list of points ensuring unique coordinates and valid minimum area
    points = draw(
        st.lists(
            st.tuples(
                st.floats(-2, 2),
                st.floats(-2, 2)
            ),
            min_size=3, max_size=3,
            unique_by=lambda x: (round(x[0], 5), round(x[1], 5))
        ).filter(lambda pts: len(pts) == 3 and Polygon(pts).area > min_area)
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
def test_run_simulation_property_test(site_count, lattice_a, boundary_condition, molecules_list, rotation_symmetries,
                                      reflection_symmetries, rotation_counts):
    """Test that run_simulation handles random valid inputs correctly.

    In this test, all inputs are varied, and even the molecules are randomly shaped. This is a property test.
    """
    rsa_config = RsaConfig(str(Path(__file__).parent / "test_data" / "config_test_periodic.json"))
    seed = 123123

    num_mols = len(molecules_list)

    if ((rotation_symmetries is None or len(rotation_symmetries) in {1, num_mols}) and
        (reflection_symmetries is None or len(reflection_symmetries) in {1, num_mols}) and
        (rotation_counts is None or len(rotation_counts) in {1, num_mols})):
        results = run_simulation(
            rsa_config,
            molecules_list=molecules_list,
            rotation_symmetries=rotation_symmetries,
            reflection_symmetries=reflection_symmetries,
            rotation_counts=rotation_counts,
            site_count=site_count,
            lattice_a=lattice_a,
            boundary_condition=boundary_condition,
            seed=seed
        )
        molecule_counters, gap_sizes, seed, flux, _ = results

        assert len(molecule_counters) == len(molecules_list)
        assert all(count >= 0 for count in molecule_counters)
        assert isinstance(gap_sizes, np.ndarray)
        assert not gap_sizes.size or gap_sizes.min() > 0
        assert isinstance(seed, int)
    else:
        with pytest.raises(ValueError,
                           match="Number of molecules, symmetries, and/or rotation counts do not match"):
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
    """Class to test errors related to custom grid generation.
    """
    def test_bad_custom_grid(self):
        """Does the custom grid raise an error when either x or y (not both) is missing?
        """
        with pytest.raises(ValueError, match="A custom grid will only be generated if 'site_x_coords', 'site_y_coords',*"):
            rsa_config = RsaConfig(str(Path(__file__).parent / "test_data" / "config_test_periodic.json"))
            run_simulation(rsa_config=rsa_config, site_x_coords=[1.1])
    def test_bad_xcoords(self):
        """Raise an error when the x coordinates are negative. Same effect as the y coordinates.
        """
        with pytest.raises(ValueError, match="Site x coordinates must be positive.*"):
            rsa_config = RsaConfig(str(Path(__file__).parent / "test_data" / "config_test_periodic.json"))
            run_simulation(rsa_config=rsa_config, site_x_coords=[-1.1], site_y_coords=[-2, 1.1], bounding_x_coord=-10, bounding_y_coord=-10)
    def test_good_custom_grid(self):
        """Does the custom grid generate correctly without errors?
        """
        rsa_config = RsaConfig(str(Path(__file__).parent / "test_data" / "config_test_periodic.json"))
        run_simulation(rsa_config=rsa_config, site_x_coords=0.9 * np.repeat(np.arange(10),10), site_y_coords=0.9 * np.tile(np.arange(10), 10), bounding_x_coord=10, bounding_y_coord=10)

def test_sequential_fluxreject():
    """Does the code handle flux-based simulation correctly?
    """
    rsa_config = RsaConfig(str(Path(__file__).parent / "test_data" / "config_test_periodic.json"))
    *_, flux, _ = run_simulation(rsa_config=rsa_config, simulation_type="sequential", include_rejected_flux=True)
    assert flux is not None  # The flux return value has to exist.
    assert isinstance(flux, np.ndarray)  # It should be a numpy array.
    assert flux.size  # The numpy array should not be empty.

def test__select_and_run():
    """When a bad simulation type/dosing scheme is selected, an error should be raised.
    """
    bad_simulation_type = "potato"
    flux = True
    with pytest.raises(ValueError, match=f"Simulation type {bad_simulation_type} with rejected_flux = {flux} is not supported."):
        _select_and_run(None, None, None, bad_simulation_type, flux, None, None)

def test_wrong_stickingprobability():
    """When the wrong sticking probability type is used, an error should be raised.
    """
    rsa_config = RsaConfig(str(Path(__file__).parent / "test_data" / "config_test_periodic.json"))
    with pytest.raises(TypeError, match="sticking_probability must be a float, list, or np.ndarray"):
        run_simulation(rsa_config=rsa_config, simulation_type="sequential", sticking_probability="one")
