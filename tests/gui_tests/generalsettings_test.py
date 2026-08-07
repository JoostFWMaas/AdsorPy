from unittest.mock import MagicMock

import numpy as np
import pytest
from shapely import Polygon

from adsorpy.gui import (
    AdsorpyGUI,
    GeneralSettings,
    PydanticPolygon,
    SurfaceParameters,
    validate_polygon,
)


# Define a real dummy function to inspect with different signature criteria
def run_simulation(standard_param="default_val", no_default_param=None, *_, **__) -> None:
    pass


def test_get_run_sim_default_success():
    """Verify successful retrieval of an existing parameter's default value."""
    mol_list_default = None
    result = GeneralSettings.get_run_sim_default("molecules_list")
    assert result is mol_list_default


def test_get_run_sim_default_missing_param_raises_key_error():
    """Verify that looking up a ghost parameter name raises a KeyError."""
    with pytest.raises(KeyError):
        GeneralSettings.get_run_sim_default("non_existent_parameter_name")


def test_prepare_simulation_inputs_success(monkeypatch, qtbot):
    """Verify AdsorpyGUI builds simulation dictionaries and extracts Polygon classes."""
    # Bind validation functions to the target file execution namespace
    monkeypatch.setattr("adsorpy.gui.validate_polygon", validate_polygon)

    gui = AdsorpyGUI()
    qtbot.addWidget(gui)

    gui.state = MagicMock()
    general_settings = GeneralSettings(gui.state)
    correct_seed = 77777
    gui.state.seed_input.setText(str(correct_seed))
    timestep_limit = 2500
    gui.state.step_limit.setValue(timestep_limit)
    lattice_type = "square"
    gui.state.surface_params = SurfaceParameters({"seed": 12, "lattice_type": lattice_type})
    coords = [[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]
    gui.state.molecule_param_list = [
        {
            "polygon": PydanticPolygon(Polygon(coords)),
            "refl_sym": False,
            "rot_sym": True,
            "rot_cnt": 6,
        },
    ]

    result = general_settings._prepare_simulation_inputs()

    assert result["seed"] == correct_seed
    assert result["timestep_limit"] == timestep_limit
    assert result["lattice_type"] == lattice_type

    # Verify key mapping transitions and polygon resolution loops
    assert len(result["molecules_list"]) == 1
    extracted_shape = result["molecules_list"][0]
    assert isinstance(extracted_shape, Polygon)
    assert np.array_equal(extracted_shape.exterior.xy, np.array(coords).T)

    assert result["reflection_symmetries"] == [False]
    assert result["rotation_symmetries"] == [True]
    assert result["rotation_counts"] == [6]
