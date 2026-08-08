# Copyright (c) 2025-2026 Contributors to the AdsorPy project.
# SPDX-License-Identifier: MIT
"""Test the GeneralSettings class of the `gui.py` module."""

from typing import ParamSpec
from unittest.mock import MagicMock

import numpy as np
import pytest
from pytestqt.qtbot import QtBot
from shapely import Polygon

from adsorpy.gui import (
    AdsorpyGUI,
    GeneralSettings,
    PydanticPolygon,
    SurfaceParameters,
)

P = ParamSpec("P")


# Define a real dummy function to inspect with different signature criteria
def run_simulation(
    no_default_param: object,
    standard_param: str = "default_val",
    *_: P.args,  # type: ignore[valid-type]
    **__: P.kwargs,  # type: ignore[valid-type]
) -> None:
    """Pretend to run a simulation with standard parameters.

    :param no_default_param: Parameter without a default value.
    :param standard_param: Standard parameter to use with a default value.
    """


def test_get_run_sim_default_success() -> None:
    """Verify successful retrieval of an existing parameter's default value."""
    mol_list_default = None
    result = GeneralSettings.get_run_sim_default("molecules_list")
    assert result is mol_list_default


def test_get_run_sim_default_missing_param_raises_key_error() -> None:
    """Verify that looking up a ghost parameter name raises a KeyError."""
    with pytest.raises(KeyError):
        GeneralSettings.get_run_sim_default("non_existent_parameter_name")


def test_prepare_simulation_inputs_success(qtbot: QtBot) -> None:
    """Verify AdsorpyGUI builds simulation dictionaries and extracts Polygon classes.

    :param qtbot: Simulates user input.
    """
    # Bind validation functions to the target file execution namespace
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

    assert result["seed"] == correct_seed  # pyright: ignore[reportTypedDictNotRequiredAccess]
    assert result["timestep_limit"] == timestep_limit  # pyright: ignore[reportTypedDictNotRequiredAccess]
    assert result["lattice_type"] == lattice_type  # pyright: ignore[reportTypedDictNotRequiredAccess]

    # Verify key mapping transitions and polygon resolution loops
    assert len(result["molecules_list"]) == 1  # pyright: ignore[reportTypedDictNotRequiredAccess, reportArgumentType]
    extracted_shape = result["molecules_list"][0]  # pyright: ignore[reportTypedDictNotRequiredAccess, reportIndexIssue, reportOptionalSubscript]
    assert isinstance(extracted_shape, Polygon)
    assert np.array_equal(extracted_shape.exterior.xy, np.array(coords).T)

    assert result["reflection_symmetries"] == [False]  # pyright: ignore[reportTypedDictNotRequiredAccess]
    assert result["rotation_symmetries"] == [True]  # pyright: ignore[reportTypedDictNotRequiredAccess]
    assert result["rotation_counts"] == [6]  # pyright: ignore[reportTypedDictNotRequiredAccess]


def test_run_simulation(qtbot: QtBot) -> None:
    """Test the run simulation function of the GUI.

    param: qtbot: Simulates user input.
    """
    gui = AdsorpyGUI()
    qtbot.addWidget(gui)

    gui.state = MagicMock()
    general_settings = GeneralSettings(gui.state)
    # The next line is required because the code will never finish otherwise. Magic mock causes issues.
    gui.state.surface_params = SurfaceParameters({"lattice_type": "hexagonal", "site_count": 10})
    coords = [[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]
    gui.state.molecule_param_list = [
        {
            "polygon": PydanticPolygon(Polygon(coords)),
            "refl_sym": False,
            "rot_sym": True,
            "rot_cnt": 6,
        },
    ]

    general_settings._on_simulation_complete = MagicMock()
    general_settings.run_simulation()
    qtbot.waitUntil(lambda: general_settings._on_simulation_complete.called, timeout=5000)  # pyright: ignore[reportUnknownLambdaType, reportAttributeAccessIssue]
    general_settings._on_simulation_complete.assert_called_once()


# def test_run_batch_simulation(qtbot: QtBot, monkeypatch: MonkeyPatch) -> None:
#     """Test the run batch simulation function of the GUI.
#
#     :param qtbot: Simulates user input.
#     """
#     gui = AdsorpyGUI()
#     qtbot.addWidget(gui)
#
#     gui.state = MagicMock()
#     general_settings = GeneralSettings(gui.state)
#     # The next line is required because the code will never finish otherwise. Magic mock causes issues.
#     gui.state.surface_params = SurfaceParameters({"lattice_type": "hexagonal", "site_count": 10})
#     coords = [[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]
#     gui.state.molecule_param_list = [
#         {
#             "polygon": PydanticPolygon(Polygon(coords)),
#             "refl_sym": False,
#             "rot_sym": True,
#             "rot_cnt": 6,
#         },
#     ]
#     general_settings.repeat_count.setValue(1)
#
#     general_settings._on_batch_simulation_complete = MagicMock()
#
#     class MockDaskClient:
#         def __init__(self, *args, **kwargs):
#             pass
#
#         def __enter__(self):
#             return self
#
#         def __exit__(self, exc_type, exc_val, exc_tb):
#             pass
#
#         def compute(self, tasks):
#             mock_future = MagicMock()
#             temp_simulation_result = (0.75, 0.65, [0.05, 0, 1])
#             mock_future.result.return_value = temp_simulation_result
#             return [mock_future]
#
#         def gather(self, futures):
#             temp_simulation_result = (0.75, 0.65, [0.05, 0, 1])
#             return [temp_simulation_result]
#
#     monkeypatch.setattr(adsorpy.gui, "Client", MockDaskClient)
#
#     general_settings.run_batch_simulation()
#     qtbot.waitUntil(
#         lambda: general_settings._on_batch_simulation_complete.called,
#         timeout=10000,
#         )
#     general_settings._on_batch_simulation_complete.assert_called_once()
