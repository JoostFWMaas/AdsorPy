# Copyright (c) 2025-2026 Contributors to the AdsorPy project.
# SPDX-License-Identifier: MIT
"""Test the GeneralSettings class of the `gui.py` module."""
# import sys
from typing import ParamSpec

# from dask import distributed
#
# if sys.version_info >= (3, 11):
#     from typing import Self
# else:
#     from typing_extensions import Self
from unittest.mock import MagicMock

import numpy as np
import pytest
from _pytest.monkeypatch import MonkeyPatch
from PySide6.QtCore import QRunnable, QThreadPool
from pytestqt.qtbot import QtBot
from shapely import Point, Polygon

from adsorpy.gui import (
    AdsorpyGUI,
    GeneralSettings,
    PydanticPolygon,
    SurfaceParameters,
)

P = ParamSpec("P")


# Define a mock function to inspect with different signature criteria.
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
            "rot_sym": 0,
            "rot_cnt": 6,
            "index": 0,
            "function_name": "",
            "label": "",
            "settings": {},
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
    assert result["rotation_symmetries"] == [0]  # pyright: ignore[reportTypedDictNotRequiredAccess]
    assert result["rotation_counts"] == [6]  # pyright: ignore[reportTypedDictNotRequiredAccess]


@pytest.mark.parametrize("iterate", range(3))  # Left in because this test was flakey.
def test_run_simulation(iterate: int, qtbot: QtBot, monkeypatch: MonkeyPatch) -> None:
    """Test the run simulation function of the GUI.

    param: qtbot: Simulates user input.
    """
    gui = AdsorpyGUI()
    qtbot.addWidget(gui)

    gui.state = MagicMock()
    general_settings = GeneralSettings(gui.state)
    # The next line is required because the code will never finish otherwise. Magic mock causes issues.
    gui.state.surface_params = SurfaceParameters({"lattice_type": "hexagonal", "site_count": 10})
    gui.state.molecule_param_list = [
        {
            "polygon": PydanticPolygon(Point((0, 0)).buffer(1.5)),
            "refl_sym": True,
            "rot_sym": 0,
            "rot_cnt": 1,
            "index": 0,
            "function_name": "",
            "label": "",
            "settings": {},
        },
    ]

    general_settings._on_simulation_complete = MagicMock()

    def run_synchronously(task: QRunnable) -> None:
        """Execute the background task immediately on the main test thread.

        Bypasses the QThreadPool asynchronous execution queue to keep the test single-threaded and deterministic.

        :param task: The background task instance waiting to be scheduled.
        """
        task.run()

    pool = QThreadPool.globalInstance()
    monkeypatch.setattr(pool, "start", run_synchronously)
    general_settings.run_simulation()
    general_settings._on_simulation_complete.assert_called_once()


# @pytest.mark.parametrize("iterate", range(3))
# def test_run_batch_simulation(iterate: int, qtbot: QtBot, monkeypatch: MonkeyPatch) -> None:
#     """Test the run batch simulation function of the GUI.
#
#     :param qtbot: Simulates user input.
#     :param monkeypatch: Pytest fixture used to intercept asynchronous dependencies.
#     """
#     gui = AdsorpyGUI()
#     qtbot.addWidget(gui)
#
#     gui.state = MagicMock()
#     general_settings = GeneralSettings(gui.state)
#
#     gui.state.surface_params = SurfaceParameters({"lattice_type": "hexagonal", "site_count": 10})
#     gui.state.molecule_param_list = [
#         {
#             "polygon": PydanticPolygon(Point((0, 0)).buffer(1.5)),
#             "refl_sym": True,
#             "rot_sym": 0,
#             "rot_cnt": 1,
#             "index": 0,
#             "function_name": "",
#             "label": "",
#             "settings": {},
#         },
#     ]
#     general_settings.repeat_count.setValue(1)
#
#     general_settings._on_batch_simulation_complete = MagicMock()
#
#     class MockDaskClient:
#         """A synchronous replacement for the distributed Dask Client.
#
#         Bypasses asynchronous background task scheduling to compute and gather
#         simulation matrices sequentially on the main thread.
#         """
#
#         def __init__(self, *args: P.args, **kwargs: P.kwargs) -> None:
#             """Initialise the mock client interface."""
#
#         def __enter__(self) -> Self:
#             """Return the context-managed instance block."""
#             return self
#
#         def __exit__(
#                 self,
#                 exc_type: type[BaseException] | None,
#                 exc_val: BaseException | None,
#                 exc_tb: object,
#         ) -> None:
#             """Handle safe scope destruction."""
#
#         def compute(self, tasks: object) -> list[MagicMock]:
#             """Mock the Dask future processing pipeline.
#
#             :param tasks: Uncomputed collection of graph execution paths.
#             :return: A list containing a resolved mock future wrapper.
#             """
#             mock_future = MagicMock()
#             temp_simulation_result: tuple[float, float, list[float]] = (0.75, 0.65, [0.05, 0.0, 1.0])
#             mock_future.result.return_value = temp_simulation_result
#             return [mock_future]
#
#         def gather(self, futures: list[object]) -> list[tuple[float, float, list[float]]]:
#             """Instantly return the requested data arrays on the calling thread.
#
#             :param futures: Managed tracker references.
#             :return: Formatted array tuple containing mock simulation metrics.
#             """
#             temp_simulation_result: tuple[float, float, list[float]] = (0.75, 0.65, [0.05, 0.0, 1.0])
#             return [temp_simulation_result]
#
#     def run_synchronously(task: QRunnable) -> None:
#         """Execute the background task immediately on the main test thread.
#
#         Bypasses the QThreadPool asynchronous queue to keep the test environment single-threaded and deterministic.
#
#         :param task: The background task instance waiting to be scheduled.
#         """
#         task.run()
#
#     def mock_as_completed(futures: list[object]) -> list[object]:
#         """Instantly pass the list of mock futures straight through without blocking.
#
#         :param futures: Collection of input mock futures.
#         :return: Handled tracker collection list.
#         """
#         return futures
#
#     # Intercept the target module with the synchronous mock class
#     monkeypatch.setattr(distributed, "Client", MockDaskClient)
#     monkeypatch.setattr(distributed, "as_completed", mock_as_completed)
#
#     pool: QThreadPool = QThreadPool.globalInstance()
#     monkeypatch.setattr(pool, "start", run_synchronously)
#
#     # Executes linearly triggers the mock callback before returning
#     general_settings.run_batch_simulation()
#
#     # Assert validation states safely without any asynchronous event loops
#     general_settings._on_batch_simulation_complete.assert_called_once()
