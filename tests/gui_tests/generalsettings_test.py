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
from PySide6.QtWidgets import QMessageBox
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


def test_on_simulation_complete(qtbot: QtBot, subtests: pytest.Subtests) -> None:
    """Test the on_simulation_complete function of the GUI.

    :param qtbot: Simulates user input.
    :param subtests: Subtest method to treat assertions as separate tests.
    """
    gui = AdsorpyGUI()
    qtbot.addWidget(gui)

    general_settings = GeneralSettings(gui.state)
    general_settings.show()  # This parent must be made visible first.

    assert not general_settings.coverage_label.isVisible(), "Label should be invisible initially."
    assert not general_settings.covered_area_label.isVisible()
    assert not general_settings.progress_bar.isVisible(), "Progress bar must start invisible."

    general_settings.progress_bar.show()

    mock_coverage = [0.5]
    mock_fraction = [0.3]

    def mock_plot(*_: P.args, **__: P.kwargs) -> None:  # type: ignore[valid-type]
        """Mock plot function."""

    def mock_analyse_gap_size(*_: P.args, **__: P.kwargs) -> None:  # type: ignore[valid-type]
        """Mock analyse gap size function."""

    class MockSimulator:
        """Mock Simulator class."""

        def __init__(self) -> None:
            """Initialise mock Simulator."""
            self.coverage = mock_coverage
            self.fraction_of_covered_area = mock_fraction
            self.svgplot_covered_grid = mock_plot
            self.analyse_gap_size = mock_analyse_gap_size

    mock_output = (MockSimulator(),)

    general_settings._on_simulation_complete(mock_output)  # pyright: ignore[reportArgumentType]

    with subtests.test(msg="Coverage label text"):
        assert general_settings.coverage_label.text() == f"Coverage: {np.sum(mock_coverage):.4f}"

    with subtests.test(msg="Covered area label text"):
        assert general_settings.covered_area_label.text() == f"Fraction of covered area: {np.sum(mock_fraction):.4f}"

    # Test widget visibility
    with subtests.test(msg="Coverage label visibility"):
        assert general_settings.coverage_label.isVisible(), "Coverage label must be visible."

    with subtests.test(msg="Covered area label visibility"):
        assert general_settings.covered_area_label.isVisible(), "Frac. cov. area label must be visible."

    with subtests.test(msg="Progress bar final visibility"):
        assert not general_settings.progress_bar.isVisible(), (
            "Progress bar must hide when finished (``finally`` block)."
        )

    # Test progress bar value
    with subtests.test(msg="Progress bar final value"):
        complete = 100
        assert general_settings.progress_bar.value() == complete, "Value should be 100 when complete."


def test_on_simulation_complete_error(qtbot: QtBot, subtests: pytest.Subtests, monkeypatch: MonkeyPatch) -> None:
    """Test the on_simulation_complete function of the GUI.

    :param qtbot: Simulates user input.
    :param subtests: Subtest method to treat assertions as separate tests.
    :param monkeypatch: MonkeyPatch object to mock attributes.
    """
    gui = AdsorpyGUI()
    qtbot.addWidget(gui)

    general_settings = GeneralSettings(gui.state)
    general_settings.show()  # This parent must be made visible first.

    assert not general_settings.coverage_label.isVisible(), "Label should be invisible initially."
    assert not general_settings.covered_area_label.isVisible()
    assert not general_settings.progress_bar.isVisible(), "Progress bar must start invisible."

    general_settings.progress_bar.show()

    mock_coverage = ["Error"]
    mock_fraction = ["Also wrong"]

    def mock_plot(*_: P.args, **__: P.kwargs) -> None:  # type: ignore[valid-type]
        """Mock plot function."""

    def mock_analyse_gap_size(*_: P.args, **__: P.kwargs) -> None:  # type: ignore[valid-type]
        """Mock analyse gap size function."""

    class MockSimulator:
        """Mock Simulator class."""

        def __init__(self) -> None:
            """Initialise mock Simulator."""
            self.coverage = mock_coverage
            self.fraction_of_covered_area = mock_fraction
            self.svgplot_covered_grid = mock_plot
            self.analyse_gap_size = mock_analyse_gap_size

    mock_output = (MockSimulator(),)

    error_called = False

    def mock_error(_: Exception) -> None:
        """Mock the error function but return nothing.

        Sets a flag denoting the calling of the function.

        :param _: Input error.
        """
        nonlocal error_called
        error_called = True

    monkeypatch.setattr(general_settings, "error", mock_error)
    general_settings._on_simulation_complete(mock_output)  # pyright: ignore[reportArgumentType]

    assert error_called, "The error function should have been called."
    assert not general_settings.progress_bar.isVisible(), "Progress bar must start invisible."


def test_on_simulation_error(qtbot: QtBot, monkeypatch: MonkeyPatch) -> None:
    """Test the on_simulation_error function of the GUI.

    :param qtbot: Simulates user input.
    :param monkeypatch: MonkeyPatch object to mock attributes.
    """
    gui = AdsorpyGUI()
    qtbot.addWidget(gui)
    general_settings = GeneralSettings(gui.state)

    error_called = False

    def mock_error(_: Exception) -> None:
        """Mock the error function but return nothing.

        Sets a flag denoting the calling of the function.

        :param _: Input error.
        """
        nonlocal error_called
        error_called = True

    monkeypatch.setattr(general_settings, "error", mock_error)
    general_settings.run_group.setDisabled(True)
    assert not general_settings.run_group.isEnabled(), "Sanity check: this should be set as disabled."

    test_exception = Exception()
    general_settings._on_simulation_error(test_exception)
    assert error_called, "The error function has to have been called."
    assert general_settings.run_group.isEnabled(), "This should have been re-enabled."


def test_export_results(qtbot: QtBot, subtests: pytest.Subtests, monkeypatch: MonkeyPatch) -> None:
    """Test the export_results function.

    :param qtbot: Simulates user input.
    :param subtests: Subtest method to treat assertions as separate tests.
    :param monkeypatch: MonkeyPatch object to mock attributes.
    """
    gui = AdsorpyGUI()
    qtbot.addWidget(gui)
    general_settings = GeneralSettings(gui.state)

    warning_called = False

    def mock_warning(*_: P.args, **__: P.kwargs) -> None:  # type: ignore[valid-type]
        """Mock warning function.

        :param _: Input args.
        :param __: Input kwargs.
        """
        nonlocal warning_called
        warning_called = True

    monkeypatch.setattr(QMessageBox, "warning", mock_warning)

    general_settings.export_results()
    assert warning_called, "The warning function should have been called."
