# Copyright (c) 2025-2026 Contributors to the AdsorPy project.
# SPDX-License-Identifier: MIT
"""Test the GeneralSettings class of the `gui.py` module."""

import json
import pickle
import sys
import zipfile
from pathlib import Path
from typing import ParamSpec

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

from unittest.mock import MagicMock

import numpy as np
import pytest
from _pytest.monkeypatch import MonkeyPatch
from PySide6.QtCore import QRunnable, QThreadPool
from PySide6.QtWidgets import QFileDialog, QMessageBox
from pytestqt.qtbot import QtBot
from shapely import Point, Polygon

from adsorpy.gui import (
    AdsorpyGUI,
    BatchSimulationInput,
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


def test_export_results_error(qtbot: QtBot, monkeypatch: MonkeyPatch) -> None:
    """Test the export_results function.

    :param qtbot: Simulates user input.
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
    assert warning_called, "The warning function should have been called because the gap size dist is empty."


@pytest.fixture
def mock_widget(qtbot: QtBot) -> GeneralSettings:
    """Mock widget function to generate mock widget for metadata and other info.

    :param qtbot: Simulates user input.
    :returns: Mocked GeneralSettings.
    """
    gui = AdsorpyGUI()
    qtbot.addWidget(gui)
    general_settings = GeneralSettings(gui.state)
    general_settings.state.gap_size_distribution = np.linspace(0, 1)
    general_settings.state.coverages = (np.arange(2), np.linspace(3, 7, num=2))  # pyright: ignore[reportAttributeAccessIssue]
    general_settings.state.fraction_of_covered_area = (np.arange(2), np.linspace(0, 3, num=2))  # pyright: ignore[reportAttributeAccessIssue]
    general_settings.input_metadata = BatchSimulationInput(repeats=10)
    return general_settings


def test_export_user_cancels_dialogue(mock_widget: GeneralSettings, monkeypatch: MonkeyPatch) -> None:
    """Test whether cancelled dialogue is handled correctly.

    :param mock_widget: Mock GeneralSettings.
    :param monkeypatch: MonkeyPatch object to mock attributes.
    """

    # Shorthand mock return value
    def mock_dialogue(*_: P.args, **__: P.kwargs) -> tuple[str, str]:  # type: ignore[valid-type]
        """Mock dialogue function.

        :param _: Input arguments, ignored.
        :param __: Input kwargs, ignored.
        """
        return "", "JSON Data Interchange (*.json);;"

    monkeypatch.setattr(QFileDialog, "getSaveFileName", mock_dialogue)

    # Track if QMessageBox.information was called
    info_called = False

    def mock_info(*_: P.args, **__: P.kwargs) -> None:  # type: ignore[valid-type]
        """Mock info function.

        :param _: Ignored args.
        :param __: Ignored kwargs.
        """
        nonlocal info_called
        info_called = True

    monkeypatch.setattr(QMessageBox, "information", mock_info)

    mock_widget.export_results()
    assert not info_called, "Should exit early without success message"


def test_export_json_format(mock_widget: GeneralSettings, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """Test whether JSON is exported correctly.

    :param mock_widget: Mock GeneralSettings.
    :param tmp_path: Path to the temporary folder.
    :param monkeypatch: MonkeyPatch object.
    """
    target_path = tmp_path / "temp_output.json"

    # Mock QFileDialog to automatically select our temp file path
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        lambda *_, **__: (str(target_path), "JSON Data Interchange (*.json);;"),  # pyright: ignore[reportUnknownLambdaType]
    )

    # Mock QMessageBox popups so they don't block execution
    monkeypatch.setattr(QMessageBox, "information", lambda *_, **__: None)  # pyright: ignore[reportUnknownLambdaType]

    # Run execution loop
    mock_widget.export_results()

    # Read back format validation
    assert target_path.exists()
    with target_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    assert "repeats" in data["metadata"]
    assert data["metadata"]["repeats"] == mock_widget.input_metadata["repeats"]  # pyright: ignore[reportTypedDictNotRequiredAccess]
    assert "Gap_size_distribution" in data
    np.testing.assert_array_equal(data["Gap_size_distribution"][:], mock_widget.state.gap_size_distribution)


def test_export_hdf5_format(mock_widget: GeneralSettings, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """Test whether hdf5 exported correctly.

    :param mock_widget: Mock GeneralSettings.
    :param tmp_path: Path to the temporary folder.
    :param monkeypatch: MonkeyPatch object.
    """
    h5py = pytest.importorskip("h5py")
    target_path = tmp_path / "temp_output.h5"

    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        lambda *_: (str(target_path), "Hierarchical Data Format (*.h5);;"),  # pyright: ignore[reportUnknownLambdaType]
    )
    monkeypatch.setattr(QMessageBox, "information", lambda *_: None)  # pyright: ignore[reportUnknownLambdaType]

    mock_widget.export_results()

    assert target_path.exists()
    with h5py.File(str(target_path), "r") as f:
        assert "repeats" in f.attrs
        assert f.attrs["repeats"] == str(mock_widget.input_metadata["repeats"])  # pyright: ignore[reportTypedDictNotRequiredAccess]
        assert "Gap_size_distribution" in f
        np.testing.assert_array_equal(f["Gap_size_distribution"][:], mock_widget.state.gap_size_distribution)


class SafeTestUnpickler(pickle.Unpickler):
    """Safely unpickle the test data."""

    @override
    def find_class(self, module: str, name: str) -> object:
        """Find the class in a module.

        :param module: The name of the module to find.
        :param name: The name of the class to find.
        :returns: The class to find.
        :raises pickle.UnpicklingError: If unable to unpickle a class safely.
        """
        if module == "builtins" or module.startswith("numpy"):
            __import__(module)
            return getattr(sys.modules[module], name)

        errmsg = f"Global '{module}.{name}' is forbidden in tests."
        raise pickle.UnpicklingError(errmsg)


def test_export_pickle_format(mock_widget: GeneralSettings, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """Test whether pickle is exported correctly.

    :param mock_widget: Mock GeneralSettings.
    :param tmp_path: Path to the temporary folder.
    :param monkeypatch: MonkeyPatch object.
    """
    target_path = tmp_path / "temp_output.pkl"

    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        lambda *_: (str(target_path), "Python Pickle Binary (*.pkl);;"),  # pyright: ignore[reportUnknownLambdaType]
    )
    monkeypatch.setattr(QMessageBox, "information", lambda *_: None)  # pyright: ignore[reportUnknownLambdaType]

    mock_widget.export_results()

    assert target_path.exists()
    with target_path.open("rb") as f:
        data = SafeTestUnpickler(f).load()

    assert "repeats" in data["metadata"]
    assert data["metadata"]["repeats"] == mock_widget.input_metadata["repeats"]  # pyright: ignore[reportTypedDictNotRequiredAccess]
    assert "Gap_size_distribution" in data
    np.testing.assert_array_equal(data["Gap_size_distribution"][:], mock_widget.state.gap_size_distribution)


def test_export_zip_csv_format(mock_widget: GeneralSettings, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """Test whether zipped csv is exported correctly.

    :param mock_widget: Mock GeneralSettings.
    :param tmp_path: Path to the temporary folder.
    :param monkeypatch: MonkeyPatch object.
    """
    target_path = tmp_path / "temp_output.zip"

    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        lambda *_: (str(target_path), "Zipped Comma Separated Values (*.zip);;"),  # pyright: ignore[reportUnknownLambdaType]
    )
    monkeypatch.setattr(QMessageBox, "information", lambda *_: None)  # pyright: ignore[reportUnknownLambdaType]

    mock_widget.export_results()

    assert target_path.exists()
    assert zipfile.is_zipfile(target_path)

    with zipfile.ZipFile(target_path, "r") as zf:
        files = zf.namelist()
        assert "metadata.txt" in files
        assert "Coverage.csv" in files
        assert "Gap_size_distribution.csv" in files

        # Read file out of memory safely to confirm structural formatting
        meta_txt = zf.read("metadata.txt").decode("utf-8")
        assert "repeats: 10" in meta_txt


def test_export_handles_io_exceptions(mock_widget: GeneralSettings, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """Test whether the export function handles exceptions correctly.

    :param mock_widget: Mock GeneralSettings.
    :param tmp_path: Path to the temporary folder.
    :param monkeypatch: MonkeyPatch object.
    """
    target_path = tmp_path / "forbidden_directory" / "temp_output.json"

    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        lambda *_: (str(target_path), "JSON Data Interchange (*.json);;"),  # pyright: ignore[reportUnknownLambdaType]
    )

    # Intercept Path.mkdir block to fake an OS-level permissions failure
    errmsg = "Permission denied."

    def mock_mkdir_fail(*_: P.args, **__: P.kwargs) -> None:  # type: ignore[valid-type]
        """Mock mkdir fail.

        :param _: Ignored args.
        :param _: Ignored kwargs.
        :raises OSError: Permission denied.
        """
        raise OSError(errmsg)

    monkeypatch.setattr(Path, "mkdir", mock_mkdir_fail)

    critical_messages = []
    monkeypatch.setattr(QMessageBox, "critical", lambda _parent, _title, msg: critical_messages.append(msg))  # pyright: ignore[reportUnknownLambdaType]

    mock_widget.export_results()

    assert len(critical_messages) == 1
    assert errmsg in critical_messages[0]
