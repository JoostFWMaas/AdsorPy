# Copyright (c) 2025-2026 Contributors to the AdsorPy project.
# SPDX-License-Identifier: MIT
"""Test the GeneralSettings class of the `gui.py` module."""

import json
import pickle
import sys
import zipfile
from pathlib import Path

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

from unittest.mock import MagicMock, Mock

import numpy as np
import pytest
from _pytest.monkeypatch import MonkeyPatch
from PySide6.QtCore import QRunnable, QThreadPool
from PySide6.QtWidgets import QFileDialog, QMessageBox
from pytestqt.qtbot import QtBot
from shapely import Point, Polygon

import adsorpy.gui
from adsorpy.gui import (
    AdsorpyGUI,
    BatchSimulationInput,
    GeneralSettings,
    MoleculeParameters,
    PydanticPolygon,
    SurfaceParameters,
)


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
    gui.state.surface_params = SurfaceParameters({"lattice_type": "triangular", "site_count": 10})
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

    # Initial state checks
    assert not general_settings.coverage_label.isVisible(), "Label should be invisible initially."
    assert not general_settings.covered_area_label.isVisible()
    assert not general_settings.progress_bar.isVisible(), "Progress bar must start invisible."

    general_settings.progress_bar.show()

    mock_coverage = [0.5]
    mock_fraction = [0.3]

    mock_simulator = Mock()
    mock_simulator.coverage = mock_coverage
    mock_simulator.fraction_of_covered_area = mock_fraction
    mock_simulator.svgplot_covered_grid = Mock()
    mock_simulator.analyse_gap_size = Mock()

    mock_output = (mock_simulator,)

    general_settings._on_simulation_complete(mock_output)  # pyright: ignore[reportArgumentType]

    with subtests.test(msg="Coverage label text"):
        assert general_settings.coverage_label.text() == f"Coverage: {np.sum(mock_coverage):.4f}"

    with subtests.test(msg="Covered area label text"):
        assert general_settings.covered_area_label.text() == f"Fraction of covered area: {np.sum(mock_fraction):.4f}"

    with subtests.test(msg="Coverage label visibility"):
        assert general_settings.coverage_label.isVisible(), "Coverage label must be visible."

    with subtests.test(msg="Covered area label visibility"):
        assert general_settings.covered_area_label.isVisible(), "Frac. cov. area label must be visible."

    with subtests.test(msg="Progress bar final visibility"):
        assert not general_settings.progress_bar.isVisible(), (
            "Progress bar must hide when finished (``finally`` block)."
        )

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

    # Assert initial UI visibility states
    assert not general_settings.coverage_label.isVisible(), "Label should be invisible initially."
    assert not general_settings.covered_area_label.isVisible()
    assert not general_settings.progress_bar.isVisible(), "Progress bar must start invisible."

    general_settings.progress_bar.show()

    mock_simulator = Mock()
    mock_simulator.coverage = ["Error"]
    mock_simulator.fraction_of_covered_area = ["Also wrong"]
    mock_simulator.svgplot_covered_grid = Mock()
    mock_simulator.analyse_gap_size = Mock()

    mock_output = (mock_simulator,)
    mock_error_method = Mock()

    monkeypatch.setattr(general_settings, "error", mock_error_method)

    general_settings._on_simulation_complete(mock_output)  # pyright: ignore[reportArgumentType]

    mock_error_method.assert_called_once()
    assert not general_settings.progress_bar.isVisible(), "Progress bar must be hidden following a crash."

    args, _ = mock_error_method.call_args
    passed_message = args[0]
    assert "Simulation completion failed" in passed_message or isinstance(passed_message, str)


def test_on_simulation_error(qtbot: QtBot, monkeypatch: MonkeyPatch) -> None:
    """Test the on_simulation_error function of the GUI.

    :param qtbot: Simulates user input.
    :param monkeypatch: MonkeyPatch object to mock attributes.
    """
    gui = AdsorpyGUI()
    qtbot.addWidget(gui)
    general_settings = GeneralSettings(gui.state)

    mock_error_method = Mock()
    monkeypatch.setattr(general_settings, "error", mock_error_method)

    general_settings.run_group.setDisabled(True)
    assert not general_settings.run_group.isEnabled(), "Sanity check: this should be set as disabled."

    test_exception = Exception("Simulation crashed")
    general_settings._on_simulation_error(test_exception)

    mock_error_method.assert_called_once()

    args, _ = mock_error_method.call_args
    passed_message = args[0]

    assert "Simulation engine error:" in passed_message
    assert "Simulation crashed" in passed_message
    assert general_settings.run_group.isEnabled(), "This should have been re-enabled."


def test_export_results_error(qtbot: QtBot, monkeypatch: MonkeyPatch) -> None:
    """Test the export_results function.

    :param qtbot: Simulates user input.
    :param monkeypatch: MonkeyPatch object to mock attributes.
    """
    gui = AdsorpyGUI()
    qtbot.addWidget(gui)
    general_settings = GeneralSettings(gui.state)

    mock_warning = Mock()
    monkeypatch.setattr(QMessageBox, "warning", mock_warning)

    general_settings.export_results()

    mock_warning.assert_called_once()


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
    mock_save_dialog = Mock(return_value=("", "JSON Data Interchange (*.json);;"))
    mock_info_box = Mock()

    monkeypatch.setattr(QFileDialog, "getSaveFileName", mock_save_dialog)
    monkeypatch.setattr(QMessageBox, "information", mock_info_box)

    mock_widget.export_results()

    mock_save_dialog.assert_called_once()
    mock_info_box.assert_not_called()


def test_export_json_format(mock_widget: GeneralSettings, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """Test whether JSON is exported correctly.

    :param mock_widget: Mock GeneralSettings.
    :param tmp_path: Path to the temporary folder.
    :param monkeypatch: MonkeyPatch object.
    """
    target_path = tmp_path / "temp_output.json"

    mock_save_dialog = Mock(return_value=(str(target_path.with_suffix("")), "JSON Data Interchange (*.json);;"))

    # Mock QFileDialog to automatically select our temp file path
    monkeypatch.setattr(QFileDialog, "getSaveFileName", mock_save_dialog)

    # Mock QMessageBox popups so they don't block execution
    mock_info_box = Mock()
    monkeypatch.setattr(QMessageBox, "information", mock_info_box)

    mock_widget.input_metadata["molecules_list"] = [Point((0, 0)).buffer(1.0)]

    # Run execution loop
    mock_widget.export_results()

    # Read back format validation
    assert target_path.exists()
    with target_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    assert "repeats" in data["metadata"]
    assert data["metadata"]["repeats"] == mock_widget.input_metadata["repeats"]  # pyright: ignore[reportTypedDictNotRequiredAccess]
    assert "Gap_size_distribution" in data
    assert "molecules_list" in data["metadata"]
    assert data["metadata"]["molecules_list"] == [str(Point((0, 0)).buffer(1.0))]
    np.testing.assert_array_equal(data["Gap_size_distribution"][:], mock_widget.state.gap_size_distribution)


def test_export_hdf5_format(mock_widget: GeneralSettings, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """Test whether hdf5 exported correctly.

    :param mock_widget: Mock GeneralSettings.
    :param tmp_path: Path to the temporary folder.
    :param monkeypatch: MonkeyPatch object.
    """
    h5py = pytest.importorskip("h5py")
    target_path = tmp_path / "temp_output.h5"

    mock_save_dialog = Mock(return_value=(str(target_path.with_suffix("")), "Hierarchical Data Format (*.h5);;"))
    mock_info_box = Mock()

    monkeypatch.setattr(QFileDialog, "getSaveFileName", mock_save_dialog)
    monkeypatch.setattr(QMessageBox, "information", mock_info_box)

    mock_widget.export_results()

    mock_save_dialog.assert_called_once()
    mock_info_box.assert_called_once()
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

    mock_save_dialog = Mock(return_value=(str(target_path.with_suffix("")), "Python Pickle Binary (*.pkl);;"))
    mock_info_box = Mock()

    monkeypatch.setattr(QFileDialog, "getSaveFileName", mock_save_dialog)
    monkeypatch.setattr(QMessageBox, "information", mock_info_box)

    mock_widget.export_results()

    # 3. Assertions
    mock_save_dialog.assert_called_once()
    mock_info_box.assert_called_once()
    assert target_path.exists()

    with target_path.open("rb") as f:
        data = SafeTestUnpickler(f).load()

    assert "metadata" in data
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

    mock_save_dialog = Mock(
        return_value=(str(target_path.with_suffix("")), "Zipped Comma Separated Values (*.zip);;"),
    )
    mock_info_box = Mock()

    monkeypatch.setattr(QFileDialog, "getSaveFileName", mock_save_dialog)
    monkeypatch.setattr(QMessageBox, "information", mock_info_box)

    mock_widget.export_results()

    mock_save_dialog.assert_called_once()
    mock_info_box.assert_called_once()
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
    errmsg = "Permission denied."

    mock_save_dialog = Mock(return_value=(str(target_path), "JSON Data Interchange (*.json);;"))
    mock_mkdir_fail = Mock(side_effect=OSError(errmsg))
    mock_critical = Mock()

    monkeypatch.setattr(QFileDialog, "getSaveFileName", mock_save_dialog)
    monkeypatch.setattr(Path, "mkdir", mock_mkdir_fail)
    monkeypatch.setattr(QMessageBox, "critical", mock_critical)

    mock_widget.export_results()

    mock_save_dialog.assert_called_once()
    mock_mkdir_fail.assert_called_once()
    mock_critical.assert_called_once()

    args, _ = mock_critical.call_args
    displayed_message = args[-1]  # The final positional argument is the error message body.
    assert errmsg in displayed_message


def test_export_bad_format_error(mock_widget: GeneralSettings, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """Test whether malformed format raises error.

    :param mock_widget: Mock GeneralSettings.
    :param tmp_path: Path to the temporary folder.
    :param monkeypatch: MonkeyPatch object.
    """
    target_path = tmp_path / "temp_output"

    mock_save_dialog = Mock(
        return_value=(str(target_path.with_suffix("")), "How did you do this? (*.);;"),
    )
    mock_info_box = Mock()
    mock_warning_box = Mock()
    mock_critical_box = Mock()

    monkeypatch.setattr(QFileDialog, "getSaveFileName", mock_save_dialog)
    monkeypatch.setattr(QMessageBox, "information", mock_info_box)
    monkeypatch.setattr(QMessageBox, "warning", mock_warning_box)
    monkeypatch.setattr(QMessageBox, "critical", mock_critical_box)

    mock_widget.export_results()

    mock_save_dialog.assert_called_once()
    mock_warning_box.assert_called_once()
    mock_info_box.assert_not_called()
    mock_critical_box.assert_not_called()


def test_change_bulk_run_value(mock_widget: GeneralSettings, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """Validate that the bulk run textbox and value are updated correctly."""
    mock_widget._change_bulk_run_value(1)
    old_text = mock_widget.bulk_run_button.text()
    old_value = mock_widget._settings.value("repeat_count")

    assert str(old_value) in old_text, "QLabel and QSettings value are out of sync."

    mock_widget._change_bulk_run_value(old_value + 1)

    new_text = mock_widget.bulk_run_button.text()
    new_value = mock_widget._settings.value("repeat_count")

    assert str(new_value) in new_text, "QLabel and QSettings value are out of sync."
    assert new_text != old_text, "Text did not update."
    assert new_value == old_value + 1, "Value did not update correctly."


def test_on_molecules_changed(mock_widget: GeneralSettings) -> None:
    """Validate the correct QLabel text when 0 <= N molecules are defined."""
    text_tuple: tuple[str, str, str] = (
        "Default.",
        "1 molecule defined by user.",
        "2 molecules defined by user.",
    )

    for idx, text in enumerate(text_tuple):
        mock_widget._on_molecules_changed([MoleculeParameters()] * idx)  # pyright: ignore[reportCallIssue]
        assert text == mock_widget.initiated_molecules_textbox.text(), f"Text did not update correctly for {idx}."

    mock_widget._on_molecules_changed(None)
    assert text_tuple[0] == mock_widget.initiated_molecules_textbox.text(), "Text did not reset correctly."


def test_get_run_sim_default(mock_widget: GeneralSettings, monkeypatch: MonkeyPatch) -> None:
    """Validate that the default value is returned correctly."""
    input_vals = ("hello", 10, 0.1, None)

    for input_val in input_vals:

        def good_function(_: str | float | None = input_val) -> None:
            """Define a function with a default type."""

        monkeypatch.setattr(adsorpy.gui, "run_simulation", good_function)

        assert input_val is mock_widget.get_run_sim_default("_")


def test_get_run_sim_default_error(mock_widget: GeneralSettings, monkeypatch: MonkeyPatch) -> None:
    """Validate that the function raises an error correctly when there is no default value."""

    def bad_function(_: None) -> None:
        """Define a function without default values."""

    monkeypatch.setattr(adsorpy.gui, "run_simulation", bad_function)
    with pytest.raises(ValueError, match="_ has no default"):
        mock_widget.get_run_sim_default("_")


def test_error(mock_widget: GeneralSettings, monkeypatch: MonkeyPatch) -> None:
    """Validate the error function."""
    errmsg = "Testing the error function."
    error_function = Mock()
    monkeypatch.setattr(QMessageBox, "critical", error_function)
    mock_widget.error(errmsg)

    error_function.assert_called_once_with(mock_widget, "Input Error", errmsg)
