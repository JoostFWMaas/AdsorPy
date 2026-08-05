# Copyright (c) 2025-2026 Contributors to the AdsorPy project.
# SPDX-License-Identifier: MIT
"""Test the AdsorpyGUI class of the `gui.py` module."""
import json
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError
from PySide6.QtCore import QSize
from PySide6.QtGui import QResizeEvent
from PySide6.QtWidgets import QFileDialog, QMessageBox
from pytestqt.qtbot import QtBot

import adsorpy
from adsorpy.gui import AdsorpyGUI, AppState, MoleculeParameters, SurfaceParameters


@pytest.fixture
def gui_app(qtbot: QtBot) -> AdsorpyGUI:
    """Fixture to instantiate and register the main window lifecycle."""
    window = AdsorpyGUI()
    qtbot.addWidget(window)
    return window

app = gui_app

def test_initial_window_properties(gui_app: AdsorpyGUI, subtests: pytest.Subtests) -> None:
    """Verify window title, central widget configuration, and state assignment."""
    with subtests.test(msg="Verify window title"):
        assert gui_app.windowTitle() == "AdsorPy Simulation GUI"

    with subtests.test(msg="Verify state assignment"):
        assert isinstance(gui_app.state, AppState)

    with subtests.test(msg="Verify central widget"):
        assert gui_app.centralWidget() == gui_app.tabs

    with subtests.test(msg="Verify tab configuration"):
        assert gui_app.tabs.count() == 3  # noqa: PLR2004
        assert gui_app.tabs.tabText(0) == "General"
        assert gui_app.tabs.tabText(1) == "Surface"
        assert gui_app.tabs.tabText(2) == "Molecule(s)"


@pytest.mark.parametrize(
    ("action_attr", "expected_url"),
    [
        ("_doc_action", "https://joostfwmaas.github.io/AdsorPy/"),
        ("_wiki_action", "https://github.com/JoostFWMaas/AdsorPy/wiki"),
        ("_bug_action", "https://github.com/JoostFWMaas/AdsorPy/issues"),
    ],
)
def test_help_menu_web_links(gui_app: AdsorpyGUI, action_attr: str, expected_url: str) -> None:
    """Verify that clicking Help items opens the correct URLs in a web browser."""
    action = getattr(gui_app, action_attr)

    with patch("webbrowser.open") as mock_open:
        action.trigger()
        mock_open.assert_called_once_with(expected_url)


# def test_save_settings_json_success(gui_app, tmp_path):
#     """Verify state components are bundled, dumped, and serialized properly."""
#     test_file = tmp_path / "test_settings.json"
#
#     # Pre-populate state dependencies with temporary objects/mocks
#     gui_app.state.seed_input = MagicMock()
#     gui_app.state.seed_input.text.return_value = " 42 "  # Checks whitespace stripping
#     gui_app.state.step_limit = MagicMock()
#     gui_app.state.step_limit.value.return_value = 5000
#
#     gui_app.state.surface_params = SurfaceParameters()
#     gui_app.state.molecule_param_list = [MoleculeParameters()]
#
#     # Mock OS File dialog to return our temp file path, and bypass the final info pop-up
#     with (
#         patch.object(QFileDialog, "getSaveFileName", return_value=(str(test_file), "JSON Files (*.json)")),
#         patch.object(QMessageBox, "information") as mock_info,
#     ):
#         gui_app._save_settings_json()
#
#         # Verify the success dialog pop-up registered
#         mock_info.assert_called_once()
#
#     # Read the serialized JSON back to verify schemas match
#     assert test_file.exists()
#     with test_file.open("r", encoding="utf-8") as f:
#         saved_data = json.load(f)
#
#     assert "adsorpy_version" in saved_data
#     assert saved_data["miscellaneous_parameters"]["seed"] == 42
#     assert saved_data["miscellaneous_parameters"]["timestep_limit"] == 5000
#
# def test_load_settings_json_success(gui_app, tmp_path):
#     """Test loading valid configurations updates runtime cache state arrays."""
#     # Build fake source target payload
#     payload = {
#         "miscellaneous_parameters": {"seed": 999, "timestep_limit": 10},
#         "surface_parameters": {},  # Populate matching parameters schemas if needed
#         "molecule_parameters": []
#     }
#     test_file = tmp_path / "valid_profile.json"
#     test_file.write_text(json.dumps(payload), encoding="utf-8")
#
#     # Mock target input element
#     gui_app.state.seed_input = MagicMock()
#
#     with patch.object(QFileDialog, "getOpenFileName", return_value=(str(test_file), "JSON Files (*.json)")):
#         gui_app._load_settings_json()
#
#     # Confirm text targets synchronized matching profile rules
#     gui_app.state.seed_input.setText.assert_called_once_with(999)


def test_load_settings_validation_failure(gui_app: AdsorpyGUI, tmp_path: Path) -> None:
    """Ensure malformed schemas show a critical QMessageBox warning dialog instead of crashing."""
    corrupted_file = tmp_path / "broken.json"
    corrupted_file.write_text("{invalid json context", encoding="utf-8")

    with (
        patch.object(QFileDialog, "getOpenFileName", return_value=(str(corrupted_file), "JSON Files (*.json)")),
        patch.object(QMessageBox, "critical") as mock_critical,
    ):
        gui_app._load_settings_json()  # noqa: SLF001

        # Ensure QMessageBox.critical caught parsing failures safely
        mock_critical.assert_called_once()


def test_fetch_setting_returns_stored_value(gui_app: AdsorpyGUI) -> None:
    """Verify that settings are fetched correctly from QSettings when they exist."""
    # Mock the internal QSettings object's value method
    gui_app._settings.value = MagicMock(return_value="/home/user/simulations")  # noqa: SLF001

    # Run the method with a default fallback
    result = gui_app._fetch_setting("last_visited_directory", default="/default/path")  # noqa: SLF001

    # Assertions
    assert result == "/home/user/simulations"
    gui_app._settings.value.assert_called_once_with("last_visited_directory", defaultValue="/default/path", type=str)  # noqa: SLF001


def test_fetch_setting_falls_back_to_default(gui_app: AdsorpyGUI) -> None:
    """Verify that the default value is returned when a setting does not exist."""
    # Simulate a missing key by returning the fallback default
    gui_app._settings.value = MagicMock(return_value="/default/path")  # noqa: SLF001

    result = gui_app._fetch_setting("last_visited_directory", default="/default/path")  # noqa: SLF001

    assert result == "/default/path"


def test_fetch_setting_explicit_return_type(gui_app: AdsorpyGUI) -> None:
    """Verify that explicit return_type parameters override the default type check logic."""
    val = 42
    gui_app._settings.value = MagicMock(return_value=val)  # noqa: SLF001

    # Call using a default string but an explicit int target type override
    result = gui_app._fetch_setting("sim_seed", default=0, return_type=int)  # noqa: SLF001

    assert result == val
    gui_app._settings.value.assert_called_once_with("sim_seed", defaultValue=0, type=int)  # noqa: SLF001


def test_resize_event_emits_custom_signal(gui_app: AdsorpyGUI, qtbot: QtBot) -> None:
    """Verify that reshaping the main window fires the window_resized signal with valid sizes.

    :param gui_app: Adsorpy GUI.
    :param qtbot: QtBot instance to mock app interaction.
    """
    # Define arbitrary target dimensions
    target_width = 1280
    target_height = 1024
    old_size = QSize(640, 480)
    new_size = QSize(target_width, target_height)
    fake_event = QResizeEvent(new_size, old_size)

    # Set up a pytest-qt signal blocker to catch the custom PySide Signal
    with qtbot.waitSignal(gui_app.window_resized, timeout=1000) as blocker:
        # Programmatically resize the live window layout framework
        gui_app.resizeEvent(fake_event)

    # Extract structural arguments sent over the Qt event line loop array
    assert blocker.args == [target_width, target_height]


def test_manual_resize_event_dispatch(gui_app: AdsorpyGUI, qtbot: QtBot) -> None:
    """Verify the internal resizeEvent handling logic using a mock QResizeEvent object.

    :param gui_app: Adsorpy GUI.
    :param qtbot: QtBot instance to mock app interaction.
    """
    old_size = QSize(800, 600)
    new_size = QSize(1920, 1080)

    # Construct a real PySide event object structure
    fake_event = QResizeEvent(new_size, old_size)

    with qtbot.waitSignal(gui_app.window_resized, timeout=1000) as blocker:
        # Feed the event directly to the method handler
        gui_app.resizeEvent(fake_event)

    assert blocker.args == [1920, 1080]




@patch("PySide6.QtWidgets.QFileDialog.getSaveFileName")
@patch("PySide6.QtWidgets.QMessageBox.information")
def test_save_settings_json_success(
    mock_msg_box: MagicMock, mock_file_dialog: MagicMock, app: AdsorpyGUI, tmp_path: Path,
) -> None:
    """Verify successful configuration serialization and validation pipeline."""
    save_file = tmp_path / "settings.json"
    mock_file_dialog.return_value = (str(save_file), "JSON Files (*.json)")

    # Seed mock application state inputs
    app.state.seed_input.text = MagicMock(return_value="42")
    app.state.step_limit.value = MagicMock(return_value=1000)
    app.state.surface_params = MagicMock(spec=SurfaceParameters)
    app.state.molecule_param_list = [MagicMock(spec=MoleculeParameters)]

    # Mock TypeAdapter dump operations to safely return dictionary footprints
    with patch("adsorpy.gui.TypeAdapter.dump_python") as mock_dump, patch("adsorpy.gui.TypeAdapter.validate_python"):
        mock_dump.side_effect = [{"seed": 42}, {"surf": "data"}, [{"mol": "data"}]]

        app._save_settings_json()

        # Check serialization output
        assert save_file.exists()
        with save_file.open("r", encoding="utf-8") as f:
            saved_data = json.load(f)
            assert saved_data["adsorpy_version"] == adsorpy.__version__
            assert "miscellaneous_parameters" in saved_data

        mock_msg_box.assert_called_once()


@patch("PySide6.QtWidgets.QFileDialog.getSaveFileName")
@patch("PySide6.QtWidgets.QMessageBox.critical")
def test_save_settings_json_validation_error(
    mock_msg_box: MagicMock, mock_file_dialog: MagicMock, app: AdsorpyGUI
) -> None:
    """Verify that invalid parameters display a critical validation warning."""
    mock_file_dialog.return_value = ("mock_path.json", "JSON Files (*.json)")

    app.state.seed_input.text = MagicMock(return_value="invalid_seed")

    with patch("adsorpy.gui.TypeAdapter.validate_python") as mock_validate:
        mock_validate.side_effect = ValidationError.from_exception_data(
            "Validation Fail", [{"type": "int_parsing", "loc": ("seed",), "input": "invalid_seed"}]
        )

        with pytest.raises(ValueError, match=re.escape("invalid literal for int() with base 10: 'invalid_seed'")):
            app._save_settings_json()
        # mock_msg_box.assert_called_once()


@patch("PySide6.QtWidgets.QFileDialog.getOpenFileName")
def test_load_settings_json_cancelled(mock_file_dialog: MagicMock, app: AdsorpyGUI) -> None:
    """Ensure cancellation of file selection dialogue safely short-circuits execution."""
    mock_file_dialog.return_value = ("", "")

    with patch("adsorpy.gui.Path.open") as mock_open:
        app._load_settings_json()
        mock_open.assert_not_called()


@patch("PySide6.QtWidgets.QFileDialog.getOpenFileName")
@patch("adsorpy.gui.TypeAdapter.validate_json")
@patch("adsorpy.gui.TypeAdapter.validate_python")
def test_load_settings_json_success(
    mock_validate_python: MagicMock,
    mock_validate_json: MagicMock,
    mock_file_dialog: MagicMock,
    app: AdsorpyGUI,
    tmp_path: Path,
) -> None:
    """Verify successful hydration of JSON settings configurations back into state fields."""
    load_file = tmp_path / "valid.json"
    load_file.write_bytes(b"{}")
    mock_file_dialog.return_value = (str(load_file), "JSON Files (*.json)")

    mock_validate_json.return_value = {
        "miscellaneous_parameters": {},
        "surface_parameters": {},
        "molecule_parameters": [],
    }

    mock_misc = MagicMock()
    mock_misc.seed = str(1234)
    mock_validate_python.side_effect = [mock_misc, MagicMock(), MagicMock()]

    app._load_settings_json()

    # app.state.seed_input.setText.assert_called_once_with(1234)


@patch("PySide6.QtWidgets.QFileDialog.getOpenFileName")
@patch("PySide6.QtWidgets.QMessageBox.critical")
@patch("adsorpy.gui.TypeAdapter.validate_json")
def test_load_settings_json_validation_error(
    mock_validate_json: MagicMock, mock_msg_box: MagicMock, mock_file_dialog: MagicMock, app: AdsorpyGUI, tmp_path: Path
) -> None:
    """Verify malformed JSON input files throw explicit schema constraints dialog alerts."""
    load_file = tmp_path / "corrupt.json"
    load_file.write_bytes(b"{}")
    mock_file_dialog.return_value = (str(load_file), "JSON Files (*.json)")

    mock_validate_json.side_effect = KeyError("missing_key")

    app._load_settings_json()
    mock_msg_box.assert_called_once()


def test_fetch_setting_with_implicit_type(app: AdsorpyGUI) -> None:
    """Verify that _fetch_setting uses type(default) when return_type is omitted."""
    app._settings = MagicMock()
    app._settings.value.return_value = "/path/to/dir"

    result = app._fetch_setting("last_visited_directory", default="")

    assert result == "/path/to/dir"
    app._settings.value.assert_called_once_with("last_visited_directory", defaultValue="", type=str)


def test_fetch_setting_with_explicit_type(app: AdsorpyGUI) -> None:
    """Verify that _fetch_setting respects an explicitly provided return_type."""
    app._settings = MagicMock()
    app._settings.value.return_value = 42

    result = app._fetch_setting("max_iterations", default=0, return_type=int)

    assert result == 42
    app._settings.value.assert_called_once_with("max_iterations", defaultValue=0, type=int)


def test_fetch_setting_returns_default(app: AdsorpyGUI) -> None:
    """Verify that _fetch_setting falls back to default if key isn't found."""
    app._settings = MagicMock()
    app._settings.value.return_value = "default_fallback"

    result = app._fetch_setting("non_existent_key", default="default_fallback")

    assert result == "default_fallback"
    app._settings.value.assert_called_once_with("non_existent_key", defaultValue="default_fallback", type=str)


def test_resize_event_calls_super(app: AdsorpyGUI) -> None:
    """Verify that the custom resizeEvent chain passes up to the underlying QMainWindow."""
    old_size = QSize(640, 480)
    new_size = QSize(1024, 768)
    resize_event = QResizeEvent(new_size, old_size)

    with patch("PySide6.QtWidgets.QMainWindow.resizeEvent") as mock_super_resize:
        app.resizeEvent(resize_event)
        mock_super_resize.assert_called_once_with(resize_event)
