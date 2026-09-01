# Copyright (c) 2025-2026 Contributors to the AdsorPy project.
# SPDX-License-Identifier: MIT
"""Test the FilePickierWidget class of the `gui.py` module."""

from collections.abc import Generator
from pathlib import Path
from unittest.mock import Mock

import pytest
from _pytest.monkeypatch import MonkeyPatch
from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import QFileDialog, QLineEdit, QPushButton
from pytestqt.qtbot import QtBot

from adsorpy.gui import FilePickerWidget


@pytest.fixture(autouse=True)
def clean_qsettings() -> Generator[None, None, None]:
    """Clear QSettings before and after each test to prevent session bleeding.

    :returns: A generator yielding control back to the test environment.
    """
    settings = QSettings("FilePickerWidget")
    settings.clear()
    yield
    settings.clear()


def test_file_picker_initialization(qtbot: QtBot, subtests: pytest.Subtests) -> None:
    """Verify default widget state and layout components.

    :param qtbot: The pytest-qt robot fixture used to manage GUI lifecycle.
    :param subtests: The pytest subtests context manager fixture.
    """
    placeholder: str = "Choose data..."
    widget: FilePickerWidget = FilePickerWidget(placeholder=placeholder)
    qtbot.addWidget(widget)

    with subtests.test(msg="Verify placeholder text"):
        assert widget.line_edit.placeholderText() == placeholder

    with subtests.test(msg="Verify line edit exists"):
        assert isinstance(widget.line_edit, QLineEdit)
        assert widget.line_edit.text() == ""

    with subtests.test(msg="Verify browse button exists"):
        assert isinstance(widget.browse_button, QPushButton)


def test_getters_and_setters(qtbot: QtBot) -> None:
    """Verify programmatic setting and getting of text values.

    :param qtbot: The pytest-qt robot fixture used to manage GUI lifecycle.
    """
    widget: FilePickerWidget = FilePickerWidget()
    qtbot.addWidget(widget)

    test_path: str = "/path/to/simulation.xyz"
    widget.setText(test_path)

    assert widget.text() == test_path
    assert widget.line_edit.text() == test_path


def test_open_file_dialog_saves_file_and_directory(qtbot: QtBot, monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    """Verify that selecting a file updates the line edit and saves the directory history.

    :param qtbot: The pytest-qt robot fixture used to simulate user actions.
    :param monkeypatch: The pytest standard fixture used to mock runtime dependencies.
    :param tmp_path: The pytest standard fixture providing a unique temporary directory path.
    """
    widget: FilePickerWidget = FilePickerWidget()
    qtbot.addWidget(widget)

    # Create a fake target file in a temporary directory
    target_dir: Path = tmp_path / "sim_data"
    target_dir.mkdir()
    fake_file: Path = target_dir / "atoms.xyz"
    fake_file.touch()

    # Mock QFileDialog.getOpenFileName to bypass the native system window
    mock_response = (str(fake_file), "XYZ File (*.xyz)")
    monkeypatch.setattr("PySide6.QtWidgets.QFileDialog.getOpenFileName", Mock(return_value=mock_response))
    # Act: Simulate a click on the browse button
    with qtbot.waitSignal(widget.browse_button.clicked, timeout=1000):
        qtbot.mouseClick(widget.browse_button, Qt.MouseButton.LeftButton)

    # Assert 1: The UI text updated correctly
    assert widget.text() == str(fake_file)

    # Assert 2: The parent directory was persistent into QSettings
    saved_dir: str = widget._fetch_setting("last_visited_directory", default="")
    assert saved_dir == str(target_dir)


def test_open_file_dialog_cancelled(qtbot: QtBot, monkeypatch: MonkeyPatch) -> None:
    """Verify that closing or cancelling the dialogue changes nothing.

    :param qtbot: The pytest-qt robot fixture used to simulate user actions.
    :param monkeypatch: The pytest standard fixture used to mock runtime dependencies.
    """
    widget: FilePickerWidget = FilePickerWidget()
    qtbot.addWidget(widget)

    widget.setText("/original/path.xyz")

    # Direct object mocking removes string errors and bypasses linting issues
    monkeypatch.setattr(QFileDialog, "getOpenFileName", Mock(return_value=("", "")))

    qtbot.mouseClick(widget.browse_button, Qt.MouseButton.LeftButton)

    # Value must remain untouched
    assert widget.text() == "/original/path.xyz"
