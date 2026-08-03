# Copyright (c) 2025-2026 Contributors to the AdsorPy project.
# SPDX-License-Identifier: MIT
"""Test the ZoomableSvgWidget class of the `gui.py` module."""

from pathlib import Path

import pytest
from _pytest.monkeypatch import MonkeyPatch
from PySide6.QtCore import QEvent, QPoint, QPointF, QSize, Qt
from PySide6.QtGui import QResizeEvent, QWheelEvent
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox, QScrollArea
from pytestqt.qtbot import QtBot, QWidget

from adsorpy.gui import ZoomableSvgWidget

VALID_SVG_BYTES: bytes = (
    b'<svg xmlns="http://w3.org" viewBox="0 0 20 20" width="20" height="20">'
    b'<rect width="20" height="20" fill="blue"/></svg>'
)


@pytest.fixture
def sample_svg_file(tmp_path: Path) -> Path:
    """Write a dummy valid SVG structure to a local disk storage layout path.

    :param tmp_path: The pytest standard fixture providing a unique temporary directory path.
    :returns: The Path instance pointing to the temporary test file.
    """
    file_path: Path = tmp_path / "graphic_render.svg"
    file_path.write_bytes(VALID_SVG_BYTES)
    return file_path


@pytest.fixture
def scrollable_widget_setup(qtbot: QtBot) -> tuple[QScrollArea, ZoomableSvgWidget]:
    """Create a realistic UI environment where ZoomableSvgWidget is nested in a QScrollArea.

    :param qtbot: The pytest-qt robot fixture managing UI lifecycles.
    :returns: A tuple holding the parent QScrollArea container and the child widget.
    """
    scroll_area: QScrollArea = QScrollArea()
    widget: ZoomableSvgWidget = ZoomableSvgWidget()
    scroll_area.setWidget(widget)

    # Configure precise dimensions and enable scrollbars explicitly for test consistency
    scroll_area.resize(400, 400)
    widget.resize(600, 600)
    scroll_area.horizontalScrollBar().setRange(0, 1000)
    scroll_area.verticalScrollBar().setRange(0, 1000)

    qtbot.addWidget(scroll_area)
    return scroll_area, widget


def test_widget_initialisation_state(qtbot: QtBot) -> None:
    """Verify default initialisation parameters and overlay widget properties.

    :param qtbot: The pytest-qt robot fixture managing UI lifecycles.
    """
    widget: ZoomableSvgWidget = ZoomableSvgWidget()
    qtbot.addWidget(widget)

    assert widget.zoom_factor == pytest.approx(1.15)
    assert widget.current_svg_path is None
    assert widget._current_svg_bytes is None  # noqa: SLF001
    assert widget.save_button.isVisible() is False
    assert widget.save_button.size() == QSize(40, 40)


def test_resize_event_pins_button_to_bottom_right(qtbot: QtBot) -> None:
    """Verify that resize event math precisely anchors the save button overlay.

    :param qtbot: The pytest-qt robot fixture managing UI lifecycles.
    """
    widget: ZoomableSvgWidget = ZoomableSvgWidget()
    qtbot.addWidget(widget)
    widget.resize(800, 800)

    old_size: QSize = QSize(600, 600)
    new_size: QSize = QSize(800, 800)
    event: QResizeEvent = QResizeEvent(new_size, old_size)
    widget.resizeEvent(event)

    margin: int = 20
    expected_x: int = 800 - widget.save_button.width() - margin
    expected_y: int = 800 - widget.save_button.height() - margin

    assert widget.save_button.x() == expected_x
    assert widget.save_button.y() == expected_y


def test_load_caches_bytes_and_toggles_visibility(qtbot: QtBot, sample_svg_file: Path) -> None:
    """Verify that loading an SVG caches bytes, captures paths, and reveals the button.

    :param qtbot: The pytest-qt robot fixture managing UI lifecycles.
    """
    widget: ZoomableSvgWidget = ZoomableSvgWidget()
    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.graphics_changed, timeout=1000) as blocker:
        widget.load_svg(sample_svg_file)

    assert blocker.args == [True]
    assert widget.current_svg_path == str(sample_svg_file)
    assert widget._current_svg_bytes == VALID_SVG_BYTES  # noqa: SLF001
    assert widget.save_button.isVisibleTo(widget) is True


def test_load_handles_raw_bytes_directly(qtbot: QtBot) -> None:
    """Verify that the overloaded load function handles direct memory byte packages.

    :param qtbot: The pytest-qt robot fixture managing UI lifecycles.
    """
    widget: ZoomableSvgWidget = ZoomableSvgWidget()
    qtbot.addWidget(widget)

    widget.load(VALID_SVG_BYTES)

    assert widget.current_svg_path is None
    assert widget._current_svg_bytes == VALID_SVG_BYTES  # noqa: SLF001
    assert widget.save_button.isVisibleTo(widget) is True


@pytest.mark.parametrize("extension", [".svg", ".png", ".jpg", ".pdf"])
def test_export_graphics_success_formats(
    qtbot: QtBot,
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    extension: str,
) -> None:
    """Verify that export routing converts and dumps all core target format profiles safely.

    :param qtbot: The pytest-qt robot fixture managing UI lifecycles.
    :param monkeypatch: The pytest mock manager engine handling runtime dependency injection.
    """
    widget: ZoomableSvgWidget = ZoomableSvgWidget()
    qtbot.addWidget(widget)
    widget.load(VALID_SVG_BYTES)

    destination_file: Path = tmp_path / f"output_export{extension}"

    mock_file_dialog: tuple[str, str] = (str(destination_file), f"Format File (*{extension})")
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args, **kwargs: mock_file_dialog)  # noqa: ARG005, # pyright: ignore[reportUnknownLambdaType]

    monkeypatch.setattr(QMessageBox, "information", lambda *args, **kwargs: None)  # noqa: ARG005, # pyright: ignore[reportUnknownLambdaType]
    monkeypatch.setattr(QMessageBox, "warning", lambda *args, **kwargs: None)  # noqa: ARG005, # pyright: ignore[reportUnknownLambdaType]

    qtbot.mouseClick(widget.save_button, Qt.MouseButton.LeftButton)

    assert destination_file.exists() is True
    assert destination_file.stat().st_size > 0


def test_export_graphics_filesystem_failure_displays_critical_dialog(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify that disk I/O errors are caught by the exception block and report via critical dialogue."""
    widget: ZoomableSvgWidget = ZoomableSvgWidget()
    qtbot.addWidget(widget)
    widget.load(VALID_SVG_BYTES)

    # Use a normal file path string that looks valid to your widget logic
    target_path = tmp_path / "test_output.svg"

    mock_file_dialog: tuple[str, str] = (str(target_path), "Scalable Vector Graphics (*.svg)")
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args, **kwargs: mock_file_dialog)  # noqa: ARG005, # pyright: ignore[reportUnknownLambdaType]

    # FORCE any file write attempt on Path objects to immediately raise an IOError
    def mock_write_bytes(self: Path, data: bytes) -> None:
        errmsg = "Simulated Disk I/O Failure"
        raise OSError(errmsg)

    def mock_open_failed(self: Path) -> None:
        errmsg = "Simulated Disk Open Failure"
        raise OSError(errmsg)

    monkeypatch.setattr(Path, "write_bytes", mock_write_bytes)
    monkeypatch.setattr(Path, "open", mock_open_failed)

    critical_triggered: bool = False

    def mock_critical(parent: QWidget, title: str, text: str) -> None:
        nonlocal critical_triggered
        # Check if "Export Failed" or "Error" is in either the title or text body
        if "Export" in title or "Error" in title or "Fail" in title:
            critical_triggered = True

    monkeypatch.setattr(QMessageBox, "critical", mock_critical)

    # Suppress the success dialogue just in case it's still called somewhere
    monkeypatch.setattr(QMessageBox, "information", lambda *args, **kwargs: QMessageBox.StandardButton.Ok)  # noqa: ARG005, # pyright: ignore[reportUnknownLambdaType]

    # Act
    qtbot.mouseClick(widget.save_button, Qt.MouseButton.LeftButton)

    # Assert
    assert critical_triggered is True, "The critical error dialog was never triggered during file write failure!"


@pytest.mark.parametrize(("delta", "scale_multiplier"), [(120, 1.15), (-120, 1.0 / 1.15)])
def test_wheel_event_zoom_with_control_modifier(
    scrollable_widget_setup: tuple[QScrollArea, ZoomableSvgWidget],
    delta: int,
    scale_multiplier: float,
) -> None:
    """Verify that scrolling with the Ctrl modifier rescales the widget and adjusts scrollbars.

    :param scrollable_widget_setup: The custom test environment nesting layout fixture.
    :param delta: The rotational scrolling movement vector parameter.
    :param scale_multiplier: The geometric multiplier factor matching calculation steps.
    """
    scroll_area, widget = scrollable_widget_setup

    # Initialize baseline coordinates and assign default positioning benchmarks
    widget.setFixedSize(400, 400)
    scroll_area.horizontalScrollBar().setValue(100)
    scroll_area.verticalScrollBar().setValue(100)

    # Create a wheel event focused at the center of the widget with the Control modifier
    wheel_event = QWheelEvent(
        QPointF(200.0, 200.0),  # Position inside the widget
        QPointF(200.0, 200.0),  # Global position coordinate footprint
        QPoint(0, 0),  # Pixel delta
        QPoint(0, delta),  # Angle delta (y-axis tracks rotation)
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.ControlModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,  # noqa: FBT003
    )

    # Calculate expected size transitions manually
    expected_width: int = int(400 * scale_multiplier)
    expected_height: int = int(400 * scale_multiplier)

    # Execute and verify the event accept transaction loop
    widget.wheelEvent(wheel_event)

    assert wheel_event.isAccepted() is True
    assert widget.width() == pytest.approx(expected_width, abs=1)
    assert widget.height() == pytest.approx(expected_height, abs=1)


def test_wheel_event_zoom_ignores_outside_scale_bounds(
    scrollable_widget_setup: tuple[QScrollArea, ZoomableSvgWidget],
) -> None:
    """Verify that zoom instructions are rejected if they break minimum or maximum boundary scales.

    :param scrollable_widget_setup: The custom test environment nesting layout fixture.
    """
    widget = scrollable_widget_setup[1]

    # Push width directly against the minimum dimension safety guard rail limit (100)
    widget.setFixedSize(105, 105)

    # Attempt an explicit zoom-out operation that would drag dimensions below 100 pixels
    zoom_out_event = QWheelEvent(
        QPointF(50.0, 50.0),
        QPointF(50.0, 50.0),
        QPoint(0, 0),
        QPoint(0, -120),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.ControlModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,  # noqa: FBT003
    )
    widget.wheelEvent(zoom_out_event)

    # Layout changes must be rejected and size configuration preserved intact
    # Ensure layout adjustments were rejected and size limits are preserved intact
    assert widget.width() == 105  # noqa: PLR2004


def test_wheel_event_horizontal_pan_with_shift_modifier(
    scrollable_widget_setup: tuple[QScrollArea, ZoomableSvgWidget],
) -> None:
    """Verify that scrolling with the Shift modifier shifts the horizontal scrollbar.

    :param scrollable_widget_setup: The custom test environment nesting layout fixture.
    """
    scroll_area, widget = scrollable_widget_setup
    h_bar = scroll_area.horizontalScrollBar()
    h_bar.setValue(200)

    # Positive angle delta scrolls left (subtracts from scroll value)
    pan_event = QWheelEvent(
        QPointF(100.0, 100.0),
        QPointF(100.0, 100.0),
        QPoint(0, 0),
        QPoint(0, 120),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.ShiftModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,  # noqa: FBT003
    )

    widget.wheelEvent(pan_event)

    assert pan_event.isAccepted() is True
    assert h_bar.value() == 200 - 120


def test_wheel_event_fallback_without_modifiers_bubbles_to_viewport(
    scrollable_widget_setup: tuple[QScrollArea, ZoomableSvgWidget],
    monkeypatch: MonkeyPatch,
) -> None:
    """Verify that normal un-modified scroll actions route up directly to the scroll viewport.

    :param scrollable_widget_setup: The custom test environment nesting layout fixture.
    :param monkeypatch: The pytest mock manager engine handling runtime dependency injection.
    """
    scroll_area, widget = scrollable_widget_setup

    event_forwarded: bool = False

    def mock_send_event(receiver: QWidget, event: QEvent) -> bool:
        nonlocal event_forwarded
        if receiver == scroll_area.viewport():
            event_forwarded = True
        return True

    # Intercept Qt's application message distribution tracking manager system call
    monkeypatch.setattr(QApplication, "sendEvent", mock_send_event)

    normal_scroll_event = QWheelEvent(
        QPointF(100.0, 100.0),
        QPointF(100.0, 100.0),
        QPoint(0, 0),
        QPoint(0, -120),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,  # noqa: FBT003
    )

    widget.wheelEvent(normal_scroll_event)
    assert event_forwarded is True
