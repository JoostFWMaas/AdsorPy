# Copyright (c) 2025-2026 Contributors to the AdsorPy project.
# SPDX-License-Identifier: MIT
"""Test the ReorderableListWidget class of the `gui.py` module."""

from unittest.mock import patch

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDropEvent
from PySide6.QtWidgets import QListWidget, QListWidgetItem
from pytestqt.qtbot import QtBot

from adsorpy.gui import ReorderableListWidget


def test_reorderable_list_widget_initialisation(qtbot: QtBot) -> None:
    """Verify that the widget initialises with internal move mode enabled.

    :param qtbot: The pytest-qt robot fixture used to manage GUI lifecycle.
    """
    widget: ReorderableListWidget = ReorderableListWidget()
    qtbot.addWidget(widget)

    # Ensure internal movement drag-and-drop flags are configured correctly
    assert widget.dragDropMode() == ReorderableListWidget.DragDropMode.InternalMove


@settings(
    deadline=None,  # Prevent slow GUI generation from failing the test
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    # Generate lists containing between 2 and 20 unique item names
    item_names=st.lists(st.text(min_size=1), min_size=2, max_size=20, unique=True),
    # Randomly select a row index to drag from and a row index to drop to
    move_indices=st.data(),
)
def test_reorderable_list_hypothesis_shuffled_moves(
    qtbot: QtBot,
    item_names: list[str],
    move_indices: st.DataObject,
) -> None:
    """Verify that any valid random drag-and-drop sequence functions correctly."""
    widget: ReorderableListWidget = ReorderableListWidget()
    qtbot.addWidget(widget)

    # Populate item entries
    items: list[QListWidgetItem] = [QListWidgetItem(name, widget) for name in item_names]
    for item in items:
        widget.addItem(item)
    total_count: int = len(item_names)

    # Draw random row selection bounds
    old_row: int = move_indices.draw(st.integers(min_value=0, max_value=total_count - 1))
    new_row: int = move_indices.draw(st.integers(min_value=0, max_value=total_count - 1))

    # Keep target item selected at its original index
    widget.setCurrentRow(old_row)
    target_item: QListWidgetItem = widget.currentItem()

    def mock_super_drop(signal: Signal) -> None:
        """Mock function for drag and drop."""
        # Mutate the widget state using the instance passed by the patch
        widget.takeItem(old_row)
        widget.insertItem(new_row, target_item)
        widget.setCurrentItem(target_item)

    mock_event: QDropEvent = QDropEvent(
        widget.rect().center(),
        Qt.DropAction.MoveAction,
        widget.mimeData([target_item]),
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )

    # Track signal emissions synchronously
    emitted_signals: list[tuple[int, int]] = []
    widget.itemsMoved.connect(lambda old, new: emitted_signals.append((old, new)))  # pyright: ignore[reportUnknownLambdaType]

    # Removed autospec=True to give the side_effect flexible argument mapping
    with patch.object(QListWidget, "dropEvent", side_effect=mock_super_drop):
        widget.dropEvent(mock_event)

    # Validate output states against Hypothesis expectations
    if old_row != new_row:
        assert len(emitted_signals) == 1, f"Expected 1 signal emission, got {len(emitted_signals)}"
        assert emitted_signals[0] == (old_row, new_row)
    else:
        assert len(emitted_signals) == 0, f"Expected 0 signal emissions when indices match, got {emitted_signals}"


def test_drop_event_no_signal_if_position_unchanged(qtbot: QtBot) -> None:
    """Verify that dropping an item back into its original index does not emit the signal.

    :param qtbot: The pytest-qt robot fixture used to manage GUI lifecycle.
    """
    widget: ReorderableListWidget = ReorderableListWidget()
    qtbot.addWidget(widget)

    item: QListWidgetItem = QListWidgetItem("Stationary Item", widget)
    widget.setCurrentItem(item)

    # Mock a drop event that resolves to the exact same position
    mock_event: QDropEvent = QDropEvent(
        widget.rect().center(),
        Qt.DropAction.MoveAction,
        widget.mimeData([item]),
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )

    # Using waitSignal with a negative timeout throws a timeout exception if the signal fires
    with pytest.raises(qtbot.TimeoutError), qtbot.waitSignal(widget.itemsMoved, timeout=200):
        widget.dropEvent(mock_event)
