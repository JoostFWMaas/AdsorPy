# Copyright (c) 2025-2026 Contributors to the AdsorPy project.
# SPDX-License-Identifier: MIT
"""Test the AutoStateMeta class of the `gui.py` module."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject
from pytestqt.qtbot import QtBot

from adsorpy.gui import AutoStateMeta


def test_autostate_meta_builtin_type_and_preinitialized_field() -> None:
    """Test whether autostate initialises correctly."""
    count: int = 42

    class TempState(QObject, metaclass=AutoStateMeta):  # type: ignore[metaclass]
        count: int
        name: str

        def __init__(self) -> None:
            super().__init__()
            # Pre-initialise _count so __call__ skips setting it to None.
            self._count = count

    state = TempState()

    assert state.count == count, "_count should keep its __init__ value instead of being overwritten with None."
    assert state.name is None, "_name was not set in __init__, so __call__ should auto-initialise it to None."

    # Verify the Qt Signal was generated with the builtin type
    received: list[int] = []
    state.countChanged.connect(received.append)
    new_count: int = 99
    state.count = new_count
    assert received == [new_count]


def test_autostate_meta_ignores_private_and_fields_annotations() -> None:
    """Verify 'fields' and '_*' annotations do not generate properties or signals."""

    class TempState(QObject, metaclass=AutoStateMeta):  # type: ignore[metaclass]
        _private_helper: int
        visible_value: str

    state = TempState()

    assert "visible_value" in TempState.fields
    assert "_private_helper" not in TempState.fields
    assert "fields" not in TempState.fields
    assert not hasattr(state, "_private_helperChanged")
    assert not hasattr(state, "fieldsChanged")


def test_autostate_meta_non_builtin_and_union_types(qtbot: QtBot, tmp_path: Path) -> None:
    """Verify Union types and non-builtin classes fall back to Signal(object)."""

    class TempState(QObject, metaclass=AutoStateMeta):  # type: ignore[metaclass]
        optional_num: int | None
        file_path: Path

    state = TempState()
    assert state.optional_num is None
    assert state.file_path is None

    test_path = tmp_path / "test.xyz"
    with qtbot.waitSignal(state.file_pathChanged, timeout=1000) as blocker:
        state.file_path = test_path

    assert blocker.args == [test_path]
    assert state.file_path == test_path


def test_autostate_meta_multiple_fields_closure_binding() -> None:
    """Verify multiple fields keep independent getters, setters, and signals."""

    class TempState(QObject, metaclass=AutoStateMeta):  # type: ignore[metaclass]
        first: int
        second: int

    state = TempState()
    first_events: list[int] = []
    second_events: list[int] = []

    state.firstChanged.connect(first_events.append)  # type: ignore[attr-defined]
    state.secondChanged.connect(second_events.append)  # type: ignore[attr-defined]
    first: int = 10
    second: int = 20
    state.first = first
    state.second = second

    assert state.first == first
    assert state.second == second
    assert first_events == [first]
    assert second_events == [second]


def test_autostate_meta_instance_isolation() -> None:
    """Verify separate instances do not share state or cross-emit signals."""

    class TempState(QObject, metaclass=AutoStateMeta):  # type: ignore[metaclass]
        label: str

    state_a = TempState()
    state_b = TempState()
    label = "alpha"
    state_a.label = label

    assert state_a.label == label
    assert state_b.label is None
