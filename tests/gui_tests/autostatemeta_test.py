# Copyright (c) 2025-2026 Contributors to the AdsorPy project.
# SPDX-License-Identifier: MIT
"""Test the AutoStateMeta class of the `gui.py` module."""
from pathlib import Path

from PySide6.QtCore import QObject
from pytestqt.qtbot import QtBot

from adsorpy.gui import AutoStateMeta

def test_autostate_meta_builtin_type_and_preinitialized_field(qtbot: QtBot) -> None:
    """Test whether autostate initialises correctly."""
    class TempState(QObject, metaclass=AutoStateMeta):
        count: int
        name: str

        def __init__(self) -> None:
            super().__init__()
            # Pre-initialise _count so __call__ skips setting it to None.
            self._count = 42

    state = TempState()

    assert state.count == 42, "_count should keep its __init__ value instead of being overwritten with None."
    assert state.name is None, "_name was not set in __init__, so __call__ should auto-initialise it to None."

    # Verify the Qt Signal was generated with the builtin type
    received: list[int] = []
    state.countChanged.connect(received.append)  # type: ignore[attr-defined]
    state.count = 99
    assert received == [99]


def test_autostate_meta_ignores_private_and_fields_annotations(qtbot: QtBot) -> None:
    """Verify 'fields' and '_*' annotations do not generate properties or signals."""
    class TempState(QObject, metaclass=AutoStateMeta):
        _private_helper: int
        visible_value: str

    state = TempState()

    assert "visible_value" in TempState.fields
    assert "_private_helper" not in TempState.fields
    assert "fields" not in TempState.fields
    assert not hasattr(state, "_private_helperChanged")
    assert not hasattr(state, "fieldsChanged")


def test_autostate_meta_non_builtin_and_union_types(qtbot: QtBot) -> None:
    """Verify Union types and non-builtin classes fall back to Signal(object)."""
    class TempState(QObject, metaclass=AutoStateMeta):
        optional_num: int | None
        file_path: Path

    state = TempState()
    assert state.optional_num is None
    assert state.file_path is None

    test_path = Path("/tmp/test.xyz")
    with qtbot.waitSignal(state.file_pathChanged, timeout=1000) as blocker:  # type: ignore[attr-defined]
        state.file_path = test_path

    assert blocker.args == [test_path]
    assert state.file_path == test_path


def test_autostate_meta_multiple_fields_closure_binding(qtbot: QtBot) -> None:
    """Verify multiple fields keep independent getters, setters, and signals."""
    class TempState(QObject, metaclass=AutoStateMeta):
        first: int
        second: int

    state = TempState()
    first_events: list[int] = []
    second_events: list[int] = []

    state.firstChanged.connect(first_events.append)  # type: ignore[attr-defined]
    state.secondChanged.connect(second_events.append)  # type: ignore[attr-defined]

    state.first = 10
    state.second = 20

    assert state.first == 10
    assert state.second == 20
    assert first_events == [10]
    assert second_events == [20]


def test_autostate_meta_instance_isolation(qtbot: QtBot) -> None:
    """Verify separate instances do not share state or cross-emit signals."""
    class TempState(QObject, metaclass=AutoStateMeta):
        label: str

    state_a = TempState()
    state_b = TempState()

    state_a.label = "alpha"

    assert state_a.label == "alpha"
    assert state_b.label is None
