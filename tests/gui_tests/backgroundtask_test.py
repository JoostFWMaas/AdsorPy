# Copyright (c) 2025-2026 Contributors to the AdsorPy project.
# SPDX-License-Identifier: MIT
"""Test the BackgroundTask and BackgroundTaskSignals classes of the `gui.py` module."""

import pytest
from _pytest.monkeypatch import MonkeyPatch
from PySide6.QtCore import QThreadPool
from pytestqt.qtbot import QtBot

from adsorpy.gui import BackgroundTask


def sample_successful_job(a: int, b: int, message: str = "done") -> str:
    """Make standard mock calculation payload.

    Multiplies a and b, appends the message.
    :param a: The first value.
    :param b: The second value.
    :param message: The message to add to the return string.
    :returns: '{a * b} - {message}' f-string.
    """
    return f"{a * b} - {message}"


def sample_failing_job() -> None:
    """Make standard mock logic loop engineered to force a target exception.

    :raises ValueError: Always.
    """
    errmsg = "Simulation payload misconfiguration"
    raise ValueError(errmsg)


def test_background_task_success_path(qtbot: QtBot) -> None:
    """Verify that a successful background execution delivers its computed payload.

    :param qtbot: The pytest-qt robot fixture managing UI thread synchronisation.
    """
    task = BackgroundTask(sample_successful_job, 10, 5, message="complete")

    # Corrected: Call waitSignal directly on the PySide signal attribute
    with qtbot.waitSignal(task.signals.finished, timeout=2000) as blocker:
        QThreadPool.globalInstance().start(task)

    # Blocker args returns a list of arguments passed to the signal emit()
    assert blocker.args == ["50 - complete"]


def test_background_task_error_catch_path(qtbot: QtBot) -> None:
    """Verify that expected target exception blocks are safely intercepted and transmitted.

    :param qtbot: The pytest-qt robot fixture managing UI thread synchronisation.
    """
    task: BackgroundTask[[], None] = BackgroundTask(sample_failing_job)

    with qtbot.waitSignal(task.signals.error, timeout=2000) as blocker:
        QThreadPool.globalInstance().start(task)

    if blocker.args:
        captured_exception = blocker.args[0]
    else:
        errmsg = "No error captured."
        raise AssertionError(errmsg)
    assert isinstance(captured_exception, ValueError)
    assert str(captured_exception) == "Simulation payload misconfiguration"


def test_background_task_unhandled_exception_bubbles() -> None:
    """Verify that exceptions outside this specific try/except tuple bubble up cleanly.

    By calling task.run() synchronously, the internal try/except contract is tested
    without polluting or crashing the global Qt background thread pool worker.
    """
    errmsg = "Somehow, this error appeared."

    def sample_runtime_crash() -> None:
        """Sample error code.

        :raises AssertionError: Always.
        """
        raise AssertionError(errmsg)  # Reasonably, code should never contain an assertion error.

    task: BackgroundTask[[], None] = BackgroundTask(sample_runtime_crash)

    # Assert that the unhandled AssertionError bubbles up normally when executing the task logic.
    with pytest.raises(AssertionError, match=errmsg):
        task.run()


def test_background_task_emit_error(monkeypatch: MonkeyPatch) -> None:
    """Test whether ``BackgroundTask`` emits an error correctly.

    :param monkeypatch: Pytest fixture to mock parameters.
    """
