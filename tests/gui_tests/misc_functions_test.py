# Copyright (c) 2025-2026 Contributors to the AdsorPy project.
# SPDX-License-Identifier: MIT
"""Test the functions of the `gui.py` module."""
import json
from collections.abc import Callable
from typing import Any

import pytest
from PySide6.QtWidgets import QDoubleSpinBox, QLineEdit, QSpinBox, QWidget
from pytestqt.qtbot import QtBot
from shapely import from_geojson
from shapely.geometry import Polygon

from src.adsorpy.gui import FilePickerWidget, extract_param_docs, set_content, validate_polygon


def test_set_content_success_cases(qtbot: QtBot, subtests: pytest.Subtests) -> None:
    """Verify that set_content correctly maps supported values to their target widgets.

    :param qtbot: The pytest-qt robot fixture used to manage GUI lifecycle.
    :param subtests: The pytest subtests context manager fixture.
    """
    spinbox: QSpinBox = QSpinBox()
    double_spinbox: QDoubleSpinBox = QDoubleSpinBox()
    line_edit: QLineEdit = QLineEdit()
    file_picker: FilePickerWidget = FilePickerWidget()

    # Register widgets with qtbot for lifecycle handling
    for w in (spinbox, double_spinbox, line_edit, file_picker):
        qtbot.addWidget(w)

    with subtests.test(msg="QSpinBox integer assignment"):
        set_content(spinbox, 42)
        assert spinbox.value() == 42

    with subtests.test(msg="QDoubleSpinBox float assignment"):
        set_content(double_spinbox, 3.14)
        assert double_spinbox.value() == pytest.approx(3.14)

    with subtests.test(msg="QLineEdit string assignment"):
        set_content(line_edit, "Hello World")
        assert line_edit.text() == "Hello World"

    with subtests.test(msg="FilePickerWidget string assignment"):
        set_content(file_picker, "/path/to/file.xyz")
        assert file_picker.text() == "/path/to/file.xyz"

    with subtests.test(msg="QLineEdit list of strings assignment"):
        set_content(line_edit, ["A", "B", "C"])
        assert line_edit.text() == "A,B,C"


@pytest.mark.parametrize(
    "widget_factory, invalid_content",
    [
        (lambda: QSpinBox(), "not an int"),
        (lambda: QDoubleSpinBox(), 12),  # Expects float, not an int
        (lambda: FilePickerWidget(), ["list", "not", "string"]),
    ],
)
def test_set_content_mismatch_raises_value_error(
    qtbot: QtBot, widget_factory: Callable[[], QWidget], invalid_content: str | list[str] | int,
) -> None:
    """Verify that incompatible widget-content pairs throw a clear ValueError.

    :param qtbot: The pytest-qt robot fixture used to manage GUI lifecycle.
    :param widget_factory: A factory function to clean-instantiate widgets per parameter pass.
    :param invalid_content: The malformed payload intended to break the pattern match.
    """
    widget: QWidget = widget_factory()
    qtbot.addWidget(widget)

    with pytest.raises(ValueError, match="Widget and content mismatch"):
        set_content(widget, invalid_content)  # type: ignore[arg-type]


def test_extract_param_docs_success() -> None:
    """Verify clean parameter descriptions extraction including multiline buffers."""

    def sample_func(x: int, y: str) -> None:
        """Process simulation parameters.

        :param x: The calculation offset variable.
        :param y: The boundary layout type index
            spanning multiple lines of text.
        :returns: Nothing.
        """

    result: dict[str, str] = extract_param_docs(sample_func)

    assert result["x"] == "The calculation offset variable."
    assert result["y"] == "The boundary layout type index spanning multiple lines of text."


def test_extract_param_docs_raises_value_error_if_missing() -> None:
    """Verify ValueError is raised if the analyzed target lacks a docstring."""

    def undocumented_func() -> None:
        pass

    with pytest.raises(ValueError, match="Docstring of undocumented_func is not defined."):
        extract_param_docs(undocumented_func)


@pytest.fixture
def sample_geojson() -> dict[str, str | list[list[list[float]]]]:
    """Provide a standard square geometry payload structured in a GeoJSON style dictionary.

    :returns: A raw dictionary describing a square polygon layout footprint.
    """
    return {"type": "Polygon", "coordinates": [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]]}


def test_validate_polygon_success_cases(sample_geojson: dict[str, Any], subtests: pytest.Subtests) -> None:
    """Verify that shape logic accepts Polygon objects, dictionary models, and raw JSON strings.

    :param sample_geojson: A preconstructed geometric square template payload fixture.
    :param subtests: The pytest subtests context manager fixture.
    """
    expected_poly: Polygon = from_geojson(json.dumps(sample_geojson))

    with subtests.test(msg="Direct Polygon pass-through"):
        res: Polygon = validate_polygon(expected_poly)
        assert res.equals(expected_poly)

    with subtests.test(msg="Dictionary model validation"):
        res = validate_polygon(sample_geojson)
        assert res.equals(expected_poly)

    with subtests.test(msg="Raw JSON string conversion"):
        raw_str: str = json.dumps(sample_geojson)
        res = validate_polygon(raw_str)
        assert res.equals(expected_poly)


def test_validate_polygon_raises_type_error_for_invalid_input() -> None:
    """Verify that unconvertible object payloads throw an explicit TypeError."""
    invalid_input: list[float] = [0.0, 0.0, 1.0, 1.0]

    with pytest.raises(TypeError, match="Cannot convert"):
        validate_polygon(invalid_input)  # type: ignore[arg-type]
