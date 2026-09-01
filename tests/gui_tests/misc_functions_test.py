# Copyright (c) 2025-2026 Contributors to the AdsorPy project.
# SPDX-License-Identifier: MIT
"""Test the functions of the `gui.py` module."""

import importlib
import json
import re
import sys
from collections.abc import Callable
from importlib.metadata import requires

import pytest
import shapely.errors
from PySide6.QtWidgets import QDoubleSpinBox, QLineEdit, QSpinBox, QWidget
from pytestqt.qtbot import QtBot
from shapely.geometry import Polygon

from adsorpy.gui import FilePickerWidget, extract_param_docs, from_geojson_str_to_polygon, set_content, validate_polygon


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
        vali = 42
        set_content(spinbox, vali)
        assert spinbox.value() == vali

    with subtests.test(msg="QDoubleSpinBox float assignment"):
        valf = 3.14
        set_content(double_spinbox, valf)
        assert double_spinbox.value() == pytest.approx(valf)

    with subtests.test(msg="QLineEdit string assignment"):
        text = "Hello World"
        set_content(line_edit, text)
        assert line_edit.text() == text

    with subtests.test(msg="FilePickerWidget string assignment"):
        file = "/path/to/file.xyz"
        set_content(file_picker, file)
        assert file_picker.text() == file

    with subtests.test(msg="QLineEdit list of strings assignment"):
        string_list = ["A", "B", "C"]
        set_content(line_edit, string_list)
        assert line_edit.text() == ",".join(string_list)


@pytest.mark.parametrize(
    ("widget_factory", "invalid_content"),
    [
        (QSpinBox, "not an int"),
        (QDoubleSpinBox, 12),  # Expects float, not an int
        (FilePickerWidget, ["list", "not", "string"]),
    ],
)
def test_set_content_mismatch_raises_value_error(
    qtbot: QtBot,
    widget_factory: Callable[[], QWidget],
    invalid_content: str | list[str] | int,
) -> None:
    """Verify that incompatible widget-content pairs throw a clear ValueError.

    :param qtbot: The pytest-qt robot fixture used to manage GUI lifecycle.
    :param widget_factory: A factory function to clean-instantiate widgets per parameter pass.
    :param invalid_content: The malformed payload intended to break the pattern match.
    """
    widget: QWidget = widget_factory()
    qtbot.addWidget(widget)

    with pytest.raises(ValueError, match="Widget and content mismatch"):
        set_content(widget, invalid_content)  # pyright: ignore[reportArgumentType]


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

    with pytest.raises(ValueError, match=r"Docstring of undocumented_func is not defined."):
        extract_param_docs(undocumented_func)


@pytest.fixture
def sample_geojson() -> dict[str, str | list[list[list[float]]]]:
    """Provide a standard square geometry payload structured in a GeoJSON style dictionary.

    :returns: A raw dictionary describing a square polygon layout footprint.
    """
    return {"type": "Polygon", "coordinates": [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]]}


def test_validate_polygon_success_cases(
    sample_geojson: dict[str, str | list[list[list[float]]]],
    subtests: pytest.Subtests,
) -> None:
    """Verify that shape logic accepts Polygon objects, dictionary models, and raw JSON strings.

    :param sample_geojson: A preconstructed geometric square template payload fixture.
    :param subtests: The pytest subtests context manager fixture.
    """
    expected_poly: Polygon = from_geojson_str_to_polygon(json.dumps(sample_geojson))

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
        validate_polygon(invalid_input)  # pyright: ignore[reportArgumentType]


def test_from_geojson_str_to_polygon_errors() -> None:
    """Test the from_geojson_str_to_polygon function error handling."""
    with pytest.raises(shapely.errors.GEOSException, match=r"ParseException: Error parsing JSON:.*"):
        from_geojson_str_to_polygon("")

    point = shapely.Point((0.0, 0.0))
    pointstr = str(shapely.to_geojson(point))

    with pytest.raises(TypeError, match="Geometry is of wrong type: Point"):
        from_geojson_str_to_polygon(pointstr)

    polygon = shapely.Polygon([(1, 1), (0, 0), (1, 0), (0, 1)])
    polygonstr = str(shapely.to_geojson(polygon))

    with pytest.raises(ValueError, match=r"Polygon is invalid. Exterior coordinates: .*"):
        from_geojson_str_to_polygon(polygonstr)


def get_gui_dependencies() -> list[str]:
    """Extract dependency names safely from the installed package metadata.

    :returns: A list of optional dependency names.
    """
    raw_requirements: list[str] = requires("adsorpy") or []
    gui_dependencies_name = "gui-deps"

    clean_deps: list[str] = []
    for req in raw_requirements:
        if re.search(rf"extra\s*==\s*['\"]{gui_dependencies_name}['\"]", req):
            # Extract only the alphabetic module name at the very beginning
            match = re.match(r"^([a-zA-Z0-9_-]+)", req)
            if match:
                clean_deps.append(match.group(1))

    return clean_deps


@pytest.mark.parametrize("missing_dep", get_gui_dependencies())
def test_missing_gui_imports(monkeypatch: pytest.MonkeyPatch, missing_dep: str) -> None:
    """Test whether missing imports are handled correctly.

    :param monkeypatch: Pytest monkeypatch fixture to mock parameters.
    :param missing_dep: The optional dependency name to remove and check.
    """
    monkeypatch.setitem(sys.modules, missing_dep, None)
    monkeypatch.delitem(sys.modules, "adsorpy.gui", raising=False)
    dask_imports = [name for name in sys.modules if missing_dep in name]
    for sub_dep in dask_imports:
        monkeypatch.setitem(sys.modules, sub_dep, None)

    with pytest.raises(ImportError, match=missing_dep):
        importlib.import_module("adsorpy.gui")
