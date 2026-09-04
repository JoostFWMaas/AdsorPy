# Copyright (c) 2025-2026 Contributors to the AdsorPy project.
# SPDX-License-Identifier: MIT
"""Test the PydanticPolygon class of the `gui.py` module."""

import json

import pytest
from pydantic import BaseModel, TypeAdapter
from shapely.geometry import Polygon

from adsorpy.gui import PydanticPolygon, from_geojson_str_to_polygon


# Create a dummy model to test field integration lifecycle safely
class SimulationGeometryModel(BaseModel):
    """Test model targeting custom geometry lifecycle pipelines."""

    footprint: PydanticPolygon


polygon_adapter = TypeAdapter(PydanticPolygon)


@pytest.fixture
def valid_geojson_dict() -> dict[str, str | list[list[list[float]]]]:
    """Provide a standard square geometry payload structured in a GeoJSON style dictionary.

    :returns: A raw dictionary describing a square polygon layout footprint.
    """
    return {
        "type": "Polygon",
        "coordinates": [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]],
    }


def test_pydantic_polygon_validation_success_types(
    valid_geojson_dict: dict[str, str | list[list[list[float]]]],
    subtests: pytest.Subtests,
) -> None:
    """Verify model can validate Shapely instances, GeoJSON dicts, or strings.

    :param valid_geojson_dict: A preconstructed geometric square template payload fixture.
    :param subtests: The pytest subtests context manager fixture.
    """
    expected_shape: Polygon = from_geojson_str_to_polygon(json.dumps(valid_geojson_dict))

    with subtests.test(msg="Validating an existing Shapely Polygon instance"):
        model_from_instance: SimulationGeometryModel = SimulationGeometryModel(
            footprint=polygon_adapter.validate_python(expected_shape),
        )
        assert isinstance(model_from_instance.footprint, Polygon)
        assert model_from_instance.footprint.equals(expected_shape)

    with subtests.test(msg="Validating a dictionary structure input"):
        model_from_dict: SimulationGeometryModel = SimulationGeometryModel(
            footprint=polygon_adapter.validate_python(valid_geojson_dict),
        )
        assert model_from_dict.footprint.equals(expected_shape)

    with subtests.test(msg="Validating a raw JSON string description input"):
        json_str: str = json.dumps(valid_geojson_dict)
        model_from_str: SimulationGeometryModel = SimulationGeometryModel(
            footprint=polygon_adapter.validate_strings(json_str),
        )
        assert model_from_str.footprint.equals(expected_shape)


def test_pydantic_polygon_validation_failure_raises_error() -> None:
    """Verify that unconvertible object payloads trigger a standard Pydantic ValidationError."""
    malformed_input: list[float] = [10.0, 20.0, 30.0]

    with pytest.raises(TypeError, match="Invalid input for Polygon"):
        SimulationGeometryModel(footprint=malformed_input)


def test_pydantic_polygon_serialisation(valid_geojson_dict: dict[str, str | list[list[list[float]]]]) -> None:
    """Verify model dump capabilities turn geometric schemas back into plain dictionaries."""
    expected_shape: Polygon = from_geojson_str_to_polygon(json.dumps(valid_geojson_dict))
    model: SimulationGeometryModel = SimulationGeometryModel(footprint=PydanticPolygon(expected_shape))

    serialised_data: dict[str, str | list[list[list[float]]]] = model.model_dump()
    reloaded_model = SimulationGeometryModel(**serialised_data)  # pyright: ignore[reportArgumentType]

    assert reloaded_model == model, "Polygon data changed by saving and loading."
