# Copyright (c) 2025-2026 Contributors to the AdsorPy project.
# SPDX-License-Identifier: MIT
"""Interface for Schrödinger."""

from __future__ import annotations

import json
from sys import version_info
from typing import Annotated, cast

if version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self  # NOQA: TC002

import shapely
from pydantic import (
    Field,
    FilePath,
    FiniteFloat,
    RootModel,
    ValidationInfo,
    model_validator,
    validate_call,
)
from shapely import Polygon

Point2D = Annotated[list[FiniteFloat], Field(min_length=2, max_length=2)]


class MoleculeParser(RootModel[dict[str, list[Point2D]]]):
    """Parses and validates molecule coordinate maps into Shapely polygons.

    Accepts a dictionary layout of {molecule_name: coordinates_list} and
    automatically applies a concave hull fallback for invalid polygons.
    """

    # Allows storing raw Shapely Polygon instances within Pydantic fields
    model_config = {"arbitrary_types_allowed": True}

    _polygons: dict[str, Polygon] = {}

    @property
    def polygons(self) -> dict[str, Polygon]:
        """Accessor to get the computed and repaired Shapely polygons.

        :returns: dictionary of polygons.
        """
        return self._polygons

    @classmethod
    @validate_call
    def from_json(
        cls,
        file_path: FilePath,
        ratio: Annotated[float, Field(ge=0.0, le=1.0)] = 0.05,
    ) -> Self:
        """Load a JSON file from a string or Path and parses the molecule footprint data.

        :param file_path: The string or Path pointing to the JSON file.
        :param ratio: Ratio between 0 and 1 (inclusive) for shapely.concave_hull.
        :returns: Validated molecule footprint.
        """
        with file_path.open("r", encoding="utf-8") as f:
            raw_data = json.load(f)

        return cls.model_validate(raw_data, context={"ratio": ratio})

    @model_validator(mode="after")
    def generate_polygons(self, info: ValidationInfo) -> Self:
        """Convert structural coordinate maps into repaired Shapely Polygons."""
        context: dict[str, float] = info.context or {}
        ratio = context.get("ratio", 0.05)

        self._polygons = {}
        # self.root holds the validated dict[str, list[Point2D]] directly
        for name, coordinates in self.root.items():
            footprint = Polygon(coordinates)

            if not footprint.is_valid:
                footprint = cast("Polygon", shapely.concave_hull(footprint, ratio=ratio))

            self._polygons[name] = footprint

        return self
