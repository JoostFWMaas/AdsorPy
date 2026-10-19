# Copyright (c) 2025-2026 Contributors to the AdsorPy project.
# SPDX-License-Identifier: MIT
"""Interface for Schrödinger."""

from __future__ import annotations

import json
from sys import version_info
from typing import Annotated, cast

import numpy as np

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
from shapely import MultiPoint, Polygon

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


class ParallelogramSurfaceTransformer:
    """Transform a parallelogram surface into a rectangle surface while preserving relative coordinates of points."""

    def __init__(self, surface_parallelogram: Polygon) -> None:
        """Initialise the parallelogram to rectangle transformer.

        :param surface_parallelogram: Original surface parallelogram.
        """
        self.original_poly = shapely.normalize(surface_parallelogram)

        coords_sequence = np.sort(self.original_poly.exterior.coords, axis=0)

        self.cut_x = coords_sequence[-2, 0]
        self.cut_width = coords_sequence[-1, 0] - self.cut_x
        self.box_width = coords_sequence[-2, 0] - coords_sequence[0, 0]
        self.box_height = coords_sequence[-1, 1] - coords_sequence[0, 1]
        self.shift_vector = np.asarray([self.box_width, 0])
        self.origin_offset = coords_sequence[0]
        self.rectangle_poly = shapely.box(0.0, 0.0, self.box_width, self.box_height)

        # This state tracking mask ensures exact index mapping on reversal
        self._was_shifted_mask: np.ndarray | None = None

    def forward_points(self, multipoint_src: MultiPoint) -> tuple[Polygon, MultiPoint]:
        """Transform points into the rectangle and record which indices moved.

        :param multipoint_src: MultiPoint collection for which the coordinates have to be transformed.
        :returns: tuple:
            1) Surface rectangle polygon.
            2) Transformed MultiPoint object.
        """
        pt_coords = np.array([pt.coords[0] for pt in multipoint_src.geoms])

        # Track exactly which point coordinates were on the left side of the cut line.
        self._was_shifted_mask = pt_coords[:, 0] > self.cut_x

        # Apply the forward shift vector to the selected coordinates.
        pt_coords[self._was_shifted_mask] -= self.shift_vector
        pt_coords -= self.origin_offset

        return self.rectangle_poly, MultiPoint(pt_coords)

    def inverse_points(self, transformed_multipoint: MultiPoint) -> tuple[Polygon, MultiPoint]:
        """Use the cached mask to snap the points back to their original position.

        :param transformed_multipoint: MultiPoint collection of transformed surface sites.
        :returns: tuple:
            1) Surface parallelogram polygon.
            2) Restored MultiPoint object.
        """
        if self._was_shifted_mask is None:
            errmsg = "Cannot invert: No forward transformation has been performed yet."
            raise ValueError(errmsg)

        pt_coords = np.array([pt.coords[0] for pt in transformed_multipoint.geoms])

        pt_coords += self.origin_offset

        # Add the shift vector to the indices that originally moved.
        pt_coords[self._was_shifted_mask] += self.shift_vector

        return self.original_poly, MultiPoint(pt_coords)
