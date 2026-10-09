# Copyright (c) 2025-2026 Contributors to the AdsorPy project.
# SPDX-License-Identifier: MIT
"""Interface for Schrödinger."""

from __future__ import annotations

import json
from sys import version_info
from typing import TYPE_CHECKING, Annotated, cast

import numpy as np
from shapely.affinity import translate
from shapely.plotting import plot_points, plot_polygon

if version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self  # NOQA: TC002

import shapely
from matplotlib import pyplot as plt
from pydantic import (
    Field,
    FilePath,
    FiniteFloat,
    RootModel,
    ValidationInfo,
    model_validator,
    validate_call,
)
from shapely import MultiPoint, Polygon, prepare

from adsorpy.run_simulation import run_simulation

if TYPE_CHECKING:
    from adsorpy.randomsequentialadsorption import Simulator

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


class ParallelogramTransformer:
    """Transform a parallelogram surface into a rectangle surface while preserving relative coordinates of points."""

    def __init__(self, surface_parallelogram: Polygon) -> None:
        """Initialise the parallelogram to rectangle transformer.

        :param surface_parallelogram: Original surface parallelogram.
        """
        self.original_poly = surface_parallelogram

        # Compute and cache transformation parameters.
        coords = list(surface_parallelogram.exterior.coords)

        bl, br, tl = map(np.array, [coords[0], coords[1], coords[3]])

        v_base = br - bl
        v_side = tl - bl
        projection_factor = np.dot(v_side, v_base) / np.dot(v_base, v_base)

        self.cut_x = float(bl[0] + projection_factor * v_base[0])
        self.shift_vector = np.array([float(v_base[0]), float(v_base[1])])

        # Build the final rectangle polygon geometry.
        min_x, min_y, _, max_y = surface_parallelogram.bounds
        cutter_box = Polygon(
            [(min_x - 1, min_y - 1), (self.cut_x, min_y - 1), (self.cut_x, max_y + 1), (min_x - 1, max_y + 1)],
        )
        left_poly = surface_parallelogram.intersection(cutter_box)
        right_poly = surface_parallelogram.difference(cutter_box)
        self.rectangle_poly = shapely.box(
            *right_poly.union(
                translate(left_poly, xoff=self.shift_vector[0], yoff=self.shift_vector[1]),
            ).bounds,
        )

        # This state tracking mask ensures exact index mapping on reversal
        self._was_shifted_mask: np.ndarray | None = None

    def forward_points(self, multipoint_src: MultiPoint) -> tuple[Polygon, MultiPoint]:
        """Transform points into the rectangle and record which indices moved.

        :param multipoint_src: MultiPoint collection for which the coordinates have to be transformed.
        :returns: tuple.
            1) Surface rectangle polygon.
            2) Transformed MultiPoint object.
        """
        pt_coords = np.array([pt.coords[0] for pt in multipoint_src.geoms])

        # Track exactly which point coordinates were on the left side of the cut line.
        self._was_shifted_mask = pt_coords[:, 0] < self.cut_x

        # Apply the forward shift vector to the selected coordinates.
        pt_coords[self._was_shifted_mask] += self.shift_vector

        return self.rectangle_poly, MultiPoint(pt_coords)

    def inverse_points(self, transformed_multipoint: MultiPoint) -> tuple[Polygon, MultiPoint]:
        """Use the cached mask to snap the points back to their original position."""
        if self._was_shifted_mask is None:
            errmsg = "Cannot invert: No forward transformation has been performed yet."
            raise ValueError(errmsg)

        pt_coords = np.array([pt.coords[0] for pt in transformed_multipoint.geoms])

        # Subtract the shift vector only from the indices that originally moved.
        pt_coords[self._was_shifted_mask] -= self.shift_vector

        return self.original_poly, MultiPoint(pt_coords)


class SimulationOutputParser:
    """Parser for the output of an adsorpy simulation."""

    def __init__(self, simulator: Simulator) -> None:
        """Initialise the result parser.

        :param simulator: The Simulator class.
        """
        self.simulator = simulator


def testvals() -> None:
    """Run a few tests."""
    parallelogram = Polygon([(0, 0), (5, 0), (7, 4), (2, 4)])
    rng = np.random.default_rng()
    rand_spam = rng.random((10, 2))
    rand_spam[:, 0] *= parallelogram.bounds[2]
    rand_spam[:, 1] *= parallelogram.bounds[3]

    prepare(parallelogram)
    filtered_spam = shapely.contains_xy(parallelogram, rand_spam)
    rand_spam = rand_spam[filtered_spam]
    original_points = MultiPoint(rand_spam)

    transformer = ParallelogramTransformer(parallelogram)

    rect_poly, rect_points = transformer.forward_points(original_points)
    xval: list[float] = []
    yval: list[float] = []
    point: shapely.Point
    for point in rect_points.geoms:
        xval.append(point.x)
        yval.append(point.y)
    box = rect_poly.bounds

    print("--- FORWARD TRANSFORMATION ---")
    print("Rectangle Points WKT:", rect_points.wkt)
    plot_polygon(rect_poly, fc="none")
    plot_points(rect_points)

    plt.show()

    output = run_simulation(
        lattice_type="custom",
        site_x_coords=np.asarray(xval),
        site_y_coords=np.asarray(yval),
        bounding_x_coord=box[2],
        bounding_y_coord=box[3],
    )[-1]

    print(output.coverage)

    restored_poly, restored_points = transformer.inverse_points(rect_points)
    print("\n--- INVERSE (UNDO) TRANSFORMATION ---")
    print("Restored Points WKT: ", restored_points.wkt)
    print("Matches Original?    ", restored_points.equals_exact(original_points, tolerance=0.0005))
    plot_polygon(restored_poly, fc="none")
    plot_points(restored_points)
    plt.show()

    # sf = SurfaceParser(affine_box)
    # print(sf.bounds)
    # plot_polygon(Polygon(affine_box))
    # plot_polygon(shapely.box(*sf.bounds))
    # from matplotlib import pyplot as plt
    # plt.show()


if __name__ == "__main__":
    testvals()
