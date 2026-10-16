# Copyright (c) 2025-2026 Contributors to the AdsorPy project.
# SPDX-License-Identifier: MIT
"""Test parallel to square surface support."""

from __future__ import annotations

import numpy as np
import shapely
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from shapely import MultiPoint
from shapely.affinity import scale, skew, translate

from adsorpy.run_simulation import run_simulation
from adsorpy.scripts.schrodinger_interface import ParallelogramSurfaceTransformer
from adsorpy.types import CoordsArray


def run_transformer_pipeline(unit_points: CoordsArray, shear_factor: float, scale_factor: float, seed: int) -> bool:
    """Core logic under test, mapping unit points into a sheared and scaled parallelogram.

    :param unit_points: Coordinate points set in the unit square.
    :param shear_factor: Shear factor to transform the unit square in the x-direction.
    :param scale_factor: Scale factor to transform the unit square in the x and y direction.
    :returns: True if there is no molecule overlap, False otherwise.
    """
    # Define a generic 1x1 box
    unit_square = shapely.box(0, 0, 1, 1)

    parallelogram = skew(unit_square, shear_factor, origin=(0, 0))
    parallelogram = scale(parallelogram, scale_factor, scale_factor, origin=(0, 0))

    transformed_x = unit_points[:, 0] + unit_points[:, 1] * np.tan(shear_factor * np.pi / 180)
    transformed_x *= scale_factor
    transformed_y = scale_factor * unit_points[:, 1]
    valid_points = np.stack([transformed_x, transformed_y], axis=1)

    original_points = MultiPoint(valid_points)
    transformer = ParallelogramSurfaceTransformer(parallelogram)

    # Forward transformation
    rect_poly, rect_points = transformer.forward_points(original_points)
    xval = [p.x for p in rect_points.geoms]
    yval = [p.y for p in rect_points.geoms]
    box = rect_poly.bounds

    # Run the simulation
    output = run_simulation(
        lattice_type="custom",
        site_x_coords=np.asarray(xval),
        site_y_coords=np.asarray(yval),
        bounding_x_coord=box[2],
        bounding_y_coord=box[3],
        boundary_condition="periodic",
        seed=seed,
    )[-1]

    # Inverse transformation and translation alignment
    _, restored_points = transformer.inverse_points(rect_points)
    existing = output.mol_data.stored_data["exists"]
    orig_coords = output.mol_data.coords[:, existing]
    orig_idx = output.mol_data.stored_data["grid_idx"][existing]
    displayable_molecules = output.mol_data.stored_data["polygon"][existing]

    for idx, (mol, orig) in enumerate(zip(displayable_molecules, orig_idx, strict=True)):
        xvals = orig_coords[0, idx] - restored_points.geoms[orig].x
        yvals = orig_coords[1, idx] - restored_points.geoms[orig].y
        displayable_molecules[idx] = translate(mol, -xvals, -yvals)

    # Check for overlapping molecules
    for idx, mol in enumerate(displayable_molecules):
        for mol2 in displayable_molecules[:idx]:
            if mol.intersects(mol2):
                return False

    return True


# Define a strategy to generate point coordinates bounded within a strict 1x1 unit box
unit_point_strategy = arrays(
    dtype=np.float64,
    shape=st.tuples(st.integers(min_value=5, max_value=200), st.just(2)),
    elements=st.floats(min_value=0.0, max_value=1.0, exclude_max=True, allow_nan=False, allow_infinity=False),
)

# Define a strategy for the X-axis shear factor
shear_strategy = st.floats(min_value=0, max_value=45)

# Define a strategy for the uniform scaling factor
scale_strategy = st.floats(min_value=1.0, max_value=10.0)

seed_strategy = st.integers(min_value=0)


@settings(deadline=None)
@given(unit_points=unit_point_strategy, shear_factor=shear_strategy, scale_factor=scale_strategy, seed=seed_strategy)
def test_transformer_with_sheared_and_scaled_points(
    unit_points: CoordsArray,
    shear_factor: float,
    scale_factor: float,
    seed: int,
) -> None:
    """Test whether the simulation runs correctly when the plot is severely sheared.

    :param unit_points: Coordinate points set in the unit square.
    :param shear_factor: Shear factor to transform the unit square in the x-direction.
    :param scale_factor: Scale factor to transform the unit square in the x and y direction.
    :param seed: Seed of the run_simulation function.
    """
    # Run core logic utilising mapped spatial distribution points
    no_overlap = run_transformer_pipeline(unit_points, shear_factor, scale_factor, seed)

    # Assert property invariant holds true
    assert no_overlap, (
        f"Molecules overlapped under inverse transform with "
        f"shear factor: {shear_factor} and scale factor: {scale_factor}"
    )
