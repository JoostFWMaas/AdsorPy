# Copyright (c) 2025-2026 Contributors to the AdsorPy project.
# SPDX-License-Identifier: MIT
"""Type aliases for the adsorpy library."""

from __future__ import annotations  # This allows for delayed hinting of classes.

from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np  # For vectorised computations (performed in C).
from numpy.typing import NDArray
from shapely import Polygon
from shapely.prepared import PreparedGeometry  # For vectorised inclusion checks.

# Definition of some frequently-used types. Not used by the compiler, just for the user, mypy, and pyright. Hello user!
BoolArray: TypeAlias = np.ndarray[tuple[int], np.dtype[np.bool_]]  # Flat Boolean array.
"""1D array of bools."""
Bools2D: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.bool_]]
"""2D array of bools."""
BufferArray: TypeAlias = np.ndarray[tuple[int, int], np.dtype[Polygon]]  # pyright: ignore[reportInvalidTypeArguments]
"""2D array of Polygons. Used for molecules with (different) buffer radii."""
CoordPair: TypeAlias = np.ndarray[tuple[Literal[2], Literal[1]], np.dtype[np.float64]]  # 2x1 array of coordinates
"""2x1 array of a single coordinate pair."""
CoordsArray: TypeAlias = np.ndarray[tuple[Literal[2], int], np.dtype[np.float64]]  # 2xN array of coords.
"""2xn array of coordinates."""
CoordsArray3D = np.ndarray[tuple[Literal[3], int], np.dtype[np.float64]]  # 2 or 3 x N array of coords.
"""3xn array of floats. 3D coordinates."""
DistArray: TypeAlias = np.ndarray[tuple[int], np.dtype[np.float64]]
"""1D array of floats. Often used to denote distances, so usually positive."""
FloatArray: TypeAlias = NDArray[np.float64]
"""Array of floats. No dimensions given."""
GeoArray: TypeAlias = np.ndarray[tuple[int], np.dtype[Polygon]]  # pyright: ignore[reportInvalidTypeArguments]
"""1D array of Polygons."""
IdxArray: TypeAlias = np.ndarray[tuple[int], np.dtype[np.int_]]  # Flat index aray of integers.
"""1D array of ints, used for indices."""
InDict: TypeAlias = dict[str, str | float | list[str] | Path | None]
"""A dictionary and the values that can be found in it."""
PrepGeoArray: TypeAlias = np.ndarray[tuple[int], np.dtype[PreparedGeometry]]  # pyright: ignore[reportInvalidTypeArguments, reportMissingTypeArgument]
"""1D array of PreparedGeometry[Polygon]s, used for fast vectorised evaluation of overlap and coordinates."""
RotMatrix: TypeAlias = np.ndarray[tuple[Literal[3], Literal[3]], np.dtype[np.float64]]
"""3x3 array of floats, used as a rotation matrix."""
