# Copyright (c) 2025-2026 Contributors to the AdsorPy project.
# SPDX-License-Identifier: MIT
"""Test the .xyz file parsing by fuzzing input."""

from collections.abc import Callable
from typing import Literal, TypeVar, cast

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import SearchStrategy

from src.adsorpy.molecule_lib import RADII, _xyz_verifier

T = TypeVar("T")

VALID_KEYS = list(RADII.keys())

# ---- Strategies ----


@st.composite
def valid_xyz_inputs(
    draw: Callable[[SearchStrategy[T]], T],
) -> tuple[
    np.ndarray[tuple[int], np.dtype[np.str_]],
    np.ndarray[tuple[int, Literal[3]], np.dtype[np.float64]],
    np.int_,
]:
    """Generate valid .xyz inputs.

    :param draw: method turning strategies into values.
    :returns:
        1) atom keys drawn from RADII.
        2) 3D atom coordinates.
        3) the atom count.
    """
    num: int = cast("int", draw(st.integers(min_value=1, max_value=50)))  # pyright: ignore[reportArgumentType]

    atomkeys: np.ndarray[tuple[int], np.dtype[np.str_]] = np.array(
        draw(st.lists(st.sampled_from(VALID_KEYS), min_size=num, max_size=num)),  # pyright: ignore[reportArgumentType]
        dtype=str,
    )

    atompos: np.ndarray[tuple[int, Literal[3]], np.dtype[np.float64]] = cast(
        "np.ndarray[tuple[int, Literal[3]], np.dtype[np.float64]]",
        draw(
            arrays(  # pyright: ignore[reportArgumentType]
                dtype=np.float64,
                shape=(num, 3),
                elements=st.floats(allow_nan=False, allow_infinity=False),
            ),
        ),
    )

    return atomkeys, atompos, np.int_(num)


@st.composite
def invalid_xyz_inputs(
    draw: Callable[[SearchStrategy[T]], T],
) -> tuple[
    np.ndarray[tuple[int], np.dtype[np.str_]],
    np.ndarray[tuple[int, Literal[1, 2, 3, 4, 5]], np.dtype[np.float64]],
    int | None,
]:
    """Generate invalid .xyz inputs.

    :param draw: method turning strategies into values.
    :returns:
        1) atom keys, atom names randomly generated.
        2) 1D-5D atom coordinates.
        3) the atom count.
    """
    num: int = cast("int", draw(st.integers(min_value=0, max_value=50)))  # pyright: ignore[reportArgumentType]

    # Maybe invalid keys
    atomkeys: np.ndarray[tuple[int], np.dtype[np.str_]] = np.array(
        draw(st.lists(st.text(min_size=1, max_size=2), min_size=num, max_size=num)),  # pyright: ignore[reportArgumentType]
        dtype=str,
    )

    # Random shape (may violate 3D rule)
    dim2 = draw(st.integers(min_value=1, max_value=5))  # pyright: ignore[reportArgumentType]
    atompos: np.ndarray[tuple[int, Literal[1, 2, 3, 4, 5]], np.dtype[np.float64]] = cast(
        "np.ndarray[tuple[int, Literal[1, 2 ,3 ,4 ,5]], np.dtype[np.float64]]",
        draw(
            arrays(  # pyright: ignore[reportCallIssue]
                dtype=np.float64,
                shape=(num, dim2),  # pyright: ignore[reportArgumentType]
                elements=st.floats(allow_nan=True, allow_infinity=True),
            ),
        ),
    )

    listed_count = cast(
        "int",
        draw(
            st.one_of(  # pyright: ignore[reportArgumentType]
                st.none(),
                st.integers(min_value=-5, max_value=100),
            ),
        ),
    )

    return atomkeys, atompos, listed_count


# ---- Tests ----


@given(valid_xyz_inputs())
def test_xyz_verifier_valid(
    data: tuple[
        np.ndarray[tuple[int], np.dtype[np.str_]],
        np.ndarray[tuple[int, Literal[3]], np.dtype[np.float64]],
        np.int_,
    ],
) -> None:
    """Test correct response to valid input.

    :param data: method turning strategies into values.
    """
    atomkeys, atompos, count = data

    # Should not raise
    _xyz_verifier(atomkeys, atompos, count)


@given(invalid_xyz_inputs())
def test_xyz_verifier_invalid(
    data: tuple[
        np.ndarray[tuple[int], np.dtype[np.str_]],
        np.ndarray[tuple[int, Literal[1, 2, 3, 4, 5]], np.dtype[np.float64]],
        np.int_,
    ],
) -> None:
    """Test correct response to invalid input.

    :param data: method turning strategies into values.
    """
    atomkeys, atompos, count = data

    # Most invalid combinations should raise
    with pytest.raises(ValueError):  # noqa: PT011
        _xyz_verifier(atomkeys, atompos, count)  # type: ignore[arg-type]
