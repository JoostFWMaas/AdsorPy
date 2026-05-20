"""Test the .xyz file parsing by fuzzing input."""

from src.adsorpy.molecule_lib import _xyz_verifier, RADII

import numpy as np
import pytest
from hypothesis import given, strategies as st
from hypothesis.extra.numpy import arrays

VALID_KEYS = list(RADII.keys())

# ---- Strategies ----

@st.composite
def valid_xyz_inputs(draw):
    n = draw(st.integers(min_value=1, max_value=50))

    atomkeys = np.array(
        draw(st.lists(st.sampled_from(VALID_KEYS), min_size=n, max_size=n)),
        dtype=str
    )

    atompos = draw(
        arrays(
            dtype=np.float64,
            shape=(n, 3),
            elements=st.floats(allow_nan=False, allow_infinity=False)
        )
    )

    return atomkeys, atompos, np.int_(n)


@st.composite
def invalid_xyz_inputs(draw):
    n = draw(st.integers(min_value=0, max_value=50))

    # Maybe invalid keys
    atomkeys = np.array(
        draw(st.lists(st.text(min_size=1, max_size=2), min_size=n, max_size=n)),
        dtype=str
    )

    # Random shape (may violate 3D rule)
    dim2 = draw(st.integers(min_value=1, max_value=5))
    atompos = draw(
        arrays(
            dtype=np.float64,
            shape=(n, dim2),
            elements=st.floats(allow_nan=True, allow_infinity=True)
        )
    )

    listed_count = draw(
        st.one_of(
            st.none(),
            st.integers(min_value=-5, max_value=100)
        )
    )

    return atomkeys, atompos, listed_count


# ---- Tests ----

@given(valid_xyz_inputs())
def test_xyz_verifier_valid(data):
    atomkeys, atompos, count = data

    # Should not raise
    _xyz_verifier(atomkeys, atompos, count)


@given(invalid_xyz_inputs())
def test_xyz_verifier_invalid(data):
    atomkeys, atompos, count = data

    # Most invalid combinations should raise
    with pytest.raises(ValueError):
        _xyz_verifier(atomkeys, atompos, count)