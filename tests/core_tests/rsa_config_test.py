# Copyright (c) 2025-2026 Contributors to the AdsorPy project.
# SPDX-License-Identifier: MIT
"""Test the `rsa_config`` module."""

import json
from pathlib import Path
from typing import TypedDict

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from adsorpy.rsa_config import LoggingConfig, RsaConfig, WrappedValue


class MockConfig(TypedDict):
    """Mock RSAConfig."""

    logging: dict[str, bool]
    sites: dict[str, int | None]
    xsize: dict[str, float | None]
    ysize: dict[str, float | None]
    zsize: dict[str, float | None]
    max_molecule_count: dict[str, int]
    lattice_a: dict[str, float]
    boundary_type: dict[str, str]
    sticking_probability: dict[str, float]


def get_valid_raw_data() -> MockConfig:
    """Get valid data to mock input.

    :return: Mock data.
    """
    return MockConfig(
        logging={"enabled": True},
        sites={"value": 100},
        xsize={"value": None},
        ysize={"value": None},
        zsize={"value": None},
        max_molecule_count={"value": 50},
        lattice_a={"value": 1.5},
        boundary_type={"value": "periodic"},
        sticking_probability={"value": 0.8},
    )


def test_strip_comments_middleware() -> None:
    """Ensure '_comment' keys are stripped from LoggingConfig and WrappedValue."""
    value = 5
    data_with_comments = {"enabled": True, "_comment_1": "ignore me", "meta_comment": "keep me"}
    config = LoggingConfig.model_validate(data_with_comments)
    assert config.enabled is True
    # _comment_1 should be stripped, meta_comment passes but is ignored by Pydantic fields

    wrapped_data = {"value": value, "_comment_test": "delete"}
    wrapped = WrappedValue[int].model_validate(wrapped_data)
    assert wrapped.value == value


@pytest.mark.parametrize(
    ("sites", "xsize", "ysize", "should_pass", "errmsg"),
    [
        (100, None, None, True, "value error"),  # Case A: Sites only.
        (None, 10.5, 10.5, True, "value error"),  # Case B: Sizes only.
        (100, 10.5, 10.5, False, "value error"),  # Case C: Both specified (Mutually Exclusive Error).
        (None, None, None, False, "value error"),  # Case D: Neither specified (Missing Dimensions Error).
        (None, 10.5, None, False, "value error"),  # Case E: Missing ysize.
        (0, None, None, False, "validation error"),  # Case F: Illegal site count.
        (None, -1.0, 1.0, False, "validation error"),  # Case H: Illegal grid dimensions.
    ],
)
def test_dimension_mutual_exclusivity(sites: int, xsize: float, ysize: float, should_pass: bool, errmsg: str) -> None:
    """Enforce correct configuration bounds for sites vs spatial dimensions."""
    base_data = get_valid_raw_data()
    base_data["sites"]["value"] = sites
    base_data["xsize"]["value"] = xsize
    base_data["ysize"]["value"] = ysize

    if should_pass:
        model = RsaConfig(**base_data)
        assert model.sites.value == sites
    else:
        with pytest.raises(ValidationError) as exc_info:
            RsaConfig(**base_data)
        assert errmsg in str(exc_info.value).lower()


def test_illegal_boundary_type() -> None:
    """Test that illegal boundary types are handled correctly."""
    base_data = get_valid_raw_data()
    base_data["boundary_type"]["value"] = "dummy type"

    with pytest.raises(ValidationError, match="dummy type"):
        RsaConfig(**base_data)


def test_json_file_loading_constructor(tmp_path: Path) -> None:
    """Verify that file-path based initialisation behaves correct.

    :param tmp_path: Temporary directory for testing.
    """
    json_file = tmp_path / "config.json"
    raw_data = get_valid_raw_data()

    json_file.write_text(json.dumps(raw_data))

    config = RsaConfig(config_path=json_file)

    config_dict = config.model_dump()
    assert config_dict == raw_data


def test_invalid_json_file_path(tmp_path: Path) -> None:
    """Verify standard Pydantic FilePath routing errors for invalid paths.

    :param tmp_path: Temporary directory for testing.
    """
    illegal_path_name = "bad_file_42.json"
    with pytest.raises(ValidationError, match=illegal_path_name):
        RsaConfig(config_path=tmp_path / illegal_path_name)


# Build strategy matches for strict validation leaves
st_boundary = st.sampled_from(["periodic", "hard", "soft"])
st_none = st.none()


@given(
    enabled=st.booleans(),
    sites=st.one_of(st_none, st.integers(min_value=1, max_value=10000)),
    xsize=st.one_of(st_none, st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False)),
    ysize=st.one_of(st_none, st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False)),
    zsize=st.one_of(st_none, st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False)),
    max_molecules=st.integers(min_value=0, max_value=5000),
    lattice_a=st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    boundary=st_boundary,
    sticking_prob=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
def test_rsa_config_properties(
    enabled: bool,
    sites: int | None,
    xsize: float | None,
    ysize: float | None,
    zsize: float | None,
    max_molecules: int,
    lattice_a: float,
    boundary: str,
    sticking_prob: float,
) -> None:
    """Fuzz input primitives globally to catch cross-cutting constraint failures.

    :param enabled: Is logging enabled?
    :param sites: Site count.
    :param xsize: Size in the x direction.
    :param ysize: Size in the y direction.
    :param zsize: Size in the z direction.
    :param max_molecules: Max number of molecules.
    :param lattice_a: Lattice constant.
    :param boundary: Boundary type.
    :param sticking_prob: Sticking probability.
    """
    # Enforce valid dimension combinations programmatically for the property runner
    has_sites = sites is not None
    has_sizes = xsize is not None and ysize is not None

    # If the randomly generated pair is invalid, explicitly force it to valid structure
    if not has_sites and not has_sizes:
        sites = 500  # Fallback valid state
    elif has_sites and has_sizes:
        xsize = None
        ysize = None

    payload = {
        "logging": {"enabled": enabled},
        "sites": {"value": sites},
        "xsize": {"value": xsize},
        "ysize": {"value": ysize},
        "zsize": {"value": zsize},
        "max_molecule_count": {"value": max_molecules},
        "lattice_a": {"value": lattice_a},
        "boundary_type": {"value": boundary},
        "sticking_probability": {"value": sticking_prob},
    }
    config = RsaConfig(**payload)
    assert config.logging.enabled == enabled
    assert config.boundary_type.value == boundary

    config_dict = config.model_dump()
    assert config_dict == payload
