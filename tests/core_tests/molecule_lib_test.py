# Copyright (c) 2025-2026 Contributors to the AdsorPy project.
# SPDX-License-Identifier: MIT
"""Test the `molecule_lib`` module."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from pathlib import Path
from typing import ParamSpec, TypeVar

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.strategies import DataObject
from pydantic import ValidationError
from shapely import MultiPolygon, Polygon

from adsorpy import molecule_lib
from adsorpy.molecule_lib import _xyz_verifier
from adsorpy.types import CoordsArray3D, StrArray

P_mol = ParamSpec("P_mol")  # Helps with static type checkers.
T = TypeVar("T")

XYZ_FILE_PATH = Path(__file__).parents[1] / "test_data" / "fluorochloromethanol.xyz"

PYDANTIC_CUSTOM_STRATEGIES = {
    "PositiveFloat": st.floats(min_value=0.1, max_value=10, allow_nan=False, allow_infinity=False),
    "NonNegativeFloat": st.floats(min_value=0, max_value=10, allow_nan=False, allow_infinity=False),
    "PositiveInt": st.integers(min_value=3, max_value=10),
    "NonNegativeInt": st.integers(min_value=0, max_value=10),
    "FilePath": st.just(XYZ_FILE_PATH),
    "str | list[str] | None": st.one_of(st.text(min_size=0, max_size=2), st.none()),
    "float | None": st.none(),
    "float": st.floats(allow_nan=False, allow_infinity=False),
}


def resolve_param_strategy(annotation: type[T]) -> st.SearchStrategy[T | object]:
    """Resolve an annotation by handling generic Pydantic wrappers first."""
    # Check if the type directly matches one of the targets
    if str(annotation) in PYDANTIC_CUSTOM_STRATEGIES:
        return PYDANTIC_CUSTOM_STRATEGIES[str(annotation)]

    # Standard strategy inference fallback
    return st.from_type(annotation)


def _discover_molecule_functions() -> tuple[Callable[P_mol, Polygon], ...]:
    """Isolate reflection logic filtering usable library structural definitions.

    :return: Tuple of functions to generate molecules.
    """
    temp_generators: dict[str, Callable[P_mol, Polygon]] = {
        name: func
        for name, func in molecule_lib.__dict__.items()
        if inspect.isfunction(func)
        and not name.startswith("_")
        and func.__module__ == molecule_lib.__name__
        and inspect.signature(func).return_annotation == "Polygon"
    }
    return tuple(temp_generators.values())


@pytest.mark.parametrize(
    "molecule_function",
    _discover_molecule_functions(),
)
def test_simple_molecule_generation_no_args(molecule_function: Callable[P_mol, Polygon]) -> None:
    """Test whether molecules/errors are generated correctly when called without arguments.

    :param molecule_function: Molecule function.
    """
    try:
        output = molecule_function()  # type: ignore[call-arg]
        assert isinstance(output, Polygon)
    except ValidationError as e:
        if "missing_argument" not in str(e):
            raise  # If the function has missing arguments, this should raise


def make_strategy_for_func(func: Callable[P_mol, Polygon]) -> st.SearchStrategy[dict[str, object]]:
    """Inspect a function's type hints and return a strategy that generates a dictionary of valid keyword arguments.

    :param func: Function to inspect.
    :returns: Strategy that generates a dictionary of valid keyword arguments.
    """
    sig = inspect.signature(func)
    strategy_mapping: dict[str, st.SearchStrategy[object]] = {}

    for name, param in sig.parameters.items():
        # Skip *args or **kwargs.
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue

        # Check if a type hint exists.
        if param.annotation != inspect.Parameter.empty:
            # Generate a strategy based on the type hint.
            strategy_mapping[name] = resolve_param_strategy(param.annotation)
        else:
            errmsg = f"Parameter '{name}' in {func.__name__} has no type hint."
            raise ValueError(errmsg)

    return st.fixed_dictionaries(strategy_mapping)


@pytest.mark.parametrize(
    "molecule_function",
    _discover_molecule_functions(),
)
@given(data=st.data())
def test_simple_molecule_generation_with_args(molecule_function: Callable[P_mol, Polygon], data: DataObject) -> None:
    """Test whether molecules/errors are generated correctly when called with arguments.

    :param molecule_function: Molecule function.
    :param data: Delayed strategy to populate function values.
    """
    strategy = make_strategy_for_func(molecule_function)
    generated_kwargs = data.draw(strategy)

    output = molecule_function(**generated_kwargs)  # type: ignore[arg-type, call-arg]
    if str(molecule_function) != "xyz_reader":
        assert isinstance(output, Polygon)
    else:
        assert isinstance(output, Polygon | MultiPolygon)


@pytest.mark.parametrize(
    ("atomkeys", "atompos", "listed_molecule_count", "expected_value_error_message"),
    [
        (np.array(["C"]), np.empty((1, 3)), None, r"The .xyz file must contain a valid molecule count on line 1."),
        (np.array(["C"]), np.empty((1, 3)), 1000, "The file promises 1000 molecules but gives 1"),
        (np.array(["Mock"]), np.empty((1, 3)), 1, r"Bad molecule types detected:"),
        (np.array(["C"]), np.empty((2, 3)), 1, "The keys and molecule coordinate lists are not of equal length."),
        (np.array(["C"]), np.full((1, 3), np.inf), 1, r"The .xyz file contains invalid coordinates."),
        (np.array(["C"]), np.empty((1, 2)), 1, r"The .xyz file must contain 3D coordinates."),
    ],
)
def test_xyz_verifier_errors(
    atomkeys: StrArray,
    atompos: CoordsArray3D,
    listed_molecule_count: int | None,
    expected_value_error_message: str,
) -> None:
    """Test whether the xyz verifier raises errors for the correct reasons.

    :param atomkeys: Atomic keys of molecules.
    :param atompos: Coordinates of molecules.
    :param listed_molecule_count: Number of molecules in list.
    :param expected_value_error_message: Expected error message.
    """
    with pytest.raises(ValueError, match=expected_value_error_message):
        _xyz_verifier(atomkeys, atompos, listed_molecule_count)
