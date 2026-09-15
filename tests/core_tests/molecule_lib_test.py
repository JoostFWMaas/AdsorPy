# Copyright (c) 2025-2026 Contributors to the AdsorPy project.
# SPDX-License-Identifier: MIT
"""Test the `molecule_lib`` module."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from pathlib import Path
from typing import ParamSpec, TypeVar
from unittest.mock import MagicMock

import numpy as np
import pytest
from _pytest.monkeypatch import MonkeyPatch
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.strategies import DataObject
from pydantic import ValidationError
from shapely import MultiPolygon, Polygon

from adsorpy import molecule_lib
from adsorpy.molecule_lib import _initialise_reader, _xyz_verifier, first_time_loader
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
    "float": st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False),
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
    if molecule_function.__name__ == "xyz_reader":
        assert isinstance(output, Polygon | MultiPolygon), "xyz_reader should produce a Polygon, or else MultiPolygon."
    else:
        assert isinstance(output, Polygon), f"{molecule_function.__name__} should produce a Polygon."


@pytest.mark.parametrize(
    ("atomkeys", "atompos", "listed_molecule_count", "expected_value_error_message"),
    [
        (np.array(["C"]), np.zeros((1, 3)), None, r"The .xyz file must contain a valid molecule count on line 1."),
        (np.array(["C"]), np.zeros((1, 3)), 1000, "The file promises 1000 molecules but gives 1"),
        (np.array(["Mock"]), np.zeros((1, 3)), 1, r"Bad molecule types detected:"),
        (np.array(["C"]), np.zeros((2, 3)), 1, "The keys and molecule coordinate lists are not of equal length."),
        (np.array(["C"]), np.full((1, 3), np.inf), 1, r"The .xyz file contains invalid coordinates."),
        (np.array(["C"]), np.zeros((1, 2)), 1, r"The .xyz file must contain 3D coordinates."),
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


@pytest.mark.parametrize(
    ("file_name", "ignore_atoms", "z_trim", "value_error_message"),
    [
        ("mockfile.bad", None, None, r"The file type is not .xyz but .bad"),
        (XYZ_FILE_PATH, None, np.inf, r"The current settings result in an empty molecule."),
        (XYZ_FILE_PATH, ["C", "H", "O", "Cl", "F"], None, r"The current settings result in an empty molecule."),
        (XYZ_FILE_PATH, ["Cl", "F"], None, None),
        (XYZ_FILE_PATH, "Cl, F", None, None),
        (XYZ_FILE_PATH, ["Cl"], None, None),
        (XYZ_FILE_PATH, "Cl", None, None),
        (XYZ_FILE_PATH, None, None, None),
    ],
)
def test_initialise_reader(
    file_name: str | Path,
    ignore_atoms: str | list[str] | None,
    z_trim: float | None,
    value_error_message: str | None,
) -> None:
    """Test whether the xyz file reader initialises correctly."""
    if value_error_message:
        with pytest.raises(ValueError, match=value_error_message):
            _initialise_reader(file_name, ignore_atoms, z_trim)
        return

    total_atom_count = 7

    ignore_atoms_len = 0
    if isinstance(ignore_atoms, str):
        ignore_atoms_len = len(ignore_atoms.split(","))
    elif isinstance(ignore_atoms, list):
        ignore_atoms_len = len(ignore_atoms)

    atomkeys, atompos = _initialise_reader(file_name, ignore_atoms, z_trim)

    assert atomkeys.size == total_atom_count - ignore_atoms_len, "Atomkeys length must match filtered length."
    assert atompos.shape[0] == total_atom_count - ignore_atoms_len


@pytest.fixture
def mock_ui_environment(monkeypatch: MonkeyPatch) -> dict[str, MagicMock]:
    """Set up standard Mocks via monkeypatching to isolate the loader from files and UI."""
    mock_reader = MagicMock(return_value=(["H"], [[0.0, 0.0, 0.0]]))
    monkeypatch.setattr(molecule_lib, "_initialise_reader", mock_reader)

    mock_adapter_instance = MagicMock()
    # Mocking standard hex return object
    mock_color = MagicMock()
    mock_color.as_hex.return_value = "#FF0000"
    mock_adapter_instance.validate_json.return_value = {"H": mock_color}

    mock_type_adapter = MagicMock(return_value=mock_adapter_instance)
    monkeypatch.setattr(molecule_lib, "TypeAdapter", mock_type_adapter)

    mock_qapp_class = MagicMock()
    mock_qapp_class.instance.return_value = None  # Force initialization branch
    monkeypatch.setattr(molecule_lib, "QApplication", mock_qapp_class)

    mock_viewer = MagicMock()
    mock_viewer.roll = 10.0
    mock_viewer.pitch = 20.0
    mock_viewer.yaw = 30.0
    mock_viewer.x_offset = 1.0
    mock_viewer.y_offset = 2.0
    mock_viewer.disabled_molecules = ["He"]
    mock_viewer.z_cutoff = 1.5
    # Configure checkbox mock behavior
    mock_viewer.z_filter_enable.isChecked.return_value = True

    mock_viewer_class = MagicMock(return_value=mock_viewer)
    monkeypatch.setattr(molecule_lib, "MoleculeViewer", mock_viewer_class)

    return {"reader": mock_reader, "adapter_instance": mock_adapter_instance, "viewer": mock_viewer}


def test_first_time_loader_success(mock_ui_environment: dict[str, MagicMock], tmp_path: Path) -> None:
    """Test standard successful execution path with valid parameters."""
    mocks = mock_ui_environment

    # Create a temporary file path to satisfy Pydantic's FilePath type checker
    temp_file = tmp_path / "molecule.xyz"
    temp_file.write_text("xyz file content placeholder")

    result = first_time_loader(
        file_name=temp_file,
        roll=0.0,
        pitch=0.0,
        yaw=0.0,
        x_offset=0.0,
        y_offset=0.0,
        ignore_atoms=None,
        z_trim=None,
        reference_lattice_spacing=1.0,
    )

    # Verify underlying operations were triggered
    mocks["reader"].assert_called_once_with(temp_file, None, None)
    mocks["viewer"].exec.assert_called_once()

    # Validate output dictionary payload matches Mocked properties
    assert result["file_name"] == str(temp_file)
    assert result["pitch"] == mocks["viewer"].pitch


def test_first_time_loader_z_trim_disabled(mock_ui_environment: dict[str, MagicMock], tmp_path: Path) -> None:
    """Test that z_trim resolves to None if the UI checkbox is unchecked."""
    mocks = mock_ui_environment
    mocks["viewer"].z_filter_enable.isChecked.return_value = False

    temp_file = tmp_path / "molecule.xyz"
    temp_file.write_text("xyz")

    result = first_time_loader(file_name=temp_file)
    assert result["z_trim"] is None


def test_first_time_loader_color_json_fail(mock_ui_environment: dict[str, MagicMock], tmp_path: Path) -> None:
    """Test gracefull warning fallback when the color JSON validation encounters errors."""
    mocks = mock_ui_environment
    # Override standard validation response to raise a FileNotFoundError exception
    mocks["adapter_instance"].validate_json.side_effect = FileNotFoundError()

    temp_file = tmp_path / "molecule.missingcolours"
    temp_file.write_text("CDEFGH")

    # Verify a warning is cleanly handled instead of blowing up execution
    with pytest.warns(UserWarning, match="Could not parse colours safely"):
        result = first_time_loader(file_name=temp_file)

    assert result["roll"] == mocks["viewer"].roll  # Verification that code still ran completely


def test_first_time_loader_pydantic_validation_error(tmp_path: Path) -> None:
    """Test that @validate_call blocks wrong structural types prior to execution."""
    with pytest.raises(ValidationError, match=r"path_not_file"):
        first_time_loader(file_name=str(tmp_path / "file.bad"))
