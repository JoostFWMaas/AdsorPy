# Copyright (c) 2025-2026 Contributors to the AdsorPy project.
# SPDX-License-Identifier: MIT
"""Schema and validator for the RSA simulation configuration using Pydantic v2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Generic, TypeAlias, TypeVar

from adsorpy.types import BoundaryConditionStrs  # noqa: TC001

if TYPE_CHECKING:
    from sys import version_info

    if version_info >= (3, 11):
        from typing import Self
    else:
        from typing_extensions import Self

from pydantic import BaseModel, FilePath, NonNegativeFloat, NonNegativeInt, PositiveInt, TypeAdapter, model_validator

JsonPrimitive: TypeAlias = float | str | int | bool | None
JsonValue: TypeAlias = "JsonPrimitive | list[JsonValue] | dict[str, JsonValue]"
RawJsonDict: TypeAlias = dict[str, JsonValue]

# Strict output leaf validation types
JsonLeaf: TypeAlias = float | str | int | list[float] | bool | None
T = TypeVar("T", bound=JsonLeaf)


class LoggingConfig(BaseModel):
    """Whether logging is enabled or not.

    :param enabled: If true: enable logging.
    """

    enabled: bool


    @model_validator(mode="before")
    @classmethod
    def strip_comments(cls, data: RawJsonDict) -> RawJsonDict:
        """Strip keys containing '_comment' from the incoming data dictionary."""
        return {k: v for k, v in data.items() if "_comment" not in k}


class WrappedValue(BaseModel, Generic[T]):
    """Generic wrapper for fields containing a nested 'value' key."""

    value: T

    @model_validator(mode="before")
    @classmethod
    def strip_comments(cls, data: RawJsonDict) -> RawJsonDict:
        """Strip keys containing '_comment' from the incoming data dictionary."""
        return {k: v for k, v in data.items() if "_comment" not in k}


class RsaConfig(BaseModel):
    """Full RSA Configuration model mirroring the JSON schema exactly.

    :param logging: Whether to enable logging.
    :param sites: The surface site count.
    :param xsize: The surface size in the x direction.
    :param ysize: The surface size in the y direction.
    :param zsize: The surface size in the z direction.
    :param max_molecule_count: The maximum number of molecules allowed in the simulation.
    :param lattice_a: The lattice spacing.
    :param boundary_type: The boundary type.
    :param sticking_probability: The sticking probability.
    """

    logging: LoggingConfig
    sites: WrappedValue[PositiveInt | None]
    xsize: WrappedValue[NonNegativeFloat | None]
    ysize: WrappedValue[NonNegativeFloat | None]
    zsize: WrappedValue[NonNegativeFloat | None]
    max_molecule_count: WrappedValue[NonNegativeInt]
    lattice_a: WrappedValue[NonNegativeFloat]
    boundary_type: WrappedValue[BoundaryConditionStrs]
    sticking_probability: WrappedValue[NonNegativeFloat]

    def __init__(self, config_path: str | FilePath | None = None, **kwargs: object) -> None:
        """Initialise the configuration.

        Supports drop-in instantiation via a positional file path string/Path object,
        or keyword arguments for testing/override setups.
        """
        if config_path is not None and not kwargs:
            validated_path = TypeAdapter(FilePath).validate_python(config_path)
            with validated_path.open("r") as f:
                data = json.load(f)
            # Route the dict parameters directly into the Pydantic init framework
            validated_model = self.model_validate(data)

            # Map parameters cleanly down to the slotted dict infrastructure
            self.__setattr__("__dict__", validated_model.__dict__)
            self.__setattr__("__pydantic_fields_set__", validated_model.__pydantic_fields_set__)
            self.__setattr__("__pydantic_extra__", validated_model.__pydantic_extra__)
            self.__setattr__("__pydantic_private__", validated_model.__pydantic_private__)
            # super().__init__(**data)
        else:
            super().__init__(**kwargs)


    @model_validator(mode="after")
    def validate_dimensions(self) -> Self:
        """Enforce mutual exclusivity between grid 'sites' and spatial 'sizes'."""
        has_sites: bool = self.sites.value is not None
        has_sizes: bool = self.xsize.value is not None and self.ysize.value is not None

        if has_sites and has_sizes:
            errmsg = "Cannot specify both 'sites' and spatial dimensions ('xsize'/'ysize')."
            raise ValueError(errmsg)
        if not has_sites and not has_sizes:
            errmsg = "You must specify either 'sites' or both 'xsize' and 'ysize'."
            raise ValueError(errmsg)
        return self

    @classmethod
    def from_file(cls, config_path: str | Path) -> RsaConfig:
        """Load and strictly validate the configuration directly from a file descriptor."""
        path = Path(config_path)
        with path.open("r") as f:
            data = json.load(f)
        return cls.model_validate(data)


    def get_value(self, item: str, required: bool) -> JsonLeaf:
        """Backward-compatible value getter matching legacy adsorpy API constraints."""
        clean_item = item.replace(".value", "")

        if clean_item == "logging":
            return self.logging.enabled

        attr: object = getattr(self, clean_item, None)
        if isinstance(attr, WrappedValue):
            # bound to JsonLeaf, guaranteed clean return type
            value: JsonLeaf = attr.value
            if value is None and required:
                errmsg = "Parameter is required but set to None."
                raise ValueError(errmsg)
            return value

        errmsg = f"Configuration has no parameter '{item}'"
        raise AttributeError(errmsg)
