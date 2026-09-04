# Copyright (c) 2025-2026 Contributors to the AdsorPy project.
# SPDX-License-Identifier: MIT
"""Reads the config json.

The config.json contains the standard values for the RSA simulations. They can be changed if the user wants to,
however, the most important values can be overridden in the run_simulation module as well.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import TypeAlias, TypeVar

# Define the allowed inner JSON primitive types
JsonLeaf: TypeAlias = float | str | int | list[float] | None
# A recursive type alias for a nested JSON dictionary layout
JsonDict: TypeAlias = dict[str, "JsonLeaf | JsonDict"]

T = TypeVar("T", float, str, int, list[float], None)


class RsaConfig:
    """Load the RSA config JSON and parse it for the simulation."""

    def __init__(self, config_path: str | Path | None = None, config: JsonDict | None = None) -> None:
        """Initialise the config reader.

        :param config_path: The path to the config file.
        :param config: The config values as a dict.
        """
        self.config_path: Path | None = Path(config_path) if config_path is not None else None
        self.__config: JsonDict = config if config is not None else {}
        self.__initialize()

    def __initialize(self) -> None:
        if self.config_path is not None:
            with self.config_path.open() as f:
                parsed: object = json.load(f)
                if isinstance(parsed, dict):
                    # Guard rail ensuring top level elements align to standard string keys
                    self.__config = parsed
                else:
                    msg = "Configuration file root must be a JSON object dictionary."
                    raise TypeError(msg)

    def to_dict(self) -> JsonDict:
        """Return the JSON as dictionary.

        :return: The JSON as dictionary.
        """
        return self.__config

    def get_item(self, item: str, required: bool = True) -> JsonLeaf | JsonDict | RsaConfig:
        """Get the item from the JSON.

        :param item: The item to be split into keys.
        :param required: Bool denoting whether it is required.
        :return: The item from the JSON.
        """
        keys = item.split(".")
        result = self.__return_key_value(self.__config, keys)
        if required and result is None:
            errmsg = f"Required item '{item}' is empty/None!"
            raise ValueError(errmsg)
        # If result is None but not required, we safely return None as part of JsonLeaf
        return result if result is not None else None

    def get_value(self, item: str, required: bool = True) -> JsonLeaf | JsonDict | RsaConfig:
        """Get the value from the JSON.

        :param item: The item to be split into keys.
        :param required: Bool denoting whether it is required.
        :return: The value of the item from the JSON.
        """
        keys = item.split(".")
        if not keys or keys[-1] != "value":
            keys.append("value")

        result = self.__return_key_value(self.__config, keys)
        if required and result is None:
            errmsg = f"A required value for '{item}' is empty/None!"
            raise ValueError(errmsg)
        return result if result is not None else None

    def __return_key_value(
        self,
        config_value: JsonLeaf | JsonDict,
        keys: list[str],
    ) -> JsonLeaf | JsonDict | RsaConfig:
        if not keys:
            if isinstance(config_value, dict):
                return RsaConfig(config_path=None, config=config_value)
            return config_value

        if isinstance(config_value, Mapping):
            first_key = keys[0]
            if first_key not in config_value:
                return None

            # Type narrowing for nested structural keys
            next_value = config_value[first_key]
            return self.__return_key_value(next_value, keys[1:])

        return None
