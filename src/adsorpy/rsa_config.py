"""Reads the config json.

The config.json contains the standard values for the RSA simulations. They can be changed if the user wants to,
however, the most important values can be overridden in the run_simulation module as well.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import ParamSpec, TypeAlias

P = ParamSpec("P")
T: TypeAlias = float | str | int | list[float]
# T = TypeVar("T", bound=float | str | list[float])


class RsaConfig:
    """Load the RSA config json and parse it for the simulation."""

    def __init__(self, config_path: str | None = None, config: dict[str, T] | None = None) -> None:
        """Initialise the config reader.

        :param config_path: The path to the config file.
        :param config: The config values as a dict.
        """
        self.config_path = Path(config_path) if config_path is not None else None
        self.__config = config if config is not None else {}
        self.__init()

    def __init(self) -> None:
        if self.config_path is not None:
            with self.config_path.open() as f:
                self.__config = json.load(f)

    def to_dict(self) -> dict[str, T]:
        """Return the json as dictionary.

        :return: The json as dictionary.
        """
        return self.__config

    def get_item(self, item: str, required: bool = True) -> T | dict[str, T] | RsaConfig | None:
        """Get the item from the json.

        :param item: The item to be split into keys.
        :param required: Bool denoting whether it is required.
        :return: The item from the json.
        """
        keys = item.split(".")
        result = self.__return_key_value(self.__config, keys)
        if required and result is None:
            errmsg = "Required result is empty/None!"
            raise ValueError(errmsg)
        return result

    def get_value(self, item: str, required: bool = True) -> T | dict[str, T] | RsaConfig | None:
        """Get the value from the json.

        :param item: The item to be split into keys.
        :param required: Bool denoting whether it is required.
        :return: The value of the item from the json.
        """
        keys = item.split(".")
        if keys[len(keys) - 1] != "value":
            keys.append("value")
        result = self.__return_key_value(self.__config, keys)
        if required and result is None:
            errmsg = "A required value is empty/None!"
            raise ValueError(errmsg)
        return result

    def __return_key_value(
        self, config_value: dict[str, T] | T, keys: list[str],
    ) -> T | dict[str, T] | RsaConfig | None:
        if not len(keys):
            if isinstance(config_value, dict):
                return RsaConfig(config_path=None, config=config_value)
            return config_value

        if isinstance(config_value, Mapping):
            if keys[0] not in config_value:
                return None

            return self.__return_key_value(config_value[keys[0]], keys[1:])
        return None
