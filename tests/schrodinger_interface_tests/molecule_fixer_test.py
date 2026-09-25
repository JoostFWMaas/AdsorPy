# Copyright (c) 2025-2026 Contributors to the AdsorPy project.
# SPDX-License-Identifier: MIT
"""Test Interface for Schrödinger."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import shapely
from shapely import Polygon

from adsorpy.scripts.schrodinger_interface import MoleculeParser


def test_molecule_parser(tmp_path: Path) -> None:
    """Test the MoleculeParser class.

    :param tmp_path: Temporary directory path.
    """
    molname = "mol1"
    disk = shapely.Point((0.0, 0.0)).buffer(1.0)
    disk_coords = shapely.get_coordinates(disk)
    disk_coords[0::2] = disk_coords[-1::2]  # make a mess out of the disk.

    assert not np.isclose(Polygon(disk_coords).area, disk.area, 0.01), "The disk should no longer be valid"

    mock_file = tmp_path / "mock_molecule.json"

    with mock_file.open("w", encoding="utf-8") as f:
        json.dump({molname: disk_coords.tolist()}, f)

    output = MoleculeParser.from_json(file_path=mock_file)

    assert np.isclose(output.polygons[molname].area, disk.area, 0.01), "The disk should be restored close to original."
