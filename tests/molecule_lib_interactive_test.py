# Copyright (c) 2025-2026 Contributors to the AdsorPy project.
# SPDX-License-Identifier: MIT
"""Test the GUI part of the `molecule_lib.py` module."""

from dataclasses import dataclass
from itertools import count
from pathlib import Path

import numpy as np  # For vectorised computations (performed in C).
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import SearchStrategy
from numpy.random import PCG64DXSM, Generator  # New random generator.
from scipy.spatial.distance import cdist
from shapely import Polygon, unary_union
from shapely.prepared import prep

import src.adsorpy.molecule_lib as mol  # Homebrew lib of molecules.
import src.adsorpy.randomsequentialadsorption as rsarun
from src.adsorpy.rsa_calculator import squared_cdist
from src.adsorpy.rsa_config import RsaConfig  # Config of the simulation.
from src.adsorpy.types import CoordsArray, GeoArray
