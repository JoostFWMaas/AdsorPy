# Copyright (c) 2025-2026 Contributors to the AdsorPy project.
# SPDX-License-Identifier: MIT
"""Contains all the molecules that can be used in this simulation.

Also includes molecule loader scripts, for which the molecule data is not included in this lib.
You need to supply your own .xyz files, or you can use preconfigured simple shapes.
"""

from __future__ import annotations

import io
import json
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Final, Literal, TypeVar, cast

import matplotlib.pyplot as plt
import numpy as np
import shapely
import shapely.affinity as aff
import svg
from pydantic import (
    FilePath,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
    StringConstraints,
    TypeAdapter,
    ValidationError,
    validate_call,
)
from pydantic_extra_types import Color
from PySide6.QtCore import QSettings, QSize, Qt
from PySide6.QtSvgWidgets import QSvgWidget
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
)
from shapely import MultiPoint, MultiPolygon, Point, Polygon
from shapely.ops import unary_union

if TYPE_CHECKING:
    from PySide6.QtGui import QFontMetrics

    from src.adsorpy.types import BoolArray, CoordsArray3D, DistArray, RotMatrix, StrArray

    T = TypeVar("T", bound=bool | int | str | float)
    Tfloat = TypeVar("Tfloat", bound=float | np.double)
    # P = ParamSpec("P")  # Helps with static type checkers.

plt.rcParams.update(
    {
        "font.size": 16,
        "xtick.top": True,
        "xtick.direction": "in",
        "xtick.major.width": 1.5,
        "xtick.minor.width": 1,
        "ytick.right": True,
        "ytick.direction": "in",
        "ytick.major.width": 1.5,
        "ytick.minor.width": 1,
        "lines.linewidth": 2.5,
        "axes.linewidth": 1.5,
        "figure.constrained_layout.use": True,
    },
)

AtomKey = Annotated[str, StringConstraints(min_length=1, max_length=2)]
"""Atom key validator. String of length 1 or 2 denoting chemical symbols."""


def _load_radii_from_json() -> dict[str, float]:
    """Load the van der Waals radii from the vdw_radii.json file.

    Uses Pydantic to validate the JSON.

    :returns: dict of the chemical symbols (keys) and van der Waals radii (values).
    """
    radii_adapter = TypeAdapter(dict[AtomKey, PositiveFloat])
    radii_json_path = Path(__file__).parent / "vdw_radii.json"
    return radii_adapter.validate_json(radii_json_path.read_bytes())


RADII: Final[dict[str, float]] = _load_radii_from_json()
"""Key-value pairs of chemical symbols and van der Waals radii.

Reference:
    S. Alvarez, "A cartography of the van der Waals territories,"
    *Dalton Trans.*, vol. 42, no. 24, pp. 8617-8636, Jun. 2013,
    doi: `10.1039/C3DT50599E <https://doi.org/10.1039/C3DT50599E>`_.
"""


@validate_call
def discorectangle(
    radius: PositiveFloat,
    distance: NonNegativeFloat,
    offx: float = 0.0,
    offy: float = 0.0,
) -> Polygon:
    """Create a disco-rectangle using the union of two circles and a rectangle.

    The circles are automatically approximated using linear segments (error of 1% or less).

    :param radius: Radius of the two circles in angstrom.
    :param distance: Distance between the two halves in angstrom.
    :param offx: X offset.
    :param offy: Y offset.
    :return: The molecule shape as a polygon.
    """
    offy *= -1.0
    offx *= -1.0
    circles = MultiPoint(
        [(offx - distance / 2.0, offy), (offx + distance / 2.0, offy)],
    ).buffer(radius)

    rectangle = Polygon(
        [
            (offx - distance / 2.0, offy - radius),
            (offx + distance / 2.0, offy - radius),
            (offx + distance / 2.0, offy + radius),
            (offx - distance / 2.0, offy + radius),
        ],
    )

    return cast("Polygon", unary_union([rectangle, circles]))


@validate_call
def circulium(
    radius: PositiveFloat,
    x_offset: float = 0.0,
    y_offset: float = 0.0,
    quad_segs: PositiveInt = 8,
) -> Polygon:
    """Create a simple circular polygon.

    The circles are automatically approximated using linear segments (error of 1% or less).

    :param radius: Radius of the circle.
    :param x_offset: The x-offset in angstrom.
    :param y_offset: The y-offset in angstrom.
    :param quad_segs: The amount of linear segments in a quarter circle.
    :return: The molecule shape as a polygon.
    """
    return Point((x_offset, y_offset)).buffer(radius, quad_segs=quad_segs)


@validate_call
def dogbonium(scale: PositiveFloat = 1.0) -> Polygon:
    """Make a molecule shaped like a bone. Used as a pathological case.

    :param scale: scale of the shape.
    :return: The molecule shape as a polygon.
    """
    positions = np.ones((4, 2))
    positions *= scale
    positions[[1, 2], 0] *= -1.0
    positions[[2, 3], 1] *= -1.0
    positions[:, 1] *= 0.5
    balls: MultiPolygon = cast("MultiPolygon", MultiPoint(positions).buffer(0.5 * scale))
    rod: Polygon = Polygon(positions)
    dogbone: Polygon = cast("Polygon", unary_union((rod, balls)))

    return dogbone


@validate_call
def polygonium(
    verts: PositiveInt = 3,
    scale: PositiveFloat = 1.0,
    roundedness: NonNegativeFloat = 0.0,
) -> Polygon:
    """Create a simple regular polygon with optional rounding.

    :param verts: The vertex count.
    :param scale: The scale factor of the polygon.
    :param roundedness: The roundedness, removes sharp corners. Theoretical limit at infinity is a disk.
    :return: The regular (rounded) polygon.
    """
    points = np.arange(verts, dtype=np.double)
    points *= 2 * np.pi
    points /= verts
    points = np.column_stack((np.sin(points), np.cos(points)))
    points *= scale

    molecule = Polygon(points)

    if roundedness > 0.0:
        center = Point((0.0, 0.0))
        fact = molecule.exterior.hausdorff_distance(center)
        molecule = molecule.buffer(roundedness, resolution=4 * verts)
        fact /= molecule.exterior.hausdorff_distance(center)
        molecule = aff.scale(molecule, xfact=fact, yfact=fact, origin=center)

    return cast("Polygon", shapely.make_valid(molecule))


@validate_call
def xyz_reader(
    file_name: FilePath,
    ignore_atoms: str | list[str] | None = None,
    x_offset: float = 0.0,
    y_offset: float = 0.0,
    roll: float = 0.0,
    pitch: float = 0.0,
    yaw: float = 0.0,
    z_trim: float | None = None,
    reference_lattice_spacing: float = 1.0,
) -> Polygon:
    """Read files in the xyz format of VASP.

    :param file_name: The name of the file, including the .xyz extension. Include the path.
    :param ignore_atoms: Atoms to ignore when making the molecule. Useful to filter out a slab.
    :param x_offset: The offset in the x direction.
    :param y_offset: The offset in the y direction.
    :param yaw: Rotation along the x-axis.
    :param pitch: Rotation along the y-axis.
    :param roll: Rotation along the z-axis.
    :param z_trim: The z value below which all molecules are removed.

    :return: The molecule polygon read from the xyz file.
    """
    atomkeys, atompos = _initialise_reader(file_name, ignore_atoms, z_trim)

    rot_mat: RotMatrix = _rotation_matrix(roll, pitch, yaw)
    coords: CoordsArray3D
    for crdx, coords in enumerate(atompos):
        atompos[crdx] = coords @ rot_mat
    atom_list = [Polygon()] * atomkeys.size

    for idx, (atm, coords) in enumerate(zip(atomkeys, atompos, strict=False)):
        atom_vdw: Polygon = Point(coords).buffer(RADII[atm])
        atom_list[idx] = atom_vdw

    molecule: Polygon = cast("Polygon", unary_union(MultiPolygon(atom_list)))
    centre = -np.mean(atompos, axis=0)
    if x_offset is not None:
        centre[0] -= x_offset
    if y_offset is not None:
        centre[1] -= y_offset

    return cast("Polygon", aff.translate(molecule, *centre))


def _rotation_matrix(roll: float | np.double, pitch: float | np.double, yaw: float | np.double) -> RotMatrix:
    """Compute the 3D rotation matrix using roll, pitch, and yaw.

    :param roll: Rotation along the x-axis.
    :param pitch: Rotation along the y-axis.
    :param yaw: Rotation along the z-axis.
    :returns: The rotation matrix.
    """
    fac: float = np.pi / 180.0
    """Conversion factor from degrees to radians."""

    rot_x: RotMatrix = np.array(
        [
            [1, 0, 0],
            [0, np.cos(roll * fac), -np.sin(roll * fac)],
            [0, np.sin(roll * fac), np.cos(roll * fac)],
        ],
    )
    rot_y: RotMatrix = np.array(
        [
            [np.cos(pitch * fac), 0, np.sin(pitch * fac)],
            [0, 1, 0],
            [-np.sin(pitch * fac), 0, np.cos(pitch * fac)],
        ],
    )
    rot_z: RotMatrix = np.array(
        [
            [np.cos(yaw * fac), -np.sin(yaw * fac), 0],
            [np.sin(yaw * fac), np.cos(yaw * fac), 0],
            [0, 0, 1],
        ],
    )

    return rot_z @ rot_y @ rot_x


class MoleculeViewer(QDialog):
    """Molecule spatial orientation and structural viewport configuration dashboard.

    Manages dynamic 3D rotational coordinate matrices (roll, pitch, yaw) for complex
    molecular clusters and applies linear structural spatial clipping filters.
    """

    def __init__(
        self,
        atomkeys: list[str] | StrArray,
        atompos: CoordsArray3D,
        colours: list[str] | StrArray,
        init_roll: float | None = None,
        init_pitch: float | None = None,
        init_yaw: float | None = None,
        init_x_offset: float | None = None,
        init_y_offset: float | None = None,
        lattice: float | None = None,
    ) -> None:
        """Initialise matrix array buffers, orientation sliders, and plotting canvas frames."""
        super().__init__()

        self._settings = QSettings("adsorpy", type(self).__name__)
        """Persistent configuration handle cached across software operational cycles."""

        # Step 1: Initialise raw variables and data placeholders via structural helper
        self.atomkeys, self.atompos, self.colours = self._init_data(
            atomkeys,
            atompos,
            colours,
            init_roll,
            init_pitch,
            init_yaw,
            init_x_offset,
            init_y_offset,
        )
        """Active slice mappings capturing parameters matching current view states."""

        self.lattice: float = lattice if lattice is not None else 1.0
        """Lattice physical constant constraint baseline scaling factor. Defaults to 1.0."""

        self.show_bonds: bool = False
        """State indicator controlling rendering toggles for chemical covalent bounds."""

        self.atom_toggles: dict[str, QCheckBox] = {}
        """Active reference mapping linking atomic symbol labels to operational checkboxes."""

        # Step 2: Build the structural layout tree
        main_vert_layout = QVBoxLayout()
        top_horizontal_layout = QHBoxLayout()

        # Generate isolated modular workspace layout panel trees
        filter_panel = self._create_filter_panel()
        plot_workspace = self._create_plot_panel()

        # Inject functional widget sub-controls directly into the module frames
        self.setup_bond_controls(filter_panel)
        self.setup_lattice_controls(filter_panel)

        # Assemble unified parent layout structural paths
        top_horizontal_layout.addLayout(filter_panel, stretch=1)
        top_horizontal_layout.addLayout(plot_workspace, stretch=5)
        main_vert_layout.addLayout(top_horizontal_layout, stretch=1)

        # Append trailing timeline/slider control panel arrays to base row
        self._add_slider_panel(main_vert_layout)

        self.setLayout(main_vert_layout)
        self.apply_atom_filters()

    @property
    def disabled_molecules(self) -> list[str] | None:
        """Get a sorted list of disabled atom name keys or None if empty.

        Identifies inactive elements via unchecked toggles and sorts them
        by ascending atomic weight indices using the master database registry.

        :return: Chronologically sorted elements or None if all toggles are active.
        """
        disabled = [atom for atom, checkbox in self.atom_toggles.items() if not checkbox.isChecked()]
        if not disabled:
            return None

        # Sorted by periodic table keys
        return sorted(disabled, key=list(RADII.keys()).index)

    def _init_data(
        self,
        atomkeys: list[str] | StrArray,
        atompos: CoordsArray3D,
        colours: list[str] | StrArray,
        init_roll: float | None = None,
        init_pitch: float | None = None,
        init_yaw: float | None = None,
        init_x_offset: float | None = None,
        init_y_offset: float | None = None,
    ) -> tuple[StrArray, CoordsArray3D, StrArray]:
        """Convert list inputs into persistent raw NumPy matrices and configure boundaries.

        :param atomkeys: list of strings of chemical symbols.
        :param atompos: Intercept array tracking three-dimensional coordinates of atoms.
        :param colours: Vector mapping hexadecimal colour tracking pointers.
        :param init_roll: Baseline matrix orientation transformation offset.
        :param init_pitch: Baseline matrix elevation transformation offset.
        :param init_yaw: Baseline matrix horizontal rotation transformation offset.
        :param init_x_offset: Baseline translation margin parallel to the abscissa.
        :param init_y_offset: Baseline translation margin parallel to the ordinate.
        :return: Tuple containing configured baseline instance data arrays.
        """
        self.orig_atomkeys = np.array(atomkeys, dtype=np.str_)
        """Immutable baseline array tracking atomic keys loaded from source files."""

        self.orig_atompos = np.array(atompos, dtype=np.double)
        """Immutable matrix holding coordinates across 3D vector parameters."""

        self.orig_colours = np.array(colours, dtype=np.str_)
        """Immutable map listing style color tokens for individual elements."""

        # Establish working instance clones to avoid operational mutations
        temp_atomkeys = self.orig_atomkeys.copy()
        temp_atompos = self.orig_atompos.copy()
        temp_colours = self.orig_colours.copy()

        # Unpack structural properties; fall back to 0.0 seamlessly if empty
        self.roll = init_roll or 0.0
        self.pitch = init_pitch or 0.0
        self.yaw = init_yaw or 0.0
        self.x_offset = init_x_offset or 0.0
        self.y_offset = init_y_offset or 0.0

        # Calculate bounding height cutoffs based on structural values
        self.min_z = float(np.min(self.orig_atompos[:, 2])) if self.orig_atompos.size else -10.0
        self.max_z = float(np.max(self.orig_atompos[:, 2])) if self.orig_atompos.size else 10.0
        self.z_cutoff = self.min_z - 0.1

        return temp_atomkeys, temp_atompos, temp_colours

    def _fetch_setting(self, name: str, default: T, return_type: type[T] | None = None) -> T:
        """Query registry fields from the application storage dictionary profile.

        :param name: Unique configuration mapping key string tracker.
        :param default: Fallback property return object assigned if name is missing.
        :param return_type: Target data object type enforcement mapping template.
        :return: Configured properties mapped to native execution type constraints.
        """
        check_type = return_type if return_type is not None else type(default)
        return cast("T", self._settings.value(name, defaultValue=default, type=check_type))

    def _create_filter_panel(self) -> QVBoxLayout:
        """Build the panel component hosting Z-cutoff adjustments and checkboxes.

        :return: Main target layout framework representing Column A.
        """
        filter_panel: QVBoxLayout = QVBoxLayout()
        filter_panel.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.bg_toggle: QPushButton = QPushButton("Toggle White Background")
        self.bg_toggle.setCheckable(True)
        self.bg_toggle.clicked.connect(self.toggle_svg_background)
        self.bg_toggle.setChecked(self._fetch_setting("bg_on", default=False))
        filter_panel.addWidget(self.bg_toggle)
        self.bg_toggle.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        hint: QSize = self.bg_toggle.sizeHint()
        self.bg_toggle.setFixedWidth(hint.width() + 20)

        # 1. Z-Cutoff Group
        z_group: QGroupBox = QGroupBox("Z-Cutoff Filter")
        z_layout: QHBoxLayout = QHBoxLayout(z_group)
        z_label: QLabel = QLabel("Z-Min Cut:")
        z_toggle_layout = QHBoxLayout()
        self.z_filter_enable = QCheckBox()
        self.z_filter_enable.setChecked(False)  # Disabled by default
        self.z_filter_enable.setToolTip("Toggle Z-cutoff filter active/inactive")
        z_layout.addLayout(z_toggle_layout)

        self.z_spinbox: QDoubleSpinBox = QDoubleSpinBox()
        self.z_spinbox.setRange(self.min_z - 1.0, self.max_z + 1.0)
        self.z_spinbox.setDecimals(2)
        self.z_spinbox.setSingleStep(0.1)
        self.z_spinbox.setValue(self.z_cutoff)
        self.z_spinbox.setEnabled(False)
        self.z_spinbox.setSuffix("Å")
        self.z_filter_enable.toggled.connect(self.z_spinbox.setEnabled)
        self.z_filter_enable.toggled.connect(self.apply_atom_filters)
        self.z_spinbox.valueChanged.connect(self.update_z_cutoff)

        z_layout.addWidget(self.z_filter_enable)

        z_layout.addWidget(z_label)
        z_layout.addWidget(self.z_spinbox)
        filter_panel.addWidget(z_group)

        # 2. Dynamic Atom Checkbox Toggles Group
        atom_group: QGroupBox = QGroupBox("Filter Atoms by Type")
        atom_checkbox_layout: QVBoxLayout = QVBoxLayout(atom_group)

        unique_atoms: list[str] = sorted(set(self.orig_atomkeys), key=list(RADII.keys()).index)

        for atom in unique_atoms:
            idx: np.ndarray = np.where(self.orig_atomkeys == atom)
            color_hex: str = self.orig_colours[idx][0] if idx[0].size else "#FFFFFF"

            item_row: QHBoxLayout = QHBoxLayout()
            item_row.setContentsMargins(0, 2, 0, 2)

            atom_checkbox: QCheckBox = QCheckBox(f"Show {atom}")
            atom_checkbox.setChecked(True)
            self.atom_toggles[atom] = atom_checkbox
            atom_checkbox.stateChanged.connect(self.apply_atom_filters)

            color_swatch: QLabel = QLabel()
            color_swatch.setFixedSize(QSize(14, 14))
            color_swatch.setStyleSheet(f"background-color: {color_hex}; border: 1px solid #555555; border-radius: 2px;")

            item_row.addWidget(color_swatch)
            item_row.addWidget(atom_checkbox)
            item_row.addStretch()
            atom_checkbox_layout.addLayout(item_row)

        filter_panel.addWidget(atom_group)
        return filter_panel

    def _create_plot_panel(self) -> QVBoxLayout:
        """Build the panel viewport frame displaying rendered molecular assets.

        :return: Main canvas workspace framework representing Column B.
        """
        plot_workspace: QVBoxLayout = QVBoxLayout()
        plot_toggle_layout: QHBoxLayout = QHBoxLayout()

        self.svg_widget: QSvgWidget = QSvgWidget()
        self.svg_widget.setMinimumSize(300, 300)
        self.svg_widget.renderer().setAspectRatioMode(Qt.AspectRatioMode.KeepAspectRatio)
        plot_toggle_layout.addWidget(self.svg_widget, alignment=Qt.AlignmentFlag.AlignCenter)
        self.toggle_svg_background(self.bg_toggle.isChecked())

        plot_workspace.addLayout(plot_toggle_layout)

        return plot_workspace

    def _add_slider_panel(self, target_layout: QVBoxLayout | QHBoxLayout) -> None:
        """Append transformation sliders directly across the bottom container.

        :param target_layout: Top-level root layout accepting row insertions.
        """
        slider_params: dict[str, tuple[int, int, bool, str]] = {
            "roll": (-180, 180, True, "°"),
            "pitch": (-180, 180, True, "°"),
            "yaw": (-180, 180, True, "°"),
            "x_offset": (-5, 5, False, " Å"),
            "y_offset": (-5, 5, False, " Å"),
        }

        max_width: int = 0
        step_size: float = 0.1
        font_metrics: QFontMetrics = self.fontMetrics()

        for param in slider_params:
            width: int = font_metrics.horizontalAdvance(param)
            max_width = max(max_width, width)

        max_width += 10

        for name, val_range in slider_params.items():
            row: QHBoxLayout = QHBoxLayout()

            label: QLabel = QLabel(name)
            label.setFixedWidth(max_width)

            slider: QSlider = QSlider(Qt.Orientation.Horizontal)
            slider.setMinimum(val_range[0] * 10)
            slider.setMaximum(val_range[1] * 10)

            box: QDoubleSpinBox = QDoubleSpinBox()
            box.setRange(float(val_range[0]), float(val_range[1]))
            box.setWrapping(val_range[2])
            box.setDecimals(2)
            box.setSingleStep(step_size)
            box.setAccelerated(True)
            box.setSuffix(val_range[3])
            box.setValue(0.0)
            box.setFixedWidth(120)

            slider.valueChanged.connect(
                lambda val, n=name, b=box: self.update_values(val, n, b),
            )

            box.valueChanged.connect(
                lambda _, s=slider, b=box, n=name: self.submit_values(s, b, n),
            )

            row.addWidget(label)
            row.addWidget(slider, 1)
            row.addWidget(box)
            target_layout.addLayout(row)

    def toggle_bonds(self, checked: bool) -> None:
        """Toggle bond rendering on or off and update the view.

        :param checked: Toggle bond rendering on or off.
        """
        self.show_bonds = checked
        self.draw()

    def update_z_cutoff(self, value: float) -> None:
        """Slot targeting real-time spinbox adjustments to update pipeline state.

        :param value: z-cutoff value to update.
        """
        self.z_cutoff = value
        self.apply_atom_filters()

    def apply_atom_filters(self) -> None:
        """Calculate boolean masks against root datasets and handle redraw requests."""
        mask: BoolArray
        if self.z_filter_enable.isChecked():
            mask = self.orig_atompos[:, 2] >= self.z_cutoff
        else:
            mask = np.ones_like(self.orig_atompos[:, 2], dtype=np.bool_)

        allowed_types: set[str] = {atom for atom, cb in self.atom_toggles.items() if cb.isChecked()}
        type_mask: np.ndarray = np.isin(self.orig_atomkeys, list(allowed_types))

        combined_mask: np.ndarray = mask & type_mask

        self.atomkeys = self.orig_atomkeys[combined_mask]
        self.atompos = self.orig_atompos[combined_mask]
        self.colours = self.orig_colours[combined_mask]

        self.draw()

    def setup_bond_controls(self, layout: QHBoxLayout | QVBoxLayout) -> None:
        """Create and connect the atomic bond visualisation toggle.

        :param layout: The QLayout instance (e.g., QVBoxLayout) where the checkbox should be added.
        """
        # Ensure the underlying rendering property exists
        if not hasattr(self, "show_bonds"):
            self.show_bonds = False

        # Initialise the checkbox widget
        self.bond_checkbox = QCheckBox("Show Atomic Bonds (visual guide)")
        self.bond_checkbox.setChecked(self.show_bonds)

        # Connect the native toggle signal directly to your slot method
        self.bond_checkbox.toggled.connect(self.toggle_bonds)

        # Insert the checkbox into the provided layout panel
        layout.addWidget(self.bond_checkbox)

    def update_values(self, val: float, name: str, box_widget: QDoubleSpinBox) -> None:
        """Unifies slider-to-backend slot to keep widgets cleanly scoped.

        :param val: The value to update the attribute to.
        :param name: The name of the parameter to update.
        :param box_widget: The QDoubleSpinBox widget to update.
        """
        v = val / 10
        # Block signals to avoid feedback looping when setting the companion value
        box_widget.blockSignals(True)  # noqa: FBT003
        box_widget.setValue(v)
        box_widget.blockSignals(False)  # noqa: FBT003

        setattr(self, name, v)
        self.draw()

    def setup_lattice_controls(self, layout: QHBoxLayout | QVBoxLayout) -> None:
        """Create and connect a standalone double spinbox for lattice spacing.

        :param layout: The layout instance where the widget should be added.
        """
        # Create a sub-layout container for clean side-by-side alignment
        container = QHBoxLayout()
        label = QLabel("Lattice Spacing:")

        # Initialise Double SpinBox (Native floats, 2 decimal precision)
        self.lattice_spin = QDoubleSpinBox()
        self.lattice_spin.setRange(0.0, 100.0)
        self.lattice_spin.setSingleStep(0.01)
        self.lattice_spin.setDecimals(2)
        self.lattice_spin.setValue(self.lattice)

        # Connect the change signal directly to the updater slot
        self.lattice_spin.valueChanged.connect(self.update_lattice_value)

        # Assemble the widgets
        container.addWidget(label)
        container.addWidget(self.lattice_spin)
        layout.addLayout(container)

    def update_lattice_value(self, float_val: float) -> None:
        """Directly sync the backend lattice variable and re-render the SVG canvas.

        :param float_val: The lattice spacing value.
        """
        self.lattice = float_val
        self.draw()

    def submit_values(self, slider_widget: QSlider, box_widget: QDoubleSpinBox, name: str) -> None:
        """Unified spinbox-to-slider slot handling native numerical typing.

        :param slider_widget: The QSlider instance.
        :param box_widget: The QDoubleSpinBox instance to link to the slider.
        :param name: The name of the parameter to update using setattr().
        """
        v = box_widget.value()
        # Block signals to avoid feedback looping when setting the companion value
        slider_widget.blockSignals(True)  # noqa: FBT003
        slider_widget.setValue(int(v * 10))
        slider_widget.blockSignals(False)  # noqa: FBT003

        setattr(self, name, v)
        self.draw()

    def toggle_svg_background(self, checked: bool) -> None:
        """Swap rendering canvas stylesheets dynamically.

        :param checked: True for background, False for foreground.
        """
        if checked:
            self.svg_widget.setStyleSheet("background-color: white; border-radius: 4px;")
        else:
            self.svg_widget.setStyleSheet("background-color: transparent;")
        self._settings.setValue("bg_on", checked)

    def transform(self) -> CoordsArray3D:
        """Transform the 3D atom coordinates using the roll, pitch, yaw, x-offset, and y-offset.

        :returns: The transformed 3D atom coordinates.
        """
        rotations = _rotation_matrix(self.roll, self.pitch, self.yaw)
        pts: CoordsArray3D = self.atompos @ rotations
        pts -= np.mean(pts, axis=0)

        pts[:, 0] -= self.x_offset
        pts[:, 1] -= self.y_offset
        return pts

    def _circles(
        self, pts: CoordsArray3D, idx1: Literal[0, 1, 2], idx2: Literal[0, 1, 2], offset: float,
    ) -> list[svg.Circle]:
        """Create svg Circles using the atom coordinates.

        :param pts: The 3D atom coordinates.
        :param idx1: The first index. 0 = x, 1 = y, 2 = z.
        :param idx2: The second index. 0 = x, 1 = y, 2 = z.
        :param offset: The offset from the first axis.
        :returns: A list of svg Circles.
        """
        scale = 80
        cx, cy = 150, 150

        px_array: DistArray = cx + scale * pts[:, idx1] + offset
        py_array: DistArray = cy - scale * pts[:, idx2]

        radii = [RADII.get(key, 0.5) * 20 for key in self.atomkeys]

        return [
            svg.Circle(cx=px, cy=py, r=rad, fill=colour, fill_opacity=0.8, stroke="black")
            for px, py, rad, colour in zip(px_array, py_array, radii, self.colours, strict=True)
        ]

    def draw(self) -> None:  # noqa: PLR0915
        """Draw the molecule projection as a 2D SVG file with 4 tiles.

        Top left: xy. Top right: yz. Bottom left: xz. Bottom right: xy van der Waals projection with reference.
        """
        pts = self.transform()
        xs: DistArray
        ys: DistArray
        zs: DistArray
        xs, ys, zs = pts.T

        # --- SVG full size ---
        width, height = 800, 800

        # Force square drawing region
        side = min(width, height)

        # Centre square inside SVG
        offset_x = (width - side) / 2
        offset_y = (height - side) / 2

        # 2x2 grid inside square
        panel = side / 2
        scale = panel * 0.45

        # --- UNIFIED BOUNDARY & BUFFER CALCULATIONS ---
        lattice_x: DistArray = self.lattice * np.array([0, 1, -1, 0.5, -0.5, 0.5, -0.5])
        lattice_y: DistArray = self.lattice * np.array(
            [0, 0, 0, np.sqrt(3) / 2, np.sqrt(3) / 2, -np.sqrt(3) / 2, -np.sqrt(3) / 2],
        )

        all_x: DistArray = np.concatenate([xs, zs, lattice_x])
        all_y: DistArray = np.concatenate([ys, zs, lattice_y])

        xmin_v: float
        xmax_v: float
        ymin_v: float
        ymax_v: float

        xmin_v, xmax_v = np.min(all_x), np.max(all_x)
        ymin_v, ymax_v = np.min(all_y), np.max(all_y)

        span = max(xmax_v - xmin_v, ymax_v - ymin_v) * 1.5
        min_comparison: float = 1e-9
        if span < min_comparison:
            span = 1.0

        cx_data = (xmin_v + xmax_v) / 2
        cy_data = (ymin_v + ymax_v) / 2

        unit_to_pixel_ratio = (scale * 2) / span

        elements: list[svg.Line | svg.Text | svg.Polygon | svg.Circle | svg.Rect] = []

        # --- 3D BOND DETECTION ---
        bond_pairs: list[tuple[int, int]] = []
        if getattr(self, "show_bonds", False):
            num_atoms = len(pts)
            # Threshold parameters
            min_dist: float = 0.4
            max_dist: float = 1.9

            for jj in range(num_atoms):
                for kk in range(jj + 1, num_atoms):
                    # Calculate true Euclidean distance in 3D space
                    dist = np.linalg.norm(pts[jj] - pts[kk])
                    if min_dist <= dist <= max_dist:
                        bond_pairs.append((jj, kk))

        def add_axis_arrows(
            panel_cx: float,
            panel_cy: float,
            label_h: str,
            label_v: str,
        ) -> list[svg.Line | svg.Polygon | svg.Text]:
            """Add arrows to indicate the axes.

            :param panel_cx: The x coordinate of the panel centre.
            :param panel_cy: The y coordinate of the panel centre.
            :param label_h: The horizontal label.
            :param label_v: The vertical label.
            :returns: List of the svg elements for the axes arrows.
            """
            base_x = panel_cx - panel * 0.45
            base_y = panel_cy + panel * 0.45
            arrow_len = 35

            elements: list[svg.Line | svg.Polygon | svg.Text] = [
                svg.Line(x1=base_x, y1=base_y, x2=base_x + arrow_len, y2=base_y, stroke="black", stroke_width=1.5),
                svg.Polygon(
                    points=[
                        svg.Point(base_x + arrow_len, base_y),
                        svg.Point(base_x + arrow_len - 6, base_y - 3),
                        svg.Point(base_x + arrow_len - 6, base_y + 3),
                    ],
                    fill="black",
                ),
                svg.Text(
                    text=label_h,
                    x=base_x + arrow_len + 5,
                    y=base_y + 4,
                    font_size=svg.Length(14, "px"),
                    font_family="sans-serif",
                    fill="black",
                ),
                svg.Line(x1=base_x, y1=base_y, x2=base_x, y2=base_y - arrow_len, stroke="black", stroke_width=1.5),
                svg.Polygon(
                    points=[
                        svg.Point(base_x, base_y - arrow_len),
                        svg.Point(base_x - 3, base_y - arrow_len + 6),
                        svg.Point(base_x + 3, base_y - arrow_len + 6),
                    ],
                    fill="black",
                ),
                svg.Text(
                    text=label_v,
                    x=base_x - 4,
                    y=base_y - arrow_len - 6,
                    text_anchor="middle",
                    font_size=svg.Length(14, "px"),
                    font_family="sans-serif",
                    fill="black",
                ),
            ]

            return elements

        def norm(val: Tfloat, center: Tfloat) -> Tfloat:
            """Calculate the norm.

            :param val: Value to calculate the norm.
            :param center: Centre point.
            :returns: The norm.
            """
            return (val - center) / span

        def scatter_proj(
            xdata: DistArray,
            ydata: DistArray,
            depth: DistArray,
            col: int,
            row: int,
        ) -> list[svg.Line | svg.Circle]:
            """Give the 2D projection scatter plot of the molecule.

            :param xdata: The x coordinates of the molecule.
            :param ydata: The y coordinates of the molecule.
            :param depth: The depth (z coordinates) of the molecule.
            :param col: The column number of the panel.
            :param row: The row number of the panel.
            :returns: The 2D scatter plot elements.
            """
            cx = offset_x + panel * (col + 0.5)
            cy = offset_y + panel * (row + 0.5)
            elements: list[svg.Line | svg.Circle] = []

            # 1. Render bonds first if enabled, so they sit visually behind the atom markers
            for idx1, idx2 in bond_pairs:
                nx1, ny1 = norm(xdata[idx1], cx_data), norm(ydata[idx1], cy_data)
                nx2, ny2 = norm(xdata[idx2], cx_data), norm(ydata[idx2], cy_data)

                # --- CRITICAL FIX: Inverted Y axis (changed '-' to '+') ---
                px1, py1 = cx + nx1 * scale * 2, cy + ny1 * scale * 2
                px2, py2 = cx + nx2 * scale * 2, cy + ny2 * scale * 2

                elements.append(
                    svg.Line(x1=px1, y1=py1, x2=px2, y2=py2, stroke="#aaaaaa", stroke_width=2, stroke_dasharray=[4, 4]),
                )

            # 2. Render depth-sorted atoms
            min_comparison: float = 1e-9
            order = np.argsort(depth)
            dmin: float
            dmax: float
            dmin, dmax = np.min(depth), np.max(depth)
            depth_range = dmax - dmin if (dmax - dmin) > min_comparison else 1.0
            size = 30 + (depth - dmin) / depth_range * 30

            nx_array = norm(xdata[order], cx_data)
            ny_array = norm(ydata[order], cy_data)

            px_array = cx + nx_array * scale * 2
            # --- CRITICAL FIX: Inverted Y axis (changed '-' to '+') ---
            py_array = cy + ny_array * scale * 2
            r_array = size[order] * 0.2

            elements.extend(
                [
                    svg.Circle(
                        cx=px,
                        cy=py,
                        r=r,
                        fill=cast("str", self.colours[ii]),
                        stroke="none",
                    )
                    for ii, px, py, r in zip(order, px_array, py_array, r_array, strict=True)
                ],
            )
            return elements

        def vdw_lattice_proj(
            xdata: DistArray,
            ydata: DistArray,
            depth: DistArray,
            col: int,
            row: int,
        ) -> list[svg.Circle]:
            """Give the van der Waals molecular projection and lattice points.

            :param xdata: The x coordinates of the molecule.
            :param ydata: The y coordinates of the molecule.
            :param depth: The depth (z coordinates) of the molecule.
            :param col: The column number of the panel.
            :param row: The row number of the panel.
            :returns: The vdW and lattice SVG elements.
            """
            cx = offset_x + panel * (col + 0.5)
            cy = offset_y + panel * (row + 0.5)
            elements: list[svg.Circle] = []

            order = np.argsort(depth)

            nx_vdw = norm(xdata[order], cx_data)
            ny_vdw = norm(ydata[order], cy_data)

            px_vdw = cx + nx_vdw * scale * 2
            py_vdw = cy + ny_vdw * scale * 2

            r_vdw = np.array([RADII[self.atomkeys[jj]] * unit_to_pixel_ratio for jj in order])

            # Generate vdW elements
            for jj, px, py, r in zip(order, px_vdw, py_vdw, r_vdw, strict=True):
                colour = self.colours[jj]
                elements.append(
                    svg.Circle(
                        cx=px,
                        cy=py,
                        r=r,
                        fill=cast("str", colour),
                        fill_opacity=0.25,
                        stroke=cast("str", colour),
                    ),
                )
                elements.append(
                    svg.Circle(
                        cx=px,
                        cy=py,
                        r=r * 0.1,
                        fill=cast("str", colour),
                        stroke="black",
                    ),
                )

            # --- lattice points ---
            for lx, ly in zip(lattice_x, lattice_y, strict=True):
                nx = norm(lx, cx_data)
                ny = norm(ly, cy_data)

                px = cx + nx * scale * 2
                py = cy + ny * scale * 2

                elements.append(svg.Circle(cx=px, cy=py, r=6, fill="black"))

            return elements

        # Render the 3 traditional scatter projections
        elements.extend(scatter_proj(xs, ys, zs, 0, 0))
        elements.extend(scatter_proj(zs, ys, xs, 1, 0))
        elements.extend(scatter_proj(xs, zs, ys, 0, 1))

        # Render the 4th panel using the new helper
        elements.extend(vdw_lattice_proj(xs, ys, zs, 1, 1))

        # --- Add Axis Arrows into the Corners ---
        elements.extend(add_axis_arrows(offset_x + panel * 0.5, offset_y + panel * 0.5, "x", "y"))
        elements.extend(add_axis_arrows(offset_x + panel * 1.5, offset_y + panel * 0.5, "z", "y"))
        elements.extend(add_axis_arrows(offset_x + panel * 0.5, offset_y + panel * 1.5, "x", "z"))
        elements.extend(add_axis_arrows(offset_x + panel * 1.5, offset_y + panel * 1.5, "x", "y"))

        # 1. Grab the active rotation matrix from the backend configuration
        rotations = _rotation_matrix(self.roll, self.pitch, self.yaw)

        # 2. Define standard, colour-coded unit directions: X (Red), Y (Green), Z (Blue)
        axes_3d = np.identity(3, dtype=float)

        # 3. Rotate the basis vectors with the exact matrix your atoms use
        rotated_axes = axes_3d @ rotations

        # Anchor point in the bottom-left quadrant area of Panel 4
        rot_base_x = (offset_x + panel * 1.5) + panel * 0.35
        rot_base_y = (offset_y + panel * 1.5) - panel * 0.35
        axis_pixel_len = 35  # Visual size of the vectors

        # Axis properties for mapping loops: colours, labels, and drawing order (by depth/Z)
        axis_meta: list[dict[str, str | np.ndarray]] = [
            {"vec": rotated_axes[0], "color": "#d32f2f", "label": "x"},  # Red X
            {"vec": rotated_axes[1], "color": "#388e3c", "label": "y"},  # Green Y
            {"vec": rotated_axes[2], "color": "#1976d2", "label": "z"},  # Blue Z
        ]
        # Sort by depth (Z-value) so background vectors don't overlap foreground elements uglily
        axis_meta.sort(key=lambda item: cast("np.ndarray", item["vec"])[2])

        for axis in axis_meta:
            vec_arr = cast("np.ndarray", axis["vec"])
            vx, vy = vec_arr[0], vec_arr[1]

            # Map components to 2D view screen (X maps right, Y maps inverted up)
            end_x = rot_base_x + vx * axis_pixel_len
            end_y = rot_base_y - vy * axis_pixel_len

            # Draw the rotated vector line segment
            elements.append(
                svg.Line(x1=rot_base_x, y1=rot_base_y, x2=end_x, y2=end_y, stroke=str(axis["color"]), stroke_width=2.5),
            )

            # Add tip circles to make the 3D projection readable
            elements.append(svg.Circle(cx=end_x, cy=end_y, r=3, fill=str(axis["color"])))

            # Label string offset
            elements.append(
                svg.Text(
                    text=str(axis["label"]),
                    x=end_x + (5 if vx >= 0 else -12),
                    y=end_y + (4 if vy <= 0 else -6),
                    font_size=svg.Length(13, "px"),
                    font_weight="bold",
                    font_family="sans-serif",
                    fill=str(axis["color"]),
                ),
            )

        # Panel borders
        elements.extend(
            [
                svg.Rect(
                    x=offset_x + c * panel,
                    y=offset_y + r * panel,
                    width=panel,
                    height=panel,
                    fill="none",
                    stroke="#888",
                )
                for c in range(2)
                for r in range(2)
            ],
        )

        svg_out = svg.SVG(
            width=svg.Length(100, "%"),
            height=svg.Length(100, "%"),
            viewBox=svg.ViewBoxSpec(0, 0, width, height),
            preserveAspectRatio=svg.PreserveAspectRatio(),
            elements=elements,
        )

        self.svg_widget.load(str(svg_out).encode("utf-8"))


@validate_call
def first_time_loader(
    file_name: FilePath,
    roll: float = 0.0,
    pitch: float = 0.0,
    yaw: float = 0.0,
    x_offset: float = 0.0,
    y_offset: float = 0.0,
    ignore_atoms: str | list[str] | None = None,
    z_trim: float | None = None,
    reference_lattice_spacing: float = 1.0,
) -> dict[str, str | float | list[str] | None]:
    """Load molecule for the first time to save settings. Uses PySide6.

    :param file_name: File path/name for the .xyz file to load the molecule from.
    :param ignore_atoms: List of atoms to ignore. Leave empty to interactively toggle all.
    :param roll: Roll angle in degrees. Leave empty for default.
    :param pitch: Pitch angle in degrees.
    :param yaw: Yaw angle in degrees.
    :param x_offset: X offset in angstrom.
    :param y_offset: Y offset in angstrom.
    :param z_trim: Z trimming factor in angstrom. Filter all atoms with lower z values.
    :param reference_lattice_spacing: spacing for the reference lattice in angstrom.
    :returns:
        1) "file_name": str,
        2) "roll": float,
        3) "pitch": float,
        4) "yaw": float,
        5) "x_offset": float,
        6) "y_offset": float,
        7) "ignore_atoms": list[str],
        8) "z_trim": float | None
    """
    atomkeys, atompos = _initialise_reader(file_name, ignore_atoms, z_trim)

    color_map_adapter = TypeAdapter(dict[AtomKey, Color])
    colour_dict: dict[str, str] = {}

    try:
        json_path = Path(__file__).parent / "molecule_colour.json"
        raw_dict = color_map_adapter.validate_json(json_path.read_bytes())
        colour_dict = {k: v.as_hex() for k, v in raw_dict.items()}

    except (ValidationError, FileNotFoundError, json.JSONDecodeError) as e:
        warnings.warn(
            f"Warning: Could not parse colours safely ({e}). Using default #FFFFFF.",
            category=UserWarning,
            stacklevel=2,
        )

    colours = np.array([colour_dict.get(k, "#FFFFFF") for k in atomkeys])

    viewer = MoleculeViewer(
        atomkeys,
        atompos,
        colours,
        roll,
        pitch,
        yaw,
        x_offset,
        y_offset,
        reference_lattice_spacing,
    )
    viewer.resize(1600, 900)
    viewer.showMaximized()
    viewer.show()

    viewer.exec()

    result: dict[str, str | float | list[str] | None] = {
        "file_name": str(file_name),
        "roll": viewer.roll,
        "pitch": viewer.pitch,
        "yaw": viewer.yaw,
        "x_offset": viewer.x_offset,
        "y_offset": viewer.y_offset,
        "ignore_atoms": viewer.disabled_molecules,
        "z_trim": viewer.z_cutoff if viewer.z_filter_enable.isChecked() else None,
    }

    # print(f"kwargs = {result}")
    return result


def _xyz_verifier(
    atomkeys: StrArray,
    atompos: CoordsArray3D,
    listed_molecule_count: np.long,
) -> None:
    """Check if the .xyz file is of the correct format.

    :param atomkeys: Atom key values (element abbreviations).
    :param atompos: Atom positions in 3D.
    :param listed_molecule_count: Number of molecules according to the file.
    :raises ValueError:
        1) If the .xyz file has no listed molecule count or an invalid count.
        2) if the .xyz file's molecule count does not match the read molecule count.
        3) if the .xyz file contains bad molecule names.
        4) if the atom keys and atom coordinate lists are not equal in length.
        5) if coordinates are nan or infinite.
        6) if the coordinates are not 3D.
    """
    errmsg: str

    if listed_molecule_count is None or listed_molecule_count < 1:
        errmsg = "The .xyz file must contain a valid molecule count on line 1."
        raise ValueError(errmsg)

    if listed_molecule_count != atomkeys.size:
        errmsg = f"The file promises {listed_molecule_count} molecules but gives {atomkeys.size}"
        raise ValueError(errmsg)

    if not np.all(badmols := np.isin(atomkeys, list(RADII.keys()))):
        errmsg = f"Bad molecule types detected: {atomkeys[~badmols]}"
        raise ValueError(errmsg)

    if atomkeys.size != atompos.shape[0]:
        errmsg = "The keys and molecule coordinate lists are not of equal length."
        raise ValueError(errmsg)

    if not np.isfinite(atompos).all():
        errmsg = "The .xyz file contains invalid coordinates."
        raise ValueError(errmsg)

    if atompos.shape[1] != 3:  # noqa: PLR2004
        errmsg = "The .xyz file must contain 3D coordinates."
        raise ValueError(errmsg)


def _initialise_reader(
    file_name: str | Path,
    ignore_atoms: str | list[str] | None = None,
    z_trim: float | None = None,
) -> tuple[StrArray, CoordsArray3D]:
    """Initialise the xyz_reader and first_time_loader.

    :param file_name: name of the file to read.
    :param ignore_atoms: list of atoms to ignore.
    :param z_trim: z value under which to remove the atoms.
    :return: a tuple of the atom keys and atom positions in 3D. Filtered.
    :raises ValueError:
        1) if the file type is not .xyz
        2) if the used settings would return an empty molecule
    """
    file_path = Path(file_name)
    if (badtype := file_path.suffix) != ".xyz":
        errmsg = f"The file type is not .xyz but {badtype}"
        raise ValueError(errmsg)
    data = np.loadtxt(file_path, dtype=str, skiprows=2)
    listed_molecule_count: np.long = cast("np.long", np.loadtxt(file_path, dtype=np.long, max_rows=1))
    atomkeys: StrArray = data[:, 0]
    atompos: CoordsArray3D = cast("CoordsArray3D", data[:, 1:].astype(np.double))

    _xyz_verifier(atomkeys, atompos, listed_molecule_count)


    mask: BoolArray | None = None
    if isinstance(ignore_atoms, str):
        ignore_atoms = ignore_atoms.split(",")
    if ignore_atoms is None:
        pass
    elif isinstance(ignore_atoms[0], str):
        mask = np.ones(atomkeys.size, dtype=np.bool_)
        for ign_atm in ignore_atoms:
            mask &= atomkeys != ign_atm

    if z_trim is not None:
        mask = np.ones(atomkeys.size, dtype=np.bool_) if mask is None else mask
        mask &= atompos[:, 2] > z_trim

    if mask is not None:
        atomkeys = atomkeys[mask]
        atompos = atompos[mask]

    if not atomkeys.size:
        errmsg = "The current settings result in an empty molecule."
        raise ValueError(errmsg)

    return atomkeys, atompos


def save_molecule_svg(molecule: Polygon, lattice: float = 1.0, filename: str | Path | io.BytesIO = "") -> None:
    """Save the molecule shape as an SVG with a locked aspect ratio."""
    rounding: int = 4

    # 1. Extract and round molecule coordinates
    coords = np.round(
        np.asarray(molecule.exterior.coords, dtype=float),
        rounding,
    )

    poly = svg.Polygon(
        points=coords.flatten().tolist(),
        fill="grey",
        stroke="none",
    )

    # 2. Compute lattice circles
    elements: list[svg.Circle] = []
    lattice_x = lattice * np.array([0, 1, -1, 0.5, -0.5, 0.5, -0.5])
    lattice_y = lattice * np.array(
        [0, 0, 0, np.sqrt(3) / 2, np.sqrt(3) / 2, -np.sqrt(3) / 2, -np.sqrt(3) / 2],
    )

    for lx, ly in zip(lattice_x, lattice_y, strict=True):
        elements.append(svg.Circle(cx=float(lx), cy=float(ly), r=lattice * 0.1, fill="black"))

    # 3. Calculate dynamic viewBox bounds
    all_x = np.concatenate([coords[:, 0], lattice_x])
    all_y = np.concatenate([coords[:, 1], lattice_y])

    padding = max(lattice, 5)
    min_x, max_x = np.min(all_x) - padding, np.max(all_x) + padding
    min_y, max_y = np.min(all_y) - padding, np.max(all_y) + padding

    view_w = max_x - min_x
    view_h = max_y - min_y

    max_dim = max(view_w, view_h)

    # Adjust centers so the content remains perfectly framed
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    min_x = center_x - (max_dim / 2)
    min_y = center_y - (max_dim / 2)
    view_w = max_dim
    view_h = max_dim

    # 4. Force Uniform Scaling Aspect Ratio
    # This prevents the container from stretching the SVG unevenly.
    aspect_ratio = svg.PreserveAspectRatio("xMidYMid")

    # 5. Construct the SVG document
    svg_out = svg.SVG(
        width=svg.Length(100, "%"),
        height=svg.Length(100, "%"),
        viewBox=svg.ViewBoxSpec(min_x, min_y, view_w, view_h),
        preserveAspectRatio=aspect_ratio,  # Injected corrected aspect behaviour
        elements=[poly, *elements],
    )

    # 6. Handle output
    svg_string = str(svg_out)

    if isinstance(filename, io.BytesIO):
        filename.write(svg_string.encode("utf-8"))
    elif filename:
        Path(filename).write_text(svg_string, encoding="utf-8")
    else:
        print(svg_string)
        warnings.warn(
            "Warning: molecule SVG file could not be written.",
            category=UserWarning,
            stacklevel=2,
        )


if __name__ == "__main__":  # Best practice
    # while (file := input("File path name, or q to quit: ")).lower() not in {"q", "quit"}:
    #     first_time_loader(file)
    file = r"C:\Users\Flash User\Downloads\xyz_examples\fluorochloromethanol.xyz"
    first_time_loader(file)
