# Copyright (c) 2025-2026 Contributors to the AdsorPy project.
# SPDX-License-Identifier: MIT
"""Contains all the molecules that can be used in this simulation.

Also includes molecule loader scripts, for which the molecule data is not included in this lib.
You need to supply your own .xyz files, or you can use preconfigured simple shapes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Literal, ParamSpec, cast

import matplotlib.pyplot as plt
import numpy as np
import shapely
import shapely.affinity as aff
import svg
from pydantic import PositiveFloat, PositiveInt
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QFontMetrics
from PySide6.QtSvgWidgets import QSvgWidget
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)
from shapely import MultiPoint, MultiPolygon, Point, Polygon
from shapely.ops import unary_union

if TYPE_CHECKING:

    from src.adsorpy.types import BoolArray

    P = ParamSpec("P")  # Helps with static type checkers.


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


# Van der Waals radii of atoms. Source: https://doi.org/10.1039/C3DT50599E
current_dir_file = Path(__file__).parent / "VdW_Radii.csv"
RADII: dict[str, float] = dict(
    np.genfromtxt(
        current_dir_file,
        dtype=[("s", "U2"), ("f", "f4")],
        encoding="utf-8-sig",
        delimiter=",",
    ),
)
"""Key-value pairs of chemical symbols and van der Waals radii."""


def discorectangle(
    radius: PositiveFloat ,
    distance: PositiveFloat ,
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


def dogbonium(scale: PositiveFloat = 1) -> Polygon:
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


def polygonium(
    verts: PositiveInt = 3,
    scale: PositiveFloat = 1.0,
    roundedness: PositiveFloat = 0.0,
) -> Polygon:
    """Create a simple regular polygon with optional rounding.

    :param verts: The vertex count.
    :param scale: The scale factor of the polygon.
    :param roundedness: The roundedness.
    :return: The regular (rounded) polygon.
    """
    points = np.arange(verts, dtype=np.float64)
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


def xyz_reader(
    file_name: str | Path,
    ignore_atoms: str | list[str] | None = None,
    x_offset: float = 0.0,
    y_offset: float = 0.0,
    roll: float = 0.0,
    pitch: float = 0.0,
    yaw: float = 0.0,
    z_trim: float | None = None,
) -> Polygon:
    """Read files in the xyz format of VASP.

    :param file_name: The name of the file, including the .xyz extension. Include the path.
    :param ignore_atoms: Atoms to ignore when making the molecule. Useful if the slab is empty.
    :param x_offset: The offset in the x direction.
    :param y_offset: The offset in the x direction.
    :param yaw: Rotation along the x-axis.
    :param pitch: Rotation along the y-axis.
    :param roll: Rotation along the z-axis.
    :param z_trim: The z value below which all molecules are removed.

    :return: The molecule polygon read from the xyz file.
    """
    atomkeys, atompos = _initialise_reader(file_name, ignore_atoms, z_trim)

    ident_mat: np.ndarray[tuple[Literal[3], Literal[3]], np.dtype[np.float64]] = cast(
        "np.ndarray[tuple[Literal[3], Literal[3]], np.dtype[np.float64]]",
        np.identity(3, dtype=np.float64),
    )

    rollmat = ident_mat.copy()
    pitchmat = ident_mat.copy()
    yawmat = ident_mat.copy()
    fac = np.pi / 180.0
    rollmat[1, [1, 2]] = np.cos(roll * fac), -np.sin(roll * fac)
    rollmat[2, [1, 2]] = np.sin(roll * fac), np.cos(roll * fac)
    pitchmat[0, [0, 2]] = np.cos(pitch * fac), np.sin(pitch * fac)
    pitchmat[2, [0, 2]] = -np.sin(pitch * fac), np.cos(pitch * fac)
    yawmat[0, [0, 1]] = np.cos(yaw * fac), -np.sin(yaw * fac)
    yawmat[1, [0, 1]] = np.sin(yaw * fac), np.cos(yaw * fac)

    ident_mat[:] = ident_mat @ yawmat @ pitchmat @ rollmat

    for crdx, coords in enumerate(atompos):
        atompos[crdx] = coords @ ident_mat
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

def rotation_matrix(roll: np.floating, pitch: np.floating, yaw: np.floating) -> np.ndarray:
    """Compute the 3D rotation matrix using roll, pitch, and yaw.

    :param roll: Rotation along the x-axis.
    :param pitch: Rotation along the y-axis.
    :param yaw: Rotation along the z-axis.
    :returns: The rotation matrix.
    """
    fac = np.pi / 180.0

    rot_x = np.array([
        [1, 0, 0],
        [0, np.cos(roll * fac), -np.sin(roll * fac)],
        [0, np.sin(roll * fac), np.cos(roll * fac)],
    ])
    rot_y = np.array([
        [np.cos(pitch * fac), 0, np.sin(pitch * fac)],
        [0, 1, 0],
        [-np.sin(pitch * fac), 0, np.cos(pitch * fac)],
    ])
    rot_z = np.array([
        [np.cos(yaw * fac), -np.sin(yaw * fac), 0],
        [np.sin(yaw * fac), np.cos(yaw * fac), 0],
        [0, 0, 1],
    ])

    return rot_z @ rot_y @ rot_x


class MoleculeViewer(QWidget):
    """Molecule orientation widget."""

    def __init__(self, atomkeys, atompos, colours, lattice) -> None:
        """Initialise the molecule orientation widget."""
        super().__init__()

        # Step 1: Initialise raw variables and data placeholders
        self.atomkeys, self.atompos, self.colours = self._init_data(atomkeys, atompos, colours)

        self.lattice: float = lattice if lattice is not None else 1.0
        self.show_bonds: bool = False
        """Bool flag: are bonds displayed or not?"""
        self.atom_toggles: dict[str, QCheckBox] = {}

        # Step 2: Build the structural layout tree
        main_vert_layout: QVBoxLayout = QVBoxLayout()
        top_horizontal_layout: QHBoxLayout = QHBoxLayout()

        # Generate structural UI modules (A, B, and C)
        filter_panel: QVBoxLayout = self._create_filter_panel()
        plot_workspace: QVBoxLayout = self._create_plot_panel()

        # Assemble Top Row (A | B)
        self.setup_bond_controls(filter_panel)
        top_horizontal_layout.addLayout(filter_panel, stretch=1)
        self.setup_lattice_controls(filter_panel)
        top_horizontal_layout.addLayout(plot_workspace, stretch=5)
        main_vert_layout.addLayout(top_horizontal_layout, stretch=1)

        # Assemble Bottom Row (-C-) directly into the layout tree
        self._add_slider_panel(main_vert_layout)

        # Finalise and trigger the initial filter pipeline pass
        self.setLayout(main_vert_layout)
        self.apply_filters()

    @property
    def disabled_molecules(self) -> list[str] | None:
        """Get a sorted list of disabled molecule names, or None if empty."""
        # Gather keys where the visual checkbox is unchecked
        disabled = [str(atom) for atom, checkbox in self.atom_toggles.items() if not checkbox.isChecked()]

        if not disabled:
            return None

        return sorted(disabled, key=list(RADII.keys()).index)

    def _init_data(
        self,
        atomkeys: list[str] | np.ndarray,
        atompos: list[list[float]] | np.ndarray,
        colours: list[str] | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Initialise internal tracking parameters, limits, and array structures.

        :param atomkeys: Atoms keys.
        :returns:
            1) Atom keys
            2) Atom positions
            3) Atom colours
        """
        self.orig_atomkeys: np.ndarray = np.array(atomkeys, dtype=str)
        self.orig_atompos: np.ndarray = np.array(atompos, dtype=np.float64)
        self.orig_colours: np.ndarray = np.array(colours, dtype=str)

        temp_atomkeys: np.ndarray = self.orig_atomkeys.copy()
        temp_atompos: np.ndarray = self.orig_atompos.copy()
        temp_colours: np.ndarray = self.orig_colours.copy()

        self.roll: float = 0.0
        self.pitch: float = 0.0
        self.yaw: float = 0.0
        self.x_offset: float = 0.0
        self.y_offset: float = 0.0

        self.min_z: float = float(np.min(self.orig_atompos[:, 2])) if self.orig_atompos.size else -10.0
        self.max_z: float = float(np.max(self.orig_atompos[:, 2])) if self.orig_atompos.size else 10.0
        self.z_cutoff: float = self.min_z - 0.1

        return temp_atomkeys, temp_atompos, temp_colours

    def _create_filter_panel(self) -> QVBoxLayout:
        """Build the panel component hosting Z-cutoff adjustments and checkboxes.

        :return: Main target layout framework representing Column A.
        """
        filter_panel: QVBoxLayout = QVBoxLayout()
        filter_panel.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.bg_toggle: QPushButton = QPushButton("Toggle White Background")
        self.bg_toggle.setCheckable(True)
        self.bg_toggle.clicked.connect(self.toggle_svg_background)
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
        self.z_filter_enable.toggled.connect(self.z_spinbox.setEnabled)
        self.z_filter_enable.toggled.connect(lambda _: self.apply_filters())
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

            cb: QCheckBox = QCheckBox(f"Show {atom}")
            cb.setChecked(True)
            self.atom_toggles[atom] = cb
            cb.stateChanged.connect(lambda state: self.apply_filters())

            color_swatch: QLabel = QLabel()
            color_swatch.setFixedSize(QSize(14, 14))
            color_swatch.setStyleSheet(f"background-color: {color_hex}; border: 1px solid #555555; border-radius: 2px;")

            item_row.addWidget(color_swatch)
            item_row.addWidget(cb)
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


        plot_workspace.addLayout(plot_toggle_layout)

        return plot_workspace

    def _add_slider_panel(self, target_layout: QVBoxLayout) -> None:
        """Append transformation sliders directly across the bottom container.

        :param target_layout: Top-level root layout accepting row insertions.
        """
        slider_params: dict[str, tuple[int, int]] = {
            "roll": (-180, 180),
            "pitch": (-180, 180),
            "yaw": (-180, 180),
            "x_offset": (-5, 5),
            "y_offset": (-5, 5),
        }

        max_width: int = 0
        font_metrics: QFontMetrics = self.fontMetrics()

        for param in slider_params:
            width: int = font_metrics.horizontalAdvance(param)
            max_width = max(max_width, width)

        max_width += 10

        for name, val_range in slider_params.items():
            row: QHBoxLayout = QHBoxLayout()

            label: QLabel = QLabel(name)
            label.setFixedWidth(max_width)

            slider: QSlider = QSlider(Qt.Horizontal)
            slider.setMinimum(val_range[0] * 10)
            slider.setMaximum(val_range[1] * 10)

            box: QDoubleSpinBox = QDoubleSpinBox()
            box.setRange(float(val_range[0]), float(val_range[1]))
            box.setDecimals(2)
            box.setSingleStep(0.1)
            box.setValue(0.0)
            box.setFixedWidth(120)

            slider.valueChanged.connect(
                lambda val, n=name, b=box: self.update_values(val, n, b),
            )

            box.editingFinished.connect(
                lambda s=slider, b=box, n=name: self.submit_values(s, b, n),
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
        """Slot targeting real-time spinbox adjustments to update pipeline state."""
        self.z_cutoff = value
        self.apply_filters()

    def apply_filters(self) -> None:
        """Calculate boolean masks against root datasets and handle redraw requests."""
        mask: np.ndarray = self.orig_atompos[:, 2] >= self.z_cutoff

        allowed_types: set[str] = {atom for atom, cb in self.atom_toggles.items() if cb.isChecked()}
        type_mask: np.ndarray = np.isin(self.orig_atomkeys, list(allowed_types))

        combined_mask: np.ndarray = mask & type_mask

        self.atomkeys = self.orig_atomkeys[combined_mask]
        self.atompos = self.orig_atompos[combined_mask]
        self.colours = self.orig_colours[combined_mask]

        self.draw()

    def setup_bond_controls(self, layout) -> None:
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

    def update_values(self, val, name, box_widget) -> None:
        """Unified slider-to-backend slot keeping widgets cleanly scoped."""
        v = val / 10
        # Block signals to avoid feedback looping when setting the companion value
        box_widget.blockSignals(True)  # noqa: FBT003
        box_widget.setValue(v)
        box_widget.blockSignals(False)  # noqa: FBT003

        setattr(self, name, v)
        self.draw()

    def setup_lattice_controls(self, layout) -> None:
        """Create and connect a standalone double spinbox for lattice spacing.

        :param layout: The QLayout instance where the widget should be added.
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
        """Directly sync the backend lattice variable and re-render the SVG canvas."""
        self.lattice = float_val
        self.draw()

    def submit_values(self, slider_widget, box_widget) -> None:
        """Unified spinbox-to-slider slot handling native numerical typing."""
        v = box_widget.value()
        # Block signals to avoid feedback looping when setting the companion value
        slider_widget.blockSignals(True)  # noqa: FBT003
        slider_widget.setValue(int(v * 10))
        slider_widget.blockSignals(False)  # noqa: FBT003

        setattr(self, slider_widget, v)
        self.draw()

    def toggle_svg_background(self, checked: bool) -> None:
        """Swap rendering canvas stylesheets dynamically.

        :param checked: True for background, False for foreground.
        """
        if checked:
            self.svg_widget.setStyleSheet("background-color: white; border-radius: 4px;")
        else:
            self.svg_widget.setStyleSheet("background-color: transparent;")

    def transform(self):
        rotations = rotation_matrix(self.roll, self.pitch, self.yaw)
        pts = self.atompos @ rotations
        pts -= np.mean(pts, axis=0)

        pts[:, 0] -= self.x_offset
        pts[:, 1] -= self.y_offset
        return pts

    def _circles(self, pts, idx1, idx2, offset_x):
        elements = []
        scale = 80
        cx, cy = 150, 150

        for (x, y, z), col, key in zip(pts, self.colours, self.atomkeys):
            coords = [x, y, z]

            px = cx + scale * coords[idx1] + offset_x
            py = cy - scale * coords[idx2]

            r = RADII.get(key, 0.5) * 20

            elements.append(svg.Circle(cx=px, cy=py, r=r, fill=col, opacity=0.8))
            elements.append(svg.Circle(cx=px, cy=py, r=r, fill="none", stroke="black"))

        return elements

    def draw(self) -> None:
        pts = self.transform()

        xs, ys, zs = pts.T

        # --- SVG full size ---
        W, H = 1000, 800

        # Force square drawing region
        side = min(W, H)

        # Center square inside SVG
        offset_x = (W - side) / 2
        offset_y = (H - side) / 2

        # 2x2 grid inside square
        panel = side / 2
        scale = panel * 0.45

        # --- UNIFIED BOUNDARY & BUFFER CALCULATIONS ---
        lattice_x = self.lattice * np.array([0, 1, -1, 0.5, -0.5, 0.5, -0.5])
        lattice_y = self.lattice * np.array(
            [0, 0, 0, np.sqrt(3) / 2, np.sqrt(3) / 2, -np.sqrt(3) / 2, -np.sqrt(3) / 2],
        )

        all_x = np.concatenate([xs, zs, lattice_x])
        all_y = np.concatenate([ys, zs, lattice_y])

        xmin_v, xmax_v = np.min(all_x), np.max(all_x)
        ymin_v, ymax_v = np.min(all_y), np.max(all_y)

        span = max(xmax_v - xmin_v, ymax_v - ymin_v) * 1.5
        if span < 1e-9:
            span = 1.0

        cx_data = (xmin_v + xmax_v) / 2
        cy_data = (ymin_v + ymax_v) / 2

        unit_to_pixel_ratio = (scale * 2) / span

        elements = []

        # --- 3D BOND DETECTION ---
        bond_pairs = []
        if getattr(self, "show_bonds", False):
            num_atoms = len(pts)
            # Threshold parameters: modify these numbers to fit your dataset's units
            min_dist = 0.4
            max_dist = 1.9

            for jj in range(num_atoms):
                for j in range(jj + 1, num_atoms):
                    # Calculate true Euclidean distance in 3D space
                    dist = np.linalg.norm(pts[jj] - pts[j])
                    if min_dist <= dist <= max_dist:
                        bond_pairs.append((jj, j))

        def add_axis_arrows(panel_cx, panel_cy, label_h, label_v):
            base_x = panel_cx - panel * 0.45
            base_y = panel_cy + panel * 0.45
            arrow_len = 35

            elements.append(
                svg.Line(x1=base_x, y1=base_y, x2=base_x + arrow_len, y2=base_y, stroke="black", stroke_width=1.5)
            )
            elements.append(
                svg.Polygon(
                    points=[
                        (base_x + arrow_len, base_y),
                        (base_x + arrow_len - 6, base_y - 3),
                        (base_x + arrow_len - 6, base_y + 3),
                    ],
                    fill="black",
                )
            )
            elements.append(
                svg.Text(
                    text=label_h,
                    x=base_x + arrow_len + 5,
                    y=base_y + 4,
                    font_size=svg.Length(14, "px"),
                    font_family="sans-serif",
                    fill="black",
                )
            )

            elements.append(
                svg.Line(x1=base_x, y1=base_y, x2=base_x, y2=base_y - arrow_len, stroke="black", stroke_width=1.5)
            )
            elements.append(
                svg.Polygon(
                    points=[
                        (base_x, base_y - arrow_len),
                        (base_x - 3, base_y - arrow_len + 6),
                        (base_x + 3, base_y - arrow_len + 6),
                    ],
                    fill="black",
                )
            )
            elements.append(
                svg.Text(
                    text=label_v,
                    x=base_x - 4,
                    y=base_y - arrow_len - 6,
                    text_anchor="middle",
                    font_size=svg.Length(14, "px"),
                    font_family="sans-serif",
                    fill="black",
                )
            )

        def norm(val, center):
            return (val - center) / span

        def scatter_proj(xdata, ydata, depth, col, row):
            cx = offset_x + panel * (col + 0.5)
            cy = offset_y + panel * (row + 0.5)

            # 1. Render bonds first if enabled, so they sit visually behind the atom markers
            for idx1, idx2 in bond_pairs:
                nx1, ny1 = norm(xdata[idx1], cx_data), norm(ydata[idx1], cy_data)
                nx2, ny2 = norm(xdata[idx2], cx_data), norm(ydata[idx2], cy_data)

                px1, py1 = cx + nx1 * scale * 2, cy - ny1 * scale * 2
                px2, py2 = cx + nx2 * scale * 2, cy - ny2 * scale * 2

                elements.append(
                    svg.Line(x1=px1, y1=py1, x2=px2, y2=py2, stroke="#aaaaaa", stroke_width=2, stroke_dasharray="4,4")
                )

            # 2. Render depth-sorted atoms
            order = np.argsort(depth)
            dmin, dmax = np.min(depth), np.max(depth)
            depth_range = dmax - dmin if (dmax - dmin) > 1e-9 else 1.0
            size = 30 + (depth - dmin) / depth_range * 30

            for ii in order:
                nx = norm(xdata[ii], cx_data)
                ny = norm(ydata[ii], cy_data)

                px = cx + nx * scale * 2
                py = cy - ny * scale * 2

                r = size[ii] * 0.2

                elements.append(
                    svg.Circle(
                        cx=px,
                        cy=py,
                        r=r,
                        fill=self.colours[ii],
                        stroke="none",
                    )
                )

        # Render the 3 traditional scatter projections
        scatter_proj(xs, ys, zs, 0, 0)
        scatter_proj(zs, ys, xs, 1, 0)
        scatter_proj(xs, zs, ys, 0, 1)

        # --- vdW (Panel 4) ---
        cx = offset_x + panel * 1.5
        cy = offset_y + panel * 1.5

        order = np.argsort(zs)

        for jj in order:
            nx = norm(xs[jj], cx_data)
            ny = norm(ys[jj], cy_data)

            px = cx + nx * scale * 2
            py = cy - ny * scale * 2

            physical_radius = RADII[self.atomkeys[jj]]
            r = physical_radius * unit_to_pixel_ratio

            elements.append(
                svg.Circle(
                    cx=px,
                    cy=py,
                    r=r,
                    fill=self.colours[jj],
                    opacity=0.25,
                )
            )

            elements.append(
                svg.Circle(
                    cx=px,
                    cy=py,
                    r=r,
                    fill="none",
                    stroke=self.colours[jj],
                )
            )

            elements.append(
                svg.Circle(
                    cx=px,
                    cy=py,
                    r=r * 0.1,
                    fill=self.colours[jj],
                    stroke="black",
                )
            )

        # --- lattice points ---
        for lx, ly in zip(lattice_x, lattice_y, strict=True):
            nx = norm(lx, cx_data)
            ny = norm(ly, cy_data)

            px = cx + nx * scale * 2
            py = cy - ny * scale * 2

            elements.append(svg.Circle(cx=px, cy=py, r=6, fill="black"))

        # --- Add Axis Arrows into the Corners ---
        add_axis_arrows(offset_x + panel * 0.5, offset_y + panel * 0.5, "x", "y")
        add_axis_arrows(offset_x + panel * 1.5, offset_y + panel * 0.5, "z", "y")
        add_axis_arrows(offset_x + panel * 0.5, offset_y + panel * 1.5, "x", "z")
        add_axis_arrows(offset_x + panel * 1.5, offset_y + panel * 1.5, "x", "y")

        # 1. Grab the active rotation matrix from your backend configuration
        rotations = rotation_matrix(self.roll, self.pitch, self.yaw)

        # 2. Define standard, color-coded unit directions: X (Red), Y (Green), Z (Blue)
        axes_3d = np.array(
            [
                [1.0, 0.0, 0.0],  # X unit vector
                [0.0, 1.0, 0.0],  # Y unit vector
                [0.0, 0.0, 1.0],  # Z unit vector
            ]
        )

        # 3. Rotate the basis vectors with the exact matrix your atoms use
        rotated_axes = axes_3d @ rotations

        # Anchor point in the bottom-left quadrant area of Panel 4
        rot_base_x = cx + panel * 0.35
        rot_base_y = cy - panel * 0.35
        axis_pixel_len = 35  # Visual size of the vectors

        # Axis properties for mapping loops: colours, labels, and drawing order (by depth/Z)
        axis_meta = [
            {"vec": rotated_axes[0], "color": "#d32f2f", "label": "x"},  # Red X
            {"vec": rotated_axes[1], "color": "#388e3c", "label": "y"},  # Green Y
            {"vec": rotated_axes[2], "color": "#1976d2", "label": "z"},  # Blue Z
        ]
        # Sort by depth (Z-value) so background vectors don't overlap foreground elements uglily
        axis_meta.sort(key=lambda item: item["vec"][2])

        for axis in axis_meta:
            vx, vy, vz = axis["vec"]

            # Map components to 2D view screen (X maps right, Y maps inverted up)
            end_x = rot_base_x + vx * axis_pixel_len
            end_y = rot_base_y - vy * axis_pixel_len

            # Draw the rotated vector line segment
            elements.append(
                svg.Line(x1=rot_base_x, y1=rot_base_y, x2=end_x, y2=end_y, stroke=axis["color"], stroke_width=2.5)
            )

            # Add tip circles to make the 3D projection readable
            elements.append(svg.Circle(cx=end_x, cy=end_y, r=3, fill=axis["color"]))

            # Label string offset
            elements.append(
                svg.Text(
                    text=axis["label"],
                    x=end_x + (5 if vx >= 0 else -12),
                    y=end_y + (4 if vy <= 0 else -6),
                    font_size=svg.Length(13, "px"),
                    font_weight="bold",
                    font_family="sans-serif",
                    fill=axis["color"],
                ),
            )

        # Panel borders
        elements.extend([
            svg.Rect(
                x=offset_x + c * panel,
                y=offset_y + r * panel,
                width=panel,
                height=panel,
                fill="none",
                stroke="#888",
            )
            for c in range(2) for r in range(2)
        ])

        svg_out = svg.SVG(
            width=svg.Length(100, "%"),
            height=svg.Length(100, "%"),
            viewBox=svg.ViewBoxSpec(0, 0, W, H),
            preserveAspectRatio=svg.PreserveAspectRatio(),
            elements=elements,
        )

        self.svg_widget.load(str(svg_out).encode("utf-8"))

def first_time_loader(
    file_name,
    ignore_atoms=None,
    x_offset=None,
    y_offset=None,
    roll=None,
    pitch=None,
    yaw=None,
    z_trim=None,
    reference_lattice_spacing=4.74,
):
    atomkeys, atompos = _initialise_reader(file_name, ignore_atoms, z_trim)

    with (Path(__file__).parent / "molecule_colour.json").open() as f:
        colourdict = json.load(f)

    colours = np.array([colourdict.get(k, "#FFFFFF") for k in atomkeys])

    app = QApplication.instance() or QApplication([])

    viewer = MoleculeViewer(
        atomkeys,
        atompos,
        colours,
        reference_lattice_spacing,
    )
    viewer.resize(1600, 900)
    viewer.showMaximized()
    viewer.show()

    app.exec()

    result = {
        "x_offset": viewer.x_offset,
        "y_offset": viewer.y_offset,
        "roll": viewer.roll,
        "pitch": viewer.pitch,
        "yaw": viewer.yaw,
        "file_name": str(file_name),
        "ignore_atoms": viewer.disabled_molecules,
        "z_trim": z_trim,
    }

    print(f"kwargs = {result}")
    return result


def _xyz_verifier(
        atomkeys: np.ndarray[tuple[int], np.dtype[np.str_]],
        atompos: np.ndarray[tuple[int, Literal[3]], np.dtype[np.float64]],
        listed_molecule_count: np.int_,
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
) -> tuple[np.ndarray[tuple[int], np.dtype[np.str_]], np.ndarray[tuple[int, Literal[3]], np.dtype[np.float64]]]:
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
    listed_molecule_count: np.int_ = cast("np.int_", np.loadtxt(file_path, dtype=np.int_, max_rows=1))
    atomkeys: np.ndarray[tuple[int], np.dtype[np.str_]] = data[:, 0]
    atompos: np.ndarray[tuple[int, Literal[3]], np.dtype[np.float64]] = cast(
        "np.ndarray[tuple[int, Literal[3]], np.dtype[np.float64]]",
        data[:, 1:].astype(np.float64),
    )

    _xyz_verifier(atomkeys, atompos, listed_molecule_count)

    mask: BoolArray | None = None
    if isinstance(ignore_atoms, str):
        mask = atomkeys != ignore_atoms
    elif ignore_atoms is None:
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


if __name__ == "__main__":  # Best practice
    # while (file := input("File path name, or q to quit: ")).lower() not in {"q", "quit"}:
    #     first_time_loader(file)
    file = r"C:\Users\Flash User\Downloads\xyz_examples\fluorochloromethanol.xyz"
    first_time_loader(file)
