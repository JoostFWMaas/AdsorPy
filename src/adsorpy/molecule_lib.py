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
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle
from matplotlib.widgets import Slider, TextBox
from shapely import MultiPoint, MultiPolygon, Point, Polygon
from shapely.ops import unary_union

from pydantic import PositiveFloat, PositiveInt
import json
from pathlib import Path
import numpy as np

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QSlider, QLabel, QLineEdit, QGraphicsScene, QDoubleSpinBox, QPushButton, QSizePolicy
)
from PySide6.QtSvgWidgets import QSvgWidget, QGraphicsSvgItem
from PySide6.QtCore import Qt

from svg import SVG, Circle

if TYPE_CHECKING:
    from matplotlib import axes, figure
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import NDArray

    from src.adsorpy.types import BoolArray, FloatArray, InDict, RotMatrix

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
    # *args: P.args,
    # **kwargs: P.kwargs,
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

# C:\Users\s137316\OneDrive - TU Eindhoven\Documents\BDEAS.xyz
def rotation_matrix(roll, pitch, yaw):
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
    def __init__(self, atomkeys, atompos, colours, lattice):
        super().__init__()

        self.atomkeys = atomkeys
        self.atompos = atompos
        self.colours = colours
        self.lattice = lattice

        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.x_offset = 0.0
        self.y_offset = 0.0

        layout = QVBoxLayout()

        # Initialize and center SVG widget
        self.svg_widget = QSvgWidget()
        self.svg_widget.setMinimumSize(300, 300)
        self.svg_widget.renderer().setAspectRatioMode(Qt.AspectRatioMode.KeepAspectRatio)
        layout.addWidget(self.svg_widget, alignment=Qt.AlignmentFlag.AlignCenter)

        # Add background toggle button
        self.bg_toggle = QPushButton("Toggle White Background")
        self.bg_toggle.setCheckable(True)
        self.bg_toggle.clicked.connect(self.toggle_svg_background)
        layout.addWidget(self.bg_toggle)
        self.bg_toggle.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        hint = self.bg_toggle.sizeHint()
        self.bg_toggle.setFixedWidth(hint.width() + 20)
        # layout.addWidget(self.bg_toggle, alignment=Qt.AlignmentFlag.AlignLeft)

        slider_params = {
            "roll": (-180, 180),
            "pitch": (-180, 180),
            "yaw": (-180, 180),
            "x_offset": (-5, 5),
            "y_offset": (-5, 5),
        }

        # Step 1: compute max width
        max_width = 0
        font_metrics = self.fontMetrics()

        for param in slider_params:
            width = font_metrics.horizontalAdvance(param)
            max_width = max(max_width, width)

        max_width += 10

        self.sliders = {}
        for name, val_range in slider_params.items():
            row = QHBoxLayout()

            label = QLabel(name)
            label.setFixedWidth(max_width)

            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(val_range[0] * 10)
            slider.setMaximum(val_range[1] * 10)

            box = QDoubleSpinBox()
            box.setRange(val_range[0], val_range[1])
            box.setDecimals(2)
            box.setSingleStep(0.1)
            box.setValue(0.0)
            box.setFixedWidth(120)

            # Connect slider to update logic using safe lambda parameter isolation
            slider.valueChanged.connect(
                lambda val, n=name, b=box: self.update_values(val, n, b)
            )

            # Connect spinbox finishing steps using safe lambda parameter isolation
            box.editingFinished.connect(
                lambda s=slider, b=box: self.submit_values(s, b)
            )

            row.addWidget(label)
            row.addWidget(slider)
            row.addWidget(box)
            layout.addLayout(row)

        self.setLayout(layout)
        self.draw()

    def update_values(self, val, name, box_widget):
        """Unified slider-to-backend slot keeping widgets cleanly scoped."""
        v = val / 10
        # Block signals to avoid feedback looping when setting the companion value
        box_widget.blockSignals(True)
        box_widget.setValue(v)
        box_widget.blockSignals(False)

        setattr(self, name, v)
        self.draw()

    def submit_values(self, slider_widget, box_widget):
        """Unified spinbox-to-slider slot handling native numerical typing."""
        v = box_widget.value()
        slider_widget.blockSignals(True)
        slider_widget.setValue(int(v * 10))
        slider_widget.blockSignals(False)

    def toggle_svg_background(self, checked):
        """Swaps rendering canvas stylesheets dynamically."""
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

            elements.append(Circle(cx=px, cy=py, r=r, fill=col, opacity=0.8))
            elements.append(Circle(cx=px, cy=py, r=r, fill="none", stroke="black"))

        return elements

    def draw(self):
        pts = self.transform()

        xs, ys, zs = pts.T

        xmin, xmax = np.min(xs), np.max(xs)
        ymin, ymax = np.min(ys), np.max(ys)
        zmin, zmax = np.min(zs), np.max(zs)

        # --- SVG full size ---
        W, H = 1000, 800

        # force square drawing region
        side = min(W, H)

        # center square inside SVG
        offset_x = (W - side) / 2
        offset_y = (H - side) / 2

        # 2x2 grid inside square
        panel = side / 2

        elements = []

        # --- helper for normalization (SHARED scale per panel) ---
        def norm(val, vmin, vmax, span, center):
            return (val - center) / span

        def get_bounds(arr):
            return np.min(arr), np.max(arr)

        # ✅ scatter projection (NOT vdW)
        def scatter_proj(xdata, ydata, depth, col, row):
            cx = panel * (col + 0.5)
            cy = panel * (row + 0.5)
            scale = panel * 0.45

            xmin_, xmax_ = get_bounds(xdata)
            ymin_, ymax_ = get_bounds(ydata)

            span = max(xmax_ - xmin_, ymax_ - ymin_)
            if span < 1e-9:
                span = 1.0

            cx_data = (xmin_ + xmax_) / 2
            cy_data = (ymin_ + ymax_) / 2

            # ✅ match your matplotlib: depth sorting + size scaling
            order = np.argsort(depth)

            # marker size scaling (same idea as your _update)
            dmin, dmax = np.min(depth), np.max(depth)
            size = 30 + (depth - dmin) / (dmax - dmin + 1e-9) * 30

            for i in order:
                nx = norm(xdata[i], xmin_, xmax_, span, cx_data)
                ny = norm(ydata[i], ymin_, ymax_, span, cy_data)

                px = cx + nx * scale * 2
                py = cy - ny * scale * 2

                r = size[i] * 0.2  # convert marker size → pixel radius

                elements.append(Circle(
                    cx=px, cy=py, r=r,
                    fill=self.colours[i],
                    stroke="none"
                ))

        # ✅ XY (depth = z)
        scatter_proj(xs, ys, zs, 0, 0)

        # ✅ ZY (depth = x)
        scatter_proj(zs, ys, xs, 1, 0)

        # ✅ XZ (depth = y)
        scatter_proj(xs, zs, ys, 0, 1)

        # --- vdW ---
        cx = offset_x + panel * 1.5
        cy = offset_y + panel * 1.5
        scale = panel * 0.45

        span = max(xmax - xmin, ymax - ymin)
        if span < 1e-9:
            span = 1.0

        cx_data = (xmin + xmax) / 2
        cy_data = (ymin + ymax) / 2

        order = np.argsort(zs)

        for i in order:
            nx = (xs[i] - cx_data) / span
            ny = (ys[i] - cy_data) / span

            px = cx + nx * scale * 2
            py = cy - ny * scale * 2

            r = RADII[self.atomkeys[i]] * scale * 0.3

            # fill
            elements.append(Circle(
                cx=px, cy=py, r=r,
                fill=self.colours[i],
                opacity=0.25
            ))

            # outline
            elements.append(Circle(
                cx=px, cy=py, r=r,
                fill="none",
                stroke=self.colours[i]
            ))

        # --- lattice points (preserved) ---
        lattice_x = self.lattice * np.array([0, 1, -1, 0.5, -0.5, 0.5, -0.5])
        lattice_y = self.lattice * np.array(
            [0, 0, 0, np.sqrt(3) / 2, np.sqrt(3) / 2, -np.sqrt(3) / 2, -np.sqrt(3) / 2]
        )

        for lx, ly in zip(lattice_x, lattice_y):
            nx = (lx - cx_data) / span
            ny = (ly - cy_data) / span

            px = cx + nx * scale * 2
            py = cy - ny * scale * 2

            elements.append(Circle(cx=px, cy=py, r=6, fill="black"))

        # panel borders (debug/visual clarity)
        from svg import Rect
        for c in range(2):
            for r in range(2):
                elements.append(Rect(
                    x=offset_x + c * panel,
                    y=offset_y + r * panel,
                    width=panel,
                    height=panel,
                    fill="none",
                    stroke="#888"
                ))

        svg = SVG(
            width="100%",
            height="100%",
            viewBox=f"0 0 {W} {H}",
            preserveAspectRatio="xMidYMid meet",
            elements=elements,
        )

        self.svg_widget.load(str(svg).encode("utf-8"))

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
        "ignore_atoms": ignore_atoms,
        "z_trim": z_trim,
    }

    print(f"kwargs = {result}")
    return result


# def _update(  # noqa: PLR0913
#     val: float,
#     ang_x: Slider,
#     ang_y: Slider,
#     ang_z: Slider,
#     translate_x: Slider,
#     translate_y: Slider,
#     box_x: TextBox,
#     box_y: TextBox,
#     box_z: TextBox,
#     box_xoff: TextBox,
#     box_yoff: TextBox,
#     atompos: FloatArray,
#     axs: list[axes.Axes],
#     fig: figure.Figure,
#     atomcolours: NDArray[np.str_],
#     atomkeys: NDArray[np.str_],
#     reference_lattice_spacing: float,
# ) -> None:
#     roll = ang_z.val
#     pitch = ang_y.val
#     yaw = ang_x.val
#     x_offset = translate_x.val
#     y_offset = translate_y.val
#
#     box_z.set_val(f"{roll:.3f}")
#     box_y.set_val(f"{pitch:.3f}")
#     box_x.set_val(f"{yaw:.3f}")
#     box_xoff.set_val(f"{x_offset:.3f}")
#     box_yoff.set_val(f"{y_offset:.3f}")
#
#     ident_mat: RotMatrix = cast("RotMatrix", np.identity(3))
#
#     rollmat = ident_mat.copy()
#     pitchmat = ident_mat.copy()
#     yawmat = ident_mat.copy()
#     fac = np.pi / 180.0
#     rollmat[1, [1, 2]] = np.cos(roll * fac), -np.sin(roll * fac)
#     rollmat[2, [1, 2]] = np.sin(roll * fac), np.cos(roll * fac)
#     pitchmat[0, [0, 2]] = np.cos(pitch * fac), np.sin(pitch * fac)
#     pitchmat[2, [0, 2]] = -np.sin(pitch * fac), np.cos(pitch * fac)
#     yawmat[0, [0, 1]] = np.cos(yaw * fac), -np.sin(yaw * fac)
#     yawmat[1, [0, 1]] = np.sin(yaw * fac), np.cos(yaw * fac)
#
#     ident_mat = ident_mat @ yawmat @ pitchmat @ rollmat
#
#     newatompos = atompos.copy()
#
#     for crdx, coords in enumerate(atompos):
#         newatompos[crdx] = coords @ ident_mat
#
#     newatompos -= np.mean(newatompos, axis=0)
#     if x_offset is not None:
#         newatompos[:, 0] -= x_offset
#     if y_offset is not None:
#         newatompos[:, 1] -= y_offset
#
#     for ax in axs:
#         ax.set_aspect("equal", "box")
#         ax.clear()
#
#     axs[0].set_xlabel("x-ax")
#     axs[0].set_ylabel("y-ax")
#     axs[1].set_xlabel("z-ax")
#     axs[1].set_ylabel("y-ax")
#     axs[2].set_xlabel("x-ax")
#     axs[2].set_ylabel("z-ax")
#     axs[3].set_xlabel("x-ax")
#     axs[3].set_ylabel("y-ax")
#
#     xs: FloatArray
#     ys: FloatArray
#     zs: FloatArray
#     xs, ys, zs = newatompos.T
#
#     xminmax = np.min(xs), np.max(xs)
#     yminmax = np.min(ys), np.max(ys)
#     zminmax = np.min(zs), np.max(zs)
#
#     xmarker = 30 + (xs - xminmax[0]) / (xminmax[1] - xminmax[0]) * 30
#     ymarker = 30 + (ys - yminmax[0]) / (yminmax[1] - yminmax[0]) * 30
#     zmarker = 30 + (zs - zminmax[0]) / (zminmax[1] - zminmax[0]) * 30
#
#     xsort = np.argsort(xs)
#     ysort = np.argsort(ys)
#     zsort = np.argsort(zs)
#
#     axs[0].scatter(xs[zsort], ys[zsort], c=atomcolours[zsort], s=zmarker[zsort])
#     axs[1].scatter(zs[xsort], ys[xsort], c=atomcolours[xsort], s=xmarker[xsort])
#     axs[2].scatter(xs[ysort], zs[ysort], c=atomcolours[ysort], s=ymarker[ysort])
#
#     circles = np.array(
#         [Circle((xval, yval), RADII[atmk]) for xval, yval, atmk in zip(xs, ys, atomkeys, strict=True)],
#     )
#     axs[3].add_collection(PatchCollection(circles[zsort], fc=atomcolours[zsort], edgecolors="none", alpha=0.25))
#     axs[3].add_collection(PatchCollection(circles[zsort], ec=atomcolours[zsort], fc="none"))
#
#     axs[3].scatter(
#         reference_lattice_spacing * np.array([0, 1, -1, 0.5, -0.5, 0.5, -0.5]),
#         reference_lattice_spacing
#         * np.array(
#             [
#                 0,
#                 0,
#                 0,
#                 np.sqrt(3) / 2,
#                 np.sqrt(3) / 2,
#                 -np.sqrt(3) / 2,
#                 -np.sqrt(3) / 2,
#             ],
#         ),
#         c="k",
#         s=200,
#     )
#     axs[3].scatter(xs[zsort], ys[zsort], c=atomcolours[zsort], edgecolors="w")
#
#     xlim: tuple[float, float] = axs[3].get_xlim()
#     ylim: tuple[float, float] = axs[3].get_ylim()
#     xmean: float = np.mean(xlim, dtype=float)
#     ymean: float = np.mean(ylim, dtype=float)
#     xlim = xlim[0] * 1.1 - 0.1 * xmean, xlim[1] * 1.1 - 0.1 * xmean
#     ylim = ylim[0] * 1.1 - 0.1 * ymean, ylim[1] * 1.1 - 0.1 * ymean
#
#     axs[3].set_xlim(xlim)
#     axs[3].set_ylim(ylim)
#
#     axs[0].set_xlim(xlim)
#     axs[0].set_ylim(ylim)
#     axs[1].set_ylim(ylim)
#     xmin, xmax = axs[1].get_xlim()
#     if xmax - xmin < 2:  # noqa: PLR2004
#         axs[1].set_xlim((xmin - 1, xmax + 1))
#     axs[2].set_xlim(xlim)
#     ymin, ymax = axs[2].get_ylim()
#     if ymax - ymin < 2:  # noqa: PLR2004
#         axs[2].set_ylim((ymin - 1, ymax + 1))
#
#     fig.canvas.blit(fig.bbox)
#     fig.canvas.flush_events()

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
        mask = np.ones(atomkeys.size, dtype=bool)
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
    file = r"C:\Users\s137316\OneDrive - TU Eindhoven\Documents\BDEAS.xyz"
    first_time_loader(file)