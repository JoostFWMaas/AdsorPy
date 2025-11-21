"""Contains all the molecules that can be used in this simulation.

Also includes molecule loader scripts, for which the molecule data is not included in this lib.
You need to supply your own .xyz files, or you can use preconfigured simple shapes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Literal, ParamSpec, TypeAlias, cast

import matplotlib.pyplot as plt
import numpy as np
import shapely
import shapely.affinity as aff
from matplotlib import axes, figure
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle
from matplotlib.widgets import Slider, TextBox
from numpy.typing import NDArray
from shapely import MultiPoint, MultiPolygon, Point, Polygon
from shapely.ops import unary_union

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

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

P = ParamSpec("P")  # Helps with static type checkers.
InDict: TypeAlias = dict[str, str | float | list[str] | Path | None]
InInDict: TypeAlias = dict[str, InDict]

# mypy: plugins = numpy.typing.mypy_plugin
# Definition of some frequently-used types. Not used by the compiler, just for the user and mypy. Hello user!
IdxArray = np.ndarray[tuple[int], np.dtype[np.int_]]  # Flat index aray of integers.
BoolArray = np.ndarray[tuple[int], np.dtype[np.bool_]]  # Flat Boolean array.
CoordPair = np.ndarray[tuple[Literal[2], Literal[1]], np.dtype[np.float64]]  # 2x1 array of coordinates
CoordsArray = np.ndarray[tuple[Literal[2], int], np.dtype[np.float64]]  # 2xN array of coords.
Bools2D = np.ndarray[tuple[int, int], np.dtype[np.bool_]]
GeoArray = np.ndarray[tuple[int], np.dtype[Polygon]]
FloatArray = NDArray[np.float64]
RotMatrix = np.ndarray[tuple[Literal[3], Literal[3]], np.dtype[np.float64]]


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
    params: list[float],
    offx: float = 0.0,
    offy: float = 0.0,
) -> Polygon:
    """Create a discorectangle using the union of two circles and a rectangle.

    The circles are automatically approximated using linear segments (error of 1% or less).

    :param params: Radius of the two circles and distance between the two halves in Angstrom.
    :param offx: X offset.
    :param offy: Y offset.
    :return: The molecule shape as a polygon.
    """
    radius: float
    distance: float
    offy *= -1.0
    offx *= -1.0
    radius, distance = params  # radius: float, distance: float
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

    return unary_union([rectangle, circles])


def circulium(
    radius: float,
    x_offset: float = 0.0,
    y_offset: float = 0.0,
    quad_segs: int = 8,
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


def dogbonium(scale: float = 1) -> Polygon:
    """Make a molecule shaped like a bone. Used as a pathological case.

    :param scale: scale of the shape.
    :return: The molecule shape as a polygon.
    """
    positions = np.ones((4, 2))
    positions *= scale
    positions[[1, 2], 0] *= -1.0
    positions[[2, 3], 1] *= -1.0
    positions[:, 1] *= 0.5
    balls: MultiPolygon = MultiPoint(positions).buffer(0.5 * scale)
    rod: Polygon = Polygon(positions)
    dogbone: Polygon = unary_union((rod, balls))

    return dogbone


def polygonium(
    verts: int = 3,
    scale: float = 1.0,
    roundedness: float = 0.0,
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

    return shapely.make_valid(molecule)


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

    molecule = MultiPolygon(atom_list)
    molecule = unary_union(molecule)
    centre = -np.mean(atompos, axis=0)
    if x_offset is not None:
        centre[0] -= x_offset
    if y_offset is not None:
        centre[1] -= y_offset

    return aff.translate(molecule, *centre)


def first_time_loader(  # noqa: PLR0915
    file_name: str | Path,
    ignore_atoms: str | list[str] | None = None,
    x_offset: float | None = None,
    y_offset: float | None = None,
    roll: float | None = None,
    pitch: float | None = None,
    yaw: float | None = None,
    z_trim: float | None = None,
    reference_lattice_spacing: float = 4.74,
) -> InDict:
    """Script to run when you first load the molecule. Shows the molecule in xy, zy, and xz perspective, and vdwaals.

    Can be used to rotate the molecule until satisfied, and will print a string that can be used for the xyz reader.

    :param file_name: File name. Include the path.
    :param ignore_atoms: Atoms to be ignored, optional. Either a string or an iterable of strings.
    :param x_offset: The x offset.
    :param y_offset: The y offset.
    :param roll: Rotation along the x-axis.
    :param pitch: Rotation along the y-axis.
    :param yaw: Rotation along the z axis.
    :param z_trim: Value below which all molecules are ignored.
    :param reference_lattice_spacing: The lattice spacing of the reference grid.
    :return: The updated dict of parameters after rotation and translation.
    """
    atomkeys, atompos = _initialise_reader(file_name, ignore_atoms, z_trim)

    with (Path(__file__).parent / "molecule_colour.json").open("r") as file:
        colourdict: dict[str, str] = dict(json.load(file))

    map_to_colour = np.vectorize(lambda name: colourdict.get(name, "#FFFFFF"))
    atomcolours: NDArray[np.str_] = map_to_colour(atomkeys)

    roll = 0.0 if roll is None else roll
    pitch = 0.0 if pitch is None else pitch
    yaw = 0.0 if yaw is None else yaw
    x_offset = 0.0 if x_offset is None else x_offset
    y_offset = 0.0 if y_offset is None else y_offset

    fig: Figure = plt.figure(figsize=(12, 10))
    axs: list[Axes] = list(np.asarray(fig.add_gridspec(3, 2).subplots()).ravel())

    for ax in axs[-2:]:
        fig.delaxes(ax)

    ax_angle_z = plt.axes(
        (0.1, 0.25, 0.65, 0.03),
        facecolor="lightgoldenrodyellow",
        transform=axs[0].transAxes,
    )
    ax_angle_y = plt.axes(
        (0.1, 0.2, 0.65, 0.03),
        facecolor="lightgoldenrodyellow",
        transform=axs[0].transAxes,
    )
    ax_angle_x = plt.axes(
        (0.1, 0.15, 0.65, 0.03),
        facecolor="lightgoldenrodyellow",
        transform=axs[0].transAxes,
    )
    trans_x = plt.axes(
        (0.1, 0.1, 0.65, 0.03),
        facecolor="lightgoldenrodyellow",
        transform=axs[0].transAxes,
    )
    trans_y = plt.axes(
        (0.1, 0.05, 0.65, 0.03),
        facecolor="lightgoldenrodyellow",
        transform=axs[0].transAxes,
    )
    ang_z_txtbx = plt.axes((0.76, 0.25, 0.1, 0.03), transform=axs[0].transAxes)
    ang_y_txtbx = plt.axes((0.76, 0.2, 0.1, 0.03), transform=axs[0].transAxes)
    ang_x_txtbx = plt.axes((0.76, 0.15, 0.1, 0.03), transform=axs[0].transAxes)
    tra_x_txtbx = plt.axes((0.76, 0.1, 0.1, 0.03), transform=axs[0].transAxes)
    tra_y_txtbx = plt.axes((0.76, 0.05, 0.1, 0.03), transform=axs[0].transAxes)

    ang_z = Slider(ax_angle_z, "roll", -180, 180, valinit=roll)
    ang_y = Slider(ax_angle_y, "pitch", -180, 180, valinit=pitch)
    ang_x = Slider(ax_angle_x, "yaw", -180, 180, valinit=yaw)
    translate_x = Slider(trans_x, "x", -5, 5, valinit=x_offset)
    translate_y = Slider(
        trans_y,
        "y",
        -5,
        5,
        valinit=y_offset if y_offset is not None else 0.0,
    )

    box_z = TextBox(ang_z_txtbx, "", initial=f"{roll}")
    box_y = TextBox(ang_y_txtbx, "", initial=f"{pitch}")
    box_x = TextBox(ang_x_txtbx, "", initial=f"{yaw}")
    box_xoff = TextBox(tra_x_txtbx, "", initial=f"{x_offset}")
    box_yoff = TextBox(tra_y_txtbx, "", initial=f"{y_offset}")

    ang_z.valtext.set_visible(False)
    ang_y.valtext.set_visible(False)
    ang_x.valtext.set_visible(False)
    translate_x.valtext.set_visible(False)
    translate_y.valtext.set_visible(False)

    def submit(text: str, slider: Slider) -> None:
        """Submit a value for one of the sliders.

        :param text: Text, must be castable to float!
        :param slider: The slider to be updated.
        :return: The new value for the slider.
        """
        slider.set_val(float(text))

    def update(val: float) -> None:
        """Update the slider value.

        :param val: The new value for the slider.
        """
        _update(
            val,
            ang_x,
            ang_y,
            ang_z,
            translate_x,
            translate_y,
            box_x,
            box_y,
            box_z,
            box_xoff,
            box_yoff,
            atompos,
            axs,
            fig,
            atomcolours,
            atomkeys,
            reference_lattice_spacing,
        )

    ident_mat: RotMatrix = cast("RotMatrix", np.identity(3, dtype=np.float64))

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

    newatompos = atompos.copy()

    for crdx, coords in enumerate(atompos):
        newatompos[crdx] = coords @ ident_mat

    newatompos -= np.mean(newatompos, axis=0)
    if x_offset is not None:
        newatompos[:, 0] -= x_offset
    if y_offset is not None:
        newatompos[:, 1] -= y_offset

    for ax in axs:
        ax.set_aspect("equal", "box")
        ax.clear()

    axs[0].set_xlabel("x-ax")
    axs[0].set_ylabel("y-ax")
    axs[1].set_xlabel("z-ax")
    axs[1].set_ylabel("y-ax")
    axs[2].set_xlabel("x-ax")
    axs[2].set_ylabel("z-ax")
    axs[3].set_xlabel("x-ax")
    axs[3].set_ylabel("y-ax")

    xs: FloatArray
    ys: FloatArray
    zs: FloatArray
    xs, ys, zs = newatompos.T

    axs[0].scatter(xs, ys, c=atomcolours)
    axs[1].scatter(zs, ys, c=atomcolours)
    axs[2].scatter(xs, zs, c=atomcolours)

    circles = [Circle((xval, yval), RADII[atmk]) for xval, yval, atmk in zip(xs, ys, atomkeys, strict=True)]
    axs[3].add_collection(PatchCollection(circles, fc=atomcolours, edgecolors="none", alpha=0.25))
    axs[3].add_collection(PatchCollection(circles, ec=atomcolours, fc="none"))
    axs[3].scatter(
        reference_lattice_spacing * np.array([0, 1, -1, 0.5, -0.5, 0.5, -0.5]),
        reference_lattice_spacing
        * np.array(
            [0, 0, 0, np.sqrt(3) / 2, np.sqrt(3) / 2, -np.sqrt(3) / 2, -np.sqrt(3) / 2],
        ),
        c="k",
        s=200,
    )
    axs[3].scatter(xs, ys, c=atomcolours, edgecolors="w")

    ang_z.on_changed(update)
    ang_y.on_changed(update)
    ang_x.on_changed(update)
    translate_x.on_changed(update)
    translate_y.on_changed(update)
    box_z.on_submit(lambda text: submit(text, ang_z))
    box_y.on_submit(lambda text: submit(text, ang_y))
    box_x.on_submit(lambda text: submit(text, ang_x))
    box_xoff.on_submit(lambda text: submit(text, translate_x))
    box_yoff.on_submit(lambda text: submit(text, translate_y))
    plt.show()

    outputparams: InDict = {
        "x_offset": translate_x.val,
        "y_offset": translate_y.val,
        "roll": ang_z.val,
        "pitch": ang_y.val,
        "yaw": ang_x.val,
        "file_name": str(file_name),
        "ignore_atoms": ignore_atoms,
        "z_trim": z_trim,
    }

    print(f"kwargs = {outputparams}")

    return outputparams


def _update(  # noqa: PLR0913
    val: float,
    ang_x: Slider,
    ang_y: Slider,
    ang_z: Slider,
    translate_x: Slider,
    translate_y: Slider,
    box_x: TextBox,
    box_y: TextBox,
    box_z: TextBox,
    box_xoff: TextBox,
    box_yoff: TextBox,
    atompos: FloatArray,
    axs: list[axes.Axes],
    fig: figure.Figure,
    atomcolours: NDArray[np.str_],
    atomkeys: NDArray[np.str_],
    reference_lattice_spacing: float,
) -> None:
    roll = ang_z.val
    pitch = ang_y.val
    yaw = ang_x.val
    x_offset = translate_x.val
    y_offset = translate_y.val

    box_z.set_val(f"{roll:.3f}")
    box_y.set_val(f"{pitch:.3f}")
    box_x.set_val(f"{yaw:.3f}")
    box_xoff.set_val(f"{x_offset:.3f}")
    box_yoff.set_val(f"{y_offset:.3f}")

    ident_mat: RotMatrix = cast("RotMatrix", np.identity(3))

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

    ident_mat = cast("RotMatrix", ident_mat @ yawmat @ pitchmat @ rollmat)

    newatompos = atompos.copy()

    for crdx, coords in enumerate(atompos):
        newatompos[crdx] = coords @ ident_mat

    newatompos -= np.mean(newatompos, axis=0)
    if x_offset is not None:
        newatompos[:, 0] -= x_offset
    if y_offset is not None:
        newatompos[:, 1] -= y_offset

    for ax in axs:
        ax.set_aspect("equal", "box")
        ax.clear()

    axs[0].set_xlabel("x-ax")
    axs[0].set_ylabel("y-ax")
    axs[1].set_xlabel("z-ax")
    axs[1].set_ylabel("y-ax")
    axs[2].set_xlabel("x-ax")
    axs[2].set_ylabel("z-ax")
    axs[3].set_xlabel("x-ax")
    axs[3].set_ylabel("y-ax")

    xs: FloatArray
    ys: FloatArray
    zs: FloatArray
    xs, ys, zs = newatompos.T

    xminmax = np.min(xs), np.max(xs)
    yminmax = np.min(ys), np.max(ys)
    zminmax = np.min(zs), np.max(zs)

    xmarker = 30 + (xs - xminmax[0]) / (xminmax[1] - xminmax[0]) * 30
    ymarker = 30 + (ys - yminmax[0]) / (yminmax[1] - yminmax[0]) * 30
    zmarker = 30 + (zs - zminmax[0]) / (zminmax[1] - zminmax[0]) * 30

    xsort = np.argsort(xs)
    ysort = np.argsort(ys)
    zsort = np.argsort(zs)

    axs[0].scatter(xs[zsort], ys[zsort], c=atomcolours[zsort], s=zmarker[zsort])
    axs[1].scatter(zs[xsort], ys[xsort], c=atomcolours[xsort], s=xmarker[xsort])
    axs[2].scatter(xs[ysort], zs[ysort], c=atomcolours[ysort], s=ymarker[ysort])

    circles = np.array(
        [Circle((xval, yval), RADII[atmk]) for xval, yval, atmk in zip(xs, ys, atomkeys, strict=True)],
    )
    axs[3].add_collection(PatchCollection(circles[zsort], fc=atomcolours[zsort], edgecolors="none", alpha=0.25))
    axs[3].add_collection(PatchCollection(circles[zsort], ec=atomcolours[zsort], fc="none"))

    axs[3].scatter(
        reference_lattice_spacing * np.array([0, 1, -1, 0.5, -0.5, 0.5, -0.5]),
        reference_lattice_spacing
        * np.array(
            [
                0,
                0,
                0,
                np.sqrt(3) / 2,
                np.sqrt(3) / 2,
                -np.sqrt(3) / 2,
                -np.sqrt(3) / 2,
            ],
        ),
        c="k",
        s=200,
    )
    axs[3].scatter(xs[zsort], ys[zsort], c=atomcolours[zsort], edgecolors="w")

    xlim: tuple[float, float] = axs[3].get_xlim()
    ylim: tuple[float, float] = axs[3].get_ylim()
    xmean: float = np.mean(xlim, dtype=float)
    ymean: float = np.mean(ylim, dtype=float)
    xlim = xlim[0] * 1.1 - 0.1 * xmean, xlim[1] * 1.1 - 0.1 * xmean
    ylim = ylim[0] * 1.1 - 0.1 * ymean, ylim[1] * 1.1 - 0.1 * ymean

    axs[3].set_xlim(xlim)
    axs[3].set_ylim(ylim)

    axs[0].set_xlim(xlim)
    axs[0].set_ylim(ylim)
    axs[1].set_ylim(ylim)
    xmin, xmax = axs[1].get_xlim()
    if xmax - xmin < 2:  # noqa: PLR2004
        axs[1].set_xlim((xmin - 1, xmax + 1))
    axs[2].set_xlim(xlim)
    ymin, ymax = axs[2].get_ylim()
    if ymax - ymin < 2:  # noqa: PLR2004
        axs[2].set_ylim((ymin - 1, ymax + 1))

    fig.canvas.blit(fig.bbox)
    fig.canvas.flush_events()


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
    :raises ValueError: 1) if the file type is not .xyz or 2) if the used settings would return an empty molecule.
    """
    file_path = Path(file_name)
    if (badtype := file_path.suffix) != ".xyz":
        errmsg = f"The file type is not .xyz but {badtype}"
        raise ValueError(errmsg)
    data = np.loadtxt(file_path, dtype=str, skiprows=2)
    atomkeys: np.ndarray[tuple[int], np.dtype[np.str_]] = cast("np.ndarray[tuple[int], np.dtype[np.str_]]", data[:, 0])
    atompos: np.ndarray[tuple[int, Literal[3]], np.dtype[np.float64]] = cast(
        "np.ndarray[tuple[int, Literal[3]], np.dtype[np.float64]]",
        data[:, 1:].astype(np.float64),
    )

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
        atomkeys = cast("np.ndarray[tuple[int], np.dtype[np.str_]]", atomkeys[mask])
        atompos = cast("np.ndarray[tuple[int, Literal[3]], np.dtype[np.float64]]", atompos[mask])

    if not atomkeys.size:
        errmsg = "The current settings result in an empty molecule."
        raise ValueError(errmsg)

    return atomkeys, atompos


if __name__ == "__main__":  # Best practice
    while (file := input("File path name, or q to quit: ")).lower() not in {"q", "quit"}:
        first_time_loader(file)
