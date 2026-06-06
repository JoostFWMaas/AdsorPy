from __future__ import annotations

from pathlib import Path

import numpy as np
import svgwrite
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


def _polygon_to_path_d(coords: FloatArray) -> str:
    """Convert an Nx2 array of coordinates into an SVG path string.

    :param coords: Array of shape (N, 2) describing polygon vertices.
    :returns: SVG path string ("d" attribute).
    """
    cmds: list[str] = [f"M {coords[0, 0]} {coords[0, 1]}"]
    cmds.extend(f"L {x} {y}" for x, y in coords[1:])
    cmds.append("Z")
    return " ".join(cmds)


def _build_transform(
    x: float,
    y: float,
    angle: float,
    reflect_x: bool,
    reflect_y: bool,
) -> str:
    """
    Build an SVG transform string for placement of a molecule.

    The order is:
    scale (reflection) → rotate → translate

    :param x: Translation in x-direction.
    :param y: Translation in y-direction.
    :param angle: Rotation angle in degrees.
    :param reflect_x: Reflect across y-axis.
    :param reflect_y: Reflect across x-axis.
    :returns: SVG transform string.
    """
    sx: float = -1.0 if reflect_x else 1.0
    sy: float = -1.0 if reflect_y else 1.0

    parts: list[str] = []

    if reflect_x or reflect_y:
        parts.append(f"scale({sx},{sy})")

    if angle != 0.0:
        parts.append(f"rotate({angle})")

    parts.append(f"translate({x},{y})")

    return " ".join(parts)


def svgplot_covered_grid(
    surf: Surface,
    amgs: list[MoleculeGroup],
    filename: str | Path,
) -> None:
    """Generate an optimized SVG of the covered surface using one base shape per molecule group.

    Each molecule instance is placed using SVG transforms (translate, rotate, reflect),
    avoiding duplication of geometry.

    :param surf: The surface object containing grid and molecule data.
    :param amgs: List of molecule groups present on the surface.
    :param filename: Output SVG file path.
    :raises OSError: If the file cannot be written.
    """
    filename = Path(filename)

    width: float = float(surf.x_max)
    height: float = float(surf.y_max)

    dwg: svgwrite.Drawing = svgwrite.Drawing(
        str(filename),
        profile="full",
        size=(width, height),
    )
    dwg.viewbox(0, 0, width, height)

    root_group = dwg.g(transform=f"scale(1,-1) translate(0,-{height})")
    dwg.add(root_group)

    # -------------------------
    # DEFINE BASE SHAPES
    # -------------------------
    shape_registry: dict[int, str] = {}

    for mol_gr in amgs:
        coords: FloatArray = np.asarray(
            mol_gr.base_polygon.exterior.coords,
            dtype=float,
        )

        shape_id: str = f"mol_{mol_gr.group_id}"
        path_d: str = _polygon_to_path_d(coords)

        path = dwg.path(
            d=path_d,
            id=shape_id,
            fill=getattr(mol_gr, "color", "blue"),
            stroke="none",
        )

        dwg.defs.add(path)
        shape_registry[mol_gr.group_id] = shape_id

    # -------------------------
    # PLACE INSTANCES
    # -------------------------
    for mol_gr in amgs:
        group = dwg.g()

        mask = (
            surf.mol_data.stored_data["exists"]
            & (surf.mol_data.stored_data["mol_group"] == mol_gr.group_id)
        )

        for molinf in surf.mol_data.stored_data[mask]:
            group_id: int = int(mol_gr.group_id)

            x: float = float(molinf["x_coord"])
            y: float = float(molinf["y_coord"])

            angle: float = float(molinf.get("angle", 0.0))
            reflect_x: bool = bool(molinf.get("reflect_x", False))
            reflect_y: bool = bool(molinf.get("reflect_y", False))

            transform: str = _build_transform(
                x=x,
                y=y,
                angle=angle,
                reflect_x=reflect_x,
                reflect_y=reflect_y,
            )

            use = dwg.use(
                href=f"#{shape_registry[group_id]}",
                transform=transform,
            )

            group.add(use)

        root_group.add(group)

    # -------------------------
    # GRID POINTS
    # -------------------------
    grid_group = dwg.g(fill="black")

    for center in surf.grid_coordinates.T:
        x, y = float(center[0]), float(center[1])
        grid_group.add(
            dwg.circle(
                center=(x, y),
                r=float(surf.lattice_a) * 0.1,
            )
        )

    root_group.add(grid_group)

    # -------------------------
    # BORDER
    # -------------------------
    if surf.bp.hard_flag:
        root_group.add(
            dwg.rect(
                insert=(0.0, 0.0),
                size=(width, height),
                stroke="black",
                fill="none",
                stroke_width=2,
            )
        )

    dwg.save()