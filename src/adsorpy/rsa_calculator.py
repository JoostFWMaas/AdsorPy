"""RSA Calculator module. Import needed for the Random Sequential Adsorption script to run.

There is no real reason for the user to need to use any of this, but the functions are accessible just in case.
"""

from typing import TYPE_CHECKING, Literal, ParamSpec, cast

import numpy as np  # For vectorised computations (performed in C).
import shapely.affinity as aff  # Install shapely via https://www.lfd.uci.edu/~gohlke/pythonlibs/#shapely
from numba import njit, prange
from numpy.typing import NDArray
from rtree.index import Index
from shapely import Polygon, STRtree
from shapely.prepared import PreparedGeometry, prep

if TYPE_CHECKING:  # When running mypy, import these classes for type checking.
    from .randomsequentialadsorption import CandidateMolecule, MoleculeGroup, Simulator

P = ParamSpec("P")  # Helps with static type checkers.

# mypy: plugins = numpy.typing.mypy_plugin
# Definition of some frequently-used types. Not used by the compiler, just for the user and mypy. Hello user!
IdxArray = np.ndarray[tuple[int], np.dtype[np.int_]]  # Flat index aray of integers.
BoolArray = np.ndarray[tuple[int], np.dtype[np.bool_]]  # Flat Boolean array.
CoordPair = np.ndarray[tuple[Literal[2], Literal[1]], np.dtype[np.float64]]  # 2x1 array of coordinates
CoordsArray = np.ndarray[tuple[Literal[2], int], np.dtype[np.float64]]  # 2xN array of coords.
CoordsArray23D = np.ndarray[tuple[Literal[2, 3], int], np.dtype[np.float64]]  # 2 or 3 x N array of coords.
Bools2D = np.ndarray[tuple[int, int], np.dtype[np.bool_]]
GeoArray = np.ndarray[tuple[int], np.dtype[Polygon]]
FloatArray = NDArray[np.float64]
DistArray = np.ndarray[tuple[int], np.dtype[np.float64]]


@njit("float64[:, :](float64[:, :], float64[:, :])", parallel=True, cache=True)  # type: ignore[misc]
def squared_cdist(coords1: CoordsArray, coords2: CoordsArray) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    """Calculate the square distance between two sets of coordinates.

    :param coords1: The first set of coordinates.
    :param coords2: The second set of coordinates.
    :return: The squared distance array.
    """
    dim1: int = coords1.shape[1]
    dim2: int = coords2.shape[1]
    distances: np.ndarray[tuple[int, int], np.dtype[np.float64]] = np.empty((dim1, dim2), dtype=np.float64)
    for ii in prange(dim1):
        for jj in prange(dim2):
            dx = coords1[0, ii] - coords2[0, jj]
            dy = coords1[1, ii] - coords2[1, jj]
            distances[ii, jj] = dx * dx + dy * dy
    return distances


def calculate_square_distance(
    candidate_coords: CoordPair,
    other_coords: CoordsArray,
    placed_index: IdxArray,
    molgridx: IdxArray,
    max_interdist: float,
    rtree: Index,
    existing: BoolArray,
) -> tuple[FloatArray, IdxArray, IdxArray]:
    """Compute the square distance for the coordinates.

    The square distance is computed between the new coordinate and the existing molecule coordinates nearby.
    The square distance can be compared to the square of the radius.
    (Computationally cheaper than taking the Euclidean distance of the entire array).

    :param candidate_coords: 2x1 array of coordinate pair.
    :param other_coords: 2xN array of coordinates of all other molecules.
    :param placed_index: N index array of molecules that have been placed.
    :param molgridx: The molecule grid index.
    :param max_interdist: The maximum distance between which molecules can still touch.
    :param rtree: The index of the RTree. If provided, uses rtree filter instead of rect filter.
    :param existing: If provided, uses existing molecule coordinates. Must be with rtree.

    :returns: Squared distance of molecule centre to other molecule centres, indices of nearby molecules, molgroup idx.
    """
    distance_boolidx_array = make_rtree_filter(
        cast("np.ndarray[tuple[Literal[2]], np.dtype[np.float64]]", candidate_coords[:, 0]),
        rtree,
        max_interdist,
        existing,
    )

    # The coordinates and molecules are filtered using the window.
    filtered_coords: CoordsArray = cast("CoordsArray", other_coords[:, distance_boolidx_array])
    neighbour_index: IdxArray = cast("IdxArray", placed_index[distance_boolidx_array])
    mol_group_index: IdxArray = cast("IdxArray", molgridx[distance_boolidx_array])

    distance_squared: DistArray
    if filtered_coords.size != 0:  # If the list of nearby coordinates is not empty, proceed.
        distance_squared = squared_cdist(candidate_coords, filtered_coords)[0]
    else:
        distance_squared = np.empty(0, dtype=np.float64)  # If there are no nearby coordinates, skip.
    return distance_squared, neighbour_index, mol_group_index


def check_outer_radius(
    distance_squared: DistArray,
    mol_gr_idx: IdxArray,
    near_index: IdxArray,
    gap_dists: DistArray,
) -> tuple[np.bool_, IdxArray, FloatArray]:
    """Check whether the outer radius is clear.

    If no molecules are present within twice the length of the longest side of the molecule,
    the molecule is approved automatically, and no shape checks need to be performed.

    :param distance_squared: 2xN array of the neighbouring molecule coordinates.
    :param mol_gr_idx: N array of the molecule group index.
    :param near_index: N array of the indices of the neighbouring molecules.
    :param gap_dists: Array of the compound radii of all molecule groups.

    :returns: Bool whether the outer radius is clear, index of values that are not clear, and array of nearby distances.
    """
    # The first step makes a boolean array, trimming values that are too far away.
    # Molecules cannot touch under any orientation if the distance between them is more than twice the minimum
    radvals: DistArray = cast("DistArray", gap_dists[mol_gr_idx])
    outer_clearflag_array: BoolArray = cast("BoolArray", distance_squared < np.square(radvals))

    neighbour_index: IdxArray = cast("IdxArray", near_index[outer_clearflag_array])
    close_distance_squared: DistArray = cast(
        "DistArray",
        distance_squared[outer_clearflag_array],
    )
    # The outer radius is empty if there are no nearby neighbours (size == 0, aka False).
    outer_radius_empty: np.bool_ = np.logical_not(close_distance_squared.size)

    return outer_radius_empty, neighbour_index, close_distance_squared


def check_hard_border(
    candidate_coords: CoordPair,
    x_max: float,
    y_max: float,
    max_radius: float,
) -> np.bool_:
    """Check whether there is guaranteed clearance between the molecule candidate and the hard border.

    :param candidate_coords: 2x1 array of coordinates of the candidate.
    :param x_max: Maximum x coordinate of the grid.
    :param y_max: Maximum y-coordinate of the grid.
    :param max_radius: Maximum radius of the molecule.

    :return: bool: is there guaranteed clearance between the molecule radius and the edge?
    """
    both_max: CoordPair = cast("CoordPair", np.array([[x_max], [y_max]], dtype=np.float64))
    both_max -= max_radius
    included: BoolArray = cast("BoolArray", max_radius < candidate_coords)
    included &= candidate_coords < both_max
    clearance: np.bool_ = np.all(included)

    return clearance


def check_hard_molecule(
    candidate_coords: CoordPair,
    x_max: float,
    y_max: float,
    fmg: "MoleculeGroup",
) -> BoolArray:
    """Check whether positioned molecules fit within the hard boundaries.

    This is done by checking whether all bounding box coordinates lie within the min/max x and y of grid.

    :param candidate_coords: 2x1 array of coordinates of the candidate.
    :param x_max: Maximum x coordinate of the grid.
    :param y_max: Maximum y-coordinate of the grid.
    :param fmg: Object of the first molecule group. The boundary parameters are needed.

    :return: Is there guaranteed clearance between the molecule polygon and the edge?
    """
    truth_table: BoolArray = cast("BoolArray", np.empty_like(fmg.bp.molecules_bounding_coords, dtype=np.bool_))
    # Check whether xmin, ymin, xmax, and ymax of the molecule touch the edge (in that order).
    truth_table[:, 0] = -fmg.bp.molecules_bounding_coords[:, 0] < candidate_coords[0]
    truth_table[:, 1] = -fmg.bp.molecules_bounding_coords[:, 1] < candidate_coords[1]
    truth_table[:, 2] = x_max > candidate_coords[0] + fmg.bp.molecules_bounding_coords[:, 2]
    truth_table[:, 3] = y_max > candidate_coords[1] + fmg.bp.molecules_bounding_coords[:, 3]
    # The index can be used if none of the sides touch the edge.
    clearance_table: BoolArray = cast("BoolArray", np.all(truth_table, axis=1))

    return clearance_table


def check_shape_overlap(
    cand: "CandidateMolecule",
    neighbour_index: IdxArray,
    simul: "Simulator",
    pmg: "MoleculeGroup",
    try_angle_first: int | None = None,
) -> tuple[bool, "CandidateMolecule"]:
    """Check whether there the new molecule overlaps with existing molecules.

    If molecules are in the intermediate range between guaranteed clearance and guaranteed overlap,
    the shape needs to be compared to other molecules.
    The first for-loop places the candidate molecule at an allowed angle in random order.
    The nested for-loop compares the placed molecule to the nearest neighbours.
    If any neighbour overlaps, the molecule rotation is rejected and a new angle is tried.
    If no neighbours overlap, the angle is accepted and the check terminates.
    If neighbours overlap for all angles, the molecule is rejected and the check ends.

    :param cand: Candidate molecule class.
    :param neighbour_index: N array of the indices of nearby neighbours.
    :param simul: The simulation class.
    :param pmg: The molecule group we are trying to place in.
    :param try_angle_first: Which angle idx value to try first. Optional.
    :returns: Bool flag that indicates whether the candidate is fully disjoint, and the candidate molecule.
    """
    no_overlap: bool = False  # Initially assume overlap is not allowed. Only changes if the contrary is proven!
    available_rotations: IdxArray = cast("IdxArray", pmg.bp.allowed_idx[pmg.bp.allowed_bools])
    positioned_molecules: GeoArray = (
        simul.mol_data.stored_mirr_data["polygon"] if simul.bp.periodic_flag else simul.mol_data.stored_data["polygon"]
    )
    pos_tree = STRtree(positioned_molecules[neighbour_index])

    # Go through this in a random order.
    random_angle_order: IdxArray = cast("IdxArray", simul.rng.permutation(available_rotations))
    if try_angle_first is not None:
        random_angle_order = cast(
            "IdxArray",
            np.delete(random_angle_order, np.nonzero(available_rotations == try_angle_first)),
        )
        random_angle_order = cast("IdxArray", np.insert(random_angle_order, 0, try_angle_first))
    ii: np.int_
    for ii in random_angle_order:
        candidate_molecule: Polygon = aff.translate(
            pmg.rotated_molecules[ii],
            *cand.coordinates[:, 0],
        )
        prepared_candidate: PreparedGeometry = prep(candidate_molecule)

        overlap_indices = pos_tree.query(candidate_molecule)  # Returns the indices of overlapping bounding boxes.

        intersect_flag: bool = False
        nn: np.int_
        for nn in neighbour_index[overlap_indices]:  # For all neighbours, check intersection.
            if prepared_candidate.intersects(positioned_molecules[nn]):
                intersect_flag = True
                break  # If there is intersection at any point, stop checking. This angle does not fit.

        if not intersect_flag:  # If intersection flag not raised for any neighbours,
            no_overlap = True  # there is no overlap and the molecule position is legal.
            cand.rot_idx = int(ii)  # Add the index of the rotated molecule
            cand.molecule = candidate_molecule
            break  # Stop checking. The random angle order guarantees a fair distribution; accept immediately.

    return no_overlap, cand


def make_rtree_filter(
    candidate_coordinates: np.ndarray[tuple[Literal[2]], np.dtype[np.float64]],
    rtree: Index,
    circumradius: float,
    existing: BoolArray,
) -> IdxArray:
    """Filter on nearby polygons using the RTree index.

    :param candidate_coordinates: Coords of the candidate molecule.
    :param rtree: Index of the nearest neighbour.
    :param circumradius: Cirucmradius of the candidate molecule.
    :param existing: BoolArray indicating if the candidate molecule exists.
    """
    minxminymaxxmaxy = (
        candidate_coordinates[0] - circumradius,
        candidate_coordinates[1] - circumradius,
        candidate_coordinates[0] + circumradius,
        candidate_coordinates[1] + circumradius,
    )

    tempresult = np.array(list(rtree.intersection(minxminymaxxmaxy)), dtype=np.int_)
    temprange = np.arange(existing.size)
    filterrange = temprange[existing]
    keyvals: dict[np.int_, np.int_] = dict(zip(filterrange, temprange, strict=False))

    return np.array([keyvals[sol] for sol in tempresult], dtype=np.int_)


def make_rectangular_filter(
    candidate_coordinates: np.ndarray[tuple[Literal[2]], np.dtype[np.float64]],
    other_coordinates: CoordsArray,
    x_offset: float,
    y_offset: float | None = None,
) -> BoolArray:
    """Make a rectangular boolean list out of x and y coordinates. Can be used to make a window.

    :param candidate_coordinates: 2 array of coordinate pair that is of interest.
    :param other_coordinates: 2xN array of coordinates that need to be filtered.
    :param x_offset: The offset over which the window is taken in x. So candidate plusminus x_off.
    :param y_offset: Same as x_offset, but in y direction. If None, then y_offset = x_offset.

    :return: Nx1 boolean array to filter the coordinates over.
    """
    candidate_x: np.float64
    candidate_y: np.float64
    candidate_x, candidate_y = candidate_coordinates

    other_x: FloatArray
    other_y: FloatArray
    other_x, other_y = other_coordinates
    y_true: float = y_offset if y_offset is not None else x_offset

    # Nota bene: X &= Y is equivalent to X = X & Y, an element-wise AND operation.
    filtered_array: BoolArray = cast("BoolArray", np.ones_like(other_x, dtype=np.bool_))
    filtered_array &= candidate_x - x_offset < other_x  # Remove everything to right.
    filtered_array &= other_x < candidate_x + x_offset  # Remove everything to the left.
    filtered_array &= candidate_y - y_true < other_y  # Remove everything above.
    filtered_array &= other_y < candidate_y + y_true  # Remove everything below.

    return filtered_array  # True for indices within window, false for indices outside.


def create_periodic_images(
    coordinates: CoordsArray23D,
    x_max: float,
    y_max: float,
    z_max: float | None = None,
) -> CoordsArray23D:
    """Create a padding of coordinates.

    Repeats the images above, below, and on the diagonals (9 tiles).

    :param coordinates: 2 by N coordinate array if 2D, otherwise 3 by N.
    :param x_max: The maximum x value of the grid.
    :param y_max: The maximum y value of the grid.
    :param z_max: The maximum z value of the grid, optional. Returns 2D grid if unfilled.

    :returns: 2 by 9*N coordinate array if 2D, otherwise 3 by 27*N array.
    """
    extended_coordinates: CoordsArray23D = coordinates.copy()
    # This creates an array of the form [[x_max, 0], [0, y_max]].
    temp_offset: np.ndarray[tuple[Literal[2, 3], Literal[2, 3]], np.dtype[np.float64]] = cast(
        "np.ndarray[tuple[Literal[2, 3], Literal[2, 3]], np.dtype[np.float64]]",
        np.diag([x_max, y_max]).reshape((2, 2)) if z_max is None else np.diag([x_max, y_max, z_max]),
    )
    offset: np.ndarray[tuple[Literal[2, 3], Literal[2, 3], Literal[1]], np.dtype[np.float64]] = cast(
        "np.ndarray[tuple[Literal[2, 3], Literal[2, 3], Literal[1]], np.dtype[np.float64]]",
        temp_offset[:, :, np.newaxis],
    )

    ii: NDArray[np.float64]
    for ii in offset:  # The first pass creates padding in x dir, the second pads in y.
        sliced_offset = cast("np.ndarray[tuple[Literal[2, 3], Literal[1]], np.dtype[np.float64]]", ii)
        extended_coordinates1: CoordsArray23D = cast("CoordsArray23D", extended_coordinates + sliced_offset)
        extended_coordinates2: CoordsArray23D = cast("CoordsArray23D", extended_coordinates - sliced_offset)
        extended_coordinates = cast(
            "CoordsArray23D",
            np.hstack((extended_coordinates, extended_coordinates1, extended_coordinates2), dtype=np.float64),
        )

    return extended_coordinates
