"""Random sequential adsorption (RSA) simulator on a grid.

In this module, all the classes are defined. This is the base of the simulation.
This module contains the lowest level methods for the library. run_simulation is preferred.
"""

from __future__ import annotations  # This allows for delayed hinting of classes.

from dataclasses import dataclass, field  # Used to define the config.
from itertools import count, cycle
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, ParamSpec, cast

import numpy as np  # For vectorised computations (performed in C).
import rsa_calculator as calc  # Library for calculation functions. Used to be static methods.
import shapely.affinity as aff
from matplotlib import pyplot as plt  # Plotting.
from matplotlib.collections import PatchCollection, PolyCollection  # To make pointers.
from matplotlib.patches import CirclePolygon, Rectangle
from numpy.typing import NDArray
from rsa_config import RsaConfig  # noqa: TC002  # Config of the simulation.
from rtree.index import Index, Property  # RTree, helps lookups!
from shapely import MultiPoint, Point, Polygon, STRtree, box, contains_xy, prepare
from shapely.prepared import PreparedGeometry  # For vectorised inclusion checks.

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

plt.rcParams.update(
    {
        "font.size": 20,
        "xtick.top": True,
        "xtick.direction": "in",
        "xtick.major.width": 3,
        "xtick.minor.width": 2,
        "ytick.right": True,
        "ytick.direction": "in",
        "ytick.major.width": 3,
        "ytick.minor.width": 2,
        "lines.linewidth": 5,
        "axes.linewidth": 3.0,
        "figure.constrained_layout.use": True,
    },
)

P = ParamSpec("P")  # Helps with static type checkers.

# mypy: plugins = numpy.typing.mypy_plugin
# Definition of some frequently-used types. Not used by the compiler, just for the user and mypy. Hello user!
IdxArray = np.ndarray[tuple[int], np.dtype[np.int_]]  # Flat index aray of integers.
BoolArray = np.ndarray[tuple[int], np.dtype[np.bool_]]  # Flat Boolean array.
CoordPair = np.ndarray[tuple[Literal[2], Literal[1]], np.dtype[np.float64]]  # 2x1 array of coordinates
CoordsArray = np.ndarray[tuple[Literal[2], int], np.dtype[np.float64]]  # 2xN array of coords.
Bools2D = np.ndarray[tuple[int, int], np.dtype[np.bool_]]
GeoArray = np.ndarray[tuple[int], np.dtype[Polygon]]
PrepGeoArray = np.ndarray[tuple[int], np.dtype[PreparedGeometry]]
FloatArray = NDArray[np.float64]
DistArray = np.ndarray[tuple[int], np.dtype[np.float64]]


@dataclass(frozen=True, slots=True)
class Config:
    """Dataclass containing the config variables read from data/config."""

    rsa_config: RsaConfig
    "The RSA config."
    sites: int | None  # Site count in the x-direction.
    "The number of sites in the x-direction."
    xsize: float | None  # The sizes are not yet properly implemented for this.
    "Under construction, do not use."
    ysize: float | None
    "Under construction, do not use."
    zsize: float | None
    "Under construction, do not use."
    max_molecule_count: int  # Maximum amount of molecules to attempt.
    "The maximum amount of molecule placements to attempt."
    lattice_a: float  # The lattice constant. Spacing between nearest points.
    "The lattice spacing in Angstrom."
    boundary_type: str  # Boundary condition: soft, hard, or periodic.
    "The boundary type. 'soft', 'hard', or 'periodic'."
    sticking_probability: float  # Sticking probability between 0 and 1. Lower probability means more rejection.
    "The sticking probability of the molecules, between 0 and 1."


class BoundaryParameters:
    """Store the boundary parameters for the surface and the molecule groups.

    Mostly empty for 'soft' boundaries, utilises different parameters for 'hard' and 'periodic' boundaries.
    """

    # Memory is reserved in slots for all variables. This increases speed and memory efficiency.
    __slots__ = (
        "allowed_bools",
        "allowed_idx",
        "biggest_radius",
        "boundary_type",
        "close_to_edge",
        "edge_flag",
        "extended_grid",
        "extended_idx",
        "extended_occupied_by",
        "extended_vacant",
        "hard_flag",
        "hard_inner",
        "mirror_counter",
        "mirrors",
        "molecules_bounding_coords",
        "molecules_flag",
        "periodic_flag",
        "soft_flag",
        "tree",
    )

    def __init__(
        self,
        boundary_type: str,
        rot_cnt: int = 0,
        dbl_max_radius: float = 0.0,
    ) -> None:
        """Initialise the boundary parameters for soft, periodic, hard.

        :param boundary_type: String denoting the boundary type.
        :param rot_cnt: Count of allowed rotations. 0 for non-molecules.
        :param dbl_max_radius: double the maximum radius of the molecule.
        """
        self.molecules_flag: Final[bool] = bool(rot_cnt)  # True if for molecules, else False.
        "True if this is the boundary parameter class for molecules, False if it used for a surface."
        self.boundary_type: Final[str] = boundary_type
        "The boundary type."
        self.soft_flag: bool = False  # Only one of these three (this line, the next, and the one after) can be True.
        "True if used for a soft boundary condition."
        self.hard_flag: bool = False
        "True if used for a hard boundary condition."
        self.periodic_flag: bool = False
        "True if used for a periodic boundary condition."
        self.soft_flag, self.hard_flag, self.periodic_flag = self.set_boundary_flags(
            self.boundary_type,
        )

        # Hard boundary parameters:
        self.hard_inner: BoolArray = np.empty(0, dtype=np.bool_)
        "All sites close to the edge of the hard boundary. These sites are True, others False."
        self.molecules_bounding_coords: np.ndarray[tuple[int, Literal[4]], np.dtype[np.float64]] = cast(
            "np.ndarray[tuple[int, Literal[4]], np.dtype[np.float64]]",
            np.empty((rot_cnt, 4), dtype=np.float64),
        )
        "Molecule bounding box coordinates: min/max x/y values."
        # Index of allowed rotations.
        self.allowed_idx: IdxArray = np.arange(rot_cnt, dtype=np.int_)
        "Index of allowed rotations. When close to a hard boundary, some rotations are no longer possible."
        # Boolean array of allowed rotations.
        self.allowed_bools: BoolArray = np.ones(rot_cnt, dtype=np.bool_)
        "Booleans belonging to the allowed rotations. When near the hard boundary, some rotations are impossible."

        # Periodic boundary parameters:
        self.extended_grid: CoordsArray = cast("CoordPair", np.empty((2, 0), dtype=np.float64))
        "The surface site coordinates of the extended (periodic) grid."
        self.extended_occupied_by: IdxArray = np.empty(0, dtype=np.int_)
        "The occupancy of the extended (periodic) grid. Filled with the indices of the molecules on the grid."
        self.extended_idx: IdxArray = np.empty(0, dtype=np.int_)
        "Index of the extended (periodic) grid."
        self.close_to_edge: BoolArray = np.empty(0, dtype=np.bool_)
        "Boolean array denoting closeness to the edge. If close to the edge, periodicity must be taken into account."
        self.extended_vacant: BoolArray = np.empty(0, dtype=np.bool_)
        "Boolean array denoting vacantness of the extended (periodic) grid. True if a site is vacant, False otherwise."
        self.edge_flag: bool = False  # Flag to indicate closeness to edge.
        "Flag indicating closeness to the edge. Reset this for every placement attempt."
        self.mirror_counter: int = 0  # Count of total amount of molecules + mirrors.
        "A counter for the molecules + mirror molecules."
        self.mirrors: IdxArray = np.empty(0, dtype=np.int_)
        "Mirror indices."
        self.biggest_radius: Final[float] = dbl_max_radius
        "The biggest radius between the two largest molecules in the simulation."
        self.tree: STRtree = STRtree([Point()])  # TODO: Figure out whether faster.
        "STRtree, currently unused."

    @staticmethod
    def set_boundary_flags(boundary_type: str) -> tuple[bool, bool, bool]:
        """Take a string of the boundary flags and turns it into any of three bools.

        Boolean checking is cheaper than string evaluation.

        :param boundary_type: The boundary type. Either soft, hard, or periodic. Throws an error otherwise.
        :return: The soft, hard, and periodic flags. One is set to True. If not, an error is thrown.
        :raises TypeError: If boundary_type is not a string.
        :raises ValueError: If boundary_type is not 'soft', 'hard', or 'periodic'.
        """
        soft_flag, hard_flag, periodic_flag = False, False, False

        if not isinstance(boundary_type, str):
            errmsg = f"The boundary_type of type {type(boundary_type)} is not a string."
            raise TypeError(errmsg)
        if boundary_type == "soft":
            soft_flag = True
        elif boundary_type == "hard":
            hard_flag = True
        elif boundary_type == "periodic":
            periodic_flag = True
        else:
            errmsg = f"The boundary_type string {boundary_type} is not 'soft', 'hard', or 'periodic'."
            raise ValueError(errmsg)

        return soft_flag, hard_flag, periodic_flag

    def generate_boundary_conditions(
        self,  # For questions about "self", see https://realpython.com/python-classes/#instance-methods-with-self
        surf: Surface,
        molgr: MoleculeGroup | None = None,
    ) -> None:
        """Generate the boundary conditions and modifies the available sites.

        If the boundary condition is soft, generate nothing.

        :param surf: The surface.
        :param molgr: The molecule group for which the boundary conditions are defined. Optional.
        """
        # This is the centre of the grid.
        centre: np.ndarray[tuple[Literal[2]], np.dtype[np.float64]] = cast(
            "np.ndarray[tuple[Literal[2]], np.dtype[np.float64]]",
            np.array([0.5 * surf.x_max, 0.5 * surf.y_max], dtype=np.float64),
        )

        if self.hard_flag and molgr is not None:
            # Check whether the molecule is guaranteed to always touch the edge.
            hard_outer: BoolArray = calc.make_rectangular_filter(
                centre,
                surf.grid_coordinates,
                *centre - molgr.min_radius,
            )
            # Check whether the molecule is guaranteed never to touch the edge.
            self.hard_inner = calc.make_rectangular_filter(
                centre,
                surf.grid_coordinates,
                *centre - molgr.max_radius,
            )
            molgr.vacant &= hard_outer  # The outer sites are no longer vacant.

        elif self.periodic_flag:
            dbl_rad: float = 2.0 * molgr.max_radius if molgr is not None else 0.0
            extended_grid: CoordsArray = calc.create_periodic_images(
                surf.grid_coordinates,
                surf.x_max,
                surf.y_max,
            )

            # A bools array to indicate which sites are guaranteed not to have mirror images.
            grid_boolsin: BoolArray = calc.make_rectangular_filter(
                centre,
                surf.grid_coordinates,
                *centre - dbl_rad,
            )

            # A bools array of sites of the extended grid. It is akin to the regular grid but with mirror images.
            extended_grid_boolsout: BoolArray = calc.make_rectangular_filter(
                centre,
                extended_grid,
                *centre + self.biggest_radius,
            )
            extended_idx: IdxArray = np.tile(surf.grid_index, reps=9).ravel()
            # The filter leaves the original indices and the indices of nearby mirror images.
            # Images have the same index number as the original site, which makes mirror lookup easy.
            temp_idx: IdxArray = extended_idx[extended_grid_boolsout].ravel()
            self.extended_idx = temp_idx
            self.extended_occupied_by = np.zeros_like(self.extended_idx, dtype=np.int_).ravel()
            # This grid has duplicate indices at mirrors.
            self.extended_grid = cast("CoordsArray", extended_grid[:, extended_grid_boolsout])
            self.tree = STRtree([Point(coord) for coord in self.extended_grid.T])
            if self.molecules_flag:
                self.extended_vacant = cast("BoolArray", np.ones_like(temp_idx, dtype=np.bool_))
                self.close_to_edge = ~grid_boolsin  # Array of sites that have mirrors.


class MoleculeGroup:
    """Create the molecule group class.

    It stores the basic shapes of rotated molecules such that they can be translated later.
    It also keeps track of several other values such as radius.
    """

    # Slots are faster by reserving memory for variables upon class creation (C optimisation).
    __slots__ = (
        "__max_rotation",
        "allowed_rotations",
        "area",
        "bp",
        "config",
        "gap_dists",
        "group_id",
        "max_radius",
        "min_radius",
        "minmax_gaps",
        "molecule",
        "molecule_counter",
        "occupied_by",
        "reflection_symmetry",
        "rot_refl_count",
        "rotated_buffer_molecules",
        "rotated_molecules",
        "rotation_count",
        "rotation_symmetry",
        "sticking_probability",
        "vacancy_count",
        "vacant",
    )

    def __init__(
        self,
        rsa_config: RsaConfig,
        molecule: Polygon,
        rotation_symmetry: int,
        reflection_symmetry: bool,
        site_count: int,
        mgc: count[int],
        rotation_count: int = 360,
        sticking_probability: float | None = None,
    ) -> None:
        """Initialise the data for molecule groups.

        Molecule groups are how data is stored per molecule footprint shape.

        :param rsa_config: RsaConfig class.
        :param molecule: (2D) polygon data representing the molecule.
        :param rotation_symmetry: Rotation symmetry of the molecule group. Set 0 for circular symmetry.
        :param reflection_symmetry: Bool denoting reflection symmetry of the molecule group.
        :param site_count: The total amount of surface sites.
        :param mgc: Molecule group counter.
        :param rotation_count: The amount of allowed rotations. If not provided, assumes 360.
        """
        self.group_id: Final[int] = next(mgc)  # Automatically assigns next mol number.
        "ID value of the molecule group. Automatically iterates when making new molecule groups."

        self.config: Final[Config] = Config(
            rsa_config=rsa_config,
            sites=rsa_config.get_value("sites", required=False),
            xsize=rsa_config.get_value("xsize", required=False),
            ysize=rsa_config.get_value("ysize", required=False),
            zsize=rsa_config.get_value("zsize", required=False),
            max_molecule_count=rsa_config.get_value("max_molecule_count"),
            lattice_a=rsa_config.get_value("lattice_a"),
            boundary_type=rsa_config.get_value("boundary_type"),
            sticking_probability=rsa_config.get_value("sticking_probability"),
        )
        "Config values."

        self.molecule: Final[Polygon] = molecule  # The molecule polygon data.
        "Molecule polygon."
        self.rotation_symmetry: Final[int] = rotation_symmetry
        "Rotation symmetry. 0 for no symmetry, 1 for circle, n (int >= 2) for n-fold."
        self.reflection_symmetry: Final[bool] = reflection_symmetry
        "Reflection symmetry. True for reflection symmetry over the y-axis, False for no reflection symmetry."
        self.area: Final[float] = self.molecule.area
        "Area of the molecule."
        centre: Point = Point((0, 0))  # Point needed to compute distance.
        self.min_radius: Final[float] = self.molecule.exterior.distance(centre)
        "Inradius of the molecule."
        self.max_radius: Final[float] = self.molecule.exterior.hausdorff_distance(centre)
        "Circumradius of the molecule."
        self.rotation_count: int = rotation_count
        "Rotation count of the molecule. How many rotations are to be considered?"
        if not rotation_symmetry:  # 0 is circle symmetry, infinite rotation:
            self.rotation_count = 1  # Only one rotation has to be attempted.
        elif not (div_modulo := np.divmod(self.rotation_count, rotation_symmetry))[1]:
            self.rotation_count = div_modulo[0]  # Divide by symmetry.
        self.__max_rotation: Final[int] = 360
        "Molecules can only rotate between 0 and 360 degrees (excluding the endpoint). Do not touch."

        # Twice as many rotations and reflections are needed if there is no reflection symmetry.
        self.rot_refl_count: Final[int] = self.rotation_count * (1 if self.reflection_symmetry else 2)
        """Rotation + reflection count. If the molecule has no reflection symmetry,
        all rotations must also be attempted while reflected. 360 rotations without reflection symmetry: 720 counts.
         """
        self.allowed_rotations: FloatArray = np.linspace(
            start=0,
            stop=self.__max_rotation,
            dtype=np.float64,
            num=self.rotation_count,
            endpoint=False,
        )
        "Rotations for the molecule."
        if not self.reflection_symmetry:
            self.allowed_rotations = np.tile(self.allowed_rotations, 2)
        self.rotated_molecules: Final[GeoArray] = cast(
            "GeoArray",
            np.empty_like(self.allowed_rotations, dtype=Polygon),
        )
        "Array of rotated molecules. Used as templates, only translation is needed to get into position."
        self.rotated_buffer_molecules: GeoArray = cast("GeoArray", np.empty_like(0, dtype=Polygon))
        """Buffer molecules are a special type of polygon used for vectorised calculations.
        They are used to determine whether surface sites are covered by these polygons."""

        self.molecule_counter: int = 0
        "Molecule counter for this molecule type."

        # Which molecule hinders what. -1 means unoccupied, -2 means unavailable.
        self.occupied_by: IdxArray = np.full(site_count, -1, dtype=np.int_)
        """Which molecule hinders what? This shows whic molecule occupies which site.
        Defaults to -1, an invalid molecule index.
        Special value -2: unoccupied by a particular molecule but still unreachable.
        """
        stickprob = sticking_probability
        self.sticking_probability: float = self.config.sticking_probability if stickprob is None else stickprob
        "Sticking probability of the molecule."

        # Initially, everything is vacant.
        self.vacant: BoolArray = np.ones(site_count, dtype=np.bool_)
        "Vacancy array for the molecule. Shows which sites are guaranteed to be unreachable (False) and which are free."

        self.vacancy_count: int = site_count
        "Counts the vacant sites. Updates per placement."

        self.bp: BoundaryParameters = BoundaryParameters(
            self.config.boundary_type,
            self.rot_refl_count,
        )
        "Boundary parameter class."
        # The list of compound max radii.
        self.gap_dists: FloatArray = np.empty(0, dtype=np.float64)
        "Distances for the molecules, measured as the sum of the circumradii.."
        # List of compound max + min radii.
        self.minmax_gaps: FloatArray = np.empty(0, dtype=np.float64)
        "Distance as the sum between the circumradius and inradius of two molecules."

    def generate_rotated_molecules(
        self,
        bopa: BoundaryParameters,
        amgs: list[MoleculeGroup],
    ) -> BoundaryParameters:
        """Generate rotated molecules and prepared buffer molecules. Generate bounding box in case of hard boundary.

        :param bopa: The boundary parameters.
        :param amgs: All molecule groups.

        :returns: The updated boundary parameters of the molecule group we are interested in.
        """
        temp_count: int = len(amgs)  # Determines how many molecule groups there are.
        #  Buffered polygons are capable of fast vectorised evaluation of inclusion of an array of points.
        self.rotated_buffer_molecules = np.empty(
            (temp_count, self.rot_refl_count),
            dtype=Polygon,
        )

        mirror_repeat: int = (not self.reflection_symmetry) + 1
        for kk in range(mirror_repeat):  # Loop twice if not symmetric, else loop once.
            for idx in np.arange(kk, self.rot_refl_count, step=mirror_repeat):
                temp_rotation: np.float64 = self.allowed_rotations[idx]
                # Define the rotated molecules. Faster than rotating them every time they are called.
                if not kk:  # On first pass, rotate the molecule.
                    self.rotated_molecules[idx] = aff.rotate(
                        self.molecule,
                        angle=temp_rotation,
                        origin=(0, 0),
                    )
                else:  # On second pass, mirror the rotated molecule.
                    self.rotated_molecules[idx] = aff.scale(
                        self.rotated_molecules[idx - 1],
                        xfact=-1,
                        origin=(0, 0),
                    )
                if bopa.hard_flag:  # The bounding box helps check boundary.
                    bopa.molecules_bounding_coords[idx] = self.rotated_molecules[idx].bounds  # xmin, ymin, xmax, ymax
                for jdx, jj in enumerate(amgs):
                    # Mols are grown by the minimum radius of all molecules, per molecule.
                    # This is used in the buffer trim step. Removes superfluous molecules.
                    self.rotated_buffer_molecules[jdx, idx] = self.rotated_molecules[idx].buffer(jj.min_radius)
        prepare(self.rotated_buffer_molecules)

        return bopa  # If the boundary condition was hard, bounding box coordinates have been added.


@dataclass(eq=False, slots=True)
class CandidateMolecule:  # Molecule is mistaken for Any by mypy.
    """Create a candidate molecule with temporary data.

    All relevant data for a candidate molecule. The molecule group index, grid index, coordinates, molecules, rotation
    index, molecule number, bool to denote closeness to the border, and existence.
    Re-used every time. All index values are illegal by default for easier debugging.
    """

    molecule_group_idx: int = -1
    "Molecule group index value. Defaults to -1, an invalid value."
    grid_index: int = -1
    "Grid index value. Defaults to -1, an invalid value."
    coordinates: CoordPair = cast("CoordPair", field(default_factory=lambda: np.empty((2, 1), dtype=np.float64)))  # noqa: RUF009
    "Coordinates of the molecule. Defaults to np.empty((2, 1), dtype=np.float64)."
    molecule: Polygon = field(default_factory=lambda: Polygon())
    "Candidate molecule. Initially empty."
    rot_idx: int = -1
    "Rotation index value. Defaults to -1, an invalid value."
    molecule_number: int = -1
    "Molecule number value. Defaults to -1, an invalid value."
    close_to_border: bool = False
    "Flag denoting closeness to the border. Defaults to False."
    exists: bool = True
    "Flag denoting existence of this molecule. Defaults to True."

    def get_canddata(
        self,
    ) -> tuple[int, bool, int, int, int, bool, float, float, Polygon]:
        """Prepare the candidate data for storage in the molecule data storage class.

        :return: Mol num, exists flag, group idx, grid idx, rot idx, close to border flag, x coord, y coord, polygon.
        """
        xcoord: float  # Has to be done like this, otherwise the type hints will cause warnings.
        ycoord: float
        xcoord, ycoord = self.coordinates.ravel()
        return (
            self.molecule_number,
            self.exists,
            self.molecule_group_idx,
            self.grid_index,
            self.rot_idx,
            self.close_to_border,
            xcoord,
            ycoord,
            self.molecule,
        )


class Simulator:
    """Perform Random Sequential Adsorption (RSA).

    The class that brings it all together.
    Places/(re)moves the molecules, updates positioning, keeps track of a lot of things.
    Allows for plotting as well, as well as gap size analysis.
    """

    __slots__ = (
        "__unclaimed",
        "bp",
        "config",
        "flux_flag",
        "minmax_rads",
        "mol_data",
        "molgrcount",
        "molgroups",
        "outer_rads",
        "rng",
        "surf",
        "total_molecule_counter",
    )

    def __init__(
        self,
        rsa_config: RsaConfig,
        include_rejected_flux: bool,
        surf: Surface,
        mol_groups: list[MoleculeGroup],
        rng: np.random.Generator,
        boundary_type: str | None = None,
    ) -> None:
        """Initialise the simulator, combine the other classes.

        :param rsa_config: RsaConfig class.
        :param include_rejected_flux: allows adsorption attempts to occupied sites.
        :param surf: Surface class.
        :param mol_groups: the molecule groups.
        :param rng: random number generator. This is needed for seeded runs (all are seeded).
        :param boundary_type: type of boundary conditions, optional. If None, defaults to config.json (periodic).
        :raises ValueError: if no molecules are provided.
        """
        self.config: Final[Config] = Config(
            rsa_config=rsa_config,
            sites=rsa_config.get_value("sites", required=False),
            xsize=rsa_config.get_value("xsize", required=False),
            ysize=rsa_config.get_value("ysize", required=False),
            zsize=rsa_config.get_value("zsize", required=False),
            max_molecule_count=rsa_config.get_value("max_molecule_count"),
            lattice_a=rsa_config.get_value("lattice_a"),
            boundary_type=rsa_config.get_value("boundary_type") if boundary_type is None else boundary_type,
            sticking_probability=rsa_config.get_value("sticking_probability"),
        )
        "Config values."

        self.rng: np.random.Generator = rng
        "Random number generator."
        self.surf: Surface = surf
        "Surface class."
        self.molgroups: list[MoleculeGroup] = mol_groups
        "List of the molecule group classes."
        self.molgrcount: int = len(self.molgroups)  # How many mol groups are there?
        "Count of the molecule group classes."
        self.bp: BoundaryParameters = BoundaryParameters(self.config.boundary_type)
        "Boundary parameter class."

        if not self.molgroups:  # If no molecules have been provided:
            errmsg = "No molecules have been provided!"
            raise ValueError(errmsg)

        self.flux_flag: bool = include_rejected_flux
        "Flag denoting whether occupied sites can be re-attempted for placement. False fails, but adds a stepcount."
        self.total_molecule_counter: np.int_ = np.int_(0)
        "Counter for all of the molecules on the surface."

        # Outer is the value of two max RADII added together. This can be of the same molecule or between molecules.
        # Minmax is the value of one max and one min radius added together, for all molecule group combinations.
        self.outer_rads: FloatArray = np.zeros(
            (self.molgrcount, self.molgrcount),
            dtype=np.float64,
        )
        "Circumradii of all of the molecules."
        self.minmax_rads: FloatArray = np.zeros(
            (self.molgrcount, self.molgrcount),
            dtype=np.float64,
        )
        "Sums of inradii and circumradii of all of the molecules."

        self._calculate_radii()

        self.mol_data: MoleculeData = MoleculeData()
        "Molecule data. This is where the values of the placed molecules and mirror molecules are stored."

        self.__unclaimed: Final[int] = -2  # Value for unreachable but __unclaimed sites.
        """Value for unreachable but unclaimed sites: sites that are not covered by a molecule but still not reachable.
         Do not change.
         """

    def _calculate_radii(self) -> None:
        """Calculate the min and max radii for the gap arrays."""
        temp_dist: FloatArray = np.array([jj.max_radius for jj in self.molgroups])
        self.outer_rads += temp_dist
        self.minmax_rads += temp_dist
        self.outer_rads += temp_dist[:, np.newaxis]
        temp_nist: FloatArray = np.array([jj.min_radius for jj in self.molgroups])
        self.minmax_rads += temp_nist[:, np.newaxis]

        for kdx, kk in enumerate(self.molgroups):
            kk.gap_dists = self.outer_rads[kdx]
            kk.minmax_gaps = self.minmax_rads[kdx]

    # @profile
    def attempt_place_molecule(
        self: Simulator,
        surf: Surface,
        pmg: MoleculeGroup,
        grid_idx: int | np.int_ | None = None,
        first_rot_idx: int | np.int_ | None = None,
    ) -> tuple[bool, int, int, int, int, list[int]]:
        """Try to place a molecule.

        It calls functions to check for adjacency and intersection.
        If the molecule does not touch any others, it is stored. Otherwise, it is discarded.

        :parameter surf: The surface.
        :parameter pmg: The primary molecule group (pmg) we want to place.
        :parameter grid_idx: A grid index value, optional. If set, try to place on this index instead of a random one.
        :parameter first_rot_idx: The first rotation to attempt, optional.
        :returns: Bool for placement of mol, the mol counter, mol group, placement index, rot index, and vacant count.
        """
        placed_flag: bool = False  # Is a molecule placed this time?
        outer_radius_empty: np.bool_  # The outer radius is checked first.
        pmg.bp.allowed_bools[:] = True  # The list of allowed bools is reset every time.
        no_overlap: bool = True  # Assume there to be no overlap between molecules.
        hard_boundary_clearance: np.bool_ = np.bool_(True)  # noqa: FBT003
        pmg.bp.edge_flag = False  # Assume not close to border.
        cand: CandidateMolecule = CandidateMolecule(molecule_group_idx=pmg.group_id)

        if pmg.sticking_probability < 1.0 and pmg.sticking_probability < self.rng.random():
            return (
                placed_flag,
                pmg.molecule_counter,
                cand.molecule_group_idx,
                cand.grid_index,
                cand.rot_idx,
                [mol.vacancy_count for mol in self.molgroups],
            )

        if self.flux_flag:  # Allows for retrying of occupied sites.
            cand.grid_index = self.rng.choice(surf.grid_index) if grid_idx is None else int(grid_idx)  # Pick a site.
            if not pmg.vacant[cand.grid_index]:  # If occupied, reject the attempt.
                return (
                    placed_flag,
                    pmg.molecule_counter,
                    cand.molecule_group_idx,
                    cand.grid_index,
                    cand.rot_idx,
                    [mol.vacancy_count for mol in self.molgroups],
                )
        elif grid_idx is None:
            free_indices: IdxArray = cast("IdxArray", surf.grid_index[pmg.vacant])
            if not free_indices.size:  # If no free sites, placement is impossible.
                return (
                    placed_flag,
                    pmg.molecule_counter,
                    cand.molecule_group_idx,
                    cand.grid_index,
                    cand.rot_idx,
                    [mol.vacancy_count for mol in self.molgroups],
                )
            cand.grid_index = self.rng.choice(free_indices)
        else:
            cand.grid_index = int(grid_idx)

        cand.coordinates[:] = surf.grid_coordinates[:, cand.grid_index, np.newaxis]

        existing: BoolArray
        placed_index: IdxArray
        mol_groups: IdxArray
        all_mol_coords: CoordsArray
        if self.bp.periodic_flag:
            existing = self.mol_data.stored_mirr_data["exists"]
            placed_index = cast("IdxArray", self.mol_data.stored_mirr_data["self_id"][existing])
            mol_groups = cast("IdxArray", self.mol_data.stored_mirr_data["mol_group"][existing])
            # Take coords of molecules + images
            all_mol_coords = cast("CoordsArray", self.mol_data.mirror_coords[:, existing])
            pmg.bp.edge_flag = cand.close_to_border = pmg.bp.close_to_edge[cand.grid_index]
            tree = self.mol_data.mirr_tree
        else:
            existing = self.mol_data.stored_data["exists"]
            placed_index = cast("IdxArray", self.mol_data.stored_data["self_id"][existing])
            all_mol_coords = cast("CoordsArray", self.mol_data.coords[:, existing])
            mol_groups = cast("IdxArray", self.mol_data.stored_data["mol_group"][existing])
            tree = self.mol_data.mol_tree

        # Function to compute the distance. Reduces the list of mols to include only nearby molecules.
        dists_squared: FloatArray  # The array of squared distances from the candidate to all other occupied sites.
        nearby_index: IdxArray  # The array of idxs of the sites close to the cand.
        mol_group_idx: IdxArray  # The array of molecule group indices of the aforementioned neighbours.
        dists_squared, nearby_index, mol_group_idx = calc.calculate_square_distance(
            cand.coordinates,
            all_mol_coords,
            placed_index,
            mol_groups,
            np.max(pmg.gap_dists),
            tree,
            existing,
        )

        neighbour_index: IdxArray  # The idxs of neighbour molecules close to the cand.
        # Checks whether there is anything near the molecule.
        # If there are no molecules within potential touching distance, no further checks need to be performed.
        # Otherwise, more checks are needed to see whether a molecule is allowed to be positioned.
        outer_radius_empty, neighbour_index, dists_squared = calc.check_outer_radius(
            dists_squared,
            mol_group_idx,
            nearby_index,
            pmg.gap_dists,
        )

        # In case of a hard boundary condition, check whether the molecule can be positioned.
        if self.bp.hard_flag:  # Check whether the site conditionally intersects.
            hard_boundary_clearance = calc.check_hard_border(
                cand.coordinates,
                surf.x_max,
                surf.y_max,
                pmg.max_radius,
            )
            if not hard_boundary_clearance:
                # Check which rotations do not intersect with the border.
                pmg.bp.allowed_bools = calc.check_hard_molecule(
                    cand.coordinates,
                    surf.x_max,
                    surf.y_max,
                    pmg,
                )
                hard_boundary_clearance = np.any(pmg.bp.allowed_bools)  # Clearance?

        # If the outer radius is empty, skip all other checks and just place.
        if outer_radius_empty and hard_boundary_clearance:
            if first_rot_idx is None:
                cand.rot_idx = self.rng.choice(
                    np.arange(pmg.rot_refl_count)[pmg.bp.allowed_bools],
                )  # Set random angle.
            else:
                cand.rot_idx = int(first_rot_idx)
            rnd_mol: Polygon = pmg.rotated_molecules[cand.rot_idx]
            cand.molecule = aff.translate(rnd_mol, *cand.coordinates.ravel())
            no_overlap = True

        if hard_boundary_clearance and not outer_radius_empty:
            # Checks whether placed molecules intersect with the new molecule.
            # This is the most expensive step of the entire script. The candidate polygon is checked against the others.
            # The first orientation without overlap is accepted. If none are found, no_overlap is False.
            no_overlap, cand = calc.check_shape_overlap(
                cand,
                neighbour_index,
                self,
                pmg,
                first_rot_idx,
            )

        if hard_boundary_clearance and no_overlap:  # If all succeeded, place.
            placed_flag = True
            self.update_placement(cand, surf, pmg, self.molgroups)
        else:
            pmg.vacant[cand.grid_index] = False  # If nothing fits, reject.
            pmg.occupied_by[cand.grid_index] = -2  # "unavailable but not occupied".
            pmg.vacancy_count -= 1

        return (
            placed_flag,
            pmg.molecule_counter,
            cand.molecule_group_idx,
            cand.grid_index,
            cand.rot_idx,
            [mol.vacancy_count for mol in self.molgroups],
        )

    def update_placement(
        self,
        cand: CandidateMolecule,
        surf: Surface,
        pmg: MoleculeGroup,
        amgs: list[MoleculeGroup],
    ) -> np.int_:
        """Update the stored molecules, coordinates, and index arrays.

        Add a new molecule.

        :param cand: The candidate molecule.
        :param surf: The surface.
        :param pmg: The molecule group that was attempted to be placed.
        :param amgs: All molecule groups.

        :return: The total placed molecule count.
        """
        cand.molecule_number = next(self.mol_data.current_mol_id)
        basic_data = cand.get_canddata()
        self.mol_data.add_entry(*basic_data)  # TODO: Make this a lot less ugly.
        cand.molecule_number = self.mol_data.last_accessed_idx

        for grp in amgs:
            grp.vacant[cand.grid_index] = False  # The chosen site is occupied.
            grp.occupied_by[cand.grid_index] = self.mol_data.last_accessed_idx

        self.total_molecule_counter += 1  # Increment total molecule counter.
        pmg.molecule_counter += 1  # Increment molecule counter for placed group.

        pmg.bp.mirrors = cast(
            "IdxArray",
            np.flatnonzero(cand.grid_index == pmg.bp.extended_idx)
            if pmg.bp.edge_flag
            else np.array([cand.grid_index], dtype=np.int_),
        )

        if pmg.bp.periodic_flag:
            pmg.bp.mirror_counter += pmg.bp.mirrors.size

            for mdx, mirror in enumerate(pmg.bp.mirrors):
                mirr_coords: CoordsArray = cast("CoordsArray", surf.bp.extended_grid[:, mirror])
                mirror_molecule: Polygon = aff.translate(
                    pmg.rotated_molecules[cand.rot_idx],
                    *mirr_coords.ravel(),
                )
                new_data = (
                    mdx,
                    *basic_data[:3],
                    mirror,
                    basic_data[4],
                    *mirr_coords.ravel(),
                    mirror_molecule,
                )
                self.mol_data.add_mirror(*new_data)

        for mol_gr in amgs:  # Clear perimeter.
            self.trim_buffer(cand.molecule_number, pmg, mol_gr, surf)
            mol_gr.vacancy_count = int(np.count_nonzero(mol_gr.vacant))

        return self.total_molecule_counter

    # @profile
    def trim_buffer(
        self,
        occupier_idx: int,
        pmg: MoleculeGroup,
        mol_group: MoleculeGroup,
        surf: Surface,
    ) -> None:
        """Trim the available sites by removing inaccessible sites.

        Take a buffered (isodistance) area around the positioned molecule of the minimum radius of the other molecule,
        then removes all coordinates within the buffer area from the list of available positions for the other.
        Nothing can be placed closer to existing molecules than this minimum radius.

        :param occupier_idx: Index of the molecule that covers the sites.
        :param pmg: Primary molecule group, the group that has been positioned.
        :param mol_group: One of the molecule groups. The group whose data is updated.
        :param surf: The Surface.

        :return: The molecule group whose availability has been updated.
        """
        minmax_gap: float = pmg.minmax_gaps[mol_group.group_id]
        boundparams = mol_group.bp
        # This adds a radius to the molecule of the min_radius and prepares it for faster evaluation.
        buffered_shape: Polygon = pmg.rotated_buffer_molecules[
            mol_group.group_id,
            self.mol_data.stored_data["rot_idx"][self.mol_data.last_accessed_idx],
        ]

        mirr_coors: CoordPair
        for ii in pmg.bp.mirrors:
            if boundparams.periodic_flag:
                mirr_coors = cast("CoordPair", boundparams.extended_grid[:, ii, np.newaxis])
            else:
                mirr_coors = cast("CoordPair", surf.grid_coordinates[:, ii, np.newaxis])
            nearby_bool_array: BoolArray = calc.make_rectangular_filter(
                mirr_coors[:, 0],
                surf.grid_coordinates,
                minmax_gap,
            )
            nearby_bool_array &= (mol_group.occupied_by == -1) | (mol_group.occupied_by == self.__unclaimed)

            # Take only the nearby free sites.
            nearby_index: IdxArray = cast("IdxArray", surf.grid_index[nearby_bool_array])
            centered_coordinates: CoordsArray = cast("CoordsArray", surf.grid_coordinates[:, nearby_index] - mirr_coors)
            nearby_bool_array[nearby_index] &= contains_xy(buffered_shape, *centered_coordinates)
            mol_group.vacant[nearby_bool_array] = False
            mol_group.occupied_by[nearby_bool_array] = occupier_idx

    # @profile
    def plot_covered_grid(
        self,
        surf: Surface,
        amgs: list[MoleculeGroup],
        save_flag: bool = False,
        plt_flag: bool = False,
        timestr: str = "",
        results_folder: str | Path = "",
    ) -> None:
        """Plot the molecules with the grid and save it as a figure.

        :param surf: The surface.
        :param amgs: All molecule groups.
        :param save_flag: True: save the figure.
        :param plt_flag: True: plot the figure.
        :param timestr: The timestring, can be used for saving the name.
        :param results_folder: The folder in which the results will be saved.
        """
        fig: Figure  # Type hinting this makes it much easier to tab-complete commands.
        ax: Axes
        fig, ax = plt.subplots(dpi=1200)
        ax.set_aspect("equal", "box")
        coords = surf.grid_coordinates
        xmax = np.max(coords[0])
        ymax = np.max(coords[1])
        ax.set_xlim(left=-0.0 * xmax, right=1.0 * xmax)
        ax.set_ylim(bottom=-0.0 * ymax, top=1.0 * ymax)
        prop_cycle = plt.rcParams["axes.prop_cycle"]
        ax.set_ylabel(r"Length ($\mathrm{\AA}$)")
        ax.set_xlabel(r"Length ($\mathrm{\AA}$)")
        colors = cycle(prop_cycle.by_key()["color"])
        ax.set_axis_off()

        if True:
            for mol_gr in amgs:
                molgr_exists_idx = self.mol_data.stored_data["exists"] & (
                    self.mol_data.stored_data["mol_group"] == mol_gr.group_id
                )

                verts: list[FloatArray] = []
                rots: list[CoordsArray] = [np.array(rot_mol.exterior.coords) for rot_mol in mol_gr.rotated_molecules]

                # molinf: NDArray[np.void]
                for molinf in self.mol_data.stored_data[molgr_exists_idx]:
                    idx: int = molinf["rot_idx"]
                    new_poly: FloatArray = rots[idx] + [
                        *molinf[["x_coord", "y_coord"]],
                    ]  # + coord
                    verts.append(new_poly)

                coll = PolyCollection(
                    verts,
                    linewidths=0.5,
                    fc=next(colors),
                    ec="none",
                    joinstyle="round",
                )
                ax.add_collection(coll)

            gridpoints = [
                CirclePolygon(xy=cast("tuple[float, float]", center), radius=surf.lattice_a * 0.1)
                for center in surf.grid_coordinates.T
            ]
            ax.add_collection(PatchCollection(gridpoints, fc="k", zorder=1))

        if surf.bp.hard_flag:  # Draw the hard border.
            rectangle_patch = Rectangle(
                (0, 0),
                surf.x_max,
                surf.y_max,
                ec="k",
                fill=False,
                linewidth=2,
            )
            ax.add_patch(rectangle_patch)

        fig.set_figwidth(5)
        fig.set_size_inches(9, 16)

        if save_flag:
            results_path = Path(results_folder) / f"{timestr}_covered_surface"
            fig.savefig(f"{results_path}.png", transparent=True)
            fig.savefig(f"{results_path}.pdf", transparent=True)

        if plt_flag:
            plt.show()

        plt.close(fig)

    def attempt_cascading_placement(
        self,
        surf: Surface,
        *molgrs: MoleculeGroup,
        site_idx: int | None = None,
    ) -> tuple[bool, int, int, int | None, int, list[int]]:
        """Try to place molecules in the order of the arguments on the same random site. Stops when one fits.

        :param surf: The surface.
        :param molgrs: The molecule groups. Placement occurs in the order of this argument.
        :param site_idx: Site to attempt placement on. If not given, pick random.

        :return: The output of Simulator.attempt_place_molecule().
        """
        # Start with a bogus output.
        output: tuple[bool, int, int, int | None, int, list[int]] = (False, -1, -1, site_idx, -1, [])
        for mols in molgrs:  # For all molecule groups in the list, try to place.
            output = self.attempt_place_molecule(
                surf,
                mols,
                grid_idx=output[3],
            )  # Try to place a molecule.
            if output[0]:
                break  # Stop when the first one has been placed.

        return output

    def attempt_random_placement(
        self,
        surf: Surface,
        *molgrs: MoleculeGroup,
        weights: FloatArray | None = None,
    ) -> tuple[bool, int, int, int, int, list[int]]:
        """Pick from a list of molecule groups and places a random one.

        :param surf: The surface.
        :param molgrs: The molecule groups.
        :param weights: The weights of the molecule groups, optional. If None, assume equal weights.

        :return: The output of Simulator.attempt_place_molecule().
        """
        distribution = None if weights is None or len(weights) != len(molgrs) else weights / np.sum(weights)
        return self.attempt_place_molecule(
            surf,
            self.rng.choice(np.asarray(molgrs), p=distribution),
        )

    def analyse_gap_size(self, surf: Surface, keepzero: bool = False) -> FloatArray:
        """Analyse the gap distance based on the distance from surface sites to the nearest molecule.

        :param surf: The surface.
        :param keepzero: Flag denoting whether to keep zero distances (inside of polygon).

        :return: The gap size distribution.
        """
        temp_molecules: GeoArray = cast(
            "GeoArray",
            self.mol_data.stored_mirr_data["polygon"][self.mol_data.stored_mirr_data["exists"]]
            if self.bp.periodic_flag
            else self.mol_data.stored_data["polygon"][self.mol_data.stored_data["exists"]],
        )

        if self.bp.hard_flag:  # This step adds the border such that it counts as a gap wall.
            inborder: Polygon = box(0, 0, surf.x_max, surf.y_max)
            border: Polygon = box(
                -0.1,
                -0.1,
                surf.x_max + 0.1,
                surf.y_max + 0.1,
            )
            border = border.difference(inborder)

            temp_molecules = np.append(temp_molecules, border)

        mol_tree = STRtree(temp_molecules)

        distance_to_grid: DistArray = np.empty(surf.grid_coordinates.shape[1], dtype=np.float64)
        grid_points = MultiPoint(surf.grid_coordinates.T)

        for grd_idx, grd_pnt in enumerate(grid_points.geoms):  # Queries tree. [1] is the distance.
            distance_to_grid[grd_idx] = mol_tree.query_nearest(grd_pnt, return_distance=True, all_matches=False)[1][0]

        # Removes all gaps of 0. distance (in a molecule). Basically removes all inaccessible/occupied sites.
        if not keepzero:
            distance_to_grid = cast("DistArray", distance_to_grid[np.nonzero(distance_to_grid)])

        return distance_to_grid


class Surface:
    """Store coordinates and occupation data."""

    __slots__ = (
        "all_site_count",
        "area",
        "bp",
        "config",
        "grid_coordinates",
        "grid_index",
        "lattice_a",
        "lattice_type",
        "simple_shape_flag",
        "sites",
        "sticking_probability",
        "tree",
        "x_max",
        "xsize",
        "y_max",
        "ysize",
        "zsize",
    )

    def __init__(
        self,
        rsa_config: RsaConfig,
        lattice_type: str = "triangular",
        site_count: int | None = None,
        lattice_a: float | None = None,
        boundary_type: str | None = None,
        sticking_probability: float | None = None,
    ) -> None:
        """Initialise the surface.

        :param rsa_config: The input parameters defined in the config.
        :param lattice_type: The type of lattice to use. Can be triangular, square, or honeycomb.
        :param site_count: The number of sites, optional. If None, defaults to the default in config.json.
        :param lattice_a: The lattice spacing, optional. If None, defaults to the default in config.json.
        :param boundary_type: The boundary type, optional. If None, defaults to the default in config.json (periodic).
        :param sticking_probability: Sticking probability. Default is 1.0 from config.
        :raise ValueError: If only x or y is provided for a custom surface. Currently unusable.
        """
        self.config: Final[Config] = Config(
            rsa_config=rsa_config,
            sites=rsa_config.get_value("sites", required=False),
            xsize=rsa_config.get_value("xsize", required=False),
            ysize=rsa_config.get_value("ysize", required=False),
            zsize=rsa_config.get_value("zsize", required=False),
            max_molecule_count=rsa_config.get_value("max_molecule_count"),
            lattice_a=rsa_config.get_value("lattice_a"),
            boundary_type=rsa_config.get_value("boundary_type") if boundary_type is None else boundary_type,
            sticking_probability=rsa_config.get_value("sticking_probability"),
        )
        "Config values."

        # Constants:
        self.lattice_type: Final[str] = lattice_type
        "Lattice type of the surface. Can be 'triangular'/'hexagonal', 'square' or 'honeycomb'."

        self.sites: int = site_count if site_count is not None else 0
        "Site count of the surface in the x-direction."
        if not self.sites:
            self.sites = self.config.sites if self.config.sites is not None else 0
        self.simple_shape_flag: Final[bool] = self.config.sites is not None
        "Currently unused. Leave True for now."
        self.xsize: float | None = self.config.xsize
        "Under construction, currently unused."
        self.ysize: float | None = self.config.ysize
        "Under construction, currently unused."
        self.zsize: float | None = self.config.zsize
        "Under construction, currently unused."

        if not self.simple_shape_flag and (self.xsize is None or self.ysize is None):
            errmsg = "When setting custom sizes, set both x and y."
            raise ValueError(errmsg)

        self.all_site_count: int = self._estimate_site_count()
        "Total site count for all molecules."

        self.lattice_a: Final[float] = self.config.lattice_a if lattice_a is None else lattice_a
        "Lattice spacing of the surface in Angstrom."
        stickprob = sticking_probability
        self.sticking_probability: Final[float] = self.config.sticking_probability if stickprob is None else stickprob
        "Sticking probability of the molecules."
        self.x_max = 0.0  # Proper value added when grid is generated.
        "Maximum x value of the surface. Adjusted properly for periodic surfaces."
        self.y_max = 0.0  # Idem ditto.
        "Maximum y value of the surface. Adjusted properly for periodic surfaces."
        self.area = 0.0  # Idem ditto.
        "Area of the surface. Adjusted properly for periodic surfaces."

        self.grid_index: IdxArray = np.arange(self.all_site_count, dtype=np.int_)
        "Grid index array."

        # Instantiated with the right size/shape:
        self.grid_coordinates: CoordsArray = cast("CoordsArray", np.empty((2, self.all_site_count), dtype=np.float64))
        "Array of the grid coordinates."

        # Class for the boundary conditions:
        self.bp = BoundaryParameters(self.config.boundary_type)
        "Boundary parameters of the surface."

        self.tree: STRtree = STRtree([Point([0, 0])])  # TODO: remove?
        "STRtree, currently unused."

    def _estimate_site_count(self) -> int:
        """Estimates the total site count.

        :returns: Total site count.
        """
        temp_all_sites = self.sites * self.sites
        if self.lattice_type in {"triangular", "hexagonal"}:
            temp_all_sites *= 2
        elif self.lattice_type == "honeycomb":
            temp_all_sites *= 4
        return temp_all_sites

    def generate_grid(self, rng: np.random.Generator | None = None) -> None:
        """Create a hexagonal grid as a 2D numpy array with x the first index and y on the second.

        :param rng: The random generator. Used when generating an amorphous surface with the Delone lib.
        :raises ValueError: If the lattice string is not supported (name and type are printed).
        :raises TypeError: If the lattice type is not supported (name and type are printed).
        """
        sqrt3: float = np.sqrt(3.0)

        x1: DistArray = np.arange(self.sites, dtype=np.float64)
        x1 *= self.lattice_a  # Scale the range by the lattice constant.
        y1: DistArray = cast("DistArray", x1 * sqrt3)  # Scale the y grid.

        if self.lattice_type in {"triangular", "hexagonal", "honeycomb"}:
            x2: DistArray = cast("DistArray", x1 + 0.5 * self.lattice_a)  # Create an off-set grid.
            y2: DistArray = cast("DistArray", x2 * sqrt3)

            x_all: DistArray = cast(
                "DistArray",
                np.hstack(
                    (np.repeat(x1, self.sites), np.repeat(x2, self.sites)),
                ),
            )
            y_all: DistArray = cast(
                "DistArray",
                np.hstack(
                    (np.tile(y1, self.sites), np.tile(y2, self.sites)),
                ),
            )

            if self.lattice_type == "honeycomb":
                x_shift: DistArray = cast("DistArray", x_all + 0.5 * self.lattice_a)
                x_all = cast("DistArray", np.hstack((x_all, x_shift)))

                y_shift0: DistArray = cast("DistArray", y_all + 0.5 * self.lattice_a / sqrt3)
                y_all = cast("DistArray", np.hstack((y_shift0, y_all)))

                x_all *= sqrt3
                y_all *= sqrt3

        elif self.lattice_type == "square":
            x_all = cast("DistArray", np.repeat(x1, self.sites))
            y_all = cast("DistArray", np.tile(x1, self.sites))

        else:
            errmsg: str = f"Unsupported lattice: {self.lattice_type} of type {type(self.lattice_type)}."
            raise ValueError(errmsg) if isinstance(self.lattice_type, str) else TypeError(errmsg)

        self.grid_coordinates = cast("CoordsArray", np.vstack((x_all, y_all)))  # (2, 2N^2) Make a coordinate array.

        self._set_xy_max(x_all, y_all, sqrt3)

        self.area = self.x_max * self.y_max

        self.grid_index = np.arange(
            self.all_site_count,
            dtype=np.int_,
        )  # Index for gridpoints.
        self.tree = STRtree(
            [Point(coord) for coord in self.grid_coordinates.T],
        )  # Never used?

    def _set_xy_max(self, x_all: DistArray, y_all: DistArray, sqrt3: float) -> None:
        """Set the x_max and y_max based on boundary condition and surface type.

        :param x_all: All x coordinates.
        :param y_all: All y coordinates.
        """
        self.x_max += cast("float", np.max(x_all)) if self.x_max == 0.0 else 0.0
        self.y_max += cast("float", np.max(y_all)) if self.y_max == 0.0 else 0.0
        if self.bp.periodic_flag:  # A periodic grid needs to be extended a little bit.
            if self.lattice_type in {"triangular", "hexagonal"}:
                self.y_max += 0.5 * self.lattice_a * sqrt3  # Half a unit is added in the max x and y size.
                self.x_max += 0.5 * self.lattice_a  # This is done to make periodic matching easier.
            elif self.lattice_type == "honeycomb":
                self.y_max += self.lattice_a
            elif self.lattice_type == "square":
                self.y_max += self.lattice_a
                self.x_max += self.lattice_a  # This is done to make periodic matching easier.

    def generate_custom_surface(
        self,
        site_x_coords: DistArray,
        site_y_coords: DistArray,
        bounding_x_coord: float,
        bounding_y_coord: float,
    ) -> None:
        """Generate a custom surface from user input.

         Ensure that the origin is (0,0) and that all values are positive. Ensure that the arrays are equal in length.

        :param site_x_coords: The x coordinates of the surface sites.
        :param site_y_coords: The y coordinates of the surface sites.
        :param bounding_x_coord: The x coordinate of the bounding box.
        :param bounding_y_coord: The y coordinate of the bounding box.
        :raise ValueError: If the x or y (bounding box) coordinates are not positive or unequal in length.
        """
        errval: str = ""  # Start with an empty error, add errors to new lines if there are multiple.
        if np.any(np.less(site_x_coords, 0.0)):
            errval += "Site x coordinates must be positive.\n"
        if np.any(np.less(site_y_coords, 0.0)):
            errval += "Site y coordinates must be positive.\n"
        if bounding_x_coord < 0.0:
            errval += "Bounding x coordinate must be positive.\n"
        if bounding_y_coord < 0.0:
            errval += "Bounding y coordinate must be positive.\n"
        if len(site_x_coords) != len(site_y_coords):
            errval += "Length of site_x_coords and site_y_coords do not match.\n"

        if errval != "":  # If there are errors, raise.
            raise ValueError(errval)

        self.x_max = bounding_x_coord
        self.y_max = bounding_y_coord
        self.area = self.x_max * self.y_max
        self.grid_coordinates = cast("CoordsArray", np.vstack((site_x_coords, site_y_coords)))
        self.all_site_count = self.grid_coordinates.shape[1]

        self.grid_index = np.arange(
            self.all_site_count,
            dtype=np.int_,
        )  # Index for gridpoints.
        self.tree = STRtree(
            [Point(coord) for coord in self.grid_coordinates.T],
        )  # TODO: Never used?

    def plot_surface_sites(self, timestr: str, directory: str | Path = "") -> None:
        """Plot the surface sites (for verification/validation).

        :param timestr: The timestring, can be used for saving.
        :param directory: The directory in which results are stored.
        """
        fig: Figure
        ax: Axes
        fig, ax = plt.subplots()
        plt.figure(dpi=1200)
        ax.set_aspect("equal", "box")
        gridpoints = [
            CirclePolygon(xy=cast("tuple[float, float]", centre), radius=self.lattice_a * 0.1)
            for centre in self.grid_coordinates.T
        ]
        ax.add_collection(PatchCollection(gridpoints, fc="k"))
        ax.set_xlim(left=0.0, right=self.x_max)
        ax.set_ylim(bottom=0.0, top=self.y_max)

        if self.bp.hard_flag:  # Add the hard boundary.
            ax.plot(
                *[[0, 0, self.x_max, self.x_max, 0], [0, self.y_max, self.y_max, 0, 0]],
            )
        surfacename = Path(directory) / (timestr + "_surface")
        fig.savefig(f"{surfacename}.png", transparent=True)
        fig.savefig(f"{surfacename}.pdf", transparent=True)
        plt.show()
        plt.close(fig)


@dataclass(slots=True)
class MoleculeData:
    """Stores the data associated with all the molecules on the surface, as well as periodic molecules."""

    max_array_length: int = 100
    "Maximum length of all arrays. Extends by this amount when the maximum is reached."
    mol_counter: count[int] = field(init=False)
    "Molecule counter."
    current_mol_id: count[int] = field(init=False)
    "Current molecule index."
    mirr_counter: count[int] = field(init=False)
    "Mirror counter. Takes steps of 4."
    current_mirror_num: int = -1
    "Current mirror index number."
    last_accessed_idx: int = -1
    "Last accessed molecule index."
    __header_names = (
        "self_id",
        "exists",
        "mol_group",
        "grid_idx",
        "rot_idx",
        "has_periodic_images",
        "x_coord",
        "y_coord",
        "polygon",
    )
    "Names of the headers."
    __column_datatypes = (int, bool, int, int, int, bool, float, float, Polygon)
    "Datatypes of the columns of the molecule struct array."
    __heads_dtypes: list[tuple[str, type]] = field(init=False)
    "Data types + names for the molecule struct array."
    __fill_vals = (-1, False, -1, -1, -1, False, 0.0, 0.0, Polygon())
    "Fill value for the new molecule. Defaults to strictly invalid values."
    stored_data: np.ndarray[tuple[int], np.dtype[np.void]] = field(init=False)
    "Molecule struct array. Here, all data for the molecules is stored."
    mol_tree: Index = field(default_factory=lambda: Index(properties=Property(dimension=2)))
    "Molecules RTree."
    __mirr_names = (
        "orig_id",
        "exists",
        "self_id",
        "mol_group",
        "grid_idx",
        "rot_idx",
        "x_coord",
        "y_coord",
        "polygon",
    )
    "Names of the columns in the mirror molecule struct array."
    __mirr_datatypes = (int, bool, int, int, int, int, float, float, Polygon)
    "Datatypes of the mirror molecule struct array."
    __mirr_heads_dtypes: list[tuple[str, type]] = field(init=False)
    "Data types + names for the mirror molecules struct array."
    __mirr_fill_vals = (-1, False, -1, -1, -1, -1, 0.0, 0.0, Polygon())
    "Fill values for the new mirror molecule. Defaults to strictly invalid values."
    stored_mirr_data: np.ndarray[tuple[int], np.dtype[np.void]] = field(init=False)
    "Mirror molecule struct array. Here, all data for the mirror molecules is stored."
    mirr_tree: Index = field(default_factory=lambda: Index(properties=Property(dimension=2)))
    "Mirror molecules RTree."
    __otomir_names = ("exists", "mirr_ids")
    "Names for the origin to mirror array."
    __otomir_types = (bool, object)
    "Types of the origin to mirror array."
    __otomir_heads_dtypes: list[tuple[str, type]] = field(init=False)
    "Data types for the original molecule to mirror (otomir) struct array."
    __otomir_fill_vals: Final[tuple[bool, list[int]]] = False, []
    "fill values for the origin to mirror array. Defaults to strictly invalid values."
    orig_to_mirrors: np.ndarray[tuple[int], np.dtype[np.void]] = field(init=False)
    "Origin to mirror struct array. Stores which mirror indices are associated with which original molecule."

    coords: CoordsArray = field(init=False)
    "Coordinates of the molecules."
    mirror_coords: CoordsArray = field(init=False)
    "Mirror coordinates of the molecules."

    def __post_init__(self) -> None:
        """Initialise the counters and the arrays after sizes are known."""
        self.mol_counter = count()
        self.current_mol_id = count()
        self.mirr_counter = count(0, 4)  # Take steps of 4.

        self.__heads_dtypes = list(
            zip(self.__header_names, self.__column_datatypes, strict=False),
        )
        self.__mirr_heads_dtypes = list(
            zip(self.__mirr_names, self.__mirr_datatypes, strict=False),
        )
        self.__otomir_heads_dtypes = list(
            zip(self.__otomir_names, self.__otomir_types, strict=False),
        )

        self.stored_data = self.make_struct_array(
            self.max_array_length,
            self.__heads_dtypes,
            self.__fill_vals,
        )
        self.stored_mirr_data = self.make_struct_array(
            self.max_array_length,
            self.__mirr_heads_dtypes,
            self.__mirr_fill_vals,
        )
        self.orig_to_mirrors = self.make_struct_array(
            self.max_array_length,
            self.__otomir_heads_dtypes,
            self.__otomir_fill_vals,
        )

        self.coords = cast("CoordPair", np.empty((2, self.max_array_length), dtype=np.float64))
        self.mirror_coords = cast("CoordPair", np.empty((2, self.max_array_length), dtype=np.float64))

    @staticmethod
    def make_struct_array(
        size: int,
        nameddtypes: list[tuple[str, type]],
        fillvals: tuple[bool | int | float | Polygon | list[int], ...],
    ) -> np.ndarray[tuple[int], np.dtype[np.void]]:
        """Make a structured array for the molecule storage.

        :param size: Size of the new array.
        :param nameddtypes: The header names and column dtypes.
        :param fillvals: The values with which the columns will be filled.
        :return: The new structured array.
        """
        struct_array: np.ndarray[tuple[int], np.dtype[np.void]] = np.empty(shape=size, dtype=nameddtypes)
        if isinstance(fillvals[-1], list):
            struct_array[:]["exists"] = fillvals[0]
            for num, _ in enumerate(struct_array):
                struct_array["mirr_ids"][num] = []
        else:
            struct_array[:] = fillvals

        return struct_array

    def add_entry(
        self,
        _: int,
        exists: bool,
        mol_group: int,
        grid_idx: int,
        rot_idx: int,
        has_periodic_images: bool,
        x_coord: float,
        y_coord: float,
        polygon: Polygon,
    ) -> None:
        """Add a new entry to the stored data. All fields are mandatory.

        :param _: Currently unused. The molecule index. Generated automatically though.  # TODO: Not used?
        :param exists: Flag to determine whether the molecule exists on the surface.
        :param mol_group: Molecule group index. Set to -1 if invalid.
        :param grid_idx: Grid site index. Set to -1 if invalid.
        :param rot_idx: Rotation index.
        :param has_periodic_images: Flag to determine whether the molecule has periodic images.
        :param x_coord: X-coordinate of the molecule centre.
        :param y_coord: Y-coordinate of the molecule centre.
        :param polygon: The molecule polygon.
        """
        self_id: int = next(self.mol_counter)
        if self_id == self.stored_data.size:  # This block extends the arrays.
            new_length: int = self.max_array_length + self.stored_data.size
            new_stored_data = self.make_struct_array(
                new_length,
                self.__heads_dtypes,
                self.__fill_vals,
            )
            new_stored_data[: self.stored_data.size] = self.stored_data
            self.stored_data = new_stored_data

            old_coords: CoordsArray = self.coords.copy()
            self.coords = cast("CoordPair", np.empty((2, new_length)))
            self.coords[:, : old_coords.shape[1]] = old_coords

            new_stored_data2 = self.make_struct_array(
                new_length,
                self.__otomir_heads_dtypes,
                self.__otomir_fill_vals,
            )
            new_stored_data2[: self.orig_to_mirrors.size] = self.orig_to_mirrors
            self.orig_to_mirrors = new_stored_data2

        temp_data = (
            self_id,
            exists,
            mol_group,
            grid_idx,
            rot_idx,
            has_periodic_images,
            x_coord,
            y_coord,
            polygon,
        )
        self.stored_data[self_id] = temp_data
        self.coords[:, self_id] = x_coord, y_coord
        self.last_accessed_idx = self_id

        self.mol_tree.insert(self_id, polygon.bounds)

    def add_mirror(
        self,
        mirr_num: int,
        orig_id: int,
        exists: bool,
        mol_group: int,
        grid_idx: int,
        rot_idx: int,
        x_coord: float,
        y_coord: float,
        polygon: Polygon,
    ) -> None:
        """Add a mirror molecule to the stored molecule data.

        :param mirr_num: The mirror number for this idx. Number between 0 and 3 (inclusive).
        :param orig_id: The ID of the molecule for which the mirror images are added.
        :param exists:  Bool, denotes whether the molecule exists.
        :param mol_group:  The molecule group ID.
        :param grid_idx: The grid index.
        :param rot_idx: The rotation index.
        :param x_coord: The x-coordinate.
        :param y_coord: The y-coordinate.
        :param polygon: The molecule polygon at the mirrored location.
        """
        if not mirr_num:  # If is the first mirror for this molecule, reserve 4 spots.
            self.current_mirror_num = next(self.mirr_counter)  # Steps of 4.

        current_mirr_idx: int = mirr_num + self.current_mirror_num
        self.orig_to_mirrors["mirr_ids"][orig_id] += (current_mirr_idx,)  # Append!
        self.orig_to_mirrors["exists"][orig_id] = exists

        if current_mirr_idx == self.stored_mirr_data.size:
            new_length: int = self.max_array_length + self.stored_mirr_data.size
            new_stored_data = self.make_struct_array(
                new_length,
                self.__mirr_heads_dtypes,
                self.__mirr_fill_vals,
            )
            new_stored_data[: self.stored_mirr_data.size] = self.stored_mirr_data
            self.stored_mirr_data = new_stored_data

            old_coords: CoordsArray = self.mirror_coords.copy()
            self.mirror_coords = cast("CoordPair", np.empty((2, new_length)))
            self.mirror_coords[:, : old_coords.shape[1]] = old_coords

        temp_data = (
            orig_id,
            exists,
            current_mirr_idx,
            mol_group,
            grid_idx,
            rot_idx,
            x_coord,
            y_coord,
            polygon,
        )
        self.stored_mirr_data[current_mirr_idx] = temp_data
        self.mirror_coords[:, current_mirr_idx] = x_coord, y_coord
        self.mirr_tree.insert(current_mirr_idx, polygon.bounds)
