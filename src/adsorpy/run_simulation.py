"""Runs the simulation.

run_simulation() should provide the user with enough customisability.
Molecules can be generated from .xyz using molecule_lib.
"""

from __future__ import annotations

from sys import version_info

if version_info >= (3, 11):
    from datetime import UTC, datetime  # For datetime stamping and seed generation.
else:
    from datetime import datetime

import time  # For timing of the script.
from itertools import count  # A simple counter, iterates with next(...).
from pathlib import Path  # For path handling in Python.
from typing import Literal, ParamSpec, TypeVar, cast  # For type hinting.

import molecule_lib as mol  # Homebrew lib of molecules and molecule footprint generation.
import numpy as np  # For vectorised computations (performed in C).
from numpy.random import PCG64DXSM, Generator  # New random generator.
from numpy.typing import NDArray
from randomsequentialadsorption import MoleculeGroup, Simulator, Surface
from rsa_config import RsaConfig  # Config of the simulation.
from shapely import Polygon  # Shapely creates and manipulates polygons.
from shapely.prepared import PreparedGeometry

P = ParamSpec("P")  # Helps with static type checkers.
T1 = TypeVar("T1")
T2 = TypeVar("T2", bound=np.generic | Polygon)  # type: ignore[explicit-any]
TargetType = np.generic | Polygon
Tn = TypeVar("Tn", bound=np.ndarray[tuple[int], np.dtype[np.generic | Polygon]])  # type: ignore[explicit-any]

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


def run_simulation(  # noqa: PLR0913
    rsa_config: RsaConfig,
    molecules_list: Polygon | list[Polygon] | np.ndarray[tuple[int], np.dtype[Polygon]] | None = None,
    rotation_symmetries: int | list[int] | np.ndarray[tuple[int], np.dtype[np.int_]] | None = None,
    reflection_symmetries: bool | list[bool] | np.ndarray[tuple[int], np.dtype[np.bool_]] | None = None,
    rotation_counts: int | list[int] | np.ndarray[tuple[int], np.dtype[np.int_]] | None = None,
    lattice_type: str = "triangular",
    site_count: int | None = None,
    lattice_a: float | None = None,
    boundary_condition: str | None = None,
    simulation_type: str = "sequential",
    dosing_distribution: list[float] | None = None,
    include_rejected_flux: bool = False,
    calculate_gap_size: bool = False,
    print_output_flag: bool = False,
    plot_output_flag: bool = False,
    seed: int | None = None,
    timestep_limit: int = 1000000,
    site_x_coords: DistArray | None = None,
    site_y_coords: DistArray | None = None,
    bounding_x_coord: float | None = None,
    bounding_y_coord: float | None = None,
    sticking_probability: float | list[float] = 1.,
) -> tuple[list[int], DistArray, int, IdxArray, IdxArray]:
    """Run the RSA simulation.

    If no molecule is provided, defaults to a circular molecule of radius 0.55 Angstrom.
    Ensure that all lists are of equal length. If only one molecule is provided, adding it to a list is not needed.
    If the simulation_type is "codosing", the dosing_distribution parameter is used, otherwise it is ignored.

    :param rsa_config: Input parameters defined in the config.
    :param molecules_list: List of molecules. Ensure this list is equal in length to the next three entries.
    :param rotation_symmetries: Rotation symmetries. 0 for circle, 1 for no symmetry, 2 twofold, 3 threefold, etc.
    :param reflection_symmetries: Mirror symmetries. True means mirror symmetry, False means no mirror symmetry.
    :param rotation_counts: Number of rotations to be considered. Preferably numbers that divide 360.
    :param lattice_type: The lattice type. "triangular", "hexagonal", or "square".
    :param site_count: The site count along one axis. Optional. If None, defaults to the value in config.json.
    :param lattice_a: The lattice spacing in Angstrom. If None, defaults to the value in config.json.
    :param boundary_condition: The boundary condition. Optional, can be soft/hard/periodic. If None, defaults to config.
    :param simulation_type: Type of simulation. Valid types are "sequential", "codosing", and "cascade".
    :param dosing_distribution: The distribution of the adsorption attempts in case of "codosing". Uniform if None.
    :param include_rejected_flux: Whether to include rejected flux in the simulation. If True, sites can be reattempted.
    :param calculate_gap_size: Whether to calculate the gap size of the simulation.
    :param print_output_flag: Toggle printing output of simulation.
    :param plot_output_flag: Toggle plotting output of simulation.
    :param seed: The seed for the simulation. If None, takes the datetime in microseconds: YYYMMDDhhmmssuuuuuu.
    :param timestep_limit: The maximum number of timesteps to simulate for when the flux is taken into account.
    :param site_x_coords: x coordinates of the sites. If this and the next four arguments are provided, generate custom.
    :param site_y_coords: y coordinates of the sites. Curstom surface overrides site_count and lattice_a.
    :param bounding_x_coord: The bounding box x coordinate value. Only use for custom surface. Leave None otherwise.
    :param bounding_y_coord: The bounding box y coordinate value. Ensure the previous three arguments are provided.
    :param sticking_probability: The sticking probability. Default is 1.0 from config. Per molecule or for all.
    :return: the amount of molecules on the surface per molecule group, the gap size distribution, and the RNG seed.
    :raises TypeError: If the sticking probability is an invalid type.
    """
    molecules_list, rotation_symmetries, reflection_symmetries, rotation_counts = _initialise_run_parameters(
        molecules_list,
        rotation_symmetries,
        reflection_symmetries,
        rotation_counts,
        simulation_type,
    )

    custom_grid_flg: bool = _error_checker(
        molecules_list,
        rotation_symmetries,
        reflection_symmetries,
        rotation_counts,
        simulation_type,
        dosing_distribution,
        boundary_condition,
        site_x_coords,
        site_y_coords,
        bounding_x_coord,
        bounding_y_coord,
    )

    # The seed is defined as the datetime in microseconds. The seed is stored so simulations can be verified.
    how_late = datetime.now(UTC) if version_info >= (3, 11) else datetime.utcnow()
    seed = int(how_late.strftime("%Y%m%d%H%M%S%f")) if seed is None else seed
    # As long as the new safer PCG64DXSM generator is not the default, override the generator.
    rng = Generator(PCG64DXSM(seed)) if isinstance(seed, int | np.int_) else seed
    # Output files of the script are datetime stamped.
    timestr = how_late.strftime("%Y%m%dT%H%M%S")
    results_folder = Path.cwd() / "Output" / f"{timestr}_RSA"

    time_count: count[int] = count()

    temp_results_folder = results_folder
    attempt_count: int = 0
    # while os.path.isdir(temp_results_folder):
    while temp_results_folder.is_dir():
        temp_results_folder = results_folder.parent / (results_folder.name + f"_{attempt_count}")
        # Add a counter to the end of the folder name.
        attempt_count += 1

    results_folder = temp_results_folder

    surf = Surface(rsa_config, lattice_type=lattice_type, lattice_a=lattice_a, site_count=site_count)
    if custom_grid_flg:
        surf.generate_custom_surface(site_x_coords, site_y_coords, bounding_x_coord, bounding_y_coord)
    else:
        surf.generate_grid(rng)

    molecules = []  # Initially, there are none.
    rot_syms = rotation_symmetries  # Rotation symmetry
    mirror_syms = reflection_symmetries  # Mirror symmetry
    rot_cnts = rotation_counts
    mgc: count[int] = count()
    if isinstance(sticking_probability, float):
        sticking_probability = [sticking_probability] * len(molecules_list)
    elif not isinstance(sticking_probability, list | np.ndarray):
        errmsg = "sticking_probability must be a float, list, or np.ndarray"
        raise TypeError(errmsg)

    pp: Polygon
    for idx, (pp, stick) in enumerate(zip(molecules_list, sticking_probability, strict=False)):
        molecules.append(
            MoleculeGroup(
                rsa_config,
                pp,
                rot_syms[idx],
                mirror_syms[idx],
                surf.all_site_count,
                mgc,
                rot_cnts[idx],
                sticking_probability=stick,
            ),
        )

    dbl_max_rad: float = 2.0 * max([cule.max_radius for cule in molecules])
    surf.bp.biggest_radius = dbl_max_rad

    surf.bp.generate_boundary_conditions(surf)  # Generate the boundary conditions.
    for molec in molecules:  # Generate the BC and molecules.
        molec.bp.biggest_radius = dbl_max_rad
        molec.bp.generate_boundary_conditions(surf, molec)
        molec.generate_rotated_molecules(molec.bp, molecules)
    sim = Simulator(
        rsa_config,
        include_rejected_flux=include_rejected_flux,
        surf=surf,
        mol_groups=molecules,
        rng=rng,
        boundary_type=boundary_condition,
    )

    all_flux, phi = _select_and_run(
        sim,
        surf,
        molecules,
        simulation_type,
        include_rejected_flux,
        time_count,
        timestep_limit,
        dosing_distribution,
    )
    gaps = _postprocessing(
        sim,
        surf,
        molecules,
        print_output_flag,
        plot_output_flag,
        calculate_gap_size,
        results_folder,
        timestr,
    )

    return [mgr.molecule_counter for mgr in sim.molgroups], gaps, seed, all_flux, phi


def _run_sequential(
    sim: Simulator,
    surf: Surface,
    molecules: list[MoleculeGroup],
    time_count: count[int],
    timestep_limit: int,
) -> None:
    for moldx, molgr in enumerate(sim.molgroups):
        while molgr.vacancy_count and next(time_count) < timestep_limit:
            sim.attempt_place_molecule(surf, molecules[moldx])


def _run_codosing(
    sim: Simulator,
    surf: Surface,
    molecules: list[MoleculeGroup],
    time_count: count[int],
    timestep_limit: int,
    dosing_distribution: list[float] | None = None,
) -> None:
    for moldx, molgr in enumerate(sim.molgroups):
        while molgr.vacancy_count and next(time_count) < timestep_limit:
            sim.attempt_random_placement(surf, *molecules[moldx:], weights=dosing_distribution)


def _run_cascade(
    sim: Simulator,
    surf: Surface,
    molecules: list[MoleculeGroup],
    time_count: count[int],
    timestep_limit: int,
) -> None:
    for moldx, molgr in enumerate(sim.molgroups):
        while molgr.vacancy_count and next(time_count) < timestep_limit:
            sim.attempt_cascading_placement(surf, *molecules[moldx:])


def _run_flux(
    sim: Simulator,
    surf: Surface,
    molecules: list[MoleculeGroup],
    timestep_limit: int,
) -> tuple[IdxArray, IdxArray]:
    """Adsorb while taking steps into account.

    :param sim: Simulator.
    :param surf: Surface.
    :param molecules: MoleculeGroups list, the molecules that will be added to the surface.
    :param timestep_limit: int, number of timesteps to simulate.
    :return: list of indices during which adsorption takes place.
    """
    all_flux: tuple[IdxArray, ...] = ()
    all_phis: list[int] = []
    for mdx, molgr in enumerate(sim.molgroups):
        mol_flux: list[int] = []
        for step in range(timestep_limit):
            if not molgr.vacancy_count:
                break  # Done when nothing is available.
            flag, *_, phis = sim.attempt_cascading_placement(surf, *molecules[mdx:])
            if flag:
                mol_flux.append(step)
                all_phis.append(np.max(phis))

        all_flux += cast("tuple[IdxArray, ...]", (np.asarray(mol_flux, dtype=np.int_),))

    return all_flux[0], np.array(all_phis, dtype=np.int_)  # FIXME: Fix the [0] later.


def _run_flux_fixedrotation(
    sim: Simulator,
    surf: Surface,
    molecules: list[MoleculeGroup],
    timestep_limit: int,
    distribution: list[float] | None = None,
) -> tuple[IdxArray, IdxArray]:
    """Adsorb while taking steps into account. Rotation is fixed and sites are varied instead.

    :param sim: Simulator.
    :param surf: Surface.
    :param molecules: MoleculeGroups list, the molecules that will be added to the surface.
    :param timestep_limit: int, number of timesteps to simulate.
    :param distribution: list of floats indicating the distribution of the molecules. Empty for uniform distribution.
    :return: list of indices during which adsorption takes place.
    """
    mol_flux: list[int] = []
    all_phis: list[int] = []
    for step in range(timestep_limit):
        if not np.any([molgroup.vacancy_count for molgroup in molecules]):
            break  # If nothing is available, terminate.

        randmol: MoleculeGroup = sim.rng.choice(molecules, p=distribution)
        if not randmol.vacancy_count:
            continue
        flag, *_, phi = sim.attempt_place_molecule(surf, randmol)
        if flag:
            mol_flux.append(step)
            all_phis.append(np.max(phi))


    return np.array(mol_flux, dtype=np.int_), np.array(all_phis, dtype=np.int_)


def _initialise_run_parameters(
    molecules_list: Polygon | list[Polygon] | np.ndarray[tuple[int], np.dtype[Polygon]] | None = None,
    rotation_symmetries: int | list[int] | np.ndarray[tuple[int], np.dtype[np.int_]] | None = None,
    reflection_symmetries: bool | list[bool] | np.ndarray[tuple[int], np.dtype[np.bool_]] | None = None,
    rotation_counts: int | list[int] | np.ndarray[tuple[int], np.dtype[np.int_]] | None = None,
    simulation_type: str = "sequential",
) -> tuple[GeoArray, IdxArray, BoolArray, IdxArray]:
    """Initialise run parameters.

    :param molecules_list: 2D Molecule footprints.
    :param rotation_symmetries: Rotation symmetries of the molecules.
    :param reflection_symmetries: Reflection symmetries of the molecules.
    :param rotation_counts: Number of rotations.
    :param simulation_type: Type of dosing scheme.
    :return: List of molecules, rotation symmetries, reflection symmetries, rotation counts.
    :raises ValueError: If the requested `simulation_type` does not exist.
    """
    molecules_list = [mol.circulium(0.55)] if molecules_list is None else molecules_list
    rotation_symmetries = [0] if rotation_symmetries is None else rotation_symmetries
    reflection_symmetries = [True] if reflection_symmetries is None else reflection_symmetries
    rotation_counts = [360] if rotation_counts is None else rotation_counts
    if simulation_type not in {"sequential", "codosing", "cascade"}:
        errmsg = "The simulation type must be either 'sequential', 'codosing', or 'cascade'."
        raise ValueError(errmsg)

    molecules_list = _turn_into_list(molecules_list, Polygon)
    rotation_symmetries = _turn_into_list(rotation_symmetries, int)
    reflection_symmetries = _turn_into_list(reflection_symmetries, bool)
    rotation_counts = _turn_into_list(rotation_counts, int)

    mol_list_size = molecules_list.size
    rot_syms = _repeater(rotation_symmetries, mol_list_size)
    refl_syms = _repeater(reflection_symmetries, mol_list_size)
    rot_cnts = _repeater(rotation_counts, mol_list_size)

    return molecules_list, rot_syms, refl_syms, rot_cnts


def _turn_into_list(val_or_list: T1 | list[T1], compare_to: type[T1]) -> np.ndarray[tuple[int], np.dtype[T2]]:  # type: ignore[explicit-any]
    """Turn a variable or a list into an array.

    :param val_or_list: value or list.
    :param compare_to: comparison type. Should be either the type of the value or the type in the list.
    :return: the 1D array of the original variable or list.
    """
    return cast(  # type: ignore[explicit-any]
        "np.ndarray[tuple[int], np.dtype[T2]]",
        np.asarray([val_or_list] if isinstance(val_or_list, compare_to) else val_or_list),
    )


def _repeater(orig_array: Tn, comparison_len: int) -> Tn:  # type: ignore[explicit-any]
    """Take the array and repeat it if it has a length of 1.

    :param orig_array: original array.
    :param comparison_len: length of the repetition.
    :return: the array repeated to the proper length.
    """
    return cast("Tn", np.repeat(orig_array, comparison_len) if orig_array.size == 1 else orig_array)  # type: ignore[explicit-any]


def _error_checker(  # noqa: PLR0913
    molecules_list: GeoArray,
    rotation_symmetries: IdxArray,
    reflection_symmetries: BoolArray,
    rotation_counts: IdxArray,
    simulation_type: str,
    dosing_distribution: list[float] | None = None,
    boundary_condition: str | None = None,
    site_x_coords: DistArray | None = None,
    site_y_coords: DistArray | None = None,
    bounding_x_coord: float | None = None,
    bounding_y_coord: float | None = None,
) -> bool:
    """Check the errors. Returns bool denoting custom grid (False if grid is not custom).

    :param molecules_list: list of molecules.
    :param rotation_symmetries: rotation symmetries.
    :param reflection_symmetries: reflection symmetries.
    :param rotation_counts: rotation counts.
    :param simulation_type: simulation type.
    :param dosing_distribution: dosing distribution.
    :param boundary_condition: boundary condition.
    :param site_x_coords: site x coordinates.
    :param site_y_coords: site y coordinates.
    :param bounding_x_coord: boundary x coordinate.
    :param bounding_y_coord: boundary y coordinate.
    :return: the custom grid flag.
    :raises ValueError: 1) if the length of the molecules, symmetries, reflections, and rotations are unequal,
    2) if the dosing distribution is not the same length as the molecule list, 3) if the provided boundary condition is
    not supported, or 4) if the custom grid x/y coordinates/boundaries are not all empty or not all provided.
    """
    if not len(molecules_list) == len(rotation_symmetries) == len(reflection_symmetries) == len(rotation_counts):
        errmsg = "Number of molecules, symmetries, and/or rotation counts do not match."
        raise ValueError(errmsg)

    if (
        simulation_type == "codosing"
        and dosing_distribution is not None
        and len(molecules_list) != len(dosing_distribution)
    ):
        errmsg = "Dosing distribution must be the same length as the molecule list."
        raise ValueError(errmsg)

    if boundary_condition is not None and boundary_condition not in {"soft", "hard", "periodic"}:
        errmsg = "Boundary condition must be either 'soft', 'hard', 'periodic', or None."
        raise ValueError(errmsg)

    custom_any_all: list[bool] = [False, True]
    for comparison in [site_x_coords, site_y_coords, bounding_x_coord, bounding_y_coord]:
        temp_result: bool = comparison is not None
        custom_any_all[0] |= temp_result
        custom_any_all[1] &= temp_result

    if custom_any_all[0] != custom_any_all[1]:  # If not all are None or not None:
        errmsg = """A custom grid will only be generated if 'site_x_coords', 'site_y_coords',
                    'bounding_x_coord' and 'bounding_y_coord' are specified. Otherwise, leave all None."""
        raise ValueError(errmsg)
    return custom_any_all[1]


def _postprocessing(
    sim: Simulator,
    surf: Surface,
    molecules: list[MoleculeGroup],
    print_output_flag: bool,
    plot_output_flag: bool,
    calculate_gap_size: bool,
    results_folder: str | Path,
    timestr: str,
) -> DistArray:
    """Perform post-processing of the data.

    :param sim: simulator.
    :param surf: surface.
    :param molecules: list of molecules.
    :param print_output_flag: True if output such as coverages and fraction of covered area should be printed.
    :param plot_output_flag: True if plots are to be shown.
    :param calculate_gap_size: True if gap size should be calculated. Can be computationally intensive.
    :param results_folder: results folder.
    :param timestr: timestr to use.
    :return: the gap size distribution. Empty array if calculate_gap_size is False.
    """
    if print_output_flag:
        for mdx, molecul in enumerate(sim.molgroups):
            print(f"Mol {mdx}, {molecul.molecule_counter}")
            print(f"Mol dens {mdx}, {100 * molecul.molecule_counter / surf.area}")
            print(f"areamol {mdx}: {molecul.area}")

    if plot_output_flag:
        Path(results_folder).mkdir(parents=True)
        sim.plot_covered_grid(
            surf,
            molecules,
            save_flag=True,
            plt_flag=True,
            timestr=f"{timestr}",
            results_folder=results_folder,
        )

    return sim.analyse_gap_size(surf) if calculate_gap_size else np.empty(0)


def _select_and_run(
    sim: Simulator,
    surf: Surface,
    molecules: list[MoleculeGroup],
    simulation_type: str,
    include_rejected_flux: bool,
    time_count: count[int],
    timestep_limit: int,
    dosing_distribution: list[float] | None = None,
) -> tuple[IdxArray, IdxArray]:
    """Select and run. A collection of the simulations. Putting them in one function helps streamline adjusting them.

    :param sim: simulator.
    :param surf: surface.
    :param molecules: list of molecules.
    :param simulation_type: simulation type.
    :param include_rejected_flux: True if rejected flux should be included.
    :param timestep_limit: number of time steps.
    :param dosing_distribution: dosing distribution. Leave empty if not needed or uniform.
    :return: Fluxes per molecule if selected.
    :raises ValueError: if the `simulation_type` (dosing scheme) and flux flag combination are not supported.
    """
    all_flux: IdxArray = np.empty(0, dtype=np.int_)
    phis: IdxArray = np.empty(0, dtype=np.int_)

    match (simulation_type, include_rejected_flux):
        case ("sequential", False):
            _run_sequential(sim, surf, molecules, time_count, timestep_limit)
        case ("codosing", False):
            _run_codosing(sim, surf, molecules, time_count, timestep_limit, dosing_distribution)
        case ("cascade", False):
            _run_cascade(sim, surf, molecules, time_count, timestep_limit)
        case ("sequential", True):
            all_flux, phis = _run_flux(sim, surf, molecules, timestep_limit)
        case ("codosing", True):
            all_flux, phis = _run_flux_fixedrotation(sim, surf, molecules, timestep_limit)
        case (_, _):
            errmsg = f"Simulation type {simulation_type} with rejected_flux = {include_rejected_flux} is not supported."
            raise ValueError(errmsg)
    return all_flux, phis


def main() -> int:
    """Run the RSA script, for demonstration purposes."""
    config_path = Path(__file__).parent / "config.json"
    rsa_config = RsaConfig(str(config_path))
    start = time.perf_counter()
    run_simulation(rsa_config, plot_output_flag=True, include_rejected_flux=False)
    end = time.perf_counter()
    totaltime = f"{(end - start):.0f} seconds elapsed since start."
    print(totaltime)

    return 0


if __name__ == "__main__":  # Best practice.
    main()
