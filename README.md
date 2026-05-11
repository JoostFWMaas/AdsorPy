# AdsorPy

[![pypi version](https://img.shields.io/pypi/v/adsorpy.svg)](https://pypi.python.org/pypi/adsorpy) <!-- [![Conda](https://img.shields.io/conda/vn/conda-forge/adsorpy)](https://anaconda.org/conda-forge/adsorpy) -->
![test results badge](https://github.com/JoostFWMaas/AdsorPy/actions/workflows/tests-ci.yml/badge.svg)
[![docs build results](https://github.com/JoostFWMaas/AdsorPy/actions/workflows/docs-ci.yml/badge.svg)](https://joostfwmaas.github.io/AdsorPy/)
[![codecov](https://codecov.io/github/JoostFWMaas/AdsorPy/graph/badge.svg?token=XBYZU63D8Y)](https://codecov.io/github/JoostFWMaas/AdsorPy)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/JoostFWMaas/AdsorPy/badge)](https://scorecard.dev/viewer/?uri=github.com/JoostFWMaas/AdsorPy)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/12810/badge)](https://www.bestpractices.dev/projects/12810)


Lattice-based random sequential adsorption (RSA) Python 3.10+ script.

In RSA, molecules arrive one by one at a surface. Adsorption takes place if the molecule does not overlap with molecules
already on the surface.
The list of available orientations for the molecule is traversed in random order until the first orientation that fits
is found, or until the list is exhausted.
All available sites are checked, and various metrics can be extracted afterwards such as the coverage, covered area, and
gap size distribution.

<img src="https://raw.githubusercontent.com/JoostFWMaas/AdsorPy/ccedff9c0a4cca1e89c4d461dbfeb91c49e38b21/images/AdsorPy_Covered_Surface.png" alt="Output of an AdsorPy run, an image of a covered surface." style="width:50%; height:auto;">

## How to use

This package can be found on PyPI as [adsorpy](https://pypi.org/project/adsorpy/), so we recommend the following: 
```bash 
pip install adsorpy
```

Run the `adsorpy.main` in order to run a simple single run using the standard disk-shaped molecule on hexagonal aluminium oxide.
New molecules can be created by running the `adsorpy.molecule_lib` or by calling the `adsorpy.molecule_lib.first_time_loader()` function. Molecules can be generated from `.xyz` files. It is
recommended to run `adsorpy.molecule_lib` directly from command line to define the molecule orientation, then store the new
molecule string for repeated use.

Run the `adsorpy.run_simulation.run_simulation()` function with the molecule footprints generated in the previous step. Output can be printed to stdout, plotted, and saved.
An example for dog-bone molecules with rotations of 360 // 12 = 30 degrees:
```python
from adsorpy.run_simulation import run_simulation
from adsorpy.molecule_lib import dogbonium
molecule = dogbonium()
output = run_simulation(molecules_list=molecule, rotation_counts=12, plot_output_flag=True)
```


User-friendliness will be updated at a later stage, allowing the user to define simulation modes, surfaces, and
molecules more easily.

Documentation (generated with `sphinx`): https://joostfwmaas.github.io/AdsorPy/

## Future additions

In a future update, the code will be expanded with diffusion, desorption, and species conversion (changing from one molecule on the surface to another).

## Design philosophy

Because AdsorPy has been made with scientific rigour in mind, the package is tested in multiple ways:
- Unit tests (`pytest`) of the code ensure correct behaviour for expected input.
- Property tests (`hypothesis`) of the most critical code components ensure correct behaviour for unexpected input as well.
- `mypy` (in `--strict` mode) and `pyright` (`strict`) ensure that the package is correctly-typed, as if it were static. The `py.typed` file, a promise that the code is type-hinted properly, is added because the code passes this test.
- `ruff` is used as a linter with almost all rules enabled (see the `pyproject.toml` for the list of exclusions and reasons).
- `slotscheck` ensures that all dataclass slots are set up correctly.
- `tach` ensures that internal dependencies are handled correctly (from low level to high), preventing circular dependencies.
- `tox` is used to run all of the aforementioned tests in parallel for multiple Python versions to ensure correct behaviour.
- CI is used for automated testing.

The package also makes use of an optional config file that falls back on standard behaviour, because configs are often used in scientific software (set-and-forget).

## Openness and academic collaboration

The script was made public for the sake of openness and academic collaboration. Please let me know if you have questions about the script, or if you have discovered any issues/bugs. 
At the end of this file, I will place a list of papers that make use of this work. Feel free to contact me if you want your work to be added to the list as well.
