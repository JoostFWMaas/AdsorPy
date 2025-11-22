# AdsorPy

Random Sequential Adsorption (RSA) Python 3.10+ script with Monte Carlo diffusion and desorption.

In RSA, molecules arrive one by one at a surface. Adsorption takes place if the molecule does not overlap with molecules
already on the surface.
The list of available orientations for the molecule is traversed in random order until the first orientation that fits
is found, or until the list is exhausted.
All available sites are checked, and various metrics can be extracted afterwards such as the coverage, covered area, and
gap size distribution.

## How to use

Run the `main.py` in order to run a simple single run using the standard Hacac molecule on hexagonal aluminium oxide.
New molecules can be created by running the `molecule_lib.py`or by calling the first_time_loader() function from that module. Molecules can be generated from `.xyz` files. It is
recommended to run `molecule_lib.py` directly from command line to define the molecule orientation, then store the new
molecule string for repeated use.

User friendliness will be updated at a later stage, allowing the user to define simulation modes, surfaces, and
molecules more easily.

## Design philosophy of AdsorPy

Because AdsorPy has been made with scientific rigour in mind, the package is tested in multiple ways:
- Unit tests (`Pytest`) of the code ensure correct behaviour for expected input.
- Property tests (`Hypothesis`) of the most critical code components ensure correct behaviour for unexpected input as well.
- `Mypy` (in `--strict`) mode ensures that the package is correctly-typed, as if it were static. The `py.typed` file--a promise that the code is type-hinted properly--is added because the code passes this test.
- `Ruff` is used as a linter with almost all rules enabled (see the pyproject.toml for the list of exlcusions and reasons).
- `Tox` is used to run all of the aforementioned tests in parallel for multiple Python versions to ensure correct behaviour.
- CI is used for automated testing every time the code is changed.

The package also makes use of an optional config file that falls back on standard behaviour, because configs are often used in scientific software (set-and-forget).

# Openness and academic collaboration

The script was made public for the sake of openness and academic collaboration. If your paper makes use of this script, feel free to contact me if there are questions. 
At the end of this file, I will place a list of papers that make use of this work. Feel free to contact me if you want your work to be added to the list as well.
