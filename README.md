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