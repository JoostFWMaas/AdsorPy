import os
import sys
from pathlib import Path

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Random Sequential Adsorption (RSA)"
copyright = "2024, J.F.W. Maas"
author = "J.F.W. Maas"
version = "1,0"
release = "1.0.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
]

templates_path = ["_templates"]
# source_dir = "."

# autodoc_member_order = "bysource"


# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "scripts/*.rst"]

automodule_path = [""]
automodule_members = True
autoclass_content = "both"

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'private-members': True,
    'special-members': '__init__',
    'inherited-members': True,
    'show-inheritance': True,
}

autodoc_member_order = "bysource"


suppress_warnings = ["config.cache"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_material"
html_title = "Random Sequential Adsorption (RSA)"

html_static_path = ["_static"]
html_style = ["style.css"]

html_theme_options = {"body_max_width": "none",
                      "color_primary": "indigo",
                      "color_accent": "blue",
                      "globaltoc_depth": 2,
                      "globaltoc_collapse": True,
                      "globaltoc_includehidden": True,
                      }
# automodule_path = ['../../src/adsorpy']

# Get the path to the directory containing the Python modules to mock
module_directory = Path("../../src/adsorpy/")


# Get the path to the directory containing the Python modules to mock

# Get a list of all .py files in the directory
py_files = module_directory.glob("*.py")

# Extract the module names from the filenames
module_names = [f.stem for f in py_files]
print(module_names)

# List of modules to mock during documentation build
autodoc_mock_imports = module_names

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
