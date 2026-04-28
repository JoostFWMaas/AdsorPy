import re
from importlib.metadata import PackageNotFoundError, metadata

try:  # This only works when running as a package.
    meta = metadata(__package__ or __name__)
except PackageNotFoundError:  # For development, this will trigger instead.
    meta = metadata("adsorpy")

__name__ = meta["Name"]
__version__ = meta["Version"]

author_and_email = meta["Author-email"]
name = re.search(r'"([^"]*)"', author_and_email)
email = re.search(r'<([^"]*)>', author_and_email)

__author__ = name[1] if name is not None else name
__author_email__ = email[1] if email is not None else email
