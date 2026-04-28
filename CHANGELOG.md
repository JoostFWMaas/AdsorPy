# Change Log
All important changes to ``AdsorPy`` will be documented in this file.

This format is based on [Keep a Changelog](http://keepachangelog.com/) and this project adheres to [Semantic Versioning](http://semver.org/).

## [unreleased]
### Added
### Changed
### Fixed
### Removed

## 1.0.3 - 2026-04-28
### Added
- Add the ``Simulator`` class as 6th return value of ``run_simulation()``.
- Add overlap and gap size subtests to the dosing scheme tests of ``run_simulation_test``.
- Add ``__version__``, ``__name__``, ``__author__``, and ``__author_email__`` to ``__init__.py``.
### Changed
- Update CI action/deploy-pages to v5. Update CI action/upload-artifact to v7.
- Move overlap and gap size test for alt simulation to subtests.
- Update dependency versions.
### Fixed
- Fix ``pyproject.toml``: ``"AdsorPy" = ["py.typed"]`` (from ``rsa-mc``, the old internal project name.)
- Turn relative imports into absolute imports to make the package pip-installable.
- Fix the type hints that broke as a result of this.
- Update ``pyproject.toml`` to have the appropriate keywords and classifiers. The project does not show up on the PyPI index but can be installed from it.
### Removed


## 1.0.2 - 2026-04-23

### Added

### Changed
- Update CI actions to move from ``node.js`` 20 to 24.

### Fixed

### Removed
- Remove unnecessary type casts.

## 1.0.1 - 2026-04-23

### Added
- Implement subtests for the test_run_alt (a test for the RSA simulator on an alternative seed).
- Add a determinism test to show identical outcome for identical seeds. 
- In ``tool.ruff.lint``: use ``future-annotations = true``, this correctly checks ``__future__`` annotation.
### Changed 
- Update version dependencies. 
- Strengthen seed comparison test. 
- Vectorise (remove a for-loop) from overlap test.

### Fixed

### Removed

## 1.0.0 - 2025-11-29
- Full release