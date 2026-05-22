# Change Log
All important changes to ``AdsorPy`` will be documented in this file.

This format is based on [Keep a Changelog](http://keepachangelog.com/). This project adheres to [Semantic Versioning](http://semver.org/).

## [unreleased]
### Added
### Changed
### Fixed
### Removed

## 1.2.2 - 2026-05-22
### Added
- Add attestation to PyPI.
- Add ``.xyz`` input validation
- Add tests with fuzzer for input validation.
- Add SBOM to PyPI CI.
- Add uv to CI.
### Changed
- Change ``typing.py`` to ``types.py`` to prevent name clash.
### Fixed
- Tighten permissions for CI
### Removed
- Remove explicit token minting in PyPI CI. This is now implicit.
- Remove pip upgrading from CI. Runners are up to date.
- Remove unnecessary info from the `.tar.gz` file.

## 1.2.1 - 2026-05-08
### Added
- Add ``tach`` for internal dependency sanitation.
- Add ``slotscheck`` to ensure correct use of slots in dataclasses.
- Add ``pydantic`` to ensure correctness of ``config.json``.
- Add intersphinx mapping for ``pydantic``, ``shapely``, ``numba``, ``matplotlib``, and ``rtree``.
### Changed
- Change lint checking version of Python from 3.13 to 3.14.
### Fixed
- Reduce the complexity of the ``tests-ci.yml`` and ``tox.toml``.
### Removed

## 1.2.0 - 2026-05-06
### Added
- Add ``pyright`` linting in strict mode.
- Add ``typing.py``.
### Changed
- Move all ``TypeAlias`` to the new ``typing.py``.
### Fixed
- Fix flux: was an ``np.ndarray``, is now a tuple of arrays. Beware: this changes the type of that return value of ``run_simulation()``.
### Removed

## 1.1.5 - 2026-05-04
### Added
- Installation can now use ``[test]``, ``[lint]``, and ``[docs]`` as optional dependencies.
- Add final ``ci-success`` step to the ``tests-ci.yml``.
- Add ``dependabot.yml`` with permission for minor/patch update pull requests.
- Add ``CODE_OF_CONDUCT.md``.
- Add ``CONTRIBUTING.md``.
- Add ``SECURITY.md``.
### Changed
- Change ``test-ci.yml`` order: first on pull_requests, then push. Change push filter from paths to paths-ignore. Now ignores all ``.md`` files and everything in docs. Tests run whenever anything else is changed. Also puts ``main`` on ignore for push. Direct pushing to ``main`` is not allowed, and not ignoring ``main`` would result in duplicate tests.
- Lower ``test_gapsize_analysis()`` in ``rsa_test.py`` to 100.
- Change parametrised tests in ``rsa_tests.py`` to improve readability of code and test results.
- Move from ``tox.ini`` to ``tox.toml``.
### Fixed
- Fix potential vulnerabilities in the CI by reducing the scope of permissions.
- All tests are now ``mypy`` and ``ruff`` compliant.
- Fix the link to the adsorpy image. The image can now be seen on PyPI.
### Removed
- ``requirements.txt``
- ``requirements_dev.txt``
- ``requirements_lint.txt``
- ``requirements_docs.txt``

## 1.1.4 - 2026-04-28
### Added
### Changed
### Fixed
- Fix default symmetry settings: no reflection symmetry, no rotation symmetry.
### Removed

## 1.1.3 - 2026-04-28
### Added
- Add ``__version__``, ``__name__``, ``__author__``, and ``__author_email__`` to ``__init__.py``.
### Changed
### Fixed
- Update ``pyproject.toml`` to have the appropriate keywords and classifiers. The project does not show up on the PyPI index but can be installed from it.
### Removed

## 1.1.2 - 2026-04-27
### Added
### Changed
### Fixed
- Turn relative imports into absolute imports to make the package pip-installable.
- Fix the type hints that broke as a result of this.
### Removed

## 1.1.1 - 2026-04-27
### Added
### Changed
- Update dependency versions.
### Fixed
- Fix ``pyproject.toml``: ``"AdsorPy" = ["py.typed"]`` (from ``rsa-mc``, the old internal project name.)
### Removed

## 1.1.0 - 2026-04-24

### Added
- Add the ``Simulator`` class as 6th return value of ``run_simulation()``.
- Add overlap and gap size subtests to the dosing scheme tests of ``run_simulation_test``.
### Changed
- Update CI action/deploy-pages to v5. Update CI action/upload-artifact to v7.
- Move overlap and gap size test for alt simulation to subtests.
### Fixed
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