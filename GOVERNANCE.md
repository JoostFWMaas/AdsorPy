# Project Governance

This document defines the governance structure, roles, and responsibilities for the AdsorPy project.

## Overview

AdsorPy is currently maintained by a single contributor as part of the @TUe-PMP research group.
The structure defined here is designed to scale as more contributors join.

---

## Roles and Responsibilities

### Maintainer

The Maintainer is responsible for the overall health, direction, and quality of the project.

**Current Maintainer:**
- Joost Maas (@JoostFWMaas)

**Responsibilities:**
- Review and merge pull requests
- Manage issue triage and prioritisation
- Define project direction and roadmap
- Ensure code quality and documentation standard
- Create and publish releases
- Act as the primary point of contact for the project

---

### Contributors

Contributors are individuals who submit issues, documentation updates, or code changes.

**Current contributors:**
- Joost Maas (@JoostFWMaas)

**Responsibilities:**
- Follow contribution guidelines (`CONTRIBUTING.md`)
- Write clear, well-documented code
- Participate in issue discussion when applicable

---

## Decision-Making process
- The Maintainer has final decision authority on:
  - Pull request approvals
  - Issue prioritisation
  - Project direction
- Major changes (e.g. architectural shifts) should:
  - Be documented in issues or proposals
  - Include rationale and discussion before implementation

---

## Contribution and Reviews
- All changes should be submitted via pull requests
- The Maintainer review and approves all pull requests
- Automated checks must pass before merging
- These checks are enforced by the repository and cannot be overridden, including by the Maintainer

---

## Releases

Releases are managed by the Maintainer.

**Process:**
- Releases are triggered by pushing a version tag (following SemVer, e.g., v1.0.0) to the repository
- Tag pushes automatically initiate CI/CD workflows
- Release creation is automated and contingent on successful CI checks
- It is not possible to produce an official release without passing all checks

---

## Transparency
- Project discussion should happen in:
  - GitHub Issues
  - Pull Requests
- Decisions should be documented in issues or commit messages where appropriate

---

## Future Governance
As the project grows, additional roles may be introduced, such as: 
- Reviewers
- Release Managers
- Module Owners

This document will be updated to reflect those changes.

---

## Contact

For questions about governance of this project:
- Open an issue in this repository
- Contact the Maintainer directly via GitHub

---

## Updates to This Document
This governance document may be updated by the Maintainer as the project evolves.
