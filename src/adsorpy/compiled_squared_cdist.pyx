# Copyright (c) 2025-2026 Contributors to the AdsorPy project.
# SPDX-License-Identifier: MIT
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: cdivision=True
"""Optimised Cython compiled version of the cdist function with explicit contiguity flags."""

import numpy as np
from cython.parallel cimport prange


def squared_cdist(double[:, :] coords1_in, double[:, :] coords2_in) -> np.ndarray[tuple[Py_ssize_t, Py_ssize_t], np.dtype[np.float64]]:
    """Calculate the square distance between two sets of coordinates.

    Optimized via Cython with OpenMP parallelisation and C-contiguous memoryviews.
    """
    cdef double[:, ::1] coords1 = np.ascontiguousarray(coords1_in, dtype=np.float64)
    cdef double[:, ::1] coords2 = np.ascontiguousarray(coords2_in, dtype=np.float64)

    cdef Py_ssize_t dim1 = coords1.shape[1]
    cdef Py_ssize_t dim2 = coords2.shape[1]

    np_distances = np.empty((dim1, dim2), dtype=np.float64)
    cdef double[:, ::1] distances = np_distances

    cdef Py_ssize_t ii, jj
    cdef double dx, dy
    cdef double c1_x, c1_y

    # Parallel outer loop using OpenMP (retains nogil execution blocks)
    for ii in prange(dim1, nogil=True):
        c1_x = coords1[0, ii]
        c1_y = coords1[1, ii]

        for jj in range(dim2):
            dx = c1_x - coords2[0, jj]
            dy = c1_y - coords2[1, jj]
            distances[ii, jj] = dx * dx + dy * dy

    return np_distances
