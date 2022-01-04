from __future__ import annotations

import warnings
from typing import Tuple

import numpy as np

from ._impl import KDTree as cKDTree


class KDTree(cKDTree):
    """Spatial KD-tree, with optional periodic boundary conditions.

    This class implements an optimized KD-tree for spatial data (3D), with optional
    support for periodic boundary conditions.
    """
    def __init__(self, points: np.ndarray, leafsize: int=10, max_threads: int=-1, boxsize: float=None, **kwargs):
        """Build a new KDTree.

        Parameters
        ----------
        points : np.ndarray
            A (N, 3) array of N points in 3 dimensions.
        leafsize : int
            The number of points in a leaf node (where we switch to brute-force search).
        max_threads : int
            The maximum number of threads to use during construction.
            -1 indicates that all available threads may be used.
        boxsize : float, optional
            If not `None`, a floating point value representing the size of the box
            to use.
        """
        super().__init__(points, leafsize, max_threads, boxsize)

        if len(kwargs) > 0:
            warnings.warn("Unrecognized keyword arguments: {}".format(kwargs))

    def query(self, points: np.ndarray, k: int=1, workers: int=1, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        if len(kwargs) > 0:
            warnings.warn("Unrecognized keyword arguments: {}".format(kwargs))

        if len(points.shape) != 2:
            shape = points.shape
            points = points.reshape((-1, shape[-1]))
        else:
            shape = None

        distances, indices = super().query(points, k, workers)

        if shape is not None:
            distances = distances.reshape(shape[:-1], k)
            indices = indices.reshape(shape[:-1], k)

        return distances, indices
