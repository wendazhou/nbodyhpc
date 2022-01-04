import numpy as np

from nbodyhpc import kdtree
import scipy.spatial

def test_kdtree_basic():
    rng = np.random.Generator(np.random.PCG64(42))

    points = rng.uniform(0, 1, size=(10000, 3))
    query_points = rng.uniform(0, 1, size=(200, 3))

    tree = kdtree.KDTree(points)
    tree_reference = scipy.spatial.KDTree(points)

    distances_ref, indices_ref = tree_reference.query(query_points, k=4)
    distances, indices = tree.query(query_points, k=4)

    assert np.allclose(distances_ref, distances)
    assert np.all(indices_ref == indices)

def test_kdtree_periodic():
    rng = np.random.Generator(np.random.PCG64(42))
    boxsize = 2.0

    points = rng.uniform(0, boxsize, size=(10000, 3)).astype(np.float32)
    query_points = rng.uniform(0, boxsize, size=(200, 3)).astype(np.float32)

    tree = kdtree.KDTree(points, boxsize=boxsize)
    tree_reference = scipy.spatial.KDTree(points, boxsize=boxsize)

    distances_ref, indices_ref = tree_reference.query(query_points, k=4)
    distances, indices = tree.query(query_points, k=4)

    assert np.allclose(distances_ref, distances)
    assert np.all(indices_ref == indices)
