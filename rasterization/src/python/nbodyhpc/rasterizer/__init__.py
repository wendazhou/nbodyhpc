from __future__ import annotations

import functools

import numpy as np

from ._impl import VulkanContainer as VulkanContainer
from ._impl import PointRenderer as PointRenderer


@functools.lru_cache(maxsize=None)
def get_default_container():
    """Obtain the default Vulkan container for this extension.

    The Vulkan container is the root of the Vulkan application, and
    is responsible for managing the Vulkan instance and the device.

    In general, you do not need to manage the container yourself, as one
    will be automatically created for you if you do not specify one.
    """
    return VulkanContainer(enable_validation_layers=False)


@functools.lru_cache(maxsize=None)
def _get_point_renderer_impl(grid_size: int, subsample_factor: int=4, container: VulkanContainer=None) -> PointRenderer:
    return PointRenderer(container, grid_size, subsample_factor)


def get_point_renderer(grid_size: int, subsample_factor: int=4, container: VulkanContainer=None) -> PointRenderer:
    """Obtain an instance of point renderer with the given parameters.

    Parameters
    ----------
    grid_size : int
        The output grid size of the renderer.
    subsample_factor : int
        The subsampling factor to use when rendering.
    container : VulkanContainer, optional
        If not `None`, the Vulkan container to use.
        Otherwise, automatically creates one if necessary, or fetches the default container.

    Returns
    -------
    PointRenderer
        A point renderer instance which can be used to render to the given grid size.
        Note that instances are cached for the same parameters.
    """
    if container is None:
        container = get_default_container()
    return _get_point_renderer_impl(grid_size, subsample_factor, container)


def render_points(positions: np.ndarray, weights: np.ndarray, radii: np.ndarray, box_size: float, grid_size: int, periodic: bool=False) -> np.ndarray:
    """Render points in a given slice.

    Parameters
    ----------
    positions : np.ndarray
        Numpy array of shape (N, 3) representing the positions of the points to render.
        Note that these are still 3-d positions, and the z-coordinate will be taken into account.
        The point will not be rendered if it is outside the slice.
    """
    renderer = get_point_renderer(grid_size)
    return renderer.render_points(positions, weights, radii, box_size, periodic)


def render_points_volume(positions: np.ndarray, weights: np.ndarray, radii: np.ndarray, box_size: float, grid_size: int, num_slices: int=None, periodic: bool=False, subsample_factor: int=4) -> np.ndarray:
    """Render points in a given volume.

    Parameters
    ----------
    positions : np.ndarray
        Numpy array representing the positions of the points to render.
        These are expected to be within the box.
    weights : np.ndarray
        Numpy array representing the weight of each point to render.
    radii : np.ndarray
        Numpy array representing the radius of each point to render.
    box_size : float
        Size of the box containing the data, defines the unit of length.
    grid_size : int
        Size of side of grid to use for rendering.
    num_slices : int, optional
        If not `None`, the number of slices to render (in the depth direction).
        Otherwise, sets the number of sizes to be the same as ``grid_size``.
    periodic : bool
        If `True`, indicates that the box is to be considered to be periodic,
        and so balls should correspondingly wrap around the edges.
        Note that it may not work if points have diameter larger than half the box size.
    subsample_factor : int
        Amount of subsampling to perform when rendering for anti-aliasing.
        Note that total computation in fragment shader scales as the cube of this quantity.

    Returns
    -------
    np.ndarray
        Numpy array of shape (grid_size, grid_size, grid_size) containing the rendered image.
    """
    if num_slices is None:
        num_slices = grid_size

    renderer = get_point_renderer(grid_size, subsample_factor)
    return renderer.render_points_volume(positions, weights, radii, num_slices, box_size, periodic)
