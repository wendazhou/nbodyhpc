from __future__ import annotations

import functools
from typing import Union, Tuple

import numpy as np

from ._impl import VulkanContainer as VulkanContainer
from ._impl import PointRenderer as PointRenderer

Extent2d = Union[int, Tuple[int, int]]

def _normalize_extent(extent: Extent2d) -> Tuple[int, int]:
    if isinstance(extent, int):
        return extent, extent
    else:
        return extent


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
def _get_point_renderer_impl(width: int, height: int, subsample_factor: int=4, container: VulkanContainer=None) -> PointRenderer:
    return PointRenderer(container, width, height, subsample_factor)


def get_point_renderer(grid_size: Extent2d, subsample_factor: int=4, container: VulkanContainer=None) -> PointRenderer:
    """Obtain an instance of point renderer with the given parameters.

    Parameters
    ----------
    grid_size : int or [int, int]
        The output grid size of the renderer, either as a square grid, or a
        tuple of width and height.
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

    height, width = _normalize_extent(grid_size)
    return _get_point_renderer_impl(width, height, subsample_factor, container)


def render_points(positions: np.ndarray, weights: np.ndarray, radii: np.ndarray, pixels_per_unit: float, grid_size: Extent2d, periodic: bool=False) -> np.ndarray:
    """Render points in a given slice.

    Parameters
    ----------
    positions : np.ndarray
        Numpy array of shape (N, 3) representing the positions of the points to render.
        Note that these are still 3-d positions, and the z-coordinate will be taken into account.
        The point will not be rendered if it is outside the slice.
    """
    renderer = get_point_renderer(grid_size)
    return renderer.render_points(positions, weights, radii, pixels_per_unit, periodic)


def render_points_volume(positions: np.ndarray, weights: np.ndarray, radii: np.ndarray, pixels_per_unit: float, grid_size: Extent2d, num_slices: int=None, periodic: bool=False, subsample_factor: int=4) -> np.ndarray:
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
    pixels_per_unit : float
        Number of pixels per unit of position.
    grid_size : int
        Size of side of grid to use for rendering.
    num_slices : int, optional
        If not `None`, the number of slices to render (in the depth direction).
        Otherwise, sets the number of sizes to be the same as ``grid_size``.
        Setting ``num_slices`` to a value less than ``grid_size`` corresponds to truncating
        the rendered volume.
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
        num_slices, _ = _normalize_extent(grid_size)

    renderer = get_point_renderer(grid_size, subsample_factor)
    return renderer.render_points_volume(positions, weights, radii, num_slices, pixels_per_unit, periodic)
