from __future__ import annotations

import functools
from typing import Union, Tuple

import numpy as np

from ._impl import VulkanContainer as VulkanContainer
from ._impl import PointRenderer as PointRenderer

Extent2d = Union[int, Tuple[int, int]]
Extent3d = Union[int, Tuple[int, int, int]]
PeriodT = Union[bool, float, Tuple[float, float, float]]

def _normalize_extent_2d(extent: Extent2d) -> Tuple[int, int]:
    if isinstance(extent, int):
        return extent, extent
    else:
        return extent

def _normalize_extent_3d(extent: Extent3d) -> Tuple[int, int, int]:
    if isinstance(extent, int):
        return extent, extent, extent
    else:
        return extent

def _normalize_period(deduced: Tuple[float, float, float], period: PeriodT) -> Tuple[float, float, float]:
    if isinstance(period, bool):
        if period:
            return deduced
        else:
            return (-1.0, -1.0, -1.0)
    elif isinstance(period, float):
        return period, period, period
    else:
        if len(period) == 2:
            # handle case for 2d
            return period[0], period[1], -1.0
        return period


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

    height, width = _normalize_extent_2d(grid_size)
    return _get_point_renderer_impl(width, height, subsample_factor, container)


def render_points(positions: np.ndarray, weights: np.ndarray, radii: np.ndarray, pixels_per_unit: float, grid_size: Extent2d, periodic: PeriodT=False) -> np.ndarray:
    """Render points in a given slice.

    Parameters
    ----------
    positions : np.ndarray
        Numpy array of shape (N, 3) representing the positions of the points to render.
        Note that these are still 3-d positions, and the z-coordinate will be taken into account.
        The point will not be rendered if it is outside the slice.
    """
    grid_x, grid_y = _normalize_extent_2d(grid_size)
    renderer = get_point_renderer((grid_x, grid_y))
    deduced = grid_x / pixels_per_unit, grid_y / pixels_per_unit, -1.0
    period = _normalize_period(deduced, periodic)
    return renderer.render_points(positions, weights, radii, pixels_per_unit, period)


def render_points_volume(positions: np.ndarray, weights: np.ndarray, radii: np.ndarray, pixels_per_unit: float,
                         grid_size: Extent3d, periodic: PeriodT=False, subsample_factor: int=4) -> np.ndarray:
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
    grid_size : int or [int, int, int]
        Size of side of grid to use for rendering. If a single integer, the grid will be a cube,
        otherwise, the grid will have the number of pixels in the given dimensions, in x, y, and z order.
    periodic : bool or float or [float, float, float]
        - If `True`, indicates that the box is to be considered to be periodic,
          and so balls should correspondingly wrap around the edges, with the box size to be deduced from the grid size.
        - If a positive floating point value, indicates the cubic box size to use.
        - If a tuple of three floating point values, indicates the box size to use in each dimension.
          In this case, negative values indicate that the given dimension is not periodic.
        Note that it may not work if points have diameter larger than half the box size.
    subsample_factor : int
        Amount of subsampling to perform when rendering for anti-aliasing.
        Note that total computation in fragment shader scales as the cube of this quantity.

    Returns
    -------
    np.ndarray
        Numpy array of shape (grid_size, grid_size, grid_size) containing the rendered image.
    """
    grid_x, grid_y, num_slices = _normalize_extent_3d(grid_size)
    deduced_box = grid_x / pixels_per_unit, grid_y / pixels_per_unit, num_slices / pixels_per_unit
    period = _normalize_period(deduced_box, periodic)

    renderer = get_point_renderer((grid_x, grid_y), subsample_factor)
    return renderer.render_points_volume(positions, weights, radii, num_slices, pixels_per_unit, period)
