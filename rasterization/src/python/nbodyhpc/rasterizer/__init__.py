from __future__ import annotations

import functools

import numpy as np

from ._impl import VulkanContainer as VulkanContainer
from ._impl import PointRenderer as PointRenderer


@functools.lru_cache(maxsize=None)
def get_default_container():
    return VulkanContainer(enable_validation_layers=False)


@functools.lru_cache(maxsize=None)
def get_point_renderer(grid_size: int, subsample_factor: int=4, container: VulkanContainer=None) -> PointRenderer:
    if container is None:
        container = get_default_container()
        return get_point_renderer(grid_size, subsample_factor, container)
    return PointRenderer(container, grid_size, subsample_factor)


def render_points_volume(positions: np.ndarray, weights: np.ndarray, radii: np.ndarray, box_size: float, grid_size: int, periodic: bool=False, subsample_factor: int=4):
    renderer = get_point_renderer(grid_size, subsample_factor)
    return renderer.render_points_volume(positions, weights, radii, box_size, periodic)
