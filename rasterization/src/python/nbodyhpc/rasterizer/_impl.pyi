# Interface file for C++ rasterizer bindings.

from typing import List
import numpy as np


class VulkanContainer:
    def __init__(self, enable_validation_layers: bool=...) -> None: ...

class PointRenderer:
    def __init__(self, vulkan_container: VulkanContainer, width: int, height: int, subsample_factor: int=...) -> None: ...

    @property
    def grid_size(self) -> int: ...

    def render_points(self, positions: np.ndarray, weights: np.ndarray, radii: np.ndarray, pixels_per_unit: float=1.0, periodic: List[float]=...) -> np.ndarray: ...
    def render_points_volume(self, positions: np.ndarray, weights: np.ndarray, radii: np.ndarray, num_slices: int, pixels_per_unit: float=1.0, periodic: List[float]=...) -> np.ndarray: ...
