# Interface file for C++ rasterizer bindings.

import numpy as np


class VulkanContainer:
    def __init__(self, enable_validation_layers: bool=...) -> None: ...

class PointRenderer:
    def __init__(self, vulkan_container: VulkanContainer, grid_size: int, subsample_factor: int=...) -> None: ...

    @property
    def grid_size(self) -> int: ...

    def render_points(self, positions: np.ndarray, weights: np.ndarray, radii: np.ndarray, box_size: float=1.0, periodic: bool=False) -> np.ndarray: ...
    def render_points_volume(self, positions: np.ndarray, weights: np.ndarray, radii: np.ndarray, box_size: float=1.0, periodic: bool=False) -> np.ndarray: ...
