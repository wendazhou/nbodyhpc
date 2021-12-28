# Utilities to rasterize point clouds into density fields.

This package implements functionality to rasterize point clouds into 2d / 3d density fields.

## Quickstart

You will need the [Vulkan SDK](https://www.lunarg.com/vulkan-sdk/) to build the package.
If you have access to rusty, you may also find a compiled version directly at
`/mnt/ceph/users/wzhou/wheels/nbodyhpc_rasterizer-0.0.1-cp38-cp38-linux_x86_64.whl`.

The main functionality is accessed through the `render_points_volume` function:
```{python}
import numpy as np
from nbodyhpc.rasterizer import render_points_volume

n_points = 10
positions = np.random.uniform(size=(n_points, 3))
radii = np.random.uniform(size=(n_points,))
weights = np.ones_like(radii)

result = render_points_volume(positions, weights, radii, box_size=1, grid_size=64)
print(result.shape)
```
For many applications, you may also wish to consider the `periodic` parameter to the `render_points_volume` function.


## Performance

Currently, the performance for the rendering (after vertex pre-processing for sorting and periodic boundary conditions)
takes about 2.5 seconds on a RTX6000 to rasterize a Camels simulation (256^3 points) at a grid resolution of 1024^3
with 16 samples per voxel.

