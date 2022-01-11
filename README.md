# High-performance libraries for working with N-body data

[![Python package](https://github.com/wendazhou/nbodyhpc/actions/workflows/python-package.yml/badge.svg)](https://github.com/wendazhou/nbodyhpc/actions/workflows/python-package.yml)

This package includes a number of high-performance libraries intended to facilitate certain workflows
for n-body simulation data, particularly in cosmology.

## Rasterizer

The [rasterizer](rasterization/) library is a Vulkan-based GPU rasterization library intended to help
compute 3d density fields from point-cloud data.

## Kd-tree

The [kdtree](kdtree/) library implements an efficient kd-tree in three dimensions, with the possibility of using periodic boundary conditions.
