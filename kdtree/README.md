# Fast Spatial KD-tree

This module implements a fast spatial kd-tree for nearest neighbor computation in 3 dimensions.
In addition to the standard KD-tree, this implementation also contains functionality to look-up in the kd-tree
using periodic boundary conditions.

## Optimizations

This implementation contains two main optimizations for better performance.
1. Tree building. During tree-building, an optimized selection (median-finding) algorithm is used.
   It is based on the Floyd-Rivest algorithm, and makes use of an AVX2-optimized partitioning function.
2. KNN lookup. During lookup, an optimized tournament tree with branchless updates is used to keep track
   of the best points. We also make use of AVX2 optimizations to accelerate distance computations.
