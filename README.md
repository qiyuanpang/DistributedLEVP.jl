# DistributedLEVP.jl
Distributed solvers for evaluating smallest eigenpairs of large sparse symmetric matrices, one of the typical usage of this package is for dimensionality reduction in spectral clustering.

March 2023 update:

Parallel Graph Signal Filter: gsf_mpi.jl

Parallel orthogonalization-free methods: gd_mpi.jl

Power Iteration Clustering: pic_sq.jl

Numerical results show that, as a dimentionality reduction method on the synthetic graph datasets from MIT Graph Challenge (http://graphchallenge.mit.edu/data-sets), the orthogonalization-free methods significantly outperform existing methods including ARPACK, LOBPCG, Graph Signal Filter, and Powering Iteration Clustering. The paper will be ready soon.

December 2022 update:

Distributed Block Chebyshev-Davidson method: bchdav_mpi.jl

For usages, please refer to ./src/bchdav_mpi.jl and the test file ./test/testBchdav_Graph.jl.

![My Image](./scaling_bchdav.png)

The scaling of the method and its component is summarized in the plot above. The matrix tested above is the normalized Laplacian of a static graph of 2*10^7 nodes in the high block overlap low block size variation category in http://graphchallenge.mit.edu/data-sets. 
