# DistributedLEVP.jl
Scalable parallel solvers for evaluating smallest eigenpairs of large sparse symmetric matrices, one of the main usages of this package is for dimensionality reduction in spectral clustering.

Available methods:

Orthogonalization-free methods: gd_mp.jl (for multithreading), gd_hybrid.jl (for multiprocessing and multithreading)

Block Chebyshev-Davidson method: bchdav_mpi.jl (multiprocessing) 

Graph Signal Filter: gsf.jl

Power Iteration Clustering: pic_sq.jl

Clustering via Power Methods: pm.jl

Papers:

[Qiyuan Pang and Haizhao Yang, Spectral Clustering via Orthogonalization-Free Methods, arXiv:2305.10356, May 2023](https://arxiv.org/abs/2305.10356)


[Qiyuan Pang and Haizhao Yang, A Distributed Block Chebyshev-Davidson Algorithm for Parallel Spectral Clustering, arXiv:2212.04443, December 2022](https://arxiv.org/abs/2212.04443)
