using Printf
using LinearAlgebra
using Test
using SparseArrays
using MKLSparse

using MPI, MAT, IterativeSolvers, Arpack, Clustering, DataStructures, DataFrames, Random

include("./utils/utils.jl")

# only one process/rank 0
MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
comm_size = MPI.Comm_size(comm)

for code in 1:1

    n_samples = parse(Int64, ARGS[1])
    k = parse(Int64, ARGS[2])
    c = ARGS[3]
    t = ARGS[4]
    name = ARGS[5]

    rownorm = true
    repeats1 = 3
    repeats2 = 3
    a0 =  2.0
    upb = 2.0 + a0
    lwb = 0.0 + a0
    log2deg = 0

    file1 = "./graph/" * c * "/" * t * "/" * string(n_samples) * "/" * t * "_" * name * "_" * string(n_samples) * "_nodes_L.mat"
    f1 = matopen(file1)
    A = read(f1, "data")
    A4err = A + sparse([ir for ir in 1:n_samples], [ir for ir in 1:n_samples], 2.0 .* ones(n_samples), n_samples, n_samples)
    A4err = (A4err+A4err')/2.0
    A += sparse([ir for ir in 1:n_samples], [ir for ir in 1:n_samples], a0 .* ones(n_samples), n_samples, n_samples)
    A = (A+A')/2.0

    filename_truth = "./graph/" * c * "/" * t * "/" * string(n_samples) * "/" * t * "_" * name * "_" * string(n_samples) * "_nodes_truePartition.mat"
    file = matopen(filename_truth)
    Y = read(file, "data")
    Iy, Jy, Vy = findnz(Y)
    Y = vec(Matrix(sparse(Iy, Jy, Vy, n_samples, 1)))
    # numcl = maximum(Y) - minimum(Y) + 1
    numcl = length(counter(Y))

    for tol in [1e-1, 1e-2, 1e-3]

        eigV_arpack = nothing
        evals_arpack = nothing
        nconv = 0
        niter = 0
        nmult = 0
        time_arpack = Inf
        for i in 1:repeats1
            time_here = @elapsed begin
            evals_arpack, eigV_arpack, nconv, niter, nmult = eigs(A, nev=k, which=:SM, tol=tol, maxiter=200)
            end
            time_arpack = min(time_arpack, time_here)
        end
        # err = norm(A*eigV_arpack - eigV_arpack*Diagonal(evals_arpack))/norm(eigV_arpack*Diagonal(evals_arpack))
        _, _, err, _ = evalEigen(A4err, eigV_arpack)
        if rank == 0
            @printf("------------ ARPACK ---------------- \n")
            @printf("N: %i class: %s type: %s \n", n_samples, c, t)
            @printf("relative error: %.2e tol: %.2e k: %i nconv: %i niter: %i, nmult: %i \n", err, tol, k, nconv, niter, nmult)
            @printf("time: %.2e node: %i blas_threads: %i \n", time_arpack, comm_size, BLAS.get_num_threads())
            for ii in 1:k
                @printf("%i-th computed eigenvalue(arpack): %.2e \n", ii, evals_arpack[ii]-a0)
            end
        end
        if rownorm
            d = 1 ./ sqrt.(sum(eigV_arpack .^ 2, dims=2))
            D = Diagonal(vec(d))
            lmul!(D, eigV_arpack)
        end
        (ari, ri, mi, vi, vm) = (0,0,0,0,0)
        for i in 1:repeats2
            # global a, ari, ri, mi, vi, vm
            VT = Matrix(transpose(eigV_arpack))
            R = kmeans(VT, numcl; maxiter=300)
            a = assignments(R)
            (ari1, ri1, _, _) = Clustering.randindex(a, Y)
            mi1 = Clustering.mutualinfo(a, Y)
            vi1 = Clustering.varinfo(a, Y)
            vm1 = Clustering.vmeasure(a, Y)
            ari, ri, mi, vi, vm = ari+ari1/repeats2, ri+ri1/repeats2, mi+mi1/repeats2, vi+vi1/repeats2, vm+vm1/repeats2
        end
        if rank == 0
            @printf("clustering results: RI = %.3e ARI = %.3e MI = %.3e VI = %.3e VM = %.3e \n\n", ri, ari, mi, vi, vm)
        end



        eigV_lobpcg = nothing
        evals_lobpcg = nothing
        time_lobpcg = Inf
        X0 = randn(n_samples, k)
        for i in 1:repeats1
            time_here = @elapsed begin
            F = lobpcg(A, false, X0; tol=tol, maxiter=200)
            evals_lobpcg = F.λ
            eigV_lobpcg = F.X
            end
            time_lobpcg = min(time_lobpcg, time_here)
        end
        # err = norm(A*eigV_lobpcg - eigV_lobpcg*Diagonal(evals_lobpcg))/norm(eigV_lobpcg*Diagonal(evals_lobpcg))
        _, _, err, _ = evalEigen(A4err, eigV_lobpcg)
        if rank == 0
            @printf("------------ LOBPCG ---------------- \n")
            @printf("N: %i class: %s type: %s \n", n_samples, c, t)
            @printf("relative error: %.2e tol: %.2e k: %i \n", err, tol, k)
            @printf("time: %.2e node: %i blas_threads: %i \n", time_lobpcg, comm_size, BLAS.get_num_threads())
            for ii in 1:k
                @printf("%i-th computed eigenvalue(lobpcg): %.2e \n", ii, evals_lobpcg[ii]-a0)
            end
        end
        if rownorm
            d = 1 ./ sqrt.(sum(eigV_lobpcg .^ 2, dims=2))
            D = Diagonal(vec(d))
            lmul!(D, eigV_lobpcg)
        end
        (ari, ri, mi, vi, vm) = (0,0,0,0,0)
        for i in 1:repeats2
            # global a, ari, ri, mi, vi, vm
            VT = Matrix(transpose(eigV_lobpcg))
            R = kmeans(VT, numcl; maxiter=300)
            a = assignments(R)
            (ari1, ri1, _, _) = Clustering.randindex(a, Y)
            mi1 = Clustering.mutualinfo(a, Y)
            vi1 = Clustering.varinfo(a, Y)
            vm1 = Clustering.vmeasure(a, Y)
            ari, ri, mi, vi, vm = ari+ari1/repeats2, ri+ri1/repeats2, mi+mi1/repeats2, vi+vi1/repeats2, vm+vm1/repeats2
        end
        if rank == 0
            @printf("clustering results: RI = %.3e ARI = %.3e MI = %.3e VI = %.3e VM = %.3e \n\n", ri, ari, mi, vi, vm)
        end



    end

end

GC.gc()
MPI.Finalize()