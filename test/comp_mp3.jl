using Printf
using LinearAlgebra
using Test
using SparseArrays
using MKLSparse

using MPI, MAT, IterativeSolvers, Arpack, Clustering, DataStructures, DataFrames, Random

include("./pm/pm.jl")
include("./gsf/gsf.jl")
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
    repeats2 = 2
    a0 = 0.0
    upb = 2.0 + a0
    lwb = 0.0 + a0
    log2deg = 0

    file1 = "./graph/" * c * "/" * t * "/" * string(n_samples) * "/" * t * "_" * name * "_" * string(n_samples) * "_nodes_L.mat"
    f1 = matopen(file1)
    A = read(f1, "data")
    W = sparse([ir for ir in 1:n_samples], [ir for ir in 1:n_samples], ones(n_samples), n_samples, n_samples) - A
    W = (W+W')/2.0
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

    evals_arpack, eigV_arpack, nconv, niter, nmult = eigs(A, nev=k, which=:SR, tol=1e-8, maxiter=200)

    for itermax in [10]
        for p in [30, 40]

            V = nothing
            iter = nothing
            lambdaest = nothing
            cputime = Dict()
            for i in 1:repeats1
                # global V, iter, cputime
                Random.seed!(1)
                V, lambdaest, iter, times = gsf(A, k; lambdaMin=lwb, lambdaMax=upb, itermax=itermax, p=p, p2=p, estkth=maximum(evals_arpack)-a0)
                if i == 1
                    for (key, val) in times
                            cputime[key] = val
                    end
                else
                    if times["main"] < cputime["main"]
                        for (key, val) in times
                            cputime[key] = val
                        end
                    end
                end
            end
            eigV_test = nothing
            evals_test = nothing
            err = nothing
            cputime_qr = 0
            for i in 1:repeats1
                eigV_test, evals_test, err, times = evalEigen(A4err, V)
                cputime_qr += times/repeats1
            end
            if rank == 0
                @printf("\n\n---------------- GSF ------------------\n")
                @printf("N: %i class: %s type: %s \n", n_samples, c, t)
                @printf("node: %i blas_threads: %i k: %i \n", comm_size, BLAS.get_num_threads(), k)
                @printf("ideal: true kth-eval: %.2e itermax: %i iter: %i p: %i \n", lambdaest, itermax, iter, p)
                @printf("relative error: %.2e \n", err)
                for ii in 1:k
                    @printf("%i-th computed eigenvalue(gsf): %.2e \n", ii, evals_test[ii]-2)
                end
                @printf("walltime pre:                 %.2e \n", cputime["pre"])
                @printf("walltime main:                %.2e \n", cputime["main"])
                @printf("   walltime estimate:         %.2e \n", cputime["estimate"])
                @printf("       walltime poly:         %.2e \n", cputime["poly"])
                @printf("   walltime M:                %.2e \n", cputime["M"])
                @printf("   walltime SVD:              %.2e \n", cputime["SVD"])
            end
            if rownorm
                d = 1 ./ sqrt.(sum(V .^ 2, dims=2))
                D = Diagonal(vec(d))
                lmul!(D, V)
            end
            (ari, ri, mi, vi, vm) = (0,0,0,0,0)
            for i in 1:repeats2
                # global a, ari, ri, mi, vi, vm
                VT = Matrix(transpose(V))
                R = kmeans(VT, numcl; maxiter=300)
                a = assignments(R)
                (ari1, ri1, _, _) = Clustering.randindex(a, Y)
                mi1 = Clustering.mutualinfo(a, Y)
                vi1 = Clustering.varinfo(a, Y)
                vm1 = Clustering.vmeasure(a, Y)
                ari, ri, mi, vi, vm = ari+ari1/repeats2, ri+ri1/repeats2, mi+mi1/repeats2, vi+vi1/repeats2, vm+vm1/repeats2
            end
            if rank == 0
                @printf("clustering results: RI = %.3e ARI = %.3e MI = %.3e VI = %.3e VM = %.3e \n", ri, ari, mi, vi, vm)
            end
            energy = 1/k*norm(eigV_test'*eigV_arpack)^2
            @printf("energy: %.2e \n\n", energy)



            V = nothing
            iter = nothing
            lambdaest = nothing
            cputime = Dict()
            for i in 1:repeats1
                # global V, iter, cputime
                Random.seed!(1)
                V, lambdaest, iter, times = gsf(A, k; lambdaMin=lwb, lambdaMax=upb, itermax=itermax, p=p, p2=p, estkth=Inf)
                if i == 1
                    for (key, val) in times
                            cputime[key] = val
                    end
                else
                    if times["main"] < cputime["main"]
                        for (key, val) in times
                            cputime[key] = val
                        end
                    end
                end
            end
            eigV_test = nothing
            evals_test = nothing
            err = nothing
            cputime_qr = 0
            for i in 1:repeats1
                eigV_test, evals_test, err, times = evalEigen(A4err, V)
                cputime_qr += times/repeats1
            end
            if rank == 0
                @printf("\n\n---------------- GSF ------------------\n")
                @printf("N: %i class: %s type: %s \n", n_samples, c, t)
                @printf("node: %i blas_threads: %i k: %i \n", comm_size, BLAS.get_num_threads(), k)
                @printf("ideal: false kth-eval: %.2e itermax: %i iter: %i p: %i \n", lambdaest, itermax, iter, p)
                @printf("relative error: %.2e \n", err)
                for ii in 1:k
                    @printf("%i-th computed eigenvalue(gsf): %.2e \n", ii, evals_test[ii]-2)
                end
                @printf("walltime pre:                 %.2e \n", cputime["pre"])
                @printf("walltime main:                %.2e \n", cputime["main"])
                @printf("   walltime estimate:         %.2e \n", cputime["estimate"])
                @printf("       walltime poly:         %.2e \n", cputime["poly"])
                @printf("   walltime M:                %.2e \n", cputime["M"])
                @printf("   walltime SVD:              %.2e \n", cputime["SVD"])
            end
            if rownorm
                d = 1 ./ sqrt.(sum(V .^ 2, dims=2))
                D = Diagonal(vec(d))
                lmul!(D, V)
            end
            (ari, ri, mi, vi, vm) = (0,0,0,0,0)
            for i in 1:repeats2
                # global a, ari, ri, mi, vi, vm
                VT = Matrix(transpose(V))
                R = kmeans(VT, numcl; maxiter=300)
                a = assignments(R)
                (ari1, ri1, _, _) = Clustering.randindex(a, Y)
                mi1 = Clustering.mutualinfo(a, Y)
                vi1 = Clustering.varinfo(a, Y)
                vm1 = Clustering.vmeasure(a, Y)
                ari, ri, mi, vi, vm = ari+ari1/repeats2, ri+ri1/repeats2, mi+mi1/repeats2, vi+vi1/repeats2, vm+vm1/repeats2
            end
            if rank == 0
                @printf("clustering results: RI = %.3e ARI = %.3e MI = %.3e VI = %.3e VM = %.3e \n", ri, ari, mi, vi, vm)
            end
            energy = 1/k*norm(eigV_test'*eigV_arpack)^2
            @printf("energy: %.2e \n\n", energy)
        end


    end

    evals_arpack, eigV_arpack, nconv, niter, nmult = eigs(W, nev=k, which=:LR, tol=1e-8, maxiter=200)
    eigV_arpack = eigV_arpack[:, k:-1:1]
    evals_arpack = evals_arpack[k:-1:1]
    
    for p in [61, 71, 81]
        V = nothing
        iter = nothing
        cputime = Inf
        for i in 1:repeats1
            # global V, iter, cputime
            Random.seed!(1)
            V, time = pm(W, k, p)
            cputime = min(cputime, time)
        end
        eigV_test = nothing
        evals_test = nothing
        err = nothing
        cputime_qr = 0
        for i in 1:repeats1
            eigV_test, evals_test, err, times = evalEigen(A4err, V)
            cputime_qr += times/repeats1
        end
        if rank == 0
            @printf("\n\n---------------- PM ------------------\n")
            @printf("N: %i class: %s type: %s \n", n_samples, c, t)
            @printf("node: %i blas_threads: %i k: %i \n", comm_size, BLAS.get_num_threads(), k)
            @printf("p: %i \n", p)
            @printf("relative error: %.2e \n", err)
            for ii in 1:k
                @printf("%i-th computed eigenvalue(pm): %.2e \n", ii, evals_test[ii]-2)
            end
            @printf("walltime main:                 %.2e \n", cputime)
        end
        if rownorm
            d = 1 ./ sqrt.(sum(V .^ 2, dims=2))
            D = Diagonal(vec(d))
            lmul!(D, V)
        end
        (ari, ri, mi, vi, vm) = (0,0,0,0,0)
        for i in 1:repeats2
            # global a, ari, ri, mi, vi, vm
            VT = Matrix(transpose(V))
            R = kmeans(VT, numcl; maxiter=300)
            a = assignments(R)
            (ari1, ri1, _, _) = Clustering.randindex(a, Y)
            mi1 = Clustering.mutualinfo(a, Y)
            vi1 = Clustering.varinfo(a, Y)
            vm1 = Clustering.vmeasure(a, Y)
            ari, ri, mi, vi, vm = ari+ari1/repeats2, ri+ri1/repeats2, mi+mi1/repeats2, vi+vi1/repeats2, vm+vm1/repeats2
        end
        if rank == 0
            @printf("clustering results: RI = %.3e ARI = %.3e MI = %.3e VI = %.3e VM = %.3e \n", ri, ari, mi, vi, vm)
        end
        energy = 1/k*norm(eigV_test'*eigV_arpack)^2
        @printf("energy: %.2e \n\n", energy)

    end

end

GC.gc()
MPI.Finalize()