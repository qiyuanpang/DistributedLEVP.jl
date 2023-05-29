using Printf
using LinearAlgebra
using Test
using SparseArrays
using MKLSparse

using MPI, MAT, IterativeSolvers, Arpack, Clustering, DataStructures, DataFrames, Random

include("./gd/gd_mp.jl")
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
    a0 = -2.0
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

    tol = 1e-2
    for itermax in [20, 30, 40]

        V = nothing
        iter = nothing
        cputime = Dict()
        for i in 1:repeats1
            # global V, iter, cputime
            Random.seed!(1)
            X = randn(n_samples, k)
            V, iter, times = ofm_mp(A, X, n_samples, alpha=0.001, beta=0.95, cg=true, linesearch=true, locking=false, tri=true, log2deg=log2deg, itermax=itermax, tol=tol)
            if i == 1
                for (key, val) in times
                        cputime[key] = val
                end
            else
                if times["main_loop"] < cputime["main_loop"]
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
            @printf("\n\n---------------- OFM ------------------\n")
            @printf("N: %i class: %s type: %s \n", n_samples, c, t)
            @printf("node: %i blas_threads: %i k: %i tol: %.2e iter: %i itermax: %i log2deg: %i nmv: %i \n", comm_size, BLAS.get_num_threads(), k, tol, iter, itermax, log2deg, cputime["nmv"])
            @printf("Tri: true CG: true linesearch: true locking: false \n")
            @printf("relative error: %.2e \n", err)
            for ii in 1:k
                @printf("%i-th computed eigenvalue(ofm): %.2e \n", ii, evals_test[ii]-2)
            end
            @printf("walltime pre:                 %.2e \n", cputime["pre"])
            @printf("walltime main_loop:           %.2e \n", cputime["main_loop"])
            @printf("   walltime checkConvergence: %.2e \n", cputime["checkConvergence"])
            @printf("   walltime Ax:               %.2e \n", cputime["Ax"])
            @printf("   walltime updateGradient:   %.2e \n", cputime["updateGradient"])
            @printf("   walltime CG:               %.2e \n", cputime["CG"])
            @printf("   walltime linesearch:       %.2e \n", cputime["linesearch"])
            @printf("walltime QR:                  %.2e \n", cputime_qr)
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


        V = nothing
        iter = nothing
        cputime = Dict()
        for i in 1:repeats1
            # global V, iter, cputime
            Random.seed!(1)
            X = randn(n_samples, k)
            V, iter, times = ofm_mp(A, X, n_samples, alpha=0.001, beta=0.95, cg=true, linesearch=true, locking=false, tri=false, log2deg=log2deg, itermax=itermax, tol=tol)
            if i == 1
                for (key, val) in times
                        cputime[key] = val
                end
            else
                if times["main_loop"] < cputime["main_loop"]
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
            @printf("\n\n---------------- OFM ------------------\n")
            @printf("N: %i class: %s type: %s \n", n_samples, c, t)
            @printf("node: %i blas_threads: %i k: %i tol: %.2e iter: %i itermax: %i log2deg: %i nmv: %i \n", comm_size, BLAS.get_num_threads(), k, tol, iter, itermax, log2deg, cputime["nmv"])
            @printf("Tri: false CG: true linesearch: true locking: false \n")
            @printf("relative error: %.2e \n", err)
            for ii in 1:k
                @printf("%i-th computed eigenvalue(ofm): %.2e \n", ii, evals_test[ii]-2)
            end
            @printf("walltime pre:                 %.2e \n", cputime["pre"])
            @printf("walltime main_loop:           %.2e \n", cputime["main_loop"])
            @printf("   walltime checkConvergence: %.2e \n", cputime["checkConvergence"])
            @printf("   walltime Ax:               %.2e \n", cputime["Ax"])
            @printf("   walltime updateGradient:   %.2e \n", cputime["updateGradient"])
            @printf("   walltime CG:               %.2e \n", cputime["CG"])
            @printf("   walltime linesearch:       %.2e \n", cputime["linesearch"])
            @printf("walltime QR:                  %.2e \n", cputime_qr)
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

        tol_here = 1e-1
        V = nothing
        iter = nothing
        cputime = Dict()
        for i in 1:repeats1
            # global V, iter, cputime
            Random.seed!(1)
            X = randn(n_samples, k)
            V, iter, times = ofm_mp(A, X, n_samples, alpha=0.001, beta=0.95, cg=true, linesearch=true, locking=true, tri=true, log2deg=log2deg, itermax=itermax, tol=tol_here)
            if i == 1
                for (key, val) in times
                        cputime[key] = val
                end
            else
                if times["main_loop"] < cputime["main_loop"]
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
            @printf("\n\n---------------- OFM ------------------\n")
            @printf("N: %i class: %s type: %s \n", n_samples, c, t)
            @printf("node: %i blas_threads: %i k: %i tol: %.2e iter: %i itermax: %i log2deg: %i nmv: %i \n", comm_size, BLAS.get_num_threads(), k, tol_here, iter, itermax, log2deg, cputime["nmv"])
            @printf("Tri: true CG: true linesearch: true locking: true \n")
            @printf("relative error: %.2e \n", err)
            for ii in 1:k
                @printf("%i-th computed eigenvalue(ofm): %.2e \n", ii, evals_test[ii]-2)
            end
            @printf("walltime pre:                 %.2e \n", cputime["pre"])
            @printf("walltime main_loop:           %.2e \n", cputime["main_loop"])
            @printf("   walltime checkConvergence: %.2e \n", cputime["checkConvergence"])
            @printf("   walltime Ax:               %.2e \n", cputime["Ax"])
            @printf("   walltime updateGradient:   %.2e \n", cputime["updateGradient"])
            @printf("   walltime CG:               %.2e \n", cputime["CG"])
            @printf("   walltime linesearch:       %.2e \n", cputime["linesearch"])
            @printf("walltime QR:                  %.2e \n", cputime_qr)
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


        tol_here = 1e-2
        V = nothing
        iter = nothing
        cputime = Dict()
        for i in 1:repeats1
            # global V, iter, cputime
            Random.seed!(1)
            X = randn(n_samples, k)
            V, iter, times = ofm_mp(A, X, n_samples, alpha=0.001, beta=0.95, cg=true, linesearch=true, locking=true, tri=true, log2deg=log2deg, itermax=itermax, tol=tol_here)
            if i == 1
                for (key, val) in times
                        cputime[key] = val
                end
            else
                if times["main_loop"] < cputime["main_loop"]
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
            @printf("\n\n---------------- OFM ------------------\n")
            @printf("N: %i class: %s type: %s \n", n_samples, c, t)
            @printf("node: %i blas_threads: %i k: %i tol: %.2e iter: %i itermax: %i log2deg: %i nmv: %i \n", comm_size, BLAS.get_num_threads(), k, tol_here, iter, itermax, log2deg, cputime["nmv"])
            @printf("Tri: true CG: true linesearch: true locking: true \n")
            @printf("relative error: %.2e \n", err)
            for ii in 1:k
                @printf("%i-th computed eigenvalue(ofm): %.2e \n", ii, evals_test[ii]-2)
            end
            @printf("walltime pre:                 %.2e \n", cputime["pre"])
            @printf("walltime main_loop:           %.2e \n", cputime["main_loop"])
            @printf("   walltime checkConvergence: %.2e \n", cputime["checkConvergence"])
            @printf("   walltime Ax:               %.2e \n", cputime["Ax"])
            @printf("   walltime updateGradient:   %.2e \n", cputime["updateGradient"])
            @printf("   walltime CG:               %.2e \n", cputime["CG"])
            @printf("   walltime linesearch:       %.2e \n", cputime["linesearch"])
            @printf("walltime QR:                  %.2e \n", cputime_qr)
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



        V = nothing
        iter = nothing
        cputime = Dict()
        for i in 1:repeats1
            # global V, iter, cputime
            Random.seed!(1)
            X = randn(n_samples, k)
            V, iter, times = omm_mp(A, X, n_samples, alpha=0.001, beta=0.95, cg=true, linesearch=true, locking=false, tri=true, log2deg=log2deg, itermax=itermax, tol=tol)
            if i == 1
                for (key, val) in times
                        cputime[key] = val
                end
            else
                if times["main_loop"] < cputime["main_loop"]
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
            @printf("\n\n---------------- OMM ------------------\n")
            @printf("N: %i class: %s type: %s \n", n_samples, c, t)
            @printf("node: %i blas_threads: %i k: %i tol: %.2e iter: %i itermax: %i log2deg: %i nmv: %i \n", comm_size, BLAS.get_num_threads(), k, tol, iter, itermax, log2deg, cputime["nmv"])
            @printf("Tri: true CG: true linesearch: true locking: false \n")
            @printf("relative error: %.2e \n", err)
            for ii in 1:k
                @printf("%i-th computed eigenvalue(omm): %.2e \n", ii, evals_test[ii]-2)
            end
            @printf("walltime pre:                 %.2e \n", cputime["pre"])
            @printf("walltime main_loop:           %.2e \n", cputime["main_loop"])
            @printf("   walltime checkConvergence: %.2e \n", cputime["checkConvergence"])
            @printf("   walltime Ax:               %.2e \n", cputime["Ax"])
            @printf("   walltime updateGradient:   %.2e \n", cputime["updateGradient"])
            @printf("   walltime CG:               %.2e \n", cputime["CG"])
            @printf("   walltime linesearch:       %.2e \n", cputime["linesearch"])
            @printf("walltime QR:                  %.2e \n", cputime_qr)
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


        V = nothing
        iter = nothing
        cputime = Dict()
        for i in 1:repeats1
            # global V, iter, cputime
            Random.seed!(1)
            X = randn(n_samples, k)
            V, iter, times = omm_mp(A, X, n_samples, alpha=0.001, beta=0.95, cg=true, linesearch=true, locking=false, tri=false, log2deg=log2deg, itermax=itermax, tol=tol)
            if i == 1
                for (key, val) in times
                        cputime[key] = val
                end
            else
                if times["main_loop"] < cputime["main_loop"]
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
            @printf("\n\n---------------- OMM ------------------\n")
            @printf("N: %i class: %s type: %s \n", n_samples, c, t)
            @printf("node: %i blas_threads: %i k: %i tol: %.2e iter: %i itermax: %i log2deg: %i nmv: %i \n", comm_size, BLAS.get_num_threads(), k, tol, iter, itermax, log2deg, cputime["nmv"])
            @printf("Tri: false CG: true linesearch: true locking: false \n")
            @printf("relative error: %.2e \n", err)
            for ii in 1:k
                @printf("%i-th computed eigenvalue(omm): %.2e \n", ii, evals_test[ii]-2)
            end
            @printf("walltime pre:                 %.2e \n", cputime["pre"])
            @printf("walltime main_loop:           %.2e \n", cputime["main_loop"])
            @printf("   walltime checkConvergence: %.2e \n", cputime["checkConvergence"])
            @printf("   walltime Ax:               %.2e \n", cputime["Ax"])
            @printf("   walltime updateGradient:   %.2e \n", cputime["updateGradient"])
            @printf("   walltime CG:               %.2e \n", cputime["CG"])
            @printf("   walltime linesearch:       %.2e \n", cputime["linesearch"])
            @printf("walltime QR:                  %.2e \n", cputime_qr)
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


        tol_here = 1e-1
        V = nothing
        iter = nothing
        cputime = Dict()
        for i in 1:repeats1
            # global V, iter, cputime
            Random.seed!(1)
            X = randn(n_samples, k)
            V, iter, times = omm_mp(A, X, n_samples, alpha=0.001, beta=0.95, cg=true, linesearch=true, locking=true, tri=true, log2deg=log2deg, itermax=itermax, tol=tol_here)
            if i == 1
                for (key, val) in times
                        cputime[key] = val
                end
            else
                if times["main_loop"] < cputime["main_loop"]
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
            @printf("\n\n---------------- OMM ------------------\n")
            @printf("N: %i class: %s type: %s \n", n_samples, c, t)
            @printf("node: %i blas_threads: %i k: %i tol: %.2e iter: %i itermax: %i log2deg: %i nmv: %i \n", comm_size, BLAS.get_num_threads(), k, tol_here, iter, itermax, log2deg, cputime["nmv"])
            @printf("Tri: true CG: true linesearch: true locking: true \n")
            @printf("relative error: %.2e \n", err)
            for ii in 1:k
                @printf("%i-th computed eigenvalue(omm): %.2e \n", ii, evals_test[ii]-2)
            end
            @printf("walltime pre:                 %.2e \n", cputime["pre"])
            @printf("walltime main_loop:           %.2e \n", cputime["main_loop"])
            @printf("   walltime checkConvergence: %.2e \n", cputime["checkConvergence"])
            @printf("   walltime Ax:               %.2e \n", cputime["Ax"])
            @printf("   walltime updateGradient:   %.2e \n", cputime["updateGradient"])
            @printf("   walltime CG:               %.2e \n", cputime["CG"])
            @printf("   walltime linesearch:       %.2e \n", cputime["linesearch"])
            @printf("walltime QR:                  %.2e \n", cputime_qr)
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


        tol_here = 1e-2
        V = nothing
        iter = nothing
        cputime = Dict()
        for i in 1:repeats1
            # global V, iter, cputime
            Random.seed!(1)
            X = randn(n_samples, k)
            V, iter, times = omm_mp(A, X, n_samples, alpha=0.001, beta=0.95, cg=true, linesearch=true, locking=true, tri=true, log2deg=log2deg, itermax=itermax, tol=tol_here)
            if i == 1
                for (key, val) in times
                        cputime[key] = val
                end
            else
                if times["main_loop"] < cputime["main_loop"]
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
            @printf("\n\n---------------- OMM ------------------\n")
            @printf("N: %i class: %s type: %s \n", n_samples, c, t)
            @printf("node: %i blas_threads: %i k: %i tol: %.2e iter: %i itermax: %i log2deg: %i nmv: %i \n", comm_size, BLAS.get_num_threads(), k, tol_here, iter, itermax, log2deg, cputime["nmv"])
            @printf("Tri: true CG: true linesearch: true locking: true \n")
            @printf("relative error: %.2e \n", err)
            for ii in 1:k
                @printf("%i-th computed eigenvalue(omm): %.2e \n", ii, evals_test[ii]-2)
            end
            @printf("walltime pre:                 %.2e \n", cputime["pre"])
            @printf("walltime main_loop:           %.2e \n", cputime["main_loop"])
            @printf("   walltime checkConvergence: %.2e \n", cputime["checkConvergence"])
            @printf("   walltime Ax:               %.2e \n", cputime["Ax"])
            @printf("   walltime updateGradient:   %.2e \n", cputime["updateGradient"])
            @printf("   walltime CG:               %.2e \n", cputime["CG"])
            @printf("   walltime linesearch:       %.2e \n", cputime["linesearch"])
            @printf("walltime QR:                  %.2e \n", cputime_qr)
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

    end

end

GC.gc()
MPI.Finalize()