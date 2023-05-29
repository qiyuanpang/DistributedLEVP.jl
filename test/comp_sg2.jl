using MPI, MAT, SparseArrays, LinearAlgebra, Printf, Random, CSV, DataFrames, Test, IterativeSolvers, Clustering, DataStructures, Arpack

include("./gd/gd_mpi.jl")

for code in 1:1
    n_samples = parse(Int64, ARGS[1])
    k = parse(Int64, ARGS[2])
    itermax1 = parse(Int64, ARGS[3])
    itermax2 = parse(Int64, ARGS[4])
    c = ARGS[5]
    t = ARGS[6]
    name = ARGS[7]
    tol = 1e-8

    rownorm = true
    repeats1 = 1
    repeats2 = 1
    a0 = -2.0
    upb = 2.0 + a0
    lwb = 0.0 + a0
    log2deg = 0


    MPI.Init()

    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    comm_size = MPI.Comm_size(comm)
    comm_size_sq = trunc(Int64, sqrt(comm_size))

    root = 0

    coords = [mod(rank, comm_size_sq), trunc(Int64, rank/comm_size_sq)]
    comm_col = MPI.Comm_split(comm, trunc(Int64, rank/comm_size_sq), rank)
    rank_col = MPI.Comm_rank(comm_col)
    comm_row = MPI.Comm_split(comm, mod(rank, comm_size_sq), rank)
    rank_row = MPI.Comm_rank(comm_row)
    r2c, counts_info, l2r, process_tree, process_tree_lvl = setup_process_tree(comm_size)
    comms_tree = create_communicators(r2c, rank, comm)

    info_cols_dist, S = split_count_local(n_samples, comm_size_sq) 


    file1 = "./graph/" * c * "/" * t * "/" * string(n_samples) * "/" * t * "_" * name * "_" * string(n_samples) * "_nodes_"
    # f = matopen(file)
    # counts = read(f, "data")

    filename_truth = "./graph/" * c * "/" * t * "/" * string(n_samples) * "/" * t * "_" * name * "_" * string(n_samples) * "_nodes_truePartition.mat"
    file = matopen(filename_truth)
    Y = read(file, "data")
    Iy, Jy, Vy = findnz(Y)
    Y = vec(Matrix(sparse(Iy, Jy, Vy, n_samples, 1)))
    # numcl = maximum(Y) - minimum(Y) + 1
    numcl = length(counter(Y))


    comm_info = Dict()
    comm_info["comm"] = comm
    # comm_T = comm_info["comm_T"]
    comm_info["comm_row"] = comm_row
    comm_info["comm_col"] = comm_col
    comm_info["rank"] = rank
    comm_info["rank_row"] = rank_row
    comm_info["rank_col"] = rank_col
    comm_info["info_cols_dist"] = info_cols_dist
    comm_info["comm_size"] = comm_size
    comm_info["comm_size_sq"] = comm_size_sq
    
    rows = [i for i in 1:n_samples]
    cols = [i for i in 1:n_samples]
    E = sparse(rows, cols, [1.0 for i in 1:n_samples], n_samples, n_samples)
    transtion = sparse(rows, cols, a0 .* ones(n_samples), n_samples, n_samples) 


    V_gather = randn(k, n_samples)
    local_A = sparse([1], [1], [0.0], n_samples, n_samples)
    V0 = randn(k, n_samples)
    time_mainloop_cum = 0.0
    for gh in [1,2,3,4,5,6,7,8,9,10]
        f1 = matopen(file1 * string(gh) * "_A.mat")
        A_gh = read(f1, "data")
        local_A += A_gh
        symmetric_A = (local_A+local_A')/2
        d = sqrt.(vec(sum(symmetric_A, dims=1)))
        d2 = zeros(n_samples)
        for ni in 1:n_samples
            if abs(d[ni]) > 0
                d2[ni] = 1/d[ni]
            end
        end
        D2 = sparse(rows, cols, d2, n_samples, n_samples)
        local_L = E - D2*symmetric_A*D2

        local_L += transtion  

        if gh == 1
            itermax = itermax1
        else
            itermax = itermax2
        end

        V = nothing
        iter = nothing
        cputime = Dict()
        for i in 1:repeats1
            V, iter, times = ofm_mpi_sq(local_L, V0, n_samples, comm_info, alpha=0.001, beta=0.95, cg=true, linesearch=true, locking=false, tri=true, log2deg=log2deg, itermax=itermax, tol=tol)
            for (key, val) in times
                if haskey(cputime, key)
                    cputime[key] += val*1.0/repeats1
                else
                    cputime[key] = val*1.0/repeats1
                end
            end
        end
        V0 .= V
        eigV_test = nothing
        evals_test = nothing
        err = nothing
        cputime_qr = 0
        for i in 1:repeats1
            # global cputime_qr, eigV_test, evals_test, err
            eigV_test, evals_test, err, times = rayleigh_local(local_L, V, n_samples, process_tree_lvl, comms_tree, counts_info, info_cols_dist, rank, rank_row, rank_col, comm_row, comm_col, root, comm_size_sq)
            cputime_qr += times/repeats1
        end
        time_mainloop_cum += cputime["main_loop"]
        if rank == 0
            @printf("---------------- OFM ------------------\n")
            @printf("N: %i class: %s type: %s sub-graph: %i \n", n_samples, c, t, gh)
            @printf("node: %i k: %i tol: %.2e iter: %i itermax: %i log2deg: %i nmv: %i \n", comm_size, k, tol, iter, itermax, log2deg, cputime["nmv"])
            @printf("Tri: true CG: true linesearch: true locking: false \n")
            @printf("relative error: %.2e \n", err)
            for ii in 1:k
                @printf("%i-th computed eigenvalue(ofm): %.2e \n", ii, evals_test[ii]-a0)
            end
            @printf("walltime pre:                 %.2e \n", cputime["pre"])
            @printf("walltime main_loop:           %.2e %.2e \n", cputime["main_loop"], time_mainloop_cum)
            @printf("   walltime checkConvergence: %.2e \n", cputime["checkConvergence"])
            @printf("   walltime Ax:               %.2e \n", cputime["Ax"])
            @printf("   walltime updateGradient:   %.2e \n", cputime["updateGradient"])
            @printf("   walltime CG:               %.2e \n", cputime["CG"])
            @printf("   walltime linesearch:       %.2e \n", cputime["linesearch"])
            @printf("walltime QR:                  %.2e \n", cputime_qr)
        end

        V_gather .= V
        if rownorm
            d = 1 ./ sqrt.(sum(V_gather .^ 2, dims=1))
            D = spdiagm(n_samples, n_samples, vec(d))
            V_gather .= V_gather*D
        end
        (ari, ri, mi, vi, vm) = (0,0,0,0,0)
        for i in 1:repeats2
            # global a, ari, ri, mi, vi, vm
            R = kmeans(V_gather, numcl; maxiter=300)
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
    @printf("\n\n")


    V_gather = randn(k, n_samples)
    local_A = sparse([1], [1], [0.0], n_samples, n_samples)
    V0 = randn(k, n_samples)
    time_mainloop_cum = 0.0
    for gh in [1,2,3,4,5,6,7,8,9,10]
        f1 = matopen(file1 * string(gh) * "_A.mat")
        A_gh = read(f1, "data")
        local_A += A_gh
        symmetric_A = (local_A+local_A')/2
        d = sqrt.(vec(sum(symmetric_A, dims=1)))
        d2 = zeros(n_samples)
        for ni in 1:n_samples
            if abs(d[ni]) > 0
                d2[ni] = 1/d[ni]
            end
        end
        D2 = sparse(rows, cols, d2, n_samples, n_samples)
        local_L = E - D2*symmetric_A*D2

        local_L += transtion  

        if gh == 1
            itermax = itermax1
        else
            itermax = itermax2
        end

        V = nothing
        iter = nothing
        cputime = Dict()
        for i in 1:repeats1
            V, iter, times = ofm_mpi_sq(local_L, V0, n_samples, comm_info, alpha=0.001, beta=0.95, cg=true, linesearch=true, locking=false, tri=false, log2deg=log2deg, itermax=itermax, tol=tol)
            for (key, val) in times
                if haskey(cputime, key)
                    cputime[key] += val*1.0/repeats1
                else
                    cputime[key] = val*1.0/repeats1
                end
            end
        end
        V0 .= V
        eigV_test = nothing
        evals_test = nothing
        err = nothing
        cputime_qr = 0
        for i in 1:repeats1
            # global cputime_qr, eigV_test, evals_test, err
            eigV_test, evals_test, err, times = rayleigh_local(local_L, V, n_samples, process_tree_lvl, comms_tree, counts_info, info_cols_dist, rank, rank_row, rank_col, comm_row, comm_col, root, comm_size_sq)
            cputime_qr += times/repeats1
        end
        time_mainloop_cum += cputime["main_loop"]
        if rank == 0
            @printf("---------------- OFM ------------------\n")
            @printf("N: %i class: %s type: %s sub-graph: %i \n", n_samples, c, t, gh)
            @printf("node: %i k: %i tol: %.2e iter: %i itermax: %i log2deg: %i nmv: %i \n", comm_size, k, tol, iter, itermax, log2deg, cputime["nmv"])
            @printf("Tri: false CG: true linesearch: true locking: false \n")
            @printf("relative error: %.2e \n", err)
            for ii in 1:k
                @printf("%i-th computed eigenvalue(ofm): %.2e \n", ii, evals_test[ii]-a0)
            end
            @printf("walltime pre:                 %.2e \n", cputime["pre"])
            @printf("walltime main_loop:           %.2e %.2e \n", cputime["main_loop"], time_mainloop_cum)
            @printf("   walltime checkConvergence: %.2e \n", cputime["checkConvergence"])
            @printf("   walltime Ax:               %.2e \n", cputime["Ax"])
            @printf("   walltime updateGradient:   %.2e \n", cputime["updateGradient"])
            @printf("   walltime CG:               %.2e \n", cputime["CG"])
            @printf("   walltime linesearch:       %.2e \n", cputime["linesearch"])
            @printf("walltime QR:                  %.2e \n", cputime_qr)
        end

        V_gather .= V
        if rownorm
            d = 1 ./ sqrt.(sum(V_gather .^ 2, dims=1))
            D = spdiagm(n_samples, n_samples, vec(d))
            V_gather .= V_gather*D
        end
        (ari, ri, mi, vi, vm) = (0,0,0,0,0)
        for i in 1:repeats2
            # global a, ari, ri, mi, vi, vm
            R = kmeans(V_gather, numcl; maxiter=300)
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
    @printf("\n\n")


    V_gather = randn(k, n_samples)
    local_A = sparse([1], [1], [0.0], n_samples, n_samples)
    V0 = randn(k, n_samples)
    time_mainloop_cum = 0.0
    for gh in [1,2,3,4,5,6,7,8,9,10]
        f1 = matopen(file1 * string(gh) * "_A.mat")
        A_gh = read(f1, "data")
        local_A += A_gh
        symmetric_A = (local_A+local_A')/2
        d = sqrt.(vec(sum(symmetric_A, dims=1)))
        d2 = zeros(n_samples)
        for ni in 1:n_samples
            if abs(d[ni]) > 0
                d2[ni] = 1/d[ni]
            end
        end
        D2 = sparse(rows, cols, d2, n_samples, n_samples)
        local_L = E - D2*symmetric_A*D2

        local_L += transtion  

        if gh == 1
            itermax = itermax1
        else
            itermax = itermax2
        end

        V = nothing
        iter = nothing
        cputime = Dict()
        for i in 1:repeats1
            V, iter, times = omm_mpi_sq(local_L, V0, n_samples, comm_info, alpha=0.001, beta=0.95, cg=true, linesearch=true, locking=false, tri=true, log2deg=log2deg, itermax=itermax, tol=tol)
            for (key, val) in times
                if haskey(cputime, key)
                    cputime[key] += val*1.0/repeats1
                else
                    cputime[key] = val*1.0/repeats1
                end
            end
        end
        V0 .= V
        eigV_test = nothing
        evals_test = nothing
        err = nothing
        cputime_qr = 0
        for i in 1:repeats1
            # global cputime_qr, eigV_test, evals_test, err
            eigV_test, evals_test, err, times = rayleigh_local(local_L, V, n_samples, process_tree_lvl, comms_tree, counts_info, info_cols_dist, rank, rank_row, rank_col, comm_row, comm_col, root, comm_size_sq)
            cputime_qr += times/repeats1
        end
        time_mainloop_cum += cputime["main_loop"]
        if rank == 0
            @printf("---------------- OMM ------------------\n")
            @printf("N: %i class: %s type: %s sub-graph: %i \n", n_samples, c, t, gh)
            @printf("node: %i k: %i tol: %.2e iter: %i itermax: %i log2deg: %i nmv: %i \n", comm_size, k, tol, iter, itermax, log2deg, cputime["nmv"])
            @printf("Tri: true CG: true linesearch: true locking: false \n")
            @printf("relative error: %.2e \n", err)
            for ii in 1:k
                @printf("%i-th computed eigenvalue(ofm): %.2e \n", ii, evals_test[ii]-a0)
            end
            @printf("walltime pre:                 %.2e \n", cputime["pre"])
            @printf("walltime main_loop:           %.2e %.2e \n", cputime["main_loop"], time_mainloop_cum)
            @printf("   walltime checkConvergence: %.2e \n", cputime["checkConvergence"])
            @printf("   walltime Ax:               %.2e \n", cputime["Ax"])
            @printf("   walltime updateGradient:   %.2e \n", cputime["updateGradient"])
            @printf("   walltime CG:               %.2e \n", cputime["CG"])
            @printf("   walltime linesearch:       %.2e \n", cputime["linesearch"])
            @printf("walltime QR:                  %.2e \n", cputime_qr)
        end

        V_gather .= V
        if rownorm
            d = 1 ./ sqrt.(sum(V_gather .^ 2, dims=1))
            D = spdiagm(n_samples, n_samples, vec(d))
            V_gather .= V_gather*D
        end
        (ari, ri, mi, vi, vm) = (0,0,0,0,0)
        for i in 1:repeats2
            # global a, ari, ri, mi, vi, vm
            R = kmeans(V_gather, numcl; maxiter=300)
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
    @printf("\n\n")
    

    V_gather = randn(k, n_samples)
    local_A = sparse([1], [1], [0.0], n_samples, n_samples)
    V0 = randn(k, n_samples)
    time_mainloop_cum = 0.0
    for gh in [1,2,3,4,5,6,7,8,9,10]
        f1 = matopen(file1 * string(gh) * "_A.mat")
        A_gh = read(f1, "data")
        local_A += A_gh
        symmetric_A = (local_A+local_A')/2
        d = sqrt.(vec(sum(symmetric_A, dims=1)))
        d2 = zeros(n_samples)
        for ni in 1:n_samples
            if abs(d[ni]) > 0
                d2[ni] = 1/d[ni]
            end
        end
        D2 = sparse(rows, cols, d2, n_samples, n_samples)
        local_L = E - D2*symmetric_A*D2

        local_L += transtion  

        if gh == 1
            itermax = itermax1
        else
            itermax = itermax2
        end

        V = nothing
        iter = nothing
        cputime = Dict()
        for i in 1:repeats1
            V, iter, times = omm_mpi_sq(local_L, V0, n_samples, comm_info, alpha=0.001, beta=0.95, cg=true, linesearch=true, locking=false, tri=false, log2deg=log2deg, itermax=itermax, tol=tol)
            for (key, val) in times
                if haskey(cputime, key)
                    cputime[key] += val*1.0/repeats1
                else
                    cputime[key] = val*1.0/repeats1
                end
            end
        end
        V0 .= V
        eigV_test = nothing
        evals_test = nothing
        err = nothing
        cputime_qr = 0
        for i in 1:repeats1
            # global cputime_qr, eigV_test, evals_test, err
            eigV_test, evals_test, err, times = rayleigh_local(local_L, V, n_samples, process_tree_lvl, comms_tree, counts_info, info_cols_dist, rank, rank_row, rank_col, comm_row, comm_col, root, comm_size_sq)
            cputime_qr += times/repeats1
        end
        time_mainloop_cum += cputime["main_loop"]
        if rank == 0
            @printf("---------------- OMM ------------------\n")
            @printf("N: %i class: %s type: %s sub-graph: %i \n", n_samples, c, t, gh)
            @printf("node: %i k: %i tol: %.2e iter: %i itermax: %i log2deg: %i nmv: %i \n", comm_size, k, tol, iter, itermax, log2deg, cputime["nmv"])
            @printf("Tri: false CG: true linesearch: true locking: false \n")
            @printf("relative error: %.2e \n", err)
            for ii in 1:k
                @printf("%i-th computed eigenvalue(ofm): %.2e \n", ii, evals_test[ii]-a0)
            end
            @printf("walltime pre:                 %.2e \n", cputime["pre"])
            @printf("walltime main_loop:           %.2e %.2e \n", cputime["main_loop"], time_mainloop_cum)
            @printf("   walltime checkConvergence: %.2e \n", cputime["checkConvergence"])
            @printf("   walltime Ax:               %.2e \n", cputime["Ax"])
            @printf("   walltime updateGradient:   %.2e \n", cputime["updateGradient"])
            @printf("   walltime CG:               %.2e \n", cputime["CG"])
            @printf("   walltime linesearch:       %.2e \n", cputime["linesearch"])
            @printf("walltime QR:                  %.2e \n", cputime_qr)
        end

        V_gather .= V
        if rownorm
            d = 1 ./ sqrt.(sum(V_gather .^ 2, dims=1))
            D = spdiagm(n_samples, n_samples, vec(d))
            V_gather .= V_gather*D
        end
        (ari, ri, mi, vi, vm) = (0,0,0,0,0)
        for i in 1:repeats2
            # global a, ari, ri, mi, vi, vm
            R = kmeans(V_gather, numcl; maxiter=300)
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
    @printf("\n\n")
    

    if rank == 0
        @printf("\n\n")
    end

    GC.gc()
    MPI.Finalize()

end