using MPI, MAT, SparseArrays, LinearAlgebra, Printf, Random, CSV, DataFrames, Test, IterativeSolvers, Clustering, DataStructures, Arpack

include("./gd/gd_mpi.jl")

for code in 1:1
    n_samples = parse(Int64, ARGS[1])
    k = parse(Int64, ARGS[2])
    tol = parse(Float64, ARGS[3])
    c = ARGS[4]
    t = ARGS[5]
    name = ARGS[6]

    rownorm = true
    repeats1 = 1
    repeats2 = 1
    a0 = 0.5
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
    local_A = sparse([1], [1], [0.0], n_samples, n_samples)
    eigV_lobpcg = zeros(n_samples, k)
    v0 = randn(n_samples, k)
    time_cum = 0.0
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
        local_L = E - D2*symmetric_A*D2 + a0 .* E

        # eigV_lobpcg = nothing
        evals_lobpcg = nothing
        nconv = 0
        niter = 0
        nmult = 0
        time_lobpcg = 0.0
        for i in 1:repeats1
            # global eigV_lobpcg, evals_lobpcg, nconv, niter, nmult, time_lobpcg
            time_lobpcg += @elapsed begin
            F = lobpcg(local_L, false, v0; tol=tol, maxiter=200)
            evals_lobpcg = F.λ
            eigV_lobpcg = F.X
            end
        end
        v0 .= eigV_lobpcg
        time_lobpcg /= repeats1
        time_cum += time_lobpcg
        # loss_ofm = ofm_loss_global(A_global, eigV_lobpcg')
        # loss_omm = omm_loss_global(A_global, eigV_lobpcg')
        err = norm(local_L*eigV_lobpcg - eigV_lobpcg*Diagonal(evals_lobpcg))/norm(eigV_lobpcg*Diagonal(evals_lobpcg))

        if rank == 0
            # @printf("actual ofm loss: %.2e \n", loss_ofm)
            # @printf("actual omm loss: %.2e \n", loss_omm)
            @printf("------------ LOBPCG ---------------- \n")
            @printf("N: %i class: %s type: %s sub-graph: %i \n", n_samples, c, t, gh)
            @printf("relative error: %.2e tol: %.2e k: %i \n", err, tol, k)
            @printf("time: %.2e time_cum: %.2e \n", time_lobpcg, time_cum)
            for ii in 1:k
                @printf("%i-th computed eigenvalue(lobpcg): %.2e \n", ii, evals_lobpcg[ii]-a0)
            end
        end

        V_gather = Array{Float64}(transpose(eigV_lobpcg))
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



    if rank == 0
        @printf("\n\n")
    end

    GC.gc()
    MPI.Finalize()

end