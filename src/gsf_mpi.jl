using Printf
using Statistics
using LowRankApprox
using MPI
using LinearAlgebra
using Test
using Random, Distributions

include("../Bchdav/bchdav_mpi.jl")

function jackson_chebyshev_filter(L, E, X, p, a0, b0, lb, ub, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col)
    A = nothing
    inds = [ind for ind = 1:size(L,1)]
    if rank_row == rank_col
        A = L - sparse(inds, inds, (lb+ub)/2 .* ones(Float64, size(L, 1)))
    else
        A = L
    end

    A = A ./ ((ub-lb)/2.0)

    a = 2.0*(a0-(ub+lb)/2)/(ub-lb)
    b = 2.0*(b0-(ub+lb)/2)/(ub-lb)

    alphap = pi/(p+2)
    g0 = 1
    g1 = cos(alphap)
    gamma0 = (acos(a)-acos(b))/pi
    gamma1 = 2.0*(sin(acos(a))-sin(acos(b)))/pi
    if p == 0
        return (g0*gamma0) .* X
    end
    if p == 1
        return (g0*gamma0) .* X + (g1*gamma1) .* SpMM_A_1_w_E(X, A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col)
    end

    t0 = deepcopy(X)
    t1 = SpMM_A_1_w_E(X, A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col)
    ans = (g0*gamma0) .* t0 + (g1*gamma1) .* t1
    j = 1
    while j <= p
        j += 1
        t2 = 2.0 .* SpMM_A_1_w_E(t1, A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col) - t0
        g2 = ((1-j/(p+2))*sin(alphap)*cos(j*alphap)+cos(alphap)*sin(j*alphap)/(p+2))/sin(alphap)
        gamma2 = 2.0*(sin(j*acos(a))-sin(j*acos(b)))/j/pi
        ans .+= (g2*gamma2) .* t2
        t0 .= t1
        t1 .= t2
    end
    ans
end


function jackson_chebyshev_filter_sq(L, E, X, p, a0, b0, lb, ub, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col)
    A = nothing
    inds = [ind for ind = 1:size(L,1)]
    if rank_row == rank_col
        A = L - sparse(inds, inds, (lb+ub)/2 .* ones(Float64, size(L, 1)))
    else
        A = L
    end

    A ./= ((ub-lb)/2.0)

    a = 2.0*(a0-(ub+lb)/2)/(ub-lb)
    b = 2.0*(b0-(ub+lb)/2)/(ub-lb)

    alphap = pi/(p+2)
    g0 = 1
    g1 = cos(alphap)
    gamma0 = (acos(a)-acos(b))/pi
    gamma1 = 2.0*(sin(acos(a))-sin(acos(b)))/pi
    if p == 0
        return (g0*gamma0) .* X
    end
    if p == 1
        return (g0*gamma0) .* X + (g1*gamma1) .* (X * A)
    end

    t0 = zeros(size(X))
    t1 = zeros(size(X))
    ans = zeros(size(X))
    t2 = zeros(size(X))
    t0 .= X
    mul!(t1, X, A)
    # ans = (g0*gamma0) .* t0 + (g1*gamma1) .* t1
    mul!(ans, g0*gamma0, t0)
    mul!(ans, g1*gamma1, t1, 1.0, 1.0)
    j = 1
    while j <= p
        j += 1
        # t2 = 2.0 .* SpMM_A_1_w_E(t1, A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col) - t0
        mul!(t2, t1, A)
        lmul!(2.0, t2)
        t2 .-= t0
        g2 = ((1-j/(p+2))*sin(alphap)*cos(j*alphap)+cos(alphap)*sin(j*alphap)/(p+2))/sin(alphap)
        gamma2 = 2.0*(sin(j*acos(a))-sin(j*acos(b)))/j/pi
        # ans .+= (g2*gamma2) .* t2
        mul!(ans, g2*gamma2, t2, 1.0, 1.0)
        t0 .= t1
        t1 .= t2
    end
    ans
end

function jackson_chebyshev_scalar(y, p, a0, b0, lb, ub)
    x = 2.0*(y-(ub+lb)/2)/(ub-lb)
    a = 2.0*(a0-(ub+lb)/2)/(ub-lb)
    b = 2.0*(b0-(ub+lb)/2)/(ub-lb)
    alphap = pi/(p+2)
    g0 = 1
    g1 = cos(alphap)
    gamma0 = (acos(a)-acos(b))/pi
    gamma1 = 2.0*(sin(acos(a))-sin(acos(b)))/pi
    if p == 0
        return g0*gamma0*1
    end
    if p == 1
        return g0*gamma0*1 + g1*gamma1*x
    end
    t0 = 1
    t1 = x
    ans = g0*gamma0*1 + g1*gamma1*x
    j = 1
    while j <= p
        j += 1
        t2 = 2.0*x*t1 - t0
        g2 = ((1-j/(p+2))*sin(alphap)*cos(j*alphap)+cos(alphap)*sin(j*alphap)/(p+2))/sin(alphap)
        gamma2 = 2.0*(sin(j*acos(a))-sin(j*acos(b)))/j/pi
        ans += g2*gamma2*t2
        t0 = t1
        t1 = t2
    end
    ans
end

function gsf_mpi(A, k, N, comm_info; lambdaMin=0, lambdaMax=2, itermax=20, p=20, p2=20, estimate=Inf)
    cputime = Dict("poly"=>0.0)

    cputime["pre"] = @elapsed begin
    comm = comm_info["comm"]
    comm_row = comm_info["comm_row"]
    comm_rol = comm_info["comm_col"]
    rank = comm_info["rank"]
    rank_row = comm_info["rank_row"]
    rank_col = comm_info["rank_col"]
    info_cols_dist = comm_info["info_cols_dist"]
    comm_size = comm_info["comm_size"]
    comm_size_sq = comm_info["comm_size_sq"]
    root = 0
    n = info_cols_dist[rank+1]

    inds = [ind for ind = 1:size(A,1)]
    E = rank_col == rank_row ? sparse(inds, inds, ones(Float64, size(A, 1)), size(A,1), size(A,2)) : sparse([1], [1], [0.0], size(A,1), size(A,2))

    lambdalb = lambdaMin
    clb = 0
    cest = 0

    distribution = Normal(0.0, 1.0/k)
    R = rand(distribution, (k, n))

    lambdaub = lambdaMax
    cub = N
    lambdaest = lambdalb + (lambdaub-lambdalb)*k/N
    end

    iter = 0
    cputime["main"] = @elapsed begin
        cputime["estimate"] = @elapsed begin
        if estimate < Inf
            lambdaest = estimate
        else
            while cest != k && iter < itermax
                lambda = lambdaest
                cputime["poly"] += @elapsed begin
                M = jackson_chebyshev_filter(A, E, R, p2, lambdaMin, lambda, lambdaMin, lambdaMax, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col)
                end
                normM = zeros(1)
                normM[1] = norm(M) ^ 2
                MPI.Allreduce!(normM, +, comm_row)
                MPI.Allreduce!(normM, +, comm_col)
                cest = normM[1]
                if cest < k
                    lambdalb = lambdaest
                else
                    lambdaub = lambdaest
                end
                if clb == cest || cub == cest
                    lambdaest = (lambdalb+lambdaub)/2.0
                else
                    if cest < k
                        clb = cest
                    else
                        cub = cest
                    end
                    lambdaest = lambdalb + (k-clb)*(lambdaub-lambdalb)/(cub-clb)
                end
                iter += 1
            end
        end
        end
        cputime["M"] = @elapsed begin
        M = jackson_chebyshev_filter(A, E, R, p, lambdaMin, lambdaest, lambdaMin, lambdaMax, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col)
        end
    end
    M, lambdaest, iter, cputime
end


function gsf_sq(A, k, N, comm_info; lambdaMin=0, lambdaMax=2, itermax=20, p=20, p2=20, estimate=Inf)
    cputime = Dict("poly"=>0.0)

    cputime["pre"] = @elapsed begin
    comm = comm_info["comm"]
    comm_row = comm_info["comm_row"]
    comm_col = comm_info["comm_col"]
    rank = comm_info["rank"]
    rank_row = comm_info["rank_row"]
    rank_col = comm_info["rank_col"]
    info_cols_dist = comm_info["info_cols_dist"]
    comm_size = comm_info["comm_size"]
    comm_size_sq = comm_info["comm_size_sq"]
    root = 0
    n = info_cols_dist[rank+1]

    inds = [ind for ind = 1:size(A,1)]
    E = rank_col == rank_row ? sparse(inds, inds, ones(Float64, size(A, 1)), size(A,1), size(A,2)) : sparse([1], [1], [0.0], size(A,1), size(A,2))

    lambdalb = lambdaMin
    clb = 0
    cest = 0

    distribution = Normal(0.0, 1.0/k)
    R = rand(distribution, (k, n))

    lambdaub = lambdaMax
    cub = N
    lambdaest = lambdalb + (lambdaub-lambdalb)*k/N
    end

    iter = 0
    cputime["main"] = @elapsed begin
        cputime["estimate"] = @elapsed begin
        if estimate < Inf
            lambdaest = estimate
        else
            while cest != k && iter < itermax
                lambda = lambdaest
                cputime["poly"] += @elapsed begin
                M = jackson_chebyshev_filter_sq(A, E, R, p2, lambdaMin, lambda, lambdaMin, lambdaMax, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col)
                end
                normM = zeros(1)
                normM[1] = norm(M) ^ 2
                cest = normM[1]
                if cest < k
                    lambdalb = lambdaest
                else
                    lambdaub = lambdaest
                end
                if clb == cest || cub == cest
                    lambdaest = (lambdalb+lambdaub)/2.0
                else
                    if cest < k
                        clb = cest
                    else
                        cub = cest
                    end
                    lambdaest = lambdalb + (k-clb)*(lambdaub-lambdalb)/(cub-clb)
                end
                iter += 1
            end
        end
        end
        cputime["M"] = @elapsed begin
        M = jackson_chebyshev_filter_sq(A, E, R, p, lambdaMin, lambdaest, lambdaMin, lambdaMax, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col)
        end
    end
    M, lambdaest, iter, cputime
end

function rayleigh(A_global, V_global)
    Vt,_ = pqr(V_global')
    # H1 = Vt'*A_global*Vt
    AV = zeros(size(Vt))
    H1 = zeros(size(Vt, 2), size(Vt, 2))
    mul!(AV, A_global, Vt)
    mul!(H1, Vt', AV)
    d, Q = eigen(H1)
    mul!(AV, Vt, Q)
    transpose(AV), d
end