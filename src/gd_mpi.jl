using Printf
using Statistics
using LowRankApprox
using MPI
using LinearAlgebra
using Test
using SparseArrays
using Tullio

include("../Bchdav/bchdav_mpi.jl")

function Cheb_filter_scalar(deg, low, high, x)
    e = (high - low)/2
    center= (high+low)/2
    
    y0 = 1
    y1 = x
    y1 = (y1 - center)/e
    y2 = y1
    for kk = 2:deg
        y2 = x*y1
        y2 = 2*(y2 - center*y1)/e - y0
        y0 = y1
        y1 = y2
    end
    # if deg % 2 == 0
    #     y2 = -y2
    # end
    y2
end

function Cheb_filter_composition_scalar(log2deg, low, high, x; base=2)
    e = (high - low)/2
    center= (high+low)/2
    deg = base ^ log2deg
    y = (x - center)/e
    
    for i in 1:log2deg
        y = Cheb_filter_scalar(base, -1.0, 1.0, y)
    end

    # if deg % 2 == 0
    #     y = -y
    # end
    y
end


function Cheb_filter_composition(log2deg, low, high, X, A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col; base=2)
    cputime = Dict()
    
    e = (high - low)/2
    center= (high+low)/2
    deg = base ^ log2deg
    inds = [ind for ind = 1:size(A,1)]
    B = nothing
    if rank_col == rank_row
        B = A - sparse(inds, inds, center .* ones(Float64, size(A, 1)), size(A,1), size(A,2))
    end
    B = A .* (1/e)
    
    Y = deepcopy(X)
    for i in 1:log2deg
        Y, times = Cheb_filter(base, -1.0, 1.0, Y, B, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col)
        for (key, val) in times
            if haskey(cputime, key)
                cputime[key] += val
            else
                cputime[key] = val
            end
        end
    end

    # if deg % 2 == 0
    #     Y .= -Y
    # end
    
    Y, cputime
end


function Cheb_filter(deg, low, high, X, A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col)
# deg should always be an odd number

    e = (high - low)/2
    center= (high+low)/2
    
    cputime = Dict("copy"=>0.0)
    cputime["SpMM"] = @elapsed begin
    Y = SpMM_A_1_w_E(X, A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col)
    end
    cputime["local_computation"] = @elapsed begin
    Y = (Y - center*X) ./ e
    end
    for kk = 2:deg-1

        cputime["SpMM"] += @elapsed begin
        Y1 = SpMM_A_1_w_E(Y, A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col)
        end
        
        cputime["local_computation"] += @elapsed begin
        Y1 = (2.0/e) .* (Y1 - center*Y) - X
        end

        cputime["copy"] += @elapsed begin
        X = deepcopy(Y)
        Y = deepcopy(Y1)
        end
    end
    
    cputime["SpMM"] += @elapsed begin
    Y1 = SpMM_A_1_w_E(Y, A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col)
    end
    cputime["local_computation"] += @elapsed begin
    Y1 = (2.0/e) .* (Y1 - center*Y) - X
    end
    
    Y1, cputime
end

function Cheb_filter_scal_scalar(deg, low, high, leftb, x)
    e = (high - low)/2
    center= (high+low)/2
    sigma = e/(leftb - center)
    tau = 2/sigma
    
    y0 = 1
    y1 = x
    y1 = (y1 - center)*(sigma/e)
    for kk = 2:deg-1
        sigma_new = 1 /(tau - sigma)
        y2 = x*y1
        y2 = (y2 - center*y1)*(2*sigma_new/e) - (sigma*sigma_new)*y0
        y0 = y1
        y1 = y2
        sigma = sigma_new
    end
    y2 = x*y1
    sigma_new = 1 /(tau - sigma)
    y2 = (y2 - center*y1)*(2*sigma_new/e) - (sigma*sigma_new)*y0
    -y2
end


function cardano(a, b, c, d0)
    p = (3.0*a*c - b^2.0)/3.0/a^2.0
    q = (2.0*b^3.0 - 9.0*a*b*c + 27.0* a^2.0 * d0)/27.0/a^3.0
    d = -4.0*p^3.0 - 27.0*q^2.0

    ans = 0.0
    if abs(d) == 0 && p == 0
        ans = 0.0
    elseif abs(d) == 0
        ans = 3.0*q/p
    elseif d < 0
        ans = cbrt(-q/2.0 + sqrt(-d/4.0/27.0)) + cbrt(-q/2.0 - sqrt(-d/4.0/27.0))
    else
        c1, c2, c3 = cbrt_comp(-q/2.0 + sqrt(Complex(-d/4.0/27.0)))
        ans1 = real(c1 - p/3.0/c1)
        ans2 = real(c2 - p/3.0/c2)
        ans3 = real(c3 - p/3.0/c3)
        ans1, ans2, ans3 = sort([ans1, ans2, ans3])
        if abs(ans1 - ans2) <= abs(ans3 - ans2)
            ans = ans3
        else
            ans = ans1
        end
    end
    # println(a, " ", b, " ", c, " ", d0, " ", ans - b/3.0/a)
    return ans - b/3.0/a
end

function cbrt_comp(x)
    cbrt(abs(x))*exp(angle(x)/3*im), cbrt(abs(x))*exp(angle(x)/3*im+2*pi/3*im), cbrt(abs(x))*exp(angle(x)/3*im-2*pi/3*im)
end

function dotMulVec(Z, X, Y)
    @tullio Z[i] = X[i,j]*Y[i,j]
end

function diagMM(A, B)
    @assert size(A) == size(B)
    V = zeros(size(A,1))
    @tullio V[i] = A[i,j]*B[j,i]
    V
end

function permutateSqM(M, k0, k1, convn)
    k2 = k0
    for i in 1:length(convn)
        if convn[i]
            M[k2+1, :], M[k0+i, :] = M[k0+i, :], M[k2+1, :]
            M[:, k2+1], M[:, k0+i] = M[:, k0+i], M[:, k2+1]
            k2 += 1
        end
    end
end

function permutateRtM(M, k0, k1, convn)
    k2 = k0
    for i in 1:length(convn)
        if convn[i]
            M[k2+1, :], M[k0+i, :] = M[k0+i, :], M[k2+1, :]
            k2 += 1
        end
    end
end


function ofm_mpi(A, X, N, comm_info; alpha=0.01, beta=0.0, cg=true, linesearch=true, locking=true, tri=true, precond=nothing, bb=nothing, log2deg=1, itermax=200, tol=1e-3, lowb=-2.0, upb=0.0, base=2)
    cputime = Dict("checkConvergence"=>0.0, "Ax"=>0.0, "CG"=>0.0, "updateGradient"=>0.0, "linesearch"=>0.0, "nmv"=>0.0)

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

    inds = [ind for ind = 1:size(A,1)]
    E = rank_col == rank_row ? sparse(inds, inds, ones(Float64, size(A, 1)), size(A,1), size(A,2)) : sparse([1], [1], [0.0], size(A,1), size(A,2))


    p, n = size(X)
    G = zeros(p, n)
    G1 = zeros(p, n)
    V = zeros(p, n)
    alphas = zeros(p)
    betas = [beta for i in 1:p]
    gX = deepcopy(X)
    K = zeros(p, p)
    mul!(K, X, X')
    MPI.Allreduce!(K, +, comm_row)
    MPI.Allreduce!(K, +, comm_col)

    VTV = nothing
    XTV = nothing
    gV = nothing
    VTAV = nothing
    VTAX = nothing
    X0 = nothing
    if linesearch
        VTV = zeros(p, p)
        XTV = zeros(p, p)
        gV = zeros(p, n)
        VTAV = zeros(p, p)
        VTAX = zeros(p, p)
    end
    if bb != nothing
        X0 = deepcopy(X)
    end


    # if log2deg > 0
    #     for i in 1:log2deg
    #         gX, _ =  Cheb_filter_composition(log2deg,  lowb, upb, gX, A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col, base=base)
    #     end
    # else
    #     gX = SpMM_A_1_w_E(gX, A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col)
    # end
    gX = SpMM_A_1_w_E(gX, A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col)

    if tri
        # G .= gX + LowerTriangular(K)*X
        mul!(G, LowerTriangular(K), X)
        G .+= gX
    else
        # G .= gX + K*X
        mul!(G, K, X)
        G .+= gX
    end

    V .= -G

    if cg
        if precond == "filter"
            # G, _ =  Cheb_filter_composition(log2deg, lowb, upb, G, A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col, base=base)
            G, _ = Cheb_filter_scal_1(base, lowb, upb, -2.0, G, A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col)
        end
    end

    if linesearch
        mul!(VTV, V, V')
        mul!(XTV, X, V')
        MPI.Allreduce!(VTV, +, comm_row)
        MPI.Allreduce!(VTV, +, comm_col)
        MPI.Allreduce!(XTV, +, comm_row)
        MPI.Allreduce!(XTV, +, comm_col)
        gV .= deepcopy(V)
        # if log2deg > 0
        #     for i in 1:log2deg
        #         gV, _ =  Cheb_filter_composition(log2deg,  lowb, upb, gV, A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col, base=base)
        #     end
        # else
        #     gV = SpMM_A_1_w_E(gV, A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col)
        # end
        gV .= SpMM_A_1_w_E(gV, A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col)
        mul!(VTAV, V, gV')
        mul!(VTAX, V, gX')
        MPI.Allreduce!(VTAV, +, comm_row)
        MPI.Allreduce!(VTAV, +, comm_col)
        MPI.Allreduce!(VTAX, +, comm_row)
        MPI.Allreduce!(VTAX, +, comm_col)
        if tri
            # a3 = cumsum(diag(VTV*UpperTriangular(VTV)))
            # a2 = cumsum(diag(VTV*UpperTriangular(XTV)+VTV*UpperTriangular(XTV')+XTV'*UpperTriangular(VTV)))
            # a1 = cumsum(diag(VTAV+XTV'*UpperTriangular(XTV')+XTV'*UpperTriangular(XTV)+VTV*UpperTriangular(K)))
            # a0 = cumsum(diag(VTAX+XTV'*UpperTriangular(K)))
            a3 = cumsum(diagMM(VTV, UpperTriangular(VTV)))
            a2 = cumsum(diagMM(VTV, UpperTriangular(XTV))+diagMM(VTV, UpperTriangular(XTV'))+diagMM(XTV', UpperTriangular(VTV)))
            a1 = cumsum(diag(VTAV)+diagMM(XTV', UpperTriangular(XTV'))+diagMM(XTV', UpperTriangular(XTV))+diagMM(VTV, UpperTriangular(K)))
            a0 = cumsum(diag(VTAX)+diagMM(XTV', UpperTriangular(K)))
            alphas = map(cardano, a3, a2, a1, a0)
        else
            # a3 = tr(VTV*VTV)
            # a2 = 3*tr(VTV*XTV)
            # a1 = tr(VTAV+XTV'*XTV'+XTV'*XTV+VTV*K)
            # a0 = tr(VTAX+XTV'*K)
            a3 = sum(diagMM(VTV, VTV))
            a2 = 3*sum(diagMM(VTV, XTV))
            a1 = sum(diag(VTAV)+diagMM(XTV', XTV')+diagMM(XTV', XTV)+diagMM(VTV, K))
            a0 = sum(diag(VTAX)+diagMM(XTV', K))
            alpha = cardano(a3,a2,a1,a0)
            alphas = [alpha for i in 1:p]
        end
    else
        alphas = [alpha for i in 1:p]
    end

    # X .+= Diagonal(alphas)*V
    Da = Diagonal(alphas)
    mul!(X, Da, V, 1.0, 1.0)
    end

    iter = 0
    kconv = 0
    cputime["main_loop"] = @elapsed begin
    while iter < itermax
        cputime["checkConvergence"] += @elapsed begin
        if locking
            norm_G = norm.(eachrow(G[kconv+1:p])) .^ 2
            norm_gX = norm.(eachrow(gX[kconv+1:p])) .^ 2
            MPI.Allreduce!(norm_G, +, comm_row)
            MPI.Allreduce!(norm_G, +, comm_col)
            MPI.Allreduce!(norm_gX, +, comm_row)
            MPI.Allreduce!(norm_gX, +, comm_col)
            norm_G = sqrt.(norm_G)
            norm_gX = cbrt.(sqrt.(norm_gX))
            norm_GgX = norm_G .* norm_gX
            convn = norm_GgX .< tol
            for ic in 1:length(convn)
                if convn[ic]
                    kconv += 1
                else
                    break
                end
            end
            # permutateRtM(X, kconv, p, convn)
            # permutateRtM(G, kconv, p, convn)
            # permutateRtM(V, kconv, p, convn)
            # permutateRtM(gX, kconv, p, convn)
            # permutateRtM(G1, kconv, p, convn)
            # permutateSqM(K, kconv, p, convn)
            # if linesearch
            #     permutateSqM(VTV, kconv, p, convn)
            #     permutateSqM(XTV, kconv, p, convn)
            #     permutateSqM(gV, kconv, p, convn)
            #     permutateSqM(VTAV, kconv, p, convn)
            #     permutateSqM(VTAX, kconv, p, convn)
            # end
            # kconv += sum(convn)
            if kconv == p
                break
            end
        end
        end

        cputime["Ax"] += @elapsed begin
        if locking
            @views gX[kconv+1:p,:] .= X[kconv+1:p,:]
            @views gX[kconv+1:p,:] .= SpMM_A_1_w_E(gX[kconv+1:p,:], A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col)
        else
            gX .= X
            gX .= SpMM_A_1_w_E(gX, A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col)
        end
        end
        cputime["nmv"] += p-kconv

        cputime["updateGradient"] += @elapsed begin
        # mul!(K, X, X')
        if locking
            @views mul!(K[kconv+1:p, kconv+1:p], X[kconv+1:p,:], X[kconv+1:p,:]')
            @views mul!(K[1:kconv, kconv+1:p], X[1:kconv,:], X[kconv+1:p,:]')
            @views K[kconv+1:p, 1:kconv] .= K[1:kconv, kconv+1:p]'
        else
            mul!(K, X, X')
        end
        MPI.Allreduce!(K, +, comm_row)
        MPI.Allreduce!(K, +, comm_col)
        if tri
            if locking
                G1[kconv+1:p,:] .= gX[kconv+1:p,:] .+ LowerTriangular(K)[kconv+1:p,:]*X
                # G1[kconv+1:p,:] .= mul!(G1[kconv+1:p,:], LowerTriangular(K)[kconv+1:p,:], X)
                # G1[kconv+1:p,:] .+= gX[kconv+1:p,:]
            else
                mul!(G1, LowerTriangular(K), X)
                G1 .+= gX
            end
        else
            # G1 .= gX + K*X
            mul!(G1, K, X)
            G1 .+= gX
        end
        end
        
        cputime["CG"] += @elapsed begin
        if cg
            if precond == "filter"
                # G1[kconv+1:p,:], _ =  Cheb_filter_composition(log2deg,  lowb, upb, G1[kconv+1:p,:], A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col, base=base)
                G1[kconv+1:p,:], _ = Cheb_filter_scal_1(base, lowb, upb, -2.0, G1[kconv+1:p,:], A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col)
            end
            # numerators = diag((G1[kconv+1:p,:] - G[kconv+1:p,:])*G1[kconv+1:p,:]')
            # denomurators = diag(G[kconv+1:p,:]*G[kconv+1:p,:]')
            numerators = zeros(p-kconv)
            denomurators = zeros(p-kconv)
            if locking
                dotMulVec(numerators, G1[kconv+1:p,:] - G[kconv+1:p,:], G1[kconv+1:p,:])
                dotMulVec(denomurators, G[kconv+1:p,:], G[kconv+1:p,:])
            else
                dotMulVec(numerators, G1 - G, G1)
                dotMulVec(denomurators, G, G)
            end
            MPI.Allreduce!(numerators, +, comm_row)
            MPI.Allreduce!(numerators, +, comm_col)
            MPI.Allreduce!(denomurators, +, comm_row)
            MPI.Allreduce!(denomurators, +, comm_col)
            betas[kconv+1:p] = numerators ./ denomurators
        end
        # G .= G1
        if locking
            # V[kconv+1:p,:] = Diagonal(betas[kconv+1:p])*V[kconv+1:p,:] - G1[kconv+1:p,:]
            Db = Diagonal(betas[kconv+1:p])
            @views lmul!(Db, V[kconv+1:p,:])
            @views V[kconv+1:p,:] .-= G1[kconv+1:p,:]
        else
            Db = Diagonal(betas)
            lmul!(Db, V)
            V .-= G1
        end
        end
        
        cputime["linesearch"] += @elapsed begin
        if linesearch
            # mul!(VTV, V, V')
            # mul!(XTV, X, V')
            if locking
                @views mul!(VTV[kconv+1:p, kconv+1:p], V[kconv+1:p,:], V[kconv+1:p,:]')
                @views mul!(VTV[1:kconv, kconv+1:p], V[1:kconv,:], V[kconv+1:p,:]')
                @views VTV[kconv+1:p, 1:kconv] .= VTV[1:kconv, kconv+1:p]'
                @views mul!(XTV[kconv+1:p, kconv+1:p], X[kconv+1:p,:], V[kconv+1:p,:]')
                @views mul!(XTV[1:kconv, kconv+1:p], X[1:kconv,:], V[kconv+1:p,:]')
                @views mul!(XTV[kconv+1:p, 1:kconv], X[kconv+1:p,:], V[1:kconv,:]')
            else
                mul!(VTV, V, V')
                mul!(XTV, X, V')
            end
            MPI.Allreduce!(VTV, +, comm_row)
            MPI.Allreduce!(VTV, +, comm_col)
            MPI.Allreduce!(XTV, +, comm_row)
            MPI.Allreduce!(XTV, +, comm_col)
            # mul!(K, X, X')
            # MPI.Allreduce!(K, +, comm_row)
            # MPI.Allreduce!(K, +, comm_col)
            if locking
                @views gV[kconv+1:p,:] .= V[kconv+1:p,:]
                @views gV[kconv+1:p,:] .= SpMM_A_1_w_E(gV[kconv+1:p,:], A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col)
            else
                gV .= V
                gV .= SpMM_A_1_w_E(gV, A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col)
            end
            cputime["nmv"] += p-kconv
            # mul!(VTAV, V, gV')
            # mul!(VTAX, V, gX')
            if locking
                @views mul!(VTAV[kconv+1:p, kconv+1:p], V[kconv+1:p,:], gV[kconv+1:p,:]')
                @views mul!(VTAV[1:kconv, kconv+1:p], V[1:kconv,:], gV[kconv+1:p,:]')
                @views VTAV[kconv+1:p, 1:kconv] .= VTAV[1:kconv, kconv+1:p]'
                @views mul!(VTAX[kconv+1:p, kconv+1:p], V[kconv+1:p,:], gX[kconv+1:p,:]')
                @views mul!(VTAX[1:kconv, kconv+1:p], V[1:kconv,:], gX[kconv+1:p,:]')
                @views mul!(VTAX[kconv+1:p, 1:kconv], V[kconv+1:p,:], gX[1:kconv,:]')
            else
                mul!(VTAV, V, gV')
                mul!(VTAX, V, gX')
            end
            MPI.Allreduce!(VTAV, +, comm_row)
            MPI.Allreduce!(VTAV, +, comm_col)
            MPI.Allreduce!(VTAX, +, comm_row)
            MPI.Allreduce!(VTAX, +, comm_col)
            if tri
                # a3 = cumsum(diag(VTV*UpperTriangular(VTV)))
                # a2 = cumsum(diag(VTV*UpperTriangular(XTV)+VTV*UpperTriangular(XTV')+XTV'*UpperTriangular(VTV)))
                # a1 = cumsum(diag(VTAV+XTV'*UpperTriangular(XTV')+XTV'*UpperTriangular(XTV)+VTV*UpperTriangular(K)))
                # a0 = cumsum(diag(VTAX+XTV'*UpperTriangular(K)))
                a3 = cumsum(diagMM(VTV, UpperTriangular(VTV)))
                a2 = cumsum(diagMM(VTV, UpperTriangular(XTV))+diagMM(VTV, UpperTriangular(XTV'))+diagMM(XTV', UpperTriangular(VTV)))
                a1 = cumsum(diag(VTAV)+diagMM(XTV', UpperTriangular(XTV'))+diagMM(XTV', UpperTriangular(XTV))+diagMM(VTV, UpperTriangular(K)))
                a0 = cumsum(diag(VTAX)+diagMM(XTV', UpperTriangular(K)))
                alphas = map(cardano, a3, a2, a1, a0)
            else
                # a3 = tr(VTV*VTV)
                # a2 = 3*tr(VTV*XTV)
                # a1 = tr(VTAV+XTV'*XTV'+XTV'*XTV+VTV*K)
                # a0 = tr(VTAX+XTV'*K)
                a3 = sum(diagMM(VTV, VTV))
                a2 = 3*sum(diagMM(VTV, XTV))
                a1 = sum(diag(VTAV)+diagMM(XTV', XTV')+diagMM(XTV', XTV)+diagMM(VTV, K))
                a0 = sum(diag(VTAX)+diagMM(XTV', K))
                alpha = cardano(a3,a2,a1,a0)
                alphas = [alpha for i in 1:p]
            end
        end

        if bb == "bb1"
            s = X - X0
            y = G1 - G
            diagss = diag(s*s')
            diagsy = diag(s*y')
            MPI.Allreduce!(diagss, +, comm_row)
            MPI.Allreduce!(diagss, +, comm_col)
            MPI.Allreduce!(diagsy, +, comm_row)
            MPI.Allreduce!(diagsy, +, comm_col)
            alphas .= diagss ./ diagsy
            # alphas .= diagsy ./ diagss
            X0 .= X
        elseif bb == "bb2"
            s = X - X0
            y = G1 - G
            diagyy = diag(y*y')
            diagsy = diag(s*y')
            MPI.Allreduce!(diagyy, +, comm_row)
            MPI.Allreduce!(diagyy, +, comm_col)
            MPI.Allreduce!(diagsy, +, comm_row)
            MPI.Allreduce!(diagsy, +, comm_col)
            alphas .= diagsy ./ diagyy
            X0 .= X
        end
        
        if locking
            X[kconv+1:p,:] .+= Diagonal(alphas[kconv+1:p])*V[kconv+1:p,:]
            # Da = Diagonal(alphas[kconv+1:p])
            # X[kconv+1:p,:] .= mul!(X[kconv+1:p,:], Da, V[kconv+1:p,:], 1.0, 1.0)
        else
            Da = Diagonal(alphas)
            mul!(X, Da, V, 1.0, 1.0)
        end
        G .= G1
        end

        iter += 1
    end
    end
    X, iter, cputime
end


function rayleigh(A_global, V_global)
    Vt,_ = pqr(V_global')
    H1 = Vt'*A_global*Vt
    d, Q = eigen(H1)
    Q'*Vt', d
end


function rayleigh_local(A, V, N, L, comms, counts_info, info_cols_dist, rank, rank_row, rank_col, comm_row, comm_col, root, comm_size_sq)
    cputime = @elapsed begin
    W = TSQR_1(V, N, L, comms, counts_info, rank, comm_row, comm_col)
    inds = [ind for ind = 1:size(A,1)]
    E = rank_col == rank_row ? sparse(inds, inds, ones(Float64, size(A, 1)), size(A,1), size(A,2)) : sparse([1], [1], [0.0], size(A,1), size(A,2))
    AW = SpMM_A_1_w_E(W, A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col)
    WAW = W*AW'
    MPI.Allreduce!(WAW, +, comm_row)
    MPI.Allreduce!(WAW, +, comm_col)
    d, Q = eigen(WAW)
    W = Q'*W
    end
    AW = SpMM_A_1_w_E(W, A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col)
    D = spdiagm(d)
    WD = D'*W
    err1 = zeros(1)
    err2 = zeros(1)
    err1[1] = norm(AW-WD)^2
    err2[1] = norm(WD)^2
    MPI.Allreduce!(err1, +, comm_row)
    MPI.Allreduce!(err1, +, comm_col)
    MPI.Allreduce!(err2, +, comm_row)
    MPI.Allreduce!(err2, +, comm_col)
    err = err1[1]/err2[1]
    W, d, err, cputime
end


function ofm_loss(A, X, X_global, info_cols_dist, comm, comm_row, comm_col, rank, rank_row, rank_col, comm_size, comm_size_sq)
    rblk = X_global[:, sum(info_cols_dist[1:rank_row*comm_size_sq])+1:sum(info_cols_dist[1:(rank_row+1)*comm_size_sq])]
    cblk = X_global[:, sum(info_cols_dist[1:rank_col*comm_size_sq])+1:sum(info_cols_dist[1:(rank_col+1)*comm_size_sq])]
    nm = zeros(1)
    nm[1] = norm(A + rblk'*cblk)^2
    MPI.Allreduce!(nm, +, comm)
    nm[1]
end

function ofm_loss_global(A_global, X_global)
    norm(A_global + X_global'*X_global)^2
end

function omm_loss_global(A_global, X_global)
    p = size(X_global, 1)
    inds = [i for i in 1:p]
    E = sparse(inds, inds, ones(Float64, p), p, p)
    tr((2*E-X_global*X_global')*X_global*A_global*X_global')
end

function omm_mpi(A, X, N, comm_info; alpha=0.01, beta=0.0, cg=true, linesearch=true, locking=true, tri=true, precond=nothing, bb=nothing, log2deg=1, itermax=200, tol=1e-3, lowb=-2.0, upb=0.0, base=2)
    cputime = Dict("checkConvergence"=>0.0, "Ax"=>0.0, "CG"=>0.0, "updateGradient"=>0.0, "linesearch"=>0.0, "nmv"=>0.0)
    
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

    inds = [ind for ind = 1:size(A,1)]
    E = rank_col == rank_row ? sparse(inds, inds, ones(Float64, size(A, 1)), size(A,1), size(A,2)) : sparse([1], [1], [0.0], size(A,1), size(A,2))


    p, n = size(X)
    G = zeros(p, n)
    G1 = zeros(p, n)
    V = zeros(p, n)
    alphas = zeros(p)
    betas = [beta for i in 1:p]
    gX = deepcopy(X)
    XTAX = zeros(p, p)
    K = zeros(p, p)
    mul!(K, X, X')
    MPI.Allreduce!(K, +, comm_row)
    MPI.Allreduce!(K, +, comm_col)

    VTV = nothing
    XTV = nothing
    gV = nothing
    VTAV = nothing
    VTAX = nothing
    X0 = nothing
    if linesearch
        VTV = zeros(p, p)
        XTV = zeros(p, p)
        gV = zeros(p, n)
        VTAV = zeros(p, p)
        VTAX = zeros(p, p)
    end
    if bb != nothing
        X0 = deepcopy(X)
    end


    # if log2deg > 0
    #     for i in 1:log2deg
    #         gX, _ =  Cheb_filter_composition(log2deg,  lowb, upb, gX, A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col, base=base)
    #     end
    # else
    #     gX = SpMM_A_1_w_E(gX, A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col)
    # end
    gX .= SpMM_A_1_w_E(gX, A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col)

    mul!(XTAX, X, gX')
    MPI.Allreduce!(XTAX, +, comm_row)
    MPI.Allreduce!(XTAX, +, comm_col)

    if tri
        # G .= 2 .* gX - LowerTriangular(K)*gX - LowerTriangular(XTAX)*X
        mul!(G, 2.0, gX)
        mul!(G, LowerTriangular(K), gX, -1.0, 1.0)
        mul!(G, LowerTriangular(XTAX), X, -1.0, 1.0)

    else
        # G .= 4 .* gX - 2 .* (XTAX*X) - 2 .* (K*gX)
        mul!(G, 2.0, gX)
        mul!(G, XTAX, X, -1.0, 1.0)
        mul!(G, K, gX, -1.0, 1.0)
        lmul!(2.0, G)
    end

    V .= -G

    if cg
        if precond == "filter"
            # G, _ =  Cheb_filter_composition(log2deg,  lowb, upb, G, A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col, base=base)
            G, _ = Cheb_filter_scal_1(base, lowb, upb, -2.0, G, A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col)
        end
    end

    if linesearch
        mul!(VTV, V, V')
        mul!(XTV, X, V')
        MPI.Allreduce!(VTV, +, comm_row)
        MPI.Allreduce!(VTV, +, comm_col)
        MPI.Allreduce!(XTV, +, comm_row)
        MPI.Allreduce!(XTV, +, comm_col)
        gV .= V
        # if log2deg > 0
        #     for i in 1:log2deg
        #         gV, _ =  Cheb_filter_composition(log2deg,  lowb, upb, gV, A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col, base=base)
        #     end
        # else
        #     gV .= SpMM_A_1_w_E(gV, A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col)
        # end
        gV .= SpMM_A_1_w_E(gV, A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col)
        mul!(VTAV, V, gV')
        mul!(VTAX, V, gX')
        MPI.Allreduce!(VTAV, +, comm_row)
        MPI.Allreduce!(VTAV, +, comm_col)
        MPI.Allreduce!(VTAX, +, comm_row)
        MPI.Allreduce!(VTAX, +, comm_col)
        if tri
            # a3 = -cumsum(diag(VTAV*UpperTriangular(VTV)+VTV*UpperTriangular(VTAV)))
            # a2 = -cumsum(diag(VTAX*UpperTriangular(VTV)+VTAV*UpperTriangular(XTV)+VTAV*UpperTriangular(XTV')+XTV'*UpperTriangular(VTAV)+VTV*UpperTriangular(VTAX')+VTV*UpperTriangular(VTAX)))
            # a1 = cumsum(diag(2*VTAV-VTAX*UpperTriangular(XTV)-VTAX*UpperTriangular(XTV')-VTAV*UpperTriangular(K)-XTV'*UpperTriangular(VTAX')-XTV'*UpperTriangular(VTAX)-VTV*UpperTriangular(XTAX)))
            # a0 = cumsum(diag(2*VTAX-VTAX*UpperTriangular(K)-XTV'*UpperTriangular(XTAX)))
            a3 = -cumsum(diagMM(VTAV, UpperTriangular(VTV))+diagMM(VTV, UpperTriangular(VTAV)))
            a2 = -cumsum(diagMM(VTAX, UpperTriangular(VTV))+diagMM(VTAV, UpperTriangular(XTV))+diagMM(VTAV, UpperTriangular(XTV'))+diagMM(XTV', UpperTriangular(VTAV))+diagMM(VTV, UpperTriangular(VTAX'))+diagMM(VTV, UpperTriangular(VTAX)))
            a1 = cumsum(diag(2*VTAV)-diagMM(VTAX, UpperTriangular(XTV))-diagMM(VTAX, UpperTriangular(XTV'))-diagMM(VTAV, UpperTriangular(K))-diagMM(XTV', UpperTriangular(VTAX'))-diagMM(XTV', UpperTriangular(VTAX))-diagMM(VTV, UpperTriangular(XTAX)))
            a0 = cumsum(diag(2*VTAX)-diagMM(VTAX, UpperTriangular(K))-diagMM(XTV', UpperTriangular(XTAX)))
            alphas = map(cardano, a3, a2, a1, a0)
        else
            # a3 = -4*tr(VTV*VTAV)
            # a2 = -6*tr(XTV'*VTAV+VTV*VTAX)
            # a1 = tr(4*VTAV-4*XTV'*VTAX'-4*XTV'*VTAX-2*VTV*XTAX-2*VTAV*K)
            # a0 = tr(4*VTAX-2*XTV'*XTAX-2*VTAX*K)
            a3 = -4*sum(diagMM(VTV, VTAV))
            a2 = -6*sum(diagMM(XTV', VTAV)+diagMM(VTV, VTAX))
            a1 = sum(diag(4*VTAV)-diagMM(4*XTV', VTAX')-diagMM(4*XTV', VTAX)-diagMM(2*VTV, XTAX)-diagMM(2*VTAV, K))
            a0 = sum(diag(4*VTAX)-diagMM(2*XTV', XTAX)-diagMM(2*VTAX, K))
            alpha = cardano(a3,a2,a1,a0)
            alphas = [alpha for i in 1:p]
        end
    else
        alphas = [alpha for i in 1:p]
    end

    # X .+= Diagonal(alphas)*V
    Da = Diagonal(alphas)
    mul!(X, Da, V, 1.0, 1.0)
    end

    iter = 0
    kconv = 0
    cputime["main_loop"] = @elapsed begin
    while iter < itermax
        cputime["checkConvergence"] += @elapsed begin
        if locking
            norm_G = norm.(eachrow(G[kconv+1:p])) .^ 2
            norm_gX = norm.(eachrow(gX[kconv+1:p])) .^ 2
            MPI.Allreduce!(norm_G, +, comm_row)
            MPI.Allreduce!(norm_G, +, comm_col)
            MPI.Allreduce!(norm_gX, +, comm_row)
            MPI.Allreduce!(norm_gX, +, comm_col)
            norm_G = sqrt.(norm_G)
            norm_gX = sqrt.(norm_gX)
            norm_GgX = norm_G .* norm_gX
            convn = norm_GgX .< tol
            for ic in 1:length(convn)
                if convn[ic]
                    kconv += 1
                else
                    break
                end
            end
            # permutateRtM(X, kconv, p, convn)
            # permutateRtM(G, kconv, p, convn)
            # permutateRtM(V, kconv, p, convn)
            # permutateRtM(gX, kconv, p, convn)
            # permutateRtM(G1, kconv, p, convn)
            # permutateSqM(K, kconv, p, convn)
            # permutateSqM(XTAX, kconv, p, convn)
            # if linesearch
            #     permutateSqM(VTV, kconv, p, convn)
            #     permutateSqM(XTV, kconv, p, convn)
            #     permutateSqM(gV, kconv, p, convn)
            #     permutateSqM(VTAV, kconv, p, convn)
            #     permutateSqM(VTAX, kconv, p, convn)
            # end
            # kconv += sum(convn)
            if kconv == p
                break
            end
        end
        end
        
        cputime["Ax"] += @elapsed begin
        if locking
            @views gX[kconv+1:p,:] .= X[kconv+1:p,:]
            @views gX[kconv+1:p,:] .= SpMM_A_1_w_E(gX[kconv+1:p,:], A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col)
        else
            gX .= X
            gX .= SpMM_A_1_w_E(gX, A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col)
        end
        end
        cputime["nmv"] += p-kconv

    
        cputime["updateGradient"] += @elapsed begin
        # mul!(K, X, X')
        if locking
            @views mul!(K[kconv+1:p, kconv+1:p], X[kconv+1:p,:], X[kconv+1:p,:]')
            @views mul!(K[1:kconv, kconv+1:p], X[1:kconv,:], X[kconv+1:p,:]')
            @views K[kconv+1:p, 1:kconv] .= K[1:kconv, kconv+1:p]'
        else
            mul!(K, X, X')
        end
        MPI.Allreduce!(K, +, comm_row)
        MPI.Allreduce!(K, +, comm_col)
        # mul!(XTAX, X, gX')
        if locking
            @views mul!(XTAX[kconv+1:p, kconv+1:p], X[kconv+1:p,:], gX[kconv+1:p,:]')
            @views mul!(XTAX[1:kconv, kconv+1:p], X[1:kconv,:], gX[kconv+1:p,:]')
            @views XTAX[kconv+1:p, 1:kconv] .= XTAX[1:kconv, kconv+1:p]'
        else
            mul!(XTAX, X, gX')
        end
        MPI.Allreduce!(XTAX, +, comm_row)
        MPI.Allreduce!(XTAX, +, comm_col)
        if tri
            # G1[kconv+1:p,:] = 2 .* gX[kconv+1:p,:] - LowerTriangular(K)[kconv+1:p,:]*gX - LowerTriangular(XTAX)[kconv+1:p,:]*X
            if locking
                G1[kconv+1:p,:] .= 2 .* gX[kconv+1:p,:] .- LowerTriangular(K)[kconv+1:p,:]*gX .- LowerTriangular(XTAX)[kconv+1:p,:]*X
                # G1[kconv+1:p,:] .= mul!(G1[kconv+1:p,:], 2.0, gX[kconv+1:p,:])
                # G1[kconv+1:p,:] .= mul!(G1[kconv+1:p,:], LowerTriangular(K)[kconv+1:p,:], gX, -1.0, 1.0)
                # G1[kconv+1:p,:] .= mul!(G1[kconv+1:p,:], LowerTriangular(XTAX)[kconv+1:p,:], X, -1.0, 1.0)
            else
                mul!(G1, 2.0, gX)
                mul!(G1, LowerTriangular(K), gX, -1.0, 1.0)
                mul!(G1, LowerTriangular(XTAX), X, -1.0, 1.0)
            end
        else
            G1 .= 4 .* gX - 2 .* (XTAX*X) - 2 .* (K*gX)
            mul!(G1, 2.0, gX)
            mul!(G1, XTAX, X, -1.0, 1.0)
            mul!(G1, K, gX, -1.0, 1.0)
            lmul!(2.0, G1)
        end
        end
        
        cputime["CG"] += @elapsed begin
        if cg
            if precond == "filter"
                # G1[kconv+1:p,:], _ =  Cheb_filter_composition(log2deg,  lowb, upb, G1[kconv+1:p,:], A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col, base=base)
                G1[kconv+1:p,:], _ = Cheb_filter_scal_1(base, lowb, upb, -2.0, G1[kconv+1:p,:], A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col)
            end
            # numerators = diag((G1[kconv+1:p,:] - G[kconv+1:p,:])*G1[kconv+1:p,:]')
            # denomurators = diag(G[kconv+1:p,:]*G[kconv+1:p,:]')
            numerators = zeros(p-kconv)
            denomurators = zeros(p-kconv)
            if locking
                dotMulVec(numerators, G1[kconv+1:p,:] - G[kconv+1:p,:], G1[kconv+1:p,:])
                dotMulVec(denomurators, G[kconv+1:p,:], G[kconv+1:p,:])
            else
                dotMulVec(numerators, G1 - G, G1)
                dotMulVec(denomurators, G, G)
            end
            MPI.Allreduce!(numerators, +, comm_row)
            MPI.Allreduce!(numerators, +, comm_col)
            MPI.Allreduce!(denomurators, +, comm_row)
            MPI.Allreduce!(denomurators, +, comm_col)
            betas[kconv+1:p] = numerators ./ denomurators
        end
        ## G .= G1
        # V[kconv+1:p,:] = Diagonal(betas[kconv+1:p])*V[kconv+1:p,:] - G1[kconv+1:p,:]
        if locking
            Db = Diagonal(betas[kconv+1:p])
            @views lmul!(Db, V[kconv+1:p,:])
            @views V[kconv+1:p,:] .-= G1[kconv+1:p,:]
        else
            Db = Diagonal(betas)
            lmul!(Db, V)
            V .-= G1
        end
        end

        cputime["linesearch"] += @elapsed begin
        if linesearch
            # mul!(VTV, V, V')
            # mul!(XTV, X, V')
            if locking
                @views mul!(VTV[kconv+1:p, kconv+1:p], V[kconv+1:p,:], V[kconv+1:p,:]')
                @views mul!(VTV[1:kconv, kconv+1:p], V[1:kconv,:], V[kconv+1:p,:]')
                @views VTV[kconv+1:p, 1:kconv] .= VTV[1:kconv, kconv+1:p]'
                @views mul!(XTV[kconv+1:p, kconv+1:p], X[kconv+1:p,:], V[kconv+1:p,:]')
                @views mul!(XTV[1:kconv, kconv+1:p], X[1:kconv,:], V[kconv+1:p,:]')
                @views mul!(XTV[kconv+1:p, 1:kconv], X[kconv+1:p,:], V[1:kconv,:]')
            else
                mul!(VTV, V, V')
                mul!(XTV, X, V')
            end
            MPI.Allreduce!(VTV, +, comm_row)
            MPI.Allreduce!(VTV, +, comm_col)
            MPI.Allreduce!(XTV, +, comm_row)
            MPI.Allreduce!(XTV, +, comm_col)
            # mul!(K, X, X')
            # MPI.Allreduce!(K, +, comm_row)
            # MPI.Allreduce!(K, +, comm_col)
            if locking
                @views gV[kconv+1:p,:] .= V[kconv+1:p,:]
                @views gV[kconv+1:p,:] .= SpMM_A_1_w_E(gV[kconv+1:p,:], A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col)
            else
                gV .= V
                gV .= SpMM_A_1_w_E(gV, A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col)
            end
            cputime["nmv"] += p-kconv
            # mul!(VTAV, V, gV')
            # mul!(VTAX, V, gX')
            if locking
                @views mul!(VTAV[kconv+1:p, kconv+1:p], V[kconv+1:p,:], gV[kconv+1:p,:]')
                @views mul!(VTAV[1:kconv, kconv+1:p], V[1:kconv,:], gV[kconv+1:p,:]')
                @views VTAV[kconv+1:p, 1:kconv] .= VTAV[1:kconv, kconv+1:p]'
                @views mul!(VTAX[kconv+1:p, kconv+1:p], V[kconv+1:p,:], gX[kconv+1:p,:]')
                @views mul!(VTAX[1:kconv, kconv+1:p], V[1:kconv,:], gX[kconv+1:p,:]')
                @views mul!(VTAX[kconv+1:p, 1:kconv], V[kconv+1:p,:], gX[1:kconv,:]')
            else
                mul!(VTAV, V, gV')
                mul!(VTAX, V, gX')
            end
            MPI.Allreduce!(VTAV, +, comm_row)
            MPI.Allreduce!(VTAV, +, comm_col)
            MPI.Allreduce!(VTAX, +, comm_row)
            MPI.Allreduce!(VTAX, +, comm_col)
            if tri
                # a3 = -cumsum(diag(VTAV*UpperTriangular(VTV)+VTV*UpperTriangular(VTAV)))
                # a2 = -cumsum(diag(VTAX*UpperTriangular(VTV)+VTAV*UpperTriangular(XTV)+VTAV*UpperTriangular(XTV')+XTV'*UpperTriangular(VTAV)+VTV*UpperTriangular(VTAX')+VTV*UpperTriangular(VTAX)))
                # a1 = cumsum(diag(2*VTAV-VTAX*UpperTriangular(XTV)-VTAX*UpperTriangular(XTV')-VTAV*UpperTriangular(K)-XTV'*UpperTriangular(VTAX')-XTV'*UpperTriangular(VTAX)-VTV*UpperTriangular(XTAX)))
                # a0 = cumsum(diag(2*VTAX-VTAX*UpperTriangular(K)-XTV'*UpperTriangular(XTAX)))
                a3 = -cumsum(diagMM(VTAV, UpperTriangular(VTV))+diagMM(VTV, UpperTriangular(VTAV)))
                a2 = -cumsum(diagMM(VTAX, UpperTriangular(VTV))+diagMM(VTAV, UpperTriangular(XTV))+diagMM(VTAV, UpperTriangular(XTV'))+diagMM(XTV', UpperTriangular(VTAV))+diagMM(VTV, UpperTriangular(VTAX'))+diagMM(VTV, UpperTriangular(VTAX)))
                a1 = cumsum(diag(2*VTAV)-diagMM(VTAX, UpperTriangular(XTV))-diagMM(VTAX, UpperTriangular(XTV'))-diagMM(VTAV, UpperTriangular(K))-diagMM(XTV', UpperTriangular(VTAX'))-diagMM(XTV', UpperTriangular(VTAX))-diagMM(VTV, UpperTriangular(XTAX)))
                a0 = cumsum(diag(2*VTAX)-diagMM(VTAX, UpperTriangular(K))-diagMM(XTV', UpperTriangular(XTAX)))
                alphas = map(cardano, a3, a2, a1, a0)
            else
                # a3 = -4*tr(VTV*VTAV)
                # a2 = -6*tr(XTV'*VTAV+VTV*VTAX)
                # a1 = tr(4*VTAV-4*XTV'*VTAX'-4*XTV'*VTAX-2*VTV*XTAX-2*VTAV*K)
                # a0 = tr(4*VTAX-2*XTV'*XTAX-2*VTAX*K)
                a3 = -4*sum(diagMM(VTV, VTAV))
                a2 = -6*sum(diagMM(XTV', VTAV)+diagMM(VTV, VTAX))
                a1 = sum(diag(4*VTAV)-diagMM(4*XTV', VTAX')-diagMM(4*XTV', VTAX)-diagMM(2*VTV, XTAX)-diagMM(2*VTAV, K))
                a0 = sum(diag(4*VTAX)-diagMM(2*XTV', XTAX)-diagMM(2*VTAX, K))
                alpha = cardano(a3,a2,a1,a0)
                alphas = [alpha for i in 1:p]
            end
        end

        if bb == "bb1"
            s = X - X0
            y = G1 - G
            diagss = diag(s*s')
            diagsy = diag(s*y')
            MPI.Allreduce!(diagss, +, comm_row)
            MPI.Allreduce!(diagss, +, comm_col)
            MPI.Allreduce!(diagsy, +, comm_row)
            MPI.Allreduce!(diagsy, +, comm_col)
            alphas .= diagss ./ diagsy
            # alphas .= diagsy ./ diagss
            X0 .= X
        elseif bb == "bb2"
            s = X - X0
            y = G1 - G
            diagyy = diag(y*y')
            diagsy = diag(s*y')
            MPI.Allreduce!(diagyy, +, comm_row)
            MPI.Allreduce!(diagyy, +, comm_col)
            MPI.Allreduce!(diagsy, +, comm_row)
            MPI.Allreduce!(diagsy, +, comm_col)
            alphas .= diagsy ./ diagyy
            X0 .= X
        end
        
        if locking
            X[kconv+1:p,:] .+= Diagonal(alphas[kconv+1:p])*V[kconv+1:p,:]
            # Da = Diagonal(alphas[kconv+1:p])
            # X[kconv+1:p,:] .= mul!(X[kconv+1:p,:], Da, V[kconv+1:p,:], 1.0, 1.0)
        else
            Da = Diagonal(alphas)
            mul!(X, Da, V, 1.0, 1.0)
        end
        G .= G1
        end

        iter += 1
    end
    end

    X, iter, cputime
end










function ofm_mpi_sq(A, X, N, comm_info; alpha=0.01, beta=0.0, cg=true, linesearch=true, locking=true, tri=true, precond=nothing, bb=nothing, log2deg=1, itermax=200, tol=1e-3, lowb=-2.0, upb=0.0, base=2)
    cputime = Dict("checkConvergence"=>0.0, "Ax"=>0.0, "CG"=>0.0, "updateGradient"=>0.0, "linesearch"=>0.0, "nmv"=>0.0)

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

    inds = [ind for ind = 1:size(A,1)]
    E = rank_col == rank_row ? sparse(inds, inds, ones(Float64, size(A, 1)), size(A,1), size(A,2)) : sparse([1], [1], [0.0], size(A,1), size(A,2))


    p, n = size(X)
    G = zeros(p, n)
    G1 = zeros(p, n)
    V = zeros(p, n)
    alphas = zeros(p)
    betas = [beta for i in 1:p]
    gX = deepcopy(X)
    K = zeros(p, p)
    mul!(K, X, X')
    # MPI.Allreduce!(K, +, comm_row)
    # MPI.Allreduce!(K, +, comm_col)

    VTV = nothing
    XTV = nothing
    gV = nothing
    VTAV = nothing
    VTAX = nothing
    X0 = nothing
    if linesearch
        VTV = zeros(p, p)
        XTV = zeros(p, p)
        gV = zeros(p, n)
        VTAV = zeros(p, p)
        VTAX = zeros(p, p)
    end
    if bb != nothing
        X0 = deepcopy(X)
    end


    # gX = SpMM_A_1_w_E(gX, A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col)
    mul!(gX, X, A)

    if tri
        # G .= gX + LowerTriangular(K)*X
        mul!(G, LowerTriangular(K), X)
        G .+= gX
    else
        # G .= gX + K*X
        mul!(G, K, X)
        G .+= gX
    end

    V .= -G

    if cg
        if precond == "filter"
            # G, _ =  Cheb_filter_composition(log2deg, lowb, upb, G, A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col, base=base)
            G, _ = Cheb_filter_scal_1(base, lowb, upb, -2.0, G, A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col)
        end
    end

    if linesearch
        mul!(VTV, V, V')
        mul!(XTV, X, V')
        # MPI.Allreduce!(VTV, +, comm_row)
        # MPI.Allreduce!(VTV, +, comm_col)
        # MPI.Allreduce!(XTV, +, comm_row)
        # MPI.Allreduce!(XTV, +, comm_col)
        # gV .= deepcopy(V)
        # gV .= SpMM_A_1_w_E(gV, A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col)
        mul!(gV, V, A)
        mul!(VTAV, V, gV')
        mul!(VTAX, V, gX')
        # MPI.Allreduce!(VTAV, +, comm_row)
        # MPI.Allreduce!(VTAV, +, comm_col)
        # MPI.Allreduce!(VTAX, +, comm_row)
        # MPI.Allreduce!(VTAX, +, comm_col)
        if tri
            # a3 = cumsum(diag(VTV*UpperTriangular(VTV)))
            # a2 = cumsum(diag(VTV*UpperTriangular(XTV)+VTV*UpperTriangular(XTV')+XTV'*UpperTriangular(VTV)))
            # a1 = cumsum(diag(VTAV+XTV'*UpperTriangular(XTV')+XTV'*UpperTriangular(XTV)+VTV*UpperTriangular(K)))
            # a0 = cumsum(diag(VTAX+XTV'*UpperTriangular(K)))
            a3 = cumsum(diagMM(VTV, UpperTriangular(VTV)))
            a2 = cumsum(diagMM(VTV, UpperTriangular(XTV))+diagMM(VTV, UpperTriangular(XTV'))+diagMM(XTV', UpperTriangular(VTV)))
            a1 = cumsum(diag(VTAV)+diagMM(XTV', UpperTriangular(XTV'))+diagMM(XTV', UpperTriangular(XTV))+diagMM(VTV, UpperTriangular(K)))
            a0 = cumsum(diag(VTAX)+diagMM(XTV', UpperTriangular(K)))
            alphas = map(cardano, a3, a2, a1, a0)
        else
            # a3 = tr(VTV*VTV)
            # a2 = 3*tr(VTV*XTV)
            # a1 = tr(VTAV+XTV'*XTV'+XTV'*XTV+VTV*K)
            # a0 = tr(VTAX+XTV'*K)
            a3 = sum(diagMM(VTV, VTV))
            a2 = 3*sum(diagMM(VTV, XTV))
            a1 = sum(diag(VTAV)+diagMM(XTV', XTV')+diagMM(XTV', XTV)+diagMM(VTV, K))
            a0 = sum(diag(VTAX)+diagMM(XTV', K))
            alpha = cardano(a3,a2,a1,a0)
            alphas = [alpha for i in 1:p]
        end
    else
        alphas = [alpha for i in 1:p]
    end

    # X .+= Diagonal(alphas)*V
    Da = Diagonal(alphas)
    mul!(X, Da, V, 1.0, 1.0)
    end

    iter = 0
    kconv = 0
    cputime["main_loop"] = @elapsed begin
    while iter < itermax
        cputime["checkConvergence"] += @elapsed begin
        if locking
            norm_G = norm.(eachrow(G[kconv+1:p])) .^ 2
            norm_gX = norm.(eachrow(gX[kconv+1:p])) .^ 2
            norm_G = sqrt.(norm_G)
            norm_gX = cbrt.(sqrt.(norm_gX))
            norm_GgX = norm_G .* norm_gX
            convn = norm_GgX .< tol
            for ic in 1:length(convn)
                if convn[ic]
                    kconv += 1
                else
                    break
                end
            end
            # permutateRtM(X, kconv, p, convn)
            # permutateRtM(G, kconv, p, convn)
            # permutateRtM(V, kconv, p, convn)
            # permutateRtM(gX, kconv, p, convn)
            # permutateRtM(G1, kconv, p, convn)
            # permutateSqM(K, kconv, p, convn)
            # if linesearch
            #     permutateSqM(VTV, kconv, p, convn)
            #     permutateSqM(XTV, kconv, p, convn)
            #     permutateSqM(gV, kconv, p, convn)
            #     permutateSqM(VTAV, kconv, p, convn)
            #     permutateSqM(VTAX, kconv, p, convn)
            # end
            # kconv += sum(convn)
            if kconv == p
                break
            end
        end
        end

        cputime["Ax"] += @elapsed begin
        if locking
            # gX[kconv+1:p,:] .= X[kconv+1:p,:]
            # gX[kconv+1:p,:] .= SpMM_A_1_w_E(gX[kconv+1:p,:], A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col)
            @views gX[kconv+1:p,:] .= X[kconv+1:p,:]*A
        else
            # gX .= X
            # gX .= SpMM_A_1_w_E(gX, A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col)
            mul!(gX, X, A)
        end
        end
        cputime["nmv"] += p-kconv

        cputime["updateGradient"] += @elapsed begin
        # mul!(K, X, X')
        if locking
            @views mul!(K[kconv+1:p, kconv+1:p], X[kconv+1:p,:], X[kconv+1:p,:]')
            @views mul!(K[1:kconv, kconv+1:p], X[1:kconv,:], X[kconv+1:p,:]')
            @views K[kconv+1:p, 1:kconv] .= K[1:kconv, kconv+1:p]'
        else
            mul!(K, X, X')
        end
        # MPI.Allreduce!(K, +, comm_row)
        # MPI.Allreduce!(K, +, comm_col)
        if tri
            if locking
                G1[kconv+1:p,:] .= gX[kconv+1:p,:] .+ LowerTriangular(K)[kconv+1:p,:]*X
                # G1[kconv+1:p,:] .= mul!(G1[kconv+1:p,:], LowerTriangular(K)[kconv+1:p,:], X)
                # G1[kconv+1:p,:] .+= gX[kconv+1:p,:]
            else
                mul!(G1, LowerTriangular(K), X)
                G1 .+= gX
            end
        else
            # G1 .= gX + K*X
            mul!(G1, K, X)
            G1 .+= gX
        end
        end
        
        cputime["CG"] += @elapsed begin
        if cg
            if precond == "filter"
                # G1[kconv+1:p,:], _ =  Cheb_filter_composition(log2deg,  lowb, upb, G1[kconv+1:p,:], A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col, base=base)
                G1[kconv+1:p,:], _ = Cheb_filter_scal_1(base, lowb, upb, -2.0, G1[kconv+1:p,:], A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col)
            end
            # numerators = diag((G1[kconv+1:p,:] - G[kconv+1:p,:])*G1[kconv+1:p,:]')
            # denomurators = diag(G[kconv+1:p,:]*G[kconv+1:p,:]')
            numerators = zeros(p-kconv)
            denomurators = zeros(p-kconv)
            if locking
                dotMulVec(numerators, G1[kconv+1:p,:] - G[kconv+1:p,:], G1[kconv+1:p,:])
                dotMulVec(denomurators, G[kconv+1:p,:], G[kconv+1:p,:])
            else
                dotMulVec(numerators, G1 - G, G1)
                dotMulVec(denomurators, G, G)
            end
            # MPI.Allreduce!(numerators, +, comm_row)
            # MPI.Allreduce!(numerators, +, comm_col)
            # MPI.Allreduce!(denomurators, +, comm_row)
            # MPI.Allreduce!(denomurators, +, comm_col)
            betas[kconv+1:p] = numerators ./ denomurators
        end
        # G .= G1
        if locking
            # V[kconv+1:p,:] = Diagonal(betas[kconv+1:p])*V[kconv+1:p,:] - G1[kconv+1:p,:]
            Db = Diagonal(betas[kconv+1:p])
            @views lmul!(Db, V[kconv+1:p,:])
            @views V[kconv+1:p,:] .-= G1[kconv+1:p,:]
        else
            Db = Diagonal(betas)
            lmul!(Db, V)
            V .-= G1
        end
        end
        
        cputime["linesearch"] += @elapsed begin
        if linesearch
            # mul!(VTV, V, V')
            # mul!(XTV, X, V')
            if locking
                @views mul!(VTV[kconv+1:p, kconv+1:p], V[kconv+1:p,:], V[kconv+1:p,:]')
                @views mul!(VTV[1:kconv, kconv+1:p], V[1:kconv,:], V[kconv+1:p,:]')
                @views VTV[kconv+1:p, 1:kconv] .= VTV[1:kconv, kconv+1:p]'
                @views mul!(XTV[kconv+1:p, kconv+1:p], X[kconv+1:p,:], V[kconv+1:p,:]')
                @views mul!(XTV[1:kconv, kconv+1:p], X[1:kconv,:], V[kconv+1:p,:]')
                @views mul!(XTV[kconv+1:p, 1:kconv], X[kconv+1:p,:], V[1:kconv,:]')
            else
                mul!(VTV, V, V')
                mul!(XTV, X, V')
            end
            # MPI.Allreduce!(VTV, +, comm_row)
            # MPI.Allreduce!(VTV, +, comm_col)
            # MPI.Allreduce!(XTV, +, comm_row)
            # MPI.Allreduce!(XTV, +, comm_col)
            # mul!(K, X, X')
            # MPI.Allreduce!(K, +, comm_row)
            # MPI.Allreduce!(K, +, comm_col)
            if locking
                # gV[kconv+1:p,:] .= V[kconv+1:p,:]
                # gV[kconv+1:p,:] .= SpMM_A_1_w_E(gV[kconv+1:p,:], A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col)
                @views gV[kconv+1:p,:] .= V[kconv+1:p,:]*A
            else
                # gV .= V
                # gV .= SpMM_A_1_w_E(gV, A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col)
                mul!(gV, V, A)
            end
            cputime["nmv"] += p-kconv
            # mul!(VTAV, V, gV')
            # mul!(VTAX, V, gX')
            if locking
                @views mul!(VTAV[kconv+1:p, kconv+1:p], V[kconv+1:p,:], gV[kconv+1:p,:]')
                @views mul!(VTAV[1:kconv, kconv+1:p], V[1:kconv,:], gV[kconv+1:p,:]')
                @views VTAV[kconv+1:p, 1:kconv] .= VTAV[1:kconv, kconv+1:p]'
                @views mul!(VTAX[kconv+1:p, kconv+1:p], V[kconv+1:p,:], gX[kconv+1:p,:]')
                @views mul!(VTAX[1:kconv, kconv+1:p], V[1:kconv,:], gX[kconv+1:p,:]')
                @views mul!(VTAX[kconv+1:p, 1:kconv], V[kconv+1:p,:], gX[1:kconv,:]')
            else
                mul!(VTAV, V, gV')
                mul!(VTAX, V, gX')
            end
            # MPI.Allreduce!(VTAV, +, comm_row)
            # MPI.Allreduce!(VTAV, +, comm_col)
            # MPI.Allreduce!(VTAX, +, comm_row)
            # MPI.Allreduce!(VTAX, +, comm_col)
            if tri
                # a3 = cumsum(diag(VTV*UpperTriangular(VTV)))
                # a2 = cumsum(diag(VTV*UpperTriangular(XTV)+VTV*UpperTriangular(XTV')+XTV'*UpperTriangular(VTV)))
                # a1 = cumsum(diag(VTAV+XTV'*UpperTriangular(XTV')+XTV'*UpperTriangular(XTV)+VTV*UpperTriangular(K)))
                # a0 = cumsum(diag(VTAX+XTV'*UpperTriangular(K)))
                a3 = cumsum(diagMM(VTV, UpperTriangular(VTV)))
                a2 = cumsum(diagMM(VTV, UpperTriangular(XTV))+diagMM(VTV, UpperTriangular(XTV'))+diagMM(XTV', UpperTriangular(VTV)))
                a1 = cumsum(diag(VTAV)+diagMM(XTV', UpperTriangular(XTV'))+diagMM(XTV', UpperTriangular(XTV))+diagMM(VTV, UpperTriangular(K)))
                a0 = cumsum(diag(VTAX)+diagMM(XTV', UpperTriangular(K)))
                alphas = map(cardano, a3, a2, a1, a0)
            else
                # a3 = tr(VTV*VTV)
                # a2 = 3*tr(VTV*XTV)
                # a1 = tr(VTAV+XTV'*XTV'+XTV'*XTV+VTV*K)
                # a0 = tr(VTAX+XTV'*K)
                a3 = sum(diagMM(VTV, VTV))
                a2 = 3*sum(diagMM(VTV, XTV))
                a1 = sum(diag(VTAV)+diagMM(XTV', XTV')+diagMM(XTV', XTV)+diagMM(VTV, K))
                a0 = sum(diag(VTAX)+diagMM(XTV', K))
                alpha = cardano(a3,a2,a1,a0)
                alphas = [alpha for i in 1:p]
            end
        end

        if bb == "bb1"
            s = X - X0
            y = G1 - G
            diagss = diag(s*s')
            diagsy = diag(s*y')
            # MPI.Allreduce!(diagss, +, comm_row)
            # MPI.Allreduce!(diagss, +, comm_col)
            # MPI.Allreduce!(diagsy, +, comm_row)
            # MPI.Allreduce!(diagsy, +, comm_col)
            alphas .= diagss ./ diagsy
            # alphas .= diagsy ./ diagss
            X0 .= X
        elseif bb == "bb2"
            s = X - X0
            y = G1 - G
            diagyy = diag(y*y')
            diagsy = diag(s*y')
            # MPI.Allreduce!(diagyy, +, comm_row)
            # MPI.Allreduce!(diagyy, +, comm_col)
            # MPI.Allreduce!(diagsy, +, comm_row)
            # MPI.Allreduce!(diagsy, +, comm_col)
            alphas .= diagsy ./ diagyy
            X0 .= X
        end
        
        if locking
            X[kconv+1:p,:] .+= Diagonal(alphas[kconv+1:p])*V[kconv+1:p,:]
            # Da = Diagonal(alphas[kconv+1:p])
            # X[kconv+1:p,:] .= mul!(X[kconv+1:p,:], Da, V[kconv+1:p,:], 1.0, 1.0)
        else
            Da = Diagonal(alphas)
            mul!(X, Da, V, 1.0, 1.0)
        end
        G .= G1
        end

        iter += 1
    end
    end
    X, iter, cputime
end





function omm_mpi_sq(A, X, N, comm_info; alpha=0.01, beta=0.0, cg=true, linesearch=true, locking=true, tri=true, precond=nothing, bb=nothing, log2deg=1, itermax=200, tol=1e-3, lowb=-2.0, upb=0.0, base=2)
    cputime = Dict("checkConvergence"=>0.0, "Ax"=>0.0, "CG"=>0.0, "updateGradient"=>0.0, "linesearch"=>0.0, "nmv"=>0.0)
    
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

    inds = [ind for ind = 1:size(A,1)]
    E = rank_col == rank_row ? sparse(inds, inds, ones(Float64, size(A, 1)), size(A,1), size(A,2)) : sparse([1], [1], [0.0], size(A,1), size(A,2))


    p, n = size(X)
    G = zeros(p, n)
    G1 = zeros(p, n)
    V = zeros(p, n)
    alphas = zeros(p)
    betas = [beta for i in 1:p]
    gX = deepcopy(X)
    XTAX = zeros(p, p)
    K = zeros(p, p)
    mul!(K, X, X')
    # MPI.Allreduce!(K, +, comm_row)
    # MPI.Allreduce!(K, +, comm_col)

    VTV = nothing
    XTV = nothing
    gV = nothing
    VTAV = nothing
    VTAX = nothing
    X0 = nothing
    if linesearch
        VTV = zeros(p, p)
        XTV = zeros(p, p)
        gV = zeros(p, n)
        VTAV = zeros(p, p)
        VTAX = zeros(p, p)
    end
    if bb != nothing
        X0 = deepcopy(X)
    end


    # gX = SpMM_A_1_w_E(gX, A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col)
    mul!(gX, X, A)

    mul!(XTAX, X, gX')
    # MPI.Allreduce!(XTAX, +, comm_row)
    # MPI.Allreduce!(XTAX, +, comm_col)

    if tri
        # G .= 2 .* gX - LowerTriangular(K)*gX - LowerTriangular(XTAX)*X
        mul!(G, 2.0, gX)
        mul!(G, LowerTriangular(K), gX, -1.0, 1.0)
        mul!(G, LowerTriangular(XTAX), X, -1.0, 1.0)

    else
        # G .= 4 .* gX - 2 .* (XTAX*X) - 2 .* (K*gX)
        mul!(G, 2.0, gX)
        mul!(G, XTAX, X, -1.0, 1.0)
        mul!(G, K, gX, -1.0, 1.0)
        lmul!(2.0, G)
    end

    V .= -G

    if cg
        if precond == "filter"
            # G, _ =  Cheb_filter_composition(log2deg,  lowb, upb, G, A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col, base=base)
            G, _ = Cheb_filter_scal_1(base, lowb, upb, -2.0, G, A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col)
        end
    end

    if linesearch
        mul!(VTV, V, V')
        mul!(XTV, X, V')
        # MPI.Allreduce!(VTV, +, comm_row)
        # MPI.Allreduce!(VTV, +, comm_col)
        # MPI.Allreduce!(XTV, +, comm_row)
        # MPI.Allreduce!(XTV, +, comm_col)
        # gV .= V
        # gV .= SpMM_A_1_w_E(gV, A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col)
        mul!(gV, V, A)
        mul!(VTAV, V, gV')
        mul!(VTAX, V, gX')
        # MPI.Allreduce!(VTAV, +, comm_row)
        # MPI.Allreduce!(VTAV, +, comm_col)
        # MPI.Allreduce!(VTAX, +, comm_row)
        # MPI.Allreduce!(VTAX, +, comm_col)
        if tri
            # a3 = -cumsum(diag(VTAV*UpperTriangular(VTV)+VTV*UpperTriangular(VTAV)))
            # a2 = -cumsum(diag(VTAX*UpperTriangular(VTV)+VTAV*UpperTriangular(XTV)+VTAV*UpperTriangular(XTV')+XTV'*UpperTriangular(VTAV)+VTV*UpperTriangular(VTAX')+VTV*UpperTriangular(VTAX)))
            # a1 = cumsum(diag(2*VTAV-VTAX*UpperTriangular(XTV)-VTAX*UpperTriangular(XTV')-VTAV*UpperTriangular(K)-XTV'*UpperTriangular(VTAX')-XTV'*UpperTriangular(VTAX)-VTV*UpperTriangular(XTAX)))
            # a0 = cumsum(diag(2*VTAX-VTAX*UpperTriangular(K)-XTV'*UpperTriangular(XTAX)))
            a3 = -cumsum(diagMM(VTAV, UpperTriangular(VTV))+diagMM(VTV, UpperTriangular(VTAV)))
            a2 = -cumsum(diagMM(VTAX, UpperTriangular(VTV))+diagMM(VTAV, UpperTriangular(XTV))+diagMM(VTAV, UpperTriangular(XTV'))+diagMM(XTV', UpperTriangular(VTAV))+diagMM(VTV, UpperTriangular(VTAX'))+diagMM(VTV, UpperTriangular(VTAX)))
            a1 = cumsum(diag(2*VTAV)-diagMM(VTAX, UpperTriangular(XTV))-diagMM(VTAX, UpperTriangular(XTV'))-diagMM(VTAV, UpperTriangular(K))-diagMM(XTV', UpperTriangular(VTAX'))-diagMM(XTV', UpperTriangular(VTAX))-diagMM(VTV, UpperTriangular(XTAX)))
            a0 = cumsum(diag(2*VTAX)-diagMM(VTAX, UpperTriangular(K))-diagMM(XTV', UpperTriangular(XTAX)))
            alphas = map(cardano, a3, a2, a1, a0)
        else
            # a3 = -4*tr(VTV*VTAV)
            # a2 = -6*tr(XTV'*VTAV+VTV*VTAX)
            # a1 = tr(4*VTAV-4*XTV'*VTAX'-4*XTV'*VTAX-2*VTV*XTAX-2*VTAV*K)
            # a0 = tr(4*VTAX-2*XTV'*XTAX-2*VTAX*K)
            a3 = -4*sum(diagMM(VTV, VTAV))
            a2 = -6*sum(diagMM(XTV', VTAV)+diagMM(VTV, VTAX))
            a1 = sum(diag(4*VTAV)-diagMM(4*XTV', VTAX')-diagMM(4*XTV', VTAX)-diagMM(2*VTV, XTAX)-diagMM(2*VTAV, K))
            a0 = sum(diag(4*VTAX)-diagMM(2*XTV', XTAX)-diagMM(2*VTAX, K))
            alpha = cardano(a3,a2,a1,a0)
            alphas = [alpha for i in 1:p]
        end
    else
        alphas = [alpha for i in 1:p]
    end

    # X .+= Diagonal(alphas)*V
    Da = Diagonal(alphas)
    mul!(X, Da, V, 1.0, 1.0)
    end

    iter = 0
    kconv = 0
    cputime["main_loop"] = @elapsed begin
    while iter < itermax
        cputime["checkConvergence"] += @elapsed begin
        if locking
            norm_G = norm.(eachrow(G[kconv+1:p])) .^ 2
            norm_gX = norm.(eachrow(gX[kconv+1:p])) .^ 2
            norm_G = sqrt.(norm_G)
            norm_gX = sqrt.(norm_gX)
            norm_GgX = norm_G .* norm_gX
            convn = norm_GgX .< tol
            for ic in 1:length(convn)
                if convn[ic]
                    kconv += 1
                else
                    break
                end
            end
            # permutateRtM(X, kconv, p, convn)
            # permutateRtM(G, kconv, p, convn)
            # permutateRtM(V, kconv, p, convn)
            # permutateRtM(gX, kconv, p, convn)
            # permutateRtM(G1, kconv, p, convn)
            # permutateSqM(K, kconv, p, convn)
            # permutateSqM(XTAX, kconv, p, convn)
            # if linesearch
            #     permutateSqM(VTV, kconv, p, convn)
            #     permutateSqM(XTV, kconv, p, convn)
            #     permutateSqM(gV, kconv, p, convn)
            #     permutateSqM(VTAV, kconv, p, convn)
            #     permutateSqM(VTAX, kconv, p, convn)
            # end
            # kconv += sum(convn)
            if kconv == p
                break
            end
        end
        end
        
        cputime["Ax"] += @elapsed begin
        if locking
            # gX[kconv+1:p,:] .= X[kconv+1:p,:]
            # gX[kconv+1:p,:] .= SpMM_A_1_w_E(gX[kconv+1:p,:], A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col)
            @views gX[kconv+1:p,:] .= X[kconv+1:p,:]*A
        else
            # gX .= X
            # gX .= SpMM_A_1_w_E(gX, A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col)
            mul!(gX, X, A)
        end
        end
        cputime["nmv"] += p-kconv

    
        cputime["updateGradient"] += @elapsed begin
        # mul!(K, X, X')
        if locking
            @views mul!(K[kconv+1:p, kconv+1:p], X[kconv+1:p,:], X[kconv+1:p,:]')
            @views mul!(K[1:kconv, kconv+1:p], X[1:kconv,:], X[kconv+1:p,:]')
            @views K[kconv+1:p, 1:kconv] .= K[1:kconv, kconv+1:p]'
        else
            mul!(K, X, X')
        end
        # MPI.Allreduce!(K, +, comm_row)
        # MPI.Allreduce!(K, +, comm_col)
        # mul!(XTAX, X, gX')
        if locking
            @views mul!(XTAX[kconv+1:p, kconv+1:p], X[kconv+1:p,:], gX[kconv+1:p,:]')
            @views mul!(XTAX[1:kconv, kconv+1:p], X[1:kconv,:], gX[kconv+1:p,:]')
            @views XTAX[kconv+1:p, 1:kconv] .= XTAX[1:kconv, kconv+1:p]'
        else
            mul!(XTAX, X, gX')
        end
        # MPI.Allreduce!(XTAX, +, comm_row)
        # MPI.Allreduce!(XTAX, +, comm_col)
        if tri
            # G1[kconv+1:p,:] = 2 .* gX[kconv+1:p,:] - LowerTriangular(K)[kconv+1:p,:]*gX - LowerTriangular(XTAX)[kconv+1:p,:]*X
            if locking
                G1[kconv+1:p,:] .= 2 .* gX[kconv+1:p,:] .- LowerTriangular(K)[kconv+1:p,:]*gX .- LowerTriangular(XTAX)[kconv+1:p,:]*X
                # G1[kconv+1:p,:] .= mul!(G1[kconv+1:p,:], 2.0, gX[kconv+1:p,:])
                # G1[kconv+1:p,:] .= mul!(G1[kconv+1:p,:], LowerTriangular(K)[kconv+1:p,:], gX, -1.0, 1.0)
                # G1[kconv+1:p,:] .= mul!(G1[kconv+1:p,:], LowerTriangular(XTAX)[kconv+1:p,:], X, -1.0, 1.0)
            else
                mul!(G1, 2.0, gX)
                mul!(G1, LowerTriangular(K), gX, -1.0, 1.0)
                mul!(G1, LowerTriangular(XTAX), X, -1.0, 1.0)
            end
        else
            G1 .= 4 .* gX - 2 .* (XTAX*X) - 2 .* (K*gX)
            mul!(G1, 2.0, gX)
            mul!(G1, XTAX, X, -1.0, 1.0)
            mul!(G1, K, gX, -1.0, 1.0)
            lmul!(2.0, G1)
        end
        end
        
        cputime["CG"] += @elapsed begin
        if cg
            if precond == "filter"
                # G1[kconv+1:p,:], _ =  Cheb_filter_composition(log2deg,  lowb, upb, G1[kconv+1:p,:], A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col, base=base)
                G1[kconv+1:p,:], _ = Cheb_filter_scal_1(base, lowb, upb, -2.0, G1[kconv+1:p,:], A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col)
            end
            # numerators = diag((G1[kconv+1:p,:] - G[kconv+1:p,:])*G1[kconv+1:p,:]')
            # denomurators = diag(G[kconv+1:p,:]*G[kconv+1:p,:]')
            numerators = zeros(p-kconv)
            denomurators = zeros(p-kconv)
            if locking
                dotMulVec(numerators, G1[kconv+1:p,:] - G[kconv+1:p,:], G1[kconv+1:p,:])
                dotMulVec(denomurators, G[kconv+1:p,:], G[kconv+1:p,:])
            else
                dotMulVec(numerators, G1 - G, G1)
                dotMulVec(denomurators, G, G)
            end
            # MPI.Allreduce!(numerators, +, comm_row)
            # MPI.Allreduce!(numerators, +, comm_col)
            # MPI.Allreduce!(denomurators, +, comm_row)
            # MPI.Allreduce!(denomurators, +, comm_col)
            betas[kconv+1:p] = numerators ./ denomurators
        end
        ## G .= G1
        # V[kconv+1:p,:] = Diagonal(betas[kconv+1:p])*V[kconv+1:p,:] - G1[kconv+1:p,:]
        if locking
            Db = Diagonal(betas[kconv+1:p])
            @views lmul!(Db, V[kconv+1:p,:])
            @views V[kconv+1:p,:] .-= G1[kconv+1:p,:]
        else
            Db = Diagonal(betas)
            lmul!(Db, V)
            V .-= G1
        end
        end

        cputime["linesearch"] += @elapsed begin
        if linesearch
            # mul!(VTV, V, V')
            # mul!(XTV, X, V')
            if locking
                @views mul!(VTV[kconv+1:p, kconv+1:p], V[kconv+1:p,:], V[kconv+1:p,:]')
                @views mul!(VTV[1:kconv, kconv+1:p], V[1:kconv,:], V[kconv+1:p,:]')
                @views VTV[kconv+1:p, 1:kconv] .= VTV[1:kconv, kconv+1:p]'
                @views mul!(XTV[kconv+1:p, kconv+1:p], X[kconv+1:p,:], V[kconv+1:p,:]')
                @views mul!(XTV[1:kconv, kconv+1:p], X[1:kconv,:], V[kconv+1:p,:]')
                @views mul!(XTV[kconv+1:p, 1:kconv], X[kconv+1:p,:], V[1:kconv,:]')
            else
                mul!(VTV, V, V')
                mul!(XTV, X, V')
            end
            # MPI.Allreduce!(VTV, +, comm_row)
            # MPI.Allreduce!(VTV, +, comm_col)
            # MPI.Allreduce!(XTV, +, comm_row)
            # MPI.Allreduce!(XTV, +, comm_col)
            # mul!(K, X, X')
            # MPI.Allreduce!(K, +, comm_row)
            # MPI.Allreduce!(K, +, comm_col)
            if locking
                # gV[kconv+1:p,:] .= V[kconv+1:p,:]
                # gV[kconv+1:p,:] .= SpMM_A_1_w_E(gV[kconv+1:p,:], A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col)
                @views gV[kconv+1:p,:] .= V[kconv+1:p,:]*A
            else
                # gV .= V
                # gV .= SpMM_A_1_w_E(gV, A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col)
                mul!(gV, V, A)
            end
            cputime["nmv"] += p-kconv
            # mul!(VTAV, V, gV')
            # mul!(VTAX, V, gX')
            if locking
                @views mul!(VTAV[kconv+1:p, kconv+1:p], V[kconv+1:p,:], gV[kconv+1:p,:]')
                @views mul!(VTAV[1:kconv, kconv+1:p], V[1:kconv,:], gV[kconv+1:p,:]')
                @views VTAV[kconv+1:p, 1:kconv] .= VTAV[1:kconv, kconv+1:p]'
                @views mul!(VTAX[kconv+1:p, kconv+1:p], V[kconv+1:p,:], gX[kconv+1:p,:]')
                @views mul!(VTAX[1:kconv, kconv+1:p], V[1:kconv,:], gX[kconv+1:p,:]')
                @views mul!(VTAX[kconv+1:p, 1:kconv], V[kconv+1:p,:], gX[1:kconv,:]')
            else
                mul!(VTAV, V, gV')
                mul!(VTAX, V, gX')
            end
            # MPI.Allreduce!(VTAV, +, comm_row)
            # MPI.Allreduce!(VTAV, +, comm_col)
            # MPI.Allreduce!(VTAX, +, comm_row)
            # MPI.Allreduce!(VTAX, +, comm_col)
            if tri
                # a3 = -cumsum(diag(VTAV*UpperTriangular(VTV)+VTV*UpperTriangular(VTAV)))
                # a2 = -cumsum(diag(VTAX*UpperTriangular(VTV)+VTAV*UpperTriangular(XTV)+VTAV*UpperTriangular(XTV')+XTV'*UpperTriangular(VTAV)+VTV*UpperTriangular(VTAX')+VTV*UpperTriangular(VTAX)))
                # a1 = cumsum(diag(2*VTAV-VTAX*UpperTriangular(XTV)-VTAX*UpperTriangular(XTV')-VTAV*UpperTriangular(K)-XTV'*UpperTriangular(VTAX')-XTV'*UpperTriangular(VTAX)-VTV*UpperTriangular(XTAX)))
                # a0 = cumsum(diag(2*VTAX-VTAX*UpperTriangular(K)-XTV'*UpperTriangular(XTAX)))
                a3 = -cumsum(diagMM(VTAV, UpperTriangular(VTV))+diagMM(VTV, UpperTriangular(VTAV)))
                a2 = -cumsum(diagMM(VTAX, UpperTriangular(VTV))+diagMM(VTAV, UpperTriangular(XTV))+diagMM(VTAV, UpperTriangular(XTV'))+diagMM(XTV', UpperTriangular(VTAV))+diagMM(VTV, UpperTriangular(VTAX'))+diagMM(VTV, UpperTriangular(VTAX)))
                a1 = cumsum(diag(2*VTAV)-diagMM(VTAX, UpperTriangular(XTV))-diagMM(VTAX, UpperTriangular(XTV'))-diagMM(VTAV, UpperTriangular(K))-diagMM(XTV', UpperTriangular(VTAX'))-diagMM(XTV', UpperTriangular(VTAX))-diagMM(VTV, UpperTriangular(XTAX)))
                a0 = cumsum(diag(2*VTAX)-diagMM(VTAX, UpperTriangular(K))-diagMM(XTV', UpperTriangular(XTAX)))
                alphas = map(cardano, a3, a2, a1, a0)
            else
                # a3 = -4*tr(VTV*VTAV)
                # a2 = -6*tr(XTV'*VTAV+VTV*VTAX)
                # a1 = tr(4*VTAV-4*XTV'*VTAX'-4*XTV'*VTAX-2*VTV*XTAX-2*VTAV*K)
                # a0 = tr(4*VTAX-2*XTV'*XTAX-2*VTAX*K)
                a3 = -4*sum(diagMM(VTV, VTAV))
                a2 = -6*sum(diagMM(XTV', VTAV)+diagMM(VTV, VTAX))
                a1 = sum(diag(4*VTAV)-diagMM(4*XTV', VTAX')-diagMM(4*XTV', VTAX)-diagMM(2*VTV, XTAX)-diagMM(2*VTAV, K))
                a0 = sum(diag(4*VTAX)-diagMM(2*XTV', XTAX)-diagMM(2*VTAX, K))
                alpha = cardano(a3,a2,a1,a0)
                alphas = [alpha for i in 1:p]
            end
        end

        if bb == "bb1"
            s = X - X0
            y = G1 - G
            diagss = diag(s*s')
            diagsy = diag(s*y')
            # MPI.Allreduce!(diagss, +, comm_row)
            # MPI.Allreduce!(diagss, +, comm_col)
            # MPI.Allreduce!(diagsy, +, comm_row)
            # MPI.Allreduce!(diagsy, +, comm_col)
            alphas .= diagss ./ diagsy
            # alphas .= diagsy ./ diagss
            X0 .= X
        elseif bb == "bb2"
            s = X - X0
            y = G1 - G
            diagyy = diag(y*y')
            diagsy = diag(s*y')
            # MPI.Allreduce!(diagyy, +, comm_row)
            # MPI.Allreduce!(diagyy, +, comm_col)
            # MPI.Allreduce!(diagsy, +, comm_row)
            # MPI.Allreduce!(diagsy, +, comm_col)
            alphas .= diagsy ./ diagyy
            X0 .= X
        end
        
        if locking
            X[kconv+1:p,:] .+= Diagonal(alphas[kconv+1:p])*V[kconv+1:p,:]
            # Da = Diagonal(alphas[kconv+1:p])
            # X[kconv+1:p,:] .= mul!(X[kconv+1:p,:], Da, V[kconv+1:p,:], 1.0, 1.0)
        else
            Da = Diagonal(alphas)
            mul!(X, Da, V, 1.0, 1.0)
        end
        G .= G1
        end

        iter += 1
    end
    end

    X, iter, cputime
end