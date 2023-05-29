using Printf
using Statistics
using LowRankApprox
using MPI
using LinearAlgebra
using Test
using SparseArrays
using Tullio
using MKLSparse

include("../utils/utils.jl")

function ofm_hybrid(A, X, N, comm_info; alpha=0.01, beta=0.0, cg=true, linesearch=true, locking=true, tri=true, precond=nothing, bb=nothing, log2deg=1, itermax=200, tol=1e-3, lowb=-2.0, upb=0.0, base=2)
    cputime = Dict("checkConvergence"=>0.0, "Ax"=>0.0, "CG"=>0.0, "updateGradient"=>0.0, "linesearch"=>0.0, "nmv"=>0.0, "updateGradient_X*XT"=>0.0, "updateGradient_Allreduce!"=>0.0, "updateGradient_G1"=>0.0, "CG_dotMulVec"=>0.0, "CG_Allreduce!"=>0.0, "CG_V"=>0.0, "linesearch_mul!"=>0.0, "linesearch_Allreduce!"=>0.0, "linesearch_AV"=>0.0, "linesearch_alphas"=>0.0, "linesearch_X"=>0.0)

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

    inds = [ind for ind = 1:size(A,1)]
    E = rank_col == rank_row ? sparse(inds, inds, ones(Float64, size(A, 1)), size(A,1), size(A,2)) : sparse([1], [1], [0.0], size(A,1), size(A,2))


    n, p = size(X)
    XT = Matrix(transpose(X))
    G = zeros(n, p)
    G1 = zeros(n, p)
    V = zeros(n, p)
    alphas = zeros(p)
    betas = [beta for i in 1:p]
    gX = deepcopy(X)
    K = zeros(p, p)
    AuX = zeros(p, n)
    mul!(K, X', X)
    MPI.Allreduce!(K, +, comm_row)
    MPI.Allreduce!(K, +, comm_col)
    UpperK = Matrix(UpperTriangular(K))

    VTV = nothing
    XTV = nothing
    gV = nothing
    VTAV = nothing
    VTAX = nothing
    X0 = nothing
    if linesearch
        VTV = zeros(p, p)
        XTV = zeros(p, p)
        gV = zeros(n, p)
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
    transpose!(AuX, gX)
    AuX = SpMM_A_1_w_E(AuX, A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col)
    transpose!(gX, AuX)

    if tri
        # G .= gX + LowerTriangular(K)*X
        UpperK = Matrix(UpperTriangular(K))
        mul!(G, X, UpperK)
        G = G + gX
    else
        # G .= gX + K*X
        mul!(G, X, K)
        G = G + gX
    end

    V = V - G

    if cg
        if precond == "filter"
            # G, _ =  Cheb_filter_composition(log2deg, lowb, upb, G, A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col, base=base)
            G, _ = Cheb_filter_scal_1(base, lowb, upb, -2.0, G, A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col)
        end
    end

    if linesearch
        mul!(VTV, V', V)
        mul!(XTV, X', V)
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
        #     gV = SpMM_A_1_w_E(gV, A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col)
        # end
        transpose!(AuX, gV)
        AuX = SpMM_A_1_w_E(AuX, A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col)
        transpose!(gV, AuX)
        mul!(VTAV, V', gV)
        mul!(VTAX, V', gX)
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
    mul!(X, V, Da, 1.0, 1.0)
    end

    iter = 0
    kconv = 0
    cputime["main_loop"] = @elapsed begin
    while iter < itermax
        cputime["checkConvergence"] += @elapsed begin
        if locking
            norm_G = norm.(eachcol(G[:, kconv+1:p])) .^ 2
            norm_gX = norm.(eachcol(gX[:, kconv+1:p])) .^ 2
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
            #     permutateRtM(gV, kconv, p, convn)
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
            @views transpose!(AuX[kconv+1:p,:], X[:,kconv+1:p])
            @views AuX[kconv+1:p,:] = SpMM_A_1_w_E(AuX[kconv+1:p,:], A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col)
            @views transpose!(gX[:,kconv+1:p], AuX[kconv+1:p,:])
        else
            transpose!(AuX, X)
            AuX = SpMM_A_1_w_E(AuX, A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col)
            transpose!(gX, AuX)
        end
        end
        cputime["nmv"] += p-kconv

        cputime["updateGradient"] += @elapsed begin
        # mul!(K, X, X')
        cputime["updateGradient_X*XT"] += @elapsed begin
        if locking
            @views mul!(K[:, kconv+1:p], X', X[:,kconv+1:p])
            @views K[kconv+1:p, 1:kconv] = K[1:kconv, kconv+1:p]'
        else
            mul!(K, X', X)

            # TullioAAT(K, X)

            # transpose!(AuX, X)
            # mul!(K, X, AuX)
        end
        end
        cputime["updateGradient_Allreduce!"] += @elapsed begin
        MPI.Allreduce!(K, +, comm_row)
        MPI.Allreduce!(K, +, comm_col)
        end
        cputime["updateGradient_G1"] += @elapsed begin
        if tri
            if locking
                # G1[kconv+1:p,:] .= gX[kconv+1:p,:] .+ LowerTriangular(K)[kconv+1:p,:]*X
                UpperK = Matrix(UpperTriangular(K))
                @views mul!(G1[:,kconv+1:p], X, UpperK[:,kconv+1:p])
                @views G1[:,kconv+1:p] = G1[:,kconv+1:p] + gX[:,kconv+1:p]
            else
                UpperK = Matrix(UpperTriangular(K))
                mul!(G1, X, UpperK)
                G1 = G1 + gX
            end
        else
            # G1 .= gX + K*X

            mul!(G1, X, K)
            G1 = G1 + gX

            # TullioAB(G1, K, X)
            # TullioSelfAdd(G1, gX)
        end
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
            cputime["CG_dotMulVec"] += @elapsed begin
            if locking
                @views dotMulVec(numerators, G1[:,kconv+1:p] - G[:,kconv+1:p], G1[:,kconv+1:p])
                @views dotMulVec(denomurators, G[:,kconv+1:p], G[:,kconv+1:p])
            else
                dotMulVec(numerators, G1 - G, G1)
                dotMulVec(denomurators, G, G)
            end
            end
            cputime["CG_Allreduce!"] += @elapsed begin
            MPI.Allreduce!(numerators, +, comm_row)
            MPI.Allreduce!(numerators, +, comm_col)
            MPI.Allreduce!(denomurators, +, comm_row)
            MPI.Allreduce!(denomurators, +, comm_col)
            betas[kconv+1:p] = numerators ./ denomurators
            end
        end
        # G .= G1
        cputime["CG_V"] += @elapsed begin 
        if locking
            # V[kconv+1:p,:] = Diagonal(betas[kconv+1:p])*V[kconv+1:p,:] - G1[kconv+1:p,:]
            Db = Diagonal(betas[kconv+1:p])
            @views rmul!(V[:,kconv+1:p], Db)
            @views V[:,kconv+1:p] = V[:,kconv+1:p] - G1[:,kconv+1:p]
        else
            Db = Diagonal(betas)
            rmul!(V, Db)
            V = V - G1
        end
        end
        end
        
        cputime["linesearch"] += @elapsed begin
        if linesearch
            # mul!(VTV, V, V')
            # mul!(XTV, X, V')
            cputime["linesearch_mul!"] += @elapsed begin
            if locking
                @views mul!(VTV[:, kconv+1:p], V', V[:,kconv+1:p])
                @views VTV[kconv+1:p, 1:kconv] = VTV[1:kconv, kconv+1:p]'
                @views mul!(XTV[:, kconv+1:p], X', V[:,kconv+1:p])
                @views mul!(XTV[kconv+1:p, 1:kconv], X[:,kconv+1:p]', V[:,1:kconv])
            else
                mul!(VTV, V', V)
                mul!(XTV, X', V)

                # TullioAAT(VTV, V)
                # TullioABT(XTV, X, V)

                # transpose!(AuX, V)
                # mul!(VTV, V, AuX)
                # mul!(XTV, X, AuX)
            end
            end
            cputime["linesearch_Allreduce!"] += @elapsed begin
            MPI.Allreduce!(VTV, +, comm_row)
            MPI.Allreduce!(VTV, +, comm_col)
            MPI.Allreduce!(XTV, +, comm_row)
            MPI.Allreduce!(XTV, +, comm_col)
            end
            # mul!(K, X, X')
            # MPI.Allreduce!(K, +, comm_row)
            # MPI.Allreduce!(K, +, comm_col)
            cputime["linesearch_AV"] += @elapsed begin
            if locking
                @views transpose!(AuX[kconv+1:p, :], V[:,kconv+1:p])
                AuX[kconv+1:p, :], cputime_AV = @views SpMM_A_1_w_E_time(AuX[kconv+1:p, :], A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col)
                @views transpose!(gV[:,kconv+1:p], AuX[kconv+1:p, :])
                for (key, val) in cputime_AV
                    if haskey(cputime, "AV_" * key)
                        cputime["AV_" * key] += val*1.0
                    else
                        cputime["AV_" * key] = val*1.0
                    end
                end
            else
                transpose!(AuX, V)
                AuX, cputime_AV = SpMM_A_1_w_E_time(AuX, A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col)
                transpose!(gV, AuX)
                for (key, val) in cputime_AV
                    if haskey(cputime, "AV_" * key)
                        cputime["AV_" * key] += val*1.0
                    else
                        cputime["AV_" * key] = val*1.0
                    end
                end
            end
            end
            cputime["nmv"] += p-kconv
            # mul!(VTAV, V, gV')
            # mul!(VTAX, V, gX')
            cputime["linesearch_mul!"] += @elapsed begin
            if locking
                @views mul!(VTAV[:, kconv+1:p], V', gV[:,kconv+1:p])
                @views VTAV[kconv+1:p, 1:kconv] = VTAV[1:kconv, kconv+1:p]'
                @views mul!(VTAX[:, kconv+1:p], V', gX[:,kconv+1:p])
                @views mul!(VTAX[kconv+1:p, 1:kconv], V[:,kconv+1:p]', gX[:,1:kconv])
            else
                mul!(VTAV, V', gV)
                mul!(VTAX, V', gX)

                # TullioABT(VTAV, V, gV)
                # TullioABT(VTAX, V, gX)

                # transpose!(AuX, gV)
                # mul!(VTAV, V, AuX)
                # transpose!(AuX, gX)
                # mul!(VTAX, V, AuX)
            end
            end
            cputime["linesearch_Allreduce!"] += @elapsed begin
            MPI.Allreduce!(VTAV, +, comm_row)
            MPI.Allreduce!(VTAV, +, comm_col)
            MPI.Allreduce!(VTAX, +, comm_row)
            MPI.Allreduce!(VTAX, +, comm_col)
            end
            cputime["linesearch_alphas"] += @elapsed begin
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
        end

        if bb == "bb1"
            s = X - X0
            y = G1 - G
            diagss = diag(s'*s)
            diagsy = diag(s'*y)
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
            diagyy = diag(y'*y)
            diagsy = diag(s'*y)
            MPI.Allreduce!(diagyy, +, comm_row)
            MPI.Allreduce!(diagyy, +, comm_col)
            MPI.Allreduce!(diagsy, +, comm_row)
            MPI.Allreduce!(diagsy, +, comm_col)
            alphas .= diagsy ./ diagyy
            X0 .= X
        end
        
        cputime["linesearch_X"] += @elapsed begin
        if locking
            # X[kconv+1:p,:] .+= Diagonal(alphas[kconv+1:p])*V[kconv+1:p,:]
            Da = Diagonal(alphas[kconv+1:p])
            @views mul!(X[:,kconv+1:p], V[:,kconv+1:p], Da, 1.0, 1.0)
        else
            Da = Diagonal(alphas)
            mul!(X, V, Da, 1.0, 1.0)
        end
        G .= G1
        end
        end

        iter += 1
    end
    end
    X, iter, cputime
end


function omm_hybrid(A, X, N, comm_info; alpha=0.01, beta=0.0, cg=true, linesearch=true, locking=true, tri=true, precond=nothing, bb=nothing, log2deg=1, itermax=200, tol=1e-3, lowb=-2.0, upb=0.0, base=2)
    cputime = Dict("checkConvergence"=>0.0, "Ax"=>0.0, "CG"=>0.0, "updateGradient"=>0.0, "linesearch"=>0.0, "nmv"=>0.0, "updateGradient_X*XT"=>0.0, "updateGradient_Allreduce!"=>0.0, "updateGradient_G1"=>0.0, "CG_dotMulVec"=>0.0, "CG_Allreduce!"=>0.0, "CG_V"=>0.0, "linesearch_mul!"=>0.0, "linesearch_Allreduce!"=>0.0, "linesearch_AV"=>0.0, "linesearch_alphas"=>0.0, "linesearch_X"=>0.0)
    
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

    inds = [ind for ind = 1:size(A,1)]
    E = rank_col == rank_row ? sparse(inds, inds, ones(Float64, size(A, 1)), size(A,1), size(A,2)) : sparse([1], [1], [0.0], size(A,1), size(A,2))


    n, p = size(X)
    G = zeros(n, p)
    G1 = zeros(n, p)
    V = zeros(n, p)
    alphas = zeros(p)
    betas = [beta for i in 1:p]
    gX = deepcopy(X)
    XT = Matrix(transpose(X))
    AuX = zeros(p, n)
    XTAX = zeros(p, p)
    K = zeros(p, p)
    mul!(K, X', X)
    MPI.Allreduce!(K, +, comm_row)
    MPI.Allreduce!(K, +, comm_col)
    UpperK = Matrix(UpperTriangular(K))

    VTV = nothing
    XTV = nothing
    gV = nothing
    VTAV = nothing
    VTAX = nothing
    X0 = nothing
    if linesearch
        VTV = zeros(p, p)
        XTV = zeros(p, p)
        gV = zeros(n, p)
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
    transpose!(AuX, gX)
    AuX = SpMM_A_1_w_E(AuX, A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col)
    transpose!(gX, AuX)

    mul!(XTAX, X', gX)
    MPI.Allreduce!(XTAX, +, comm_row)
    MPI.Allreduce!(XTAX, +, comm_col)
    UpperXTAX = Matrix(UpperTriangular(XTAX))

    if tri
        # G .= 2 .* gX - LowerTriangular(K)*gX - LowerTriangular(XTAX)*X
        mul!(G, 2.0, gX)
        mul!(G, gX, UpperK, -1.0, 1.0)
        mul!(G, X, UpperXTAX, -1.0, 1.0)

    else
        # G .= 4 .* gX - 2 .* (XTAX*X) - 2 .* (K*gX)
        mul!(G, 2.0, gX)
        mul!(G, X, XTAX -1.0, 1.0)
        mul!(G, gX, K, -1.0, 1.0)
        lmul!(2.0, G)
    end

    V = V - G

    if cg
        if precond == "filter"
            # G, _ =  Cheb_filter_composition(log2deg,  lowb, upb, G, A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col, base=base)
            G, _ = Cheb_filter_scal_1(base, lowb, upb, -2.0, G, A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col)
        end
    end

    if linesearch
        mul!(VTV, V', V)
        mul!(XTV, X', V)
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
        transpose!(AuX, gV)
        AuX = SpMM_A_1_w_E(AuX, A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col)
        transpose!(gV, AuX)
        mul!(VTAV, V', gV)
        mul!(VTAX, V', gX)
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
    mul!(X, V, Da, 1.0, 1.0)
    end

    iter = 0
    kconv = 0
    cputime["main_loop"] = @elapsed begin
    while iter < itermax
        cputime["checkConvergence"] += @elapsed begin
        if locking
            norm_G = norm.(eachcol(G[:, kconv+1:p])) .^ 2
            norm_gX = norm.(eachcol(gX[:, kconv+1:p])) .^ 2
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
            #     permutateRtM(gV, kconv, p, convn)
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
            @views transpose!(AuX[kconv+1:p,:], X[:,kconv+1:p])
            @views AuX[kconv+1:p,:] = SpMM_A_1_w_E(AuX[kconv+1:p,:], A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col)
            @views transpose!(gX[:,kconv+1:p], AuX[kconv+1:p,:])
        else
            transpose!(AuX, X)
            AuX = SpMM_A_1_w_E(AuX, A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col)
            transpose!(gX, AuX)
        end
        end
        cputime["nmv"] += p-kconv

    
        cputime["updateGradient"] += @elapsed begin
        # mul!(K, X, X')
        cputime["updateGradient_X*XT"] += @elapsed begin
        if locking
            @views mul!(K[:, kconv+1:p], X', X[:,kconv+1:p])
            @views K[kconv+1:p, 1:kconv] = K[1:kconv, kconv+1:p]'
        else
            mul!(K, X', X)

            # TullioAAT(K, X)
        end
        end
        cputime["updateGradient_Allreduce!"] += @elapsed begin
        MPI.Allreduce!(K, +, comm_row)
        MPI.Allreduce!(K, +, comm_col)
        end
        # mul!(XTAX, X, gX')
        cputime["updateGradient_X*XT"] += @elapsed begin
        if locking
            @views mul!(XTAX[:, kconv+1:p], X', gX[:,kconv+1:p])
            @views XTAX[kconv+1:p, 1:kconv] = XTAX[1:kconv, kconv+1:p]'
        else
            mul!(XTAX, X', gX)

            # TullioABT(K, X, gX)
        end
        end
        cputime["updateGradient_Allreduce!"] += @elapsed begin
        MPI.Allreduce!(XTAX, +, comm_row)
        MPI.Allreduce!(XTAX, +, comm_col)
        end
        cputime["updateGradient_G1"] += @elapsed begin
        if tri
            # G1[kconv+1:p,:] = 2 .* gX[kconv+1:p,:] - LowerTriangular(K)[kconv+1:p,:]*gX - LowerTriangular(XTAX)[kconv+1:p,:]*X
            if locking
                # G1[kconv+1:p,:] .= 2 .* gX[kconv+1:p,:] .- LowerTriangular(K)[kconv+1:p,:]*gX .- LowerTriangular(XTAX)[kconv+1:p,:]*X
                UpperK = Matrix(UpperTriangular(K))
                UpperXTAX = Matrix(UpperTriangular(XTAX))
                @views mul!(G1[:,kconv+1:p], 2.0, gX[:,kconv+1:p])
                @views mul!(G1[:,kconv+1:p], gX, UpperK[:,kconv+1:p], -1.0, 1.0)
                @views mul!(G1[:,kconv+1:p], X, UpperXTAX[:,kconv+1:p], -1.0, 1.0)
            else
                UpperK = Matrix(UpperTriangular(K))
                UpperXTAX = Matrix(UpperTriangular(XTAX))
                mul!(G1, 2.0, gX)
                mul!(G1, gX, UpperK, -1.0, 1.0)
                mul!(G1, X, UpperXTAX, -1.0, 1.0)
            end
        else
            # G1 .= 4 .* gX - 2 .* (XTAX*X) - 2 .* (K*gX)
            mul!(G1, 2.0, gX)
            mul!(G1, X, XTAX -1.0, 1.0)
            mul!(G1, gX, K, -1.0, 1.0)
            lmul!(2.0, G1)
        end
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
            cputime["CG_dotMulVec"] += @elapsed begin
            if locking
                @views dotMulVec(numerators, G1[:,kconv+1:p] - G[:,kconv+1:p], G1[:,kconv+1:p])
                @views dotMulVec(denomurators, G[:,kconv+1:p], G[:,kconv+1:p])
            else
                dotMulVec(numerators, G1 - G, G1)
                dotMulVec(denomurators, G, G)
            end
            end
            cputime["CG_Allreduce!"] += @elapsed begin
            MPI.Allreduce!(numerators, +, comm_row)
            MPI.Allreduce!(numerators, +, comm_col)
            MPI.Allreduce!(denomurators, +, comm_row)
            MPI.Allreduce!(denomurators, +, comm_col)
            betas[kconv+1:p] = numerators ./ denomurators
            end
        end
        ## G .= G1
        # V[kconv+1:p,:] = Diagonal(betas[kconv+1:p])*V[kconv+1:p,:] - G1[kconv+1:p,:]
        cputime["CG_V"] += @elapsed begin 
        if locking
            Db = Diagonal(betas[kconv+1:p])
            @views rmul!(V[:,kconv+1:p], Db)
            @views V[:,kconv+1:p] = V[:,kconv+1:p] - G1[:,kconv+1:p]
        else
            Db = Diagonal(betas)
            rmul!(V, Db)
            V = V - G1
        end
        end
        end

        cputime["linesearch"] += @elapsed begin
        if linesearch
            # mul!(VTV, V, V')
            # mul!(XTV, X, V')
            cputime["linesearch_mul!"] += @elapsed begin
            if locking
                @views mul!(VTV[:, kconv+1:p], V', V[:,kconv+1:p])
                @views VTV[kconv+1:p, 1:kconv] = VTV[1:kconv, kconv+1:p]'
                @views mul!(XTV[:, kconv+1:p], X', V[:,kconv+1:p])
                @views mul!(XTV[kconv+1:p, 1:kconv], X[:,kconv+1:p]', V[:,1:kconv])
            else
                mul!(VTV, V', V)
                mul!(XTV, X', V)

                # TullioAAT(VTV, V)
                # TullioABT(XTV, X, V)
            end
            end
            cputime["linesearch_Allreduce!"] += @elapsed begin
            MPI.Allreduce!(VTV, +, comm_row)
            MPI.Allreduce!(VTV, +, comm_col)
            MPI.Allreduce!(XTV, +, comm_row)
            MPI.Allreduce!(XTV, +, comm_col)
            end
            # mul!(K, X, X')
            # MPI.Allreduce!(K, +, comm_row)
            # MPI.Allreduce!(K, +, comm_col)
            cputime["linesearch_AV"] += @elapsed begin
            if locking
                @views transpose!(AuX[kconv+1:p,:], V[:,kconv+1:p])
                AuX[kconv+1:p,:], cputime_AV = @views SpMM_A_1_w_E_time(AuX[kconv+1:p,:], A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col)
                @views transpose!(gV[:,kconv+1:p], AuX[kconv+1:p,:])
                for (key, val) in cputime_AV
                    if haskey(cputime, "AV_" * key)
                        cputime["AV_" * key] += val*1.0
                    else
                        cputime["AV_" * key] = val*1.0
                    end
                end
            else
                transpose!(AuX, V)
                AuX, cputime_AV = SpMM_A_1_w_E_time(AuX, A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col)
                transpose!(gV, AuX)
                for (key, val) in cputime_AV
                    if haskey(cputime, "AV_" * key)
                        cputime["AV_" * key] += val*1.0
                    else
                        cputime["AV_" * key] = val*1.0
                    end
                end
            end
            end
            cputime["nmv"] += p-kconv
            # mul!(VTAV, V, gV')
            # mul!(VTAX, V, gX')
            cputime["linesearch_mul!"] += @elapsed begin
            if locking
                @views mul!(VTAV[:, kconv+1:p], V', gV[:,kconv+1:p])
                @views VTAV[kconv+1:p, 1:kconv] = VTAV[1:kconv, kconv+1:p]'
                @views mul!(VTAX[:, kconv+1:p], V', gX[:,kconv+1:p])
                @views mul!(VTAX[kconv+1:p, 1:kconv], V[:,kconv+1:p]', gX[:,1:kconv])
            else
                mul!(VTAV, V', gV)
                mul!(VTAX, V', gX)

                # TullioABT(VTAV, V, gV)
                # TullioABT(VTAX, V, gX)
            end
            end
            cputime["linesearch_Allreduce!"] += @elapsed begin
            MPI.Allreduce!(VTAV, +, comm_row)
            MPI.Allreduce!(VTAV, +, comm_col)
            MPI.Allreduce!(VTAX, +, comm_row)
            MPI.Allreduce!(VTAX, +, comm_col)
            end
            cputime["linesearch_alphas"] += @elapsed begin
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
        end

        if bb == "bb1"
            s = X - X0
            y = G1 - G
            diagss = diag(s'*s)
            diagsy = diag(s'*y)
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
            diagyy = diag(y'*y)
            diagsy = diag(s'*y)
            MPI.Allreduce!(diagyy, +, comm_row)
            MPI.Allreduce!(diagyy, +, comm_col)
            MPI.Allreduce!(diagsy, +, comm_row)
            MPI.Allreduce!(diagsy, +, comm_col)
            alphas .= diagsy ./ diagyy
            X0 .= X
        end
        
        cputime["linesearch_X"] += @elapsed begin
        if locking
            # X[kconv+1:p,:] .+= Diagonal(alphas[kconv+1:p])*V[kconv+1:p,:]
            Da = Diagonal(alphas[kconv+1:p])
            @views mul!(X[:,kconv+1:p], V[:,kconv+1:p], Da, 1.0, 1.0)
        else
            Da = Diagonal(alphas)
            mul!(X, V, Da, 1.0, 1.0)
        end
        G .= G1
        end
        end

        iter += 1
    end
    end

    X, iter, cputime
end

