using Printf
using LinearAlgebra
using Test
using SparseArrays
using MKLSparse

include("../utils/utils.jl")

function ofm_mp(A, X, N; alpha=0.01, beta=0.0, cg=true, linesearch=true, locking=true, tri=true, precond=nothing, bb=nothing, log2deg=1, itermax=200, tol=1e-3, lowb=-2.0, upb=0.0, base=2)
    cputime = Dict("checkConvergence"=>0.0, "Ax"=>0.0, "CG"=>0.0, "updateGradient"=>0.0, "linesearch"=>0.0, "nmv"=>0.0)

    cputime["pre"] = @elapsed begin

    n, p = size(X)
    G = zeros(n, p)
    G1 = zeros(n, p)
    V = zeros(n, p)
    alphas = zeros(p)
    betas = [beta for i in 1:p]
    gX = deepcopy(X)
    K = zeros(p, p)
    UpperK = zeros(p, p)
    mul!(K, X', X)

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


    # gX = SpMM_A_1_w_E(gX, A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col)
    mul!(gX, A, X)

    if tri
        # G .= gX + LowerTriangular(K)*X
        UpperK .= Matrix(UpperTriangular(K))
        mul!(G, X, UpperK)
        G = G + gX
    else
        # G .= gX + K*X
        mul!(G, X, K)
        G = G + gX
    end

    V .= -G

    if cg
        if precond == "filter"
            # G, _ = Cheb_filter_scal_1(base, lowb, upb, -2.0, G, A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col)
        end
    end

    if linesearch
        mul!(VTV, V', V)
        mul!(XTV, X', V)
        mul!(gV, A, V)
        mul!(VTAV, V', gV)
        mul!(VTAX, V', gX)
        if tri
            a3 = cumsum(diagMM(VTV, UpperTriangular(VTV)))
            a2 = cumsum(diagMM(VTV, UpperTriangular(XTV))+diagMM(VTV, UpperTriangular(XTV'))+diagMM(XTV', UpperTriangular(VTV)))
            a1 = cumsum(diag(VTAV)+diagMM(XTV', UpperTriangular(XTV'))+diagMM(XTV', UpperTriangular(XTV))+diagMM(VTV, UpperTriangular(K)))
            a0 = cumsum(diag(VTAX)+diagMM(XTV', UpperTriangular(K)))
            alphas = map(cardano, a3, a2, a1, a0)
        else
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
            norm_G = norm.(eachcol(G[:,kconv+1:p]))
            norm_gX = norm.(eachcol(gX[:,kconv+1:p]))
            norm_gX = cbrt.(norm_gX)
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
            # gX[kconv+1:p,:] .= X[kconv+1:p,:]
            # gX[kconv+1:p,:] .= SpMM_A_1_w_E(gX[kconv+1:p,:], A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col)
            @views mul!(gX[:, kconv+1:p], A, X[:, kconv+1:p])
        else
            # gX .= X
            # gX .= SpMM_A_1_w_E(gX, A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col)
            mul!(gX, A, X)
        end
        end
        cputime["nmv"] += p-kconv

        cputime["updateGradient"] += @elapsed begin
        # mul!(K, X, X')
        if locking
            @views mul!(K[:, kconv+1:p], X', X[:,kconv+1:p])
            @views K[kconv+1:p, 1:kconv] = K[1:kconv, kconv+1:p]'
        else
            mul!(K, X', X)
        end
        # MPI.Allreduce!(K, +, comm_row)
        # MPI.Allreduce!(K, +, comm_col)
        if tri
            if locking
                UpperK = Matrix(UpperTriangular(K))
                @views mul!(G1[:,kconv+1:p], X, UpperK[:,kconv+1:p])
                @views G1[:,kconv+1:p] = G1[:,kconv+1:p] + gX[:,kconv+1:p]
                # G1[kconv+1:p,:] .= mul!(G1[kconv+1:p,:], LowerTriangular(K)[kconv+1:p,:], X)
                # G1[kconv+1:p,:] .+= gX[kconv+1:p,:]
            else
                UpperK = Matrix(UpperTriangular(K))
                mul!(G1, X, UpperK)
                G1 = G1 + gX
            end
        else
            # G1 .= gX + K*X
            mul!(G1, X, K)
            G1 = G1 + gX
        end
        end
        
        cputime["CG"] += @elapsed begin
        if cg
            if precond == "filter"
                # G1[kconv+1:p,:], _ = Cheb_filter_scal_1(base, lowb, upb, -2.0, G1[kconv+1:p,:], A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col)
            end
            # numerators = diag((G1[kconv+1:p,:] - G[kconv+1:p,:])*G1[kconv+1:p,:]')
            # denomurators = diag(G[kconv+1:p,:]*G[kconv+1:p,:]')
            numerators = zeros(p-kconv)
            denomurators = zeros(p-kconv)
            if locking
                @views dotMulVec(numerators, G1[:,kconv+1:p] - G[:,kconv+1:p], G1[:,kconv+1:p])
                @views dotMulVec(denomurators, G[:,kconv+1:p], G[:,kconv+1:p])
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
            @views rmul!(V[:,kconv+1:p], Db)
            @views V[:,kconv+1:p] = V[:,kconv+1:p] - G1[:,kconv+1:p]
        else
            Db = Diagonal(betas)
            rmul!(V, Db)
            V = V - G1
        end
        end
        
        cputime["linesearch"] += @elapsed begin
        if linesearch
            if locking
                @views mul!(VTV[:, kconv+1:p], V', V[:,kconv+1:p])
                @views VTV[kconv+1:p, 1:kconv] = VTV[1:kconv, kconv+1:p]'
                @views mul!(XTV[:, kconv+1:p], X', V[:,kconv+1:p])
                @views mul!(XTV[kconv+1:p, 1:kconv], X[:,kconv+1:p]', V[:,1:kconv])
            else
                mul!(VTV, V', V)
                mul!(XTV, X', V)
            end
            if locking
                # gV[kconv+1:p,:] .= V[kconv+1:p,:]
                # gV[kconv+1:p,:] .= SpMM_A_1_w_E(gV[kconv+1:p,:], A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col)
                @views mul!(gV[:,kconv+1:p], A, V[:,kconv+1:p])
            else
                # gV .= V
                # gV .= SpMM_A_1_w_E(gV, A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col)
                mul!(gV, A, V)
            end
            cputime["nmv"] += p-kconv
            if locking
                @views mul!(VTAV[:, kconv+1:p], V', gV[:,kconv+1:p])
                @views VTAV[kconv+1:p, 1:kconv] = VTAV[1:kconv, kconv+1:p]'
                @views mul!(VTAX[:, kconv+1:p], V', gX[:,kconv+1:p])
                @views mul!(VTAX[kconv+1:p, 1:kconv], V[:,kconv+1:p]', gX[:,1:kconv])
            else
                mul!(VTAV, V', gV)
                mul!(VTAX, V', gX)
            end
            if tri
                a3 = cumsum(diagMM(VTV, UpperTriangular(VTV)))
                a2 = cumsum(diagMM(VTV, UpperTriangular(XTV))+diagMM(VTV, UpperTriangular(XTV'))+diagMM(XTV', UpperTriangular(VTV)))
                a1 = cumsum(diag(VTAV)+diagMM(XTV', UpperTriangular(XTV'))+diagMM(XTV', UpperTriangular(XTV))+diagMM(VTV, UpperTriangular(K)))
                a0 = cumsum(diag(VTAX)+diagMM(XTV', UpperTriangular(K)))
                alphas = map(cardano, a3, a2, a1, a0)
            else
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
            diagss = diag(s'*s)
            diagsy = diag(s'*y)
            alphas .= diagss ./ diagsy
            # alphas .= diagsy ./ diagss
            X0 .= X
        elseif bb == "bb2"
            s = X - X0
            y = G1 - G
            diagyy = diag(y'*y)
            diagsy = diag(s'*y)
            alphas .= diagsy ./ diagyy
            X0 .= X
        end
        
        if locking
            Da = Diagonal(alphas[kconv+1:p])
            @views mul!(X[:,kconv+1:p], V[:,kconv+1:p], Da, 1.0, 1.0)
        else
            Da = Diagonal(alphas)
            mul!(X, V, Da, 1.0, 1.0)
        end
        G .= G1
        end

        iter += 1
    end
    end
    X, iter, cputime
end





function omm_mp(A, X, N; alpha=0.01, beta=0.0, cg=true, linesearch=true, locking=true, tri=true, precond=nothing, bb=nothing, log2deg=1, itermax=200, tol=1e-3, lowb=-2.0, upb=0.0, base=2)
    cputime = Dict("checkConvergence"=>0.0, "Ax"=>0.0, "CG"=>0.0, "updateGradient"=>0.0, "linesearch"=>0.0, "nmv"=>0.0)
    
    cputime["pre"] = @elapsed begin

    n, p = size(X)
    G = zeros(n, p)
    G1 = zeros(n, p)
    V = zeros(n, p)
    alphas = zeros(p)
    betas = [beta for i in 1:p]
    gX = deepcopy(X)
    XTAX = zeros(p, p)
    K = zeros(p, p)
    mul!(K, X', X)
    UpperK = Matrix(UpperTriangular(K))
    UpperXTAX = Matrix(UpperTriangular(XTAX))

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


    # gX = SpMM_A_1_w_E(gX, A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col)
    mul!(gX, A, X)

    mul!(XTAX, X', gX)

    if tri
        # G .= 2 .* gX - LowerTriangular(K)*gX - LowerTriangular(XTAX)*X
        UpperK = Matrix(UpperTriangular(K))
        UpperXTAX = Matrix(UpperTriangular(XTAX))
        mul!(G, 2.0, gX)
        mul!(G, gX, UpperK, -1.0, 1.0)
        mul!(G, X, UpperXTAX, -1.0, 1.0)

    else
        # G .= 4 .* gX - 2 .* (XTAX*X) - 2 .* (K*gX)
        mul!(G, 2.0, gX)
        mul!(G, X, XTAX, -1.0, 1.0)
        mul!(G, gX, K, -1.0, 1.0)
        rmul!(G, 2.0)
    end

    V .= -G

    if cg
        if precond == "filter"
            # G, _ = Cheb_filter_scal_1(base, lowb, upb, -2.0, G, A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col)
        end
    end

    if linesearch
        mul!(VTV, V', V)
        mul!(XTV, X', V)
        mul!(gV, A, V)
        mul!(VTAV, V', gV)
        mul!(VTAX, V', gX)
        if tri
            a3 = -cumsum(diagMM(VTAV, UpperTriangular(VTV))+diagMM(VTV, UpperTriangular(VTAV)))
            a2 = -cumsum(diagMM(VTAX, UpperTriangular(VTV))+diagMM(VTAV, UpperTriangular(XTV))+diagMM(VTAV, UpperTriangular(XTV'))+diagMM(XTV', UpperTriangular(VTAV))+diagMM(VTV, UpperTriangular(VTAX'))+diagMM(VTV, UpperTriangular(VTAX)))
            a1 = cumsum(diag(2*VTAV)-diagMM(VTAX, UpperTriangular(XTV))-diagMM(VTAX, UpperTriangular(XTV'))-diagMM(VTAV, UpperTriangular(K))-diagMM(XTV', UpperTriangular(VTAX'))-diagMM(XTV', UpperTriangular(VTAX))-diagMM(VTV, UpperTriangular(XTAX)))
            a0 = cumsum(diag(2*VTAX)-diagMM(VTAX, UpperTriangular(K))-diagMM(XTV', UpperTriangular(XTAX)))
            alphas = map(cardano, a3, a2, a1, a0)
        else
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
            norm_G = norm.(eachcol(G[:, kconv+1:p]))
            norm_gX = norm.(eachcol(gX[:, kconv+1:p]))
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
            # gX[kconv+1:p,:] .= X[kconv+1:p,:]
            # gX[kconv+1:p,:] .= SpMM_A_1_w_E(gX[kconv+1:p,:], A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col)
            @views mul!(gX[:,kconv+1:p], A, X[:,kconv+1:p])
        else
            # gX .= X
            # gX .= SpMM_A_1_w_E(gX, A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col)
            mul!(gX, A, X)
        end
        end
        cputime["nmv"] += p-kconv

    
        cputime["updateGradient"] += @elapsed begin
        # mul!(K, X, X')
        if locking
            @views mul!(K[:, kconv+1:p], X', X[:,kconv+1:p])
            @views K[kconv+1:p, 1:kconv] = K[1:kconv, kconv+1:p]'
        else
            mul!(K, X', X)
        end
        if locking
            @views mul!(XTAX[:, kconv+1:p], X', gX[:,kconv+1:p])
            @views XTAX[kconv+1:p, 1:kconv] = XTAX[1:kconv, kconv+1:p]'
        else
            mul!(XTAX, X', gX)
        end
        if tri
            # G1[kconv+1:p,:] = 2 .* gX[kconv+1:p,:] - LowerTriangular(K)[kconv+1:p,:]*gX - LowerTriangular(XTAX)[kconv+1:p,:]*X
            if locking
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
            mul!(G1, 2.0, gX)
            mul!(G1, X, XTAX, -1.0, 1.0)
            mul!(G1, gX, K, -1.0, 1.0)
            rmul!(G1, 2.0)
        end
        end
        
        cputime["CG"] += @elapsed begin
        if cg
            if precond == "filter"
                # G1[kconv+1:p,:], _ = Cheb_filter_scal_1(base, lowb, upb, -2.0, G1[kconv+1:p,:], A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col)
            end
            # numerators = diag((G1[kconv+1:p,:] - G[kconv+1:p,:])*G1[kconv+1:p,:]')
            # denomurators = diag(G[kconv+1:p,:]*G[kconv+1:p,:]')
            numerators = zeros(p-kconv)
            denomurators = zeros(p-kconv)
            if locking
                @views dotMulVec(numerators, G1[:,kconv+1:p] - G[:,kconv+1:p], G1[:,kconv+1:p])
                @views dotMulVec(denomurators, G[:,kconv+1:p], G[:,kconv+1:p])
            else
                dotMulVec(numerators, G1 - G, G1)
                dotMulVec(denomurators, G, G)
            end
            betas[kconv+1:p] = numerators ./ denomurators
        end
        ## G .= G1
        # V[kconv+1:p,:] = Diagonal(betas[kconv+1:p])*V[kconv+1:p,:] - G1[kconv+1:p,:]
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

        cputime["linesearch"] += @elapsed begin
        if linesearch
            # mul!(VTV, V, V')
            # mul!(XTV, X, V')
            if locking
                @views mul!(VTV[:, kconv+1:p], V', V[:,kconv+1:p])
                @views VTV[kconv+1:p, 1:kconv] = VTV[1:kconv, kconv+1:p]'
                @views mul!(XTV[:, kconv+1:p], X', V[:,kconv+1:p])
                @views mul!(XTV[kconv+1:p, 1:kconv], X[:,kconv+1:p]', V[:,1:kconv])
            else
                mul!(VTV, V', V)
                mul!(XTV, X', V)
            end
            if locking
                # gV[kconv+1:p,:] .= V[kconv+1:p,:]
                # gV[kconv+1:p,:] .= SpMM_A_1_w_E(gV[kconv+1:p,:], A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col)
                @views mul!(gV[:,kconv+1:p], A, V[:,kconv+1:p])
            else
                # gV .= V
                # gV .= SpMM_A_1_w_E(gV, A, E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col)
                mul!(gV, A, V)
            end
            cputime["nmv"] += p-kconv
            if locking
                @views mul!(VTAV[:, kconv+1:p], V', gV[:,kconv+1:p])
                @views VTAV[kconv+1:p, 1:kconv] = VTAV[1:kconv, kconv+1:p]'
                @views mul!(VTAX[:, kconv+1:p], V', gX[:,kconv+1:p])
                @views mul!(VTAX[kconv+1:p, 1:kconv], V[:,kconv+1:p]', gX[:,1:kconv])
            else
                mul!(VTAV, V', gV)
                mul!(VTAX, V', gX)
            end
            if tri
                a3 = -cumsum(diagMM(VTAV, UpperTriangular(VTV))+diagMM(VTV, UpperTriangular(VTAV)))
                a2 = -cumsum(diagMM(VTAX, UpperTriangular(VTV))+diagMM(VTAV, UpperTriangular(XTV))+diagMM(VTAV, UpperTriangular(XTV'))+diagMM(XTV', UpperTriangular(VTAV))+diagMM(VTV, UpperTriangular(VTAX'))+diagMM(VTV, UpperTriangular(VTAX)))
                a1 = cumsum(diag(2*VTAV)-diagMM(VTAX, UpperTriangular(XTV))-diagMM(VTAX, UpperTriangular(XTV'))-diagMM(VTAV, UpperTriangular(K))-diagMM(XTV', UpperTriangular(VTAX'))-diagMM(XTV', UpperTriangular(VTAX))-diagMM(VTV, UpperTriangular(XTAX)))
                a0 = cumsum(diag(2*VTAX)-diagMM(VTAX, UpperTriangular(K))-diagMM(XTV', UpperTriangular(XTAX)))
                alphas = map(cardano, a3, a2, a1, a0)
            else
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
            diagss = diag(s'*s)
            diagsy = diag(s'*y)
            alphas .= diagss ./ diagsy
            # alphas .= diagsy ./ diagss
            X0 .= X
        elseif bb == "bb2"
            s = X - X0
            y = G1 - G
            diagyy = diag(y'*y)
            diagsy = diag(s'*y)
            alphas .= diagsy ./ diagyy
            X0 .= X
        end
        
        if locking
            Da = Diagonal(alphas[kconv+1:p])
            @views mul!(X[:,kconv+1:p], V[:,kconv+1:p], Da, 1.0, 1.0)
        else
            Da = Diagonal(alphas)
            mul!(X, V, Da, 1.0, 1.0)
        end
        G .= G1
        end

        iter += 1
    end
    end

    X, iter, cputime
end