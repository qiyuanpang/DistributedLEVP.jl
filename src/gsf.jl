using Printf
using Statistics
using LowRankApprox
using MPI
using LinearAlgebra
using Test
using Random, Distributions

include("../utils/utils.jl")

function gsf(A, k; lambdaMin=0, lambdaMax=2, itermax=20, p=20, p2=20, estkth=Inf)
    cputime = Dict("poly"=>0.0)

    cputime["pre"] = @elapsed begin
    n = size(A,1)

    lambdalb = lambdaMin
    clb = 0
    cest = 0

    distribution = Normal(0.0, 1.0/k)
    R = rand(distribution, (n, k))

    lambdaub = lambdaMax
    cub = n
    lambdaest = lambdalb + (lambdaub-lambdalb)*k/n
    end
    
    iter = 0
    cputime["main"] = @elapsed begin
        cputime["estimate"] = @elapsed begin
        if estkth < Inf
            lambdaest = estkth
        else
            while cest != k && iter < itermax
                lambda = lambdaest
                cputime["poly"] += @elapsed begin
                M = jackson_chebyshev_filter(A, R, p2, lambdaMin, lambda, lambdaMin, lambdaMax)
                end
                cest = norm(M) ^ 2
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
        M = jackson_chebyshev_filter(A, R, p, lambdaMin, lambdaest, lambdaMin, lambdaMax)
        end
        cputime["SVD"] = @elapsed begin
        F = svd(M)
        M = F.U
        end
    end
    M, lambdaest, iter, cputime
end