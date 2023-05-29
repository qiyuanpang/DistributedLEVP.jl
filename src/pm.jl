using LinearAlgebra
using SparseArrays
using MKLSparse

function pm(L, k, p; S=nothing)
    n = size(L, 1)
    V = zeros(n, k)
    if S == nothing
        S = randn(n, k)
    end
    cputime = @elapsed begin
        for i in 1:p
            mul!(V, L, S)
            S .= V
        end
        F = svd(V)
        V = F.U
    end
    V, cputime
end