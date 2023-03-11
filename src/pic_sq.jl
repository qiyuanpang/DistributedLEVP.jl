using LinearAlgebra
using SparseArrays


function pic_sq(W, k, tol)
    N = size(W, 2)
    v0 = randn(N, 1)
    v1 = zeros(N, 1)
    mul!(v1, W, v0)
    nv1 = maximum(abs.(v1))
    lmul!(1/nv1, v1)
    d1 = abs.(v0-v1)
    d0 = zeros(N, 1)
    err = norm(d1 - d0)
    iter = 1
    while err > tol
        d0 .= d1
        v0 .= v1
        mul!(v1, W, v0)
        nv1 = maximum(abs.(v1))
        lmul!(1/nv1, v1)
        d1 .= abs.(v0-v1)
        err = norm(d1 - d0)
        iter += 1
    end
    v1, iter, err
end