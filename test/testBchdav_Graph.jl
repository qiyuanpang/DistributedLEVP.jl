using MPI, MAT, SparseArrays, LinearAlgebra, Printf, Random, CSV, DataFrames, Test

include("./Bchdav/bchdav_mpi.jl")

n_samples = parse(Int64, ARGS[1])
k = parse(Int64, ARGS[2])
sparisity = 0.005
degree = parse(Int, ARGS[3])
tol = parse(Float64, ARGS[4])
blk = parse(Int, ARGS[5])
c = ARGS[6]
t = ARGS[7]
name = ARGS[8]
repeats = 1
a0 = 0.0
upb = 2.0
lwb = 0.0

function split_count(N::Integer, n::Integer)
    q,r = divrem(N, n)
    return [i <= r ? q+1 : q for i = 1:n]
end

function split_count_local(N::Integer, n::Integer)
    counts = zeros(Int64, n*n)
    counts1 = split_count(N, n)
    S = zeros(Int64, 2, n*n)
    for i in 1:n # cols
        counts2 = split_count(counts1[i], n)
        counts[(i-1)*n+1:i*n] = counts2
        for j in 1:n # rows
            S[1, (i-1)*n+j] = counts1[j]
            S[2, (i-1)*n+j] = counts1[i]
        end
    end
    return counts, S
end

function Scatterv_counts_col(I, N, rows)
    n = length(rows)
    counts = zeros(Int64, n)
    M = length(I)
    ind = 1
    for j in 1:M
        i = I[j]
        while i > sum(rows[1:ind])
            ind += 1
        end
        counts[ind] += 1
    end
    return counts
end


MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
comm_size = MPI.Comm_size(comm)
comm_size_sq = trunc(Int64, sqrt(comm_size))

root = 0

# dims = MPI.Dims_create!(comm_size, [0,comm_size_sq])
# comm_cart = MPI.Cart_create(comm, dims, [1, 0], 1)
# coords = MPI.Cart_coords(comm_cart)

coords = [mod(rank, comm_size_sq), trunc(Int64, rank/comm_size_sq)]
comm_col = MPI.Comm_split(comm, trunc(Int64, rank/comm_size_sq), rank)
rank_col = MPI.Comm_rank(comm_col)
comm_row = MPI.Comm_split(comm, mod(rank, comm_size_sq), rank)
rank_row = MPI.Comm_rank(comm_row)

info_cols_dist, S = split_count_local(n_samples, comm_size_sq) 
transpose = false

# @show info_cols_dist

# for i in 1:comm_size
#     if rank == i-1
#         @show rank rank_col rank_row
#     end
#     MPI.Barrier(comm)
# end

# counts = zeros(Int64, comm_size_sq, comm_size_sq)

file = "./graph/" * c * "/" * t * "/" * string(n_samples) * "/" * t * "_" * name * "_" * string(n_samples) * "_nodes_L_dvd.mat"
# file = "sparsedata/sparse" * string(n_samples) * "/sparseL" * string(n_samples) * what * "_dvd.mat"
# df2 = DataFrame(CSV.File(file))
# counts = reshape(df2.I, (comm_size_sq, comm_size_sq))
f = matopen(file)
counts = read(f, "data")

if rank_col == root
    global counts
    fname = "./graph/" * c * "/" * t * "/" * string(n_samples) * "/" * t * "_" * name * "_" * string(n_samples) * "_nodes_L_" * string(rank_row) * ".mat"
    # fname = "sparsedata/sparse" * string(n_samples) * "/sparseL" * string(n_samples) * what * "_" * string(rank_row) * ".mat"
    # df = DataFrame(CSV.File(fname))
    # I = broadcast(Int64, df.I)
    # J = broadcast(Int64, df.J)
    # V = df.V
    fl = matopen(fname)
    AT = read(fl, "data")
    J, I, V = findnz(AT)
    # rows = S[1, rank_row*comm_size_sq+1:(rank_row+1)*comm_size_sq]
    # counts[:, rank_row+1] = Scatterv_counts_col(I, n_samples, rows)
    I = reshape(I, (1,length(I)))
    J = reshape(J, (1,length(J)))
    V = reshape(V, (1,length(V)))

    # VBuffer for scatter
    I_vbuf = VBuffer(I, vec(counts[:, rank_row+1]))
    J_vbuf = VBuffer(J, vec(counts[:, rank_row+1]))
    V_vbuf = VBuffer(V, vec(counts[:, rank_row+1]))

else
    # these variables can be set to `nothing` on non-root processes

    I_vbuf = VBuffer(nothing)
    J_vbuf = VBuffer(nothing)
    V_vbuf = VBuffer(nothing)

end

MPI.Barrier(comm)

local_I = MPI.Scatterv!(I_vbuf, zeros(Int64, 1, counts[rank_col+1, rank_row+1]), root, comm_col)
local_I = reshape(local_I, length(local_I))
dI = ones(Int64, length(local_I))*sum(S[1, 1+comm_size_sq*rank_row:1+comm_size_sq*rank_row+rank_col-1])
local_I = broadcast(-, local_I, dI)
local_J = MPI.Scatterv!(J_vbuf, zeros(Int64, 1, counts[rank_col+1, rank_row+1]), root, comm_col)
local_J = reshape(local_J, length(local_J))
local_V = MPI.Scatterv!(V_vbuf, zeros(Float64, 1, counts[rank_col+1, rank_row+1]), root, comm_col)
local_V = reshape(local_V, length(local_V))
# @printf("%i, %i, %i , %i, %i, %i, %i \n", minimum(local_I), maximum(local_I), S[1, rank+1], size(local_I, 1), rank_col, rank_row, counts[rank_col+1, rank_row+1])
local_A = sparse(local_I, local_J, local_V, S[1, rank+1], S[2, rank+1]) # == A(rank_col, rank_row)

if rank_row == rank_col
    local_A += sparse([ir for ir in 1:S[1, rank+1]], [ir for ir in 1:S[1, rank+1]], a0 .* ones(S[1, rank+1]), S[1, rank+1], S[2, rank+1])
end
local_A_T = local_A'


maxnnz = zeros(1)
minnnz = zeros(1)
ttnnz = zeros(1)
maxnnz[1] = length(local_V)
minnnz[1] = length(local_V)
ttnnz[1] = length(local_V)
MPI.Allreduce!(maxnnz, max, comm)
MPI.Allreduce!(minnnz, min, comm)
MPI.Allreduce!(ttnnz, +, comm)
if rank == 0
    @printf("max imb: %i * %i / %i = %f \n", maxnnz[1], comm_size, ttnnz[1], maxnnz[1]*comm_size/ttnnz[1])
    @printf("min imb: %i * %i / %i = %f \n", minnnz[1], comm_size, ttnnz[1], minnnz[1]*comm_size/ttnnz[1])
    @printf("upper imb ratio: %.2e \n", (maxnnz[1]*comm_size - ttnnz[1])/ttnnz[1])
    @printf("inner imb ratio: %.2e \n", (minnnz[1]*comm_size - ttnnz[1])/ttnnz[1])
end


if rank_col == root
    I = nothing
    J = nothing
    V = nothing
    # X = nothing
end

# local_info_cols = info_cols_dist[rank_row*comm_size_sq+1:rank_row*comm_size_sq+comm_size_sq]
# Y_gather = zeros(Float64, local_size_A[1], local_size_X[1])
# X_gather_T = zeros(Float64, local_size_A[2], local_size_X[1])

# global X_gather = Array{Float64}(undef, (local_size_X[1], sum(local_info_cols)))
# _counts = vcat([local_size_X[1] for i in 1:length(local_info_cols[:])]', info_cols_dist[rank_row*comm_size_sq+1:(rank_row+1)*comm_size_sq]')
# global X_gather_vbuf = VBuffer(X_gather, vec(prod(_counts, dims=1)))
# global Y_gather_T = zeros(Float64, local_size_X[1], local_size_A[1])
# _counts = vcat([local_size_X[1] for i in 1:length(local_info_cols[:])]', info_cols_dist[rank_col*comm_size_sq+1:(rank_col+1)*comm_size_sq]')
# global Y_gather_T_vbuf = VBuffer(Y_gather_T, vec(prod(_counts, dims=1)))

MPI.Barrier(comm)


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


# v = ones(info_cols_dist[rank+1]) + Array(range(sum(info_cols_dist[1:rank])+1, sum(info_cols_dist[1:rank+1]), info_cols_dist[rank+1])) ./ n_samples
v = ones(info_cols_dist[rank+1])
V0 = repeat(v', k, 1)
opts = Dict([("polym", degree), ("tol", tol), ("itmax", 200), ("kmore", 0), ("blk", blk), ("upb", upb+a0), ("lwb", lwb+a0), ("low_nwb", lwb+(upb-lwb)/10)])

elapsedTime = nothing
evals = nothing
eigV = nothing

# X = zeros(Float64, blk, info_cols_dist[rank+1])
# Y = zeros(Float64, blk, info_cols_dist[rank_col*comm_size_sq+rank_row+1])
# Y_gather = zeros(Float64, size(local_A, 1), blk)
# X_gather_T = zeros(Float64, size(local_A, 2), blk)
# local_info_cols = info_cols_dist[rank_row*comm_size_sq+1:rank_row*comm_size_sq+comm_size_sq]
# X_gather = Array{Float64}(undef, (blk, sum(local_info_cols)))
# _counts = vcat([blk for i in 1:length(local_info_cols[:])]', info_cols_dist[rank_row*comm_size_sq+1:(rank_row+1)*comm_size_sq]')
# X_gather_vbuf = VBuffer(X_gather, vec(prod(_counts, dims=1)))
# Y_gather_T = zeros(Float64, blk, size(local_A,1))
# _counts = vcat([blk for i in 1:length(local_info_cols[:])]', info_cols_dist[rank_col*comm_size_sq+1:(rank_col+1)*comm_size_sq]')
# Y_gather_T_vbuf = VBuffer(Y_gather_T, vec(prod(_counts, dims=1)))

MPI.Barrier(comm)

cputime = @elapsed begin
for ii in 1:repeats
    global elapsedTime, evals, eigV
    evals, eigV, kconv, history, walltime = dbchdav(local_A, n_samples, k, comm_info, opts; verb=false)
    for (key, val) in walltime
        if haskey(elapsedTime, key)
            elapsedTime[key] += val*1.0/repeats
        else
            elapsedTime[key] = val*1.0/repeats
        end
    end
end
end
cputime /= repeats

MPI.Barrier(comm)

# # test SpMM_A_0
# eigV = ones(Float64, size(X))
# for ii = 1:blk
#     eigV[ii, :] = eigV[ii, :]*(ii*1.0)
# end
# Target = ones(Float64, size(X))
# for ii = 1:blk
#     Target[ii, :] = Target[ii, :]*(ii*2.0)
# end
# if rank == root
#     Target[:, 1] = Target[:, 1]/2.0
# end
# if rank == comm_size - 1
#     Target[:, end] = Target[:, end]/2.0
# end


inds = [ind for ind = 1:size(local_A,1)]
local_E = rank_col == rank_row ? sparse(inds, inds, ones(Float64, size(local_A, 1)), size(local_A,1), size(local_A,2)) : sparse([1], [1], [0.0], size(local_A,1), size(local_A,2))
AeigV = SpMM_A_0(eigV, local_A, local_E, info_cols_dist, rank_row, rank_col, comm_size_sq, root, comm_row, comm_col)
res = AeigV' - eigV'*Diagonal(evals)
err = zeros(1)
err[1] = norm(res)^2
MPI.Allreduce!(err, +, comm_row)
MPI.Allreduce!(err, +, comm_col)
err[1] = sqrt(err[1])

maxtime = zeros(Float64, 10, 1)
mintime = zeros(Float64, 10, 1)
maxtime[1] = cputime
maxtime[2] = elapsedTime["total"]
maxtime[3] = elapsedTime["Pre_loop"]
maxtime[4] = elapsedTime["main_loop"]
maxtime[5] = elapsedTime["Cheb_filter_scal"]
maxtime[6] = elapsedTime["SpMM"]
maxtime[7] = elapsedTime["TSQR"]
maxtime[8] = elapsedTime["Norm"]
maxtime[9] = elapsedTime["Hn"]
maxtime[10] = elapsedTime["Inner_prod"]
mintime[1] = cputime
mintime[2] = elapsedTime["total"]
mintime[3] = elapsedTime["Pre_loop"]
mintime[4] = elapsedTime["main_loop"]
mintime[5] = elapsedTime["Cheb_filter_scal"]
mintime[6] = elapsedTime["SpMM"]
mintime[7] = elapsedTime["TSQR"]
mintime[8] = elapsedTime["Norm"]
mintime[9] = elapsedTime["Hn"]
mintime[10] = elapsedTime["Inner_prod"]

MPI.Allreduce!(maxtime, max, comm)
MPI.Allreduce!(mintime, min, comm)

if rank == root
    @printf("=========== problem size = %i #vectors = %i tol = %.2e blk = %i =========== \n", n_samples, k, tol, blk)
    @printf("=========== graph type: %s %s =========== \n", c, t)
    @printf("#nods(2D):               %i \n", comm_size)
    @printf("degree:                  %i \n", degree)
    @printf("dbchdav wall time(s):       min: %.2e max: %.2e \n", mintime[1], maxtime[1])
    @printf("total wall time(s):         min: %.2e max: %.2e \n", mintime[2], maxtime[2])
    @printf("pre loop wall time(s):      min: %.2e max: %.2e \n", mintime[3], maxtime[3])
    @printf("main loop wall time(s):     min: %.2e max: %.2e \n", mintime[4], maxtime[4])
    @printf("   Filter wall time(s):     min: %.2e max: %.2e / %i = %.2e \n", mintime[5], maxtime[5], elapsedTime["Cheb_filter_scal_n"], maxtime[5]/elapsedTime["Cheb_filter_scal_n"])
    @printf("   SpMM wall time(s):       min: %.2e max: %.2e / %i = %.2e \n", mintime[6], maxtime[6], elapsedTime["SpMM_n"], maxtime[6]/elapsedTime["SpMM_n"])
    @printf("   TSQR wall time(s):       min: %.2e max: %.2e / %i = %.2e \n", mintime[7], maxtime[7], elapsedTime["TSQR_n"], maxtime[7]/elapsedTime["TSQR_n"])
    @printf("   Norm wall time(s):       min: %.2e max: %.2e / %i = %.2e \n", mintime[8], maxtime[8], elapsedTime["Norm_n"], maxtime[8]/elapsedTime["Norm_n"])
    @printf("   Hn wall time(s):         min: %.2e max: %.2e / %i = %.2e \n", mintime[9], maxtime[9], elapsedTime["Hn_n"], maxtime[9]/elapsedTime["Hn_n"])
    @printf("   Inner_prod wall time(s): min: %.2e max: %.2e / %i = %.2e \n", mintime[10], maxtime[10], elapsedTime["Inner_prod_n"], maxtime[10]/elapsedTime["Inner_prod_n"])
    @printf("relative error: %.2e \n", err[1])  
    for ii in 1:k
        @printf("%i-th computed eigenvalue: %.2e \n", ii, evals[ii])
    end  
    @printf("\n \n")
end

GC.gc()
MPI.Finalize()

# for i = 0:comm_size-1
#     if rank == i
#         @show rank rank_col rank_row local_Y'
#     end
#     MPI.Barrier(comm)
# end



# local_Y = zeros(Float64, local_size_X[2], local_size_X[1])
# larger_Y = zeros(Float64, local_size_A[1], local_size_X[1])

# # MPI.Barrier(comm)

# for i = 0:comm_size-1
#     if rank == i
#         @show rank larger_Y
#     end
#     # MPI.Barrier(comm)
# end
