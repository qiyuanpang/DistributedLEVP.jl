import numpy as np
from scipy import sparse, io
import scipy as sp
import sys
import h5py
import time

def split_count(N, n):
    q,r = divmod(N, n)
    return [q+1 if i < r else q for i in range(n)]

def findnz_local(A, comm_size_sq, nnz):
    N = np.size(A,0)
    counts = split_count(N, comm_size_sq)
    I = np.zeros(nnz)
    J = np.zeros(nnz)
    V = np.zeros(nnz)
    col_start = col_end = 0
    row_start = row_end = 0
    counts_local = np.zeros(comm_size_sq, dtype=int)
    dvd = np.zeros((comm_size_sq, comm_size_sq), dtype=int)
    n = 0
    for j in range(comm_size_sq):
        col_start = sum(counts[:j])
        col_end = sum(counts[:j+1])
        n1 = 0
        for i in range(comm_size_sq):
            row_start = sum(counts[:i])
            row_end = sum(counts[:i+1])
            I1, J1, V1 = sparse.find(A[row_start:row_end, col_start:col_end])
            m = len(I1)
            I[n:n+m] = I1 + row_start
            J[n:n+m] = J1
            V[n:n+m] = V1
            # print(i, j, min(I[n:n+m]), max(I[n:n+m]), row_start, row_end-row_start, len(I1))
            n1 += m
            n += m
            dvd[i, j] = m
        counts_local[j] = n1
    return I, J, V, counts_local, counts, dvd

start = time.time()

n_samples = int(sys.argv[1])
comm_size_sq = int(sys.argv[2])
# what = sys.argv[3]
c = sys.argv[3]
t = sys.argv[4]
name = sys.argv[5]

fname = "./graph/" + c + "/" + t + "/" + str(n_samples) + "/" + t + "_" + name + "_" + str(n_samples) + "_nodes_"
I = np.load(fname + "I.npy")
J = np.load(fname + "J.npy")
V = np.load(fname + "V.npy")
# I = np.squeeze(fI['I'])
# J = np.squeeze(fJ['J'])
# V = np.squeeze(fV['V'])

# print(np.shape(V), np.shape(I), np.shape(J))
A = sparse.coo_matrix((V, (I, J)), shape=(n_samples, n_samples))
A = A.tocsc()

nnz = len(V)
I, J, V, nnzs, cols, dvd = findnz_local(A, comm_size_sq, nnz)


s = 0
for i in range(comm_size_sq):
    I1 = I[sum(nnzs[:i]):sum(nnzs[:i+1])]
    J1 = J[sum(nnzs[:i]):sum(nnzs[:i+1])]
    V1 = V[sum(nnzs[:i]):sum(nnzs[:i+1])]
    # for j in range(comm_size_sq):
        # print(j, i, min(I1[sum(dvd[:j, i]):sum(dvd[:j+1, i])]), max(I1[sum(dvd[:j, i]):sum(dvd[:j+1, i])]))
    A1 = sparse.coo_matrix((V1, (I1, J1)), shape=(n_samples, cols[i])).T
    tosave = fname + "L_" + str(i) + ".mat"
    # sparse.save_npz(tosave, A1)
    # np.save(tosave, A1)
    M = {"data": A1}
    io.savemat(tosave, M, do_compression=True)

dvd = sparse.coo_matrix(dvd)
tosave = fname + "L_dvd.mat"
# sparse.save_npz(tosave, dvd)
# np.save(tosave, dvd)
M = {"data": dvd}
io.savemat(tosave, M)

end = time.time()

print(fname, " processed and saved within ", end-start, " seconds! ")
