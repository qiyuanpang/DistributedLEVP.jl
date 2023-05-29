import sys
import pandas as pd
from scipy import sparse, io
import numpy as np

sizes = [18571154]
classes = ["LBOLBSV", "LBOHBSV", "HBOLBSV", "HBOHBSV"]
types = ["static"]
names = {"static": {}}
names["static"] = {c: ("low" if c[0] == "L" else "high") + "Overlap_" + ("low" if c[3] == "L" else "high") + "BlockSizeVar" for c in classes}
classes = ["MAWI"]
names["static"]["MAWI"] = "mawi"
names["static"]["GRAPH500"] = "graph500-scale24-ef16"

for c in classes:
    for t in types:
        for N in sizes:

            filename_p1 = "./"+c+"/"+t+"/"+str(N)+"/"+t+"_"+names[t][c]+"_"
            filename = filename_p1+str(N)+"_nodes.tsv"
            fields = ["row", "col", "val"]
            data = pd.read_csv(filename, delimiter="\t", names=fields)

            fields = data.columns.tolist()
            I1 = data[fields[0]].to_numpy() - 1
            J1 = data[fields[1]].to_numpy() - 1
            V1 = data[fields[2]].to_numpy()

            # print("N = ", N, " sparsity = ", len(I1)*1.0/N/N)
            
            tosave = filename[:-4] + "_L.mat"
            tosave1 = filename[:-4] + "_I.npy"
            tosave2 = filename[:-4] + "_J.npy"
            tosave3 = filename[:-4] + "_V.npy"

            A = sparse.coo_matrix((V1, (I1, J1)), shape=(N, N))
            A = (A + A.transpose()) / 2
            D = np.squeeze(np.sqrt(np.array(np.sum(A, axis=1))))
            D2 = np.zeros(N)
            for i in range(N):
                if abs(D[i]) > 0:
                    D2[i] = 1/D[i]
            I2, J2 = np.array([i for i in range(N)]), np.array([i for i in range(N)])
            D2inv = sparse.coo_matrix((D2, (I2, J2)), shape=(N, N))
            Identity = sparse.coo_matrix((D*D2, (I2, J2)), shape=(N, N))
            L = Identity - D2inv.dot(A.dot(D2inv))
            L = (L + L.transpose()) / 2

            I3, J3 = L.nonzero()
            V3 = L.data

            np.save(tosave1, I3)
            print(tosave1, " saved!!")

            np.save(tosave2, J3)
            print(tosave2, " saved!!")

            np.save(tosave3, V3)
            print(tosave3, " saved!!")

            if N <= 5000000:
                M = {"data": L}
                io.savemat(tosave, M, do_compression=True)
                print(tosave, " saved!!")


            #filename_truth = filename_p1+str(N)+"_nodes_truePartition.tsv"
            #fields = ["row", "cluster"]
            #data = pd.read_csv(filename_truth, delimiter="\t", names=fields)

            #I2 = data[fields[0]].to_numpy() - 1
            #J2 = np.array([0 for _ in range(N)])
            #V2 = data[fields[1]].to_numpy()

            #tosave = filename_truth[:-4] + ".mat"
            #target = sparse.coo_matrix((V2, (I2, J2)), shape=(N, 1))
            #M = {"data": target}
            #io.savemat(tosave, M, do_compression=True)

            #print(filename_truth, " saved!!")

            sys.stdout.flush()