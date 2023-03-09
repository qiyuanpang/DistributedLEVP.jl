import networkx as nx
import numpy as np
from networkx.readwrite import json_graph
from scipy import sparse, io
import os
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap

def rand1Probs(r, state, a):
    probs = np.array(sparse.random(r, r, density=2/r, random_state=state).todense() * a)
    epsilon = np.ones((r,r)) * 1e-7
    epsilon = (epsilon+epsilon.transpose())/2
    for i in range(r):
        probs[i,i] = 0.05
    probs = (probs+probs.transpose())/2
    probs = probs + epsilon
    return probs

def decayingOffDiagonal(k, scale, diagonal):
    probs = np.zeros((k,k))
    epsilon = np.ones((k,k)) * 1e-7
    epsilon = (epsilon+epsilon.transpose())/2
    for i in range(k):
        for j in range(k):
            probs[i,j] = norm.pdf(np.exp(abs(i-j))-1, scale=scale)
    for i in range(k):
        probs[i,i] = diagonal[i]
    probs = (probs+probs.transpose())/2
    probs = probs + epsilon
    return probs


def decayingOffDiagonal3(k, scale, diagonal):
    probs = np.zeros((k,k))
    epsilon = np.ones((k,k)) * 1e-7
    epsilon = (epsilon+epsilon.transpose())/2
    for i in range(k):
        for j in range(k):
            probs[i,j] = norm.pdf(np.exp(abs(i-j))-1, scale=scale)
    for i in range(k):
        probs[i,i] = diagonal[i]
    probs[0, k-1] = norm.pdf(np.exp(abs(1))-1, scale=scale)
    probs[k-1, 0] = norm.pdf(np.exp(abs(1))-1, scale=scale)
    probs = (probs+probs.transpose())/2
    probs = probs + epsilon
    return probs


def decayingOffDiagonal4(k, scale, diagonal):
    probs = np.zeros((k,k))
    epsilon = np.ones((k,k)) * 1e-7
    epsilon = (epsilon+epsilon.transpose())/2
    for i in range(k):
        for j in range(k):
            probs[i,j] = norm.pdf(np.exp(abs(i-j))-1, scale=scale)
    for i in range(k):
        probs[i,i] = diagonal[i]
    for i in range(k):
        for j in range(k):
            if abs(j-i) == k//2:
                probs[i,j] = norm.pdf(np.exp(1)-1, scale=scale)/2
    probs = (probs+probs.transpose())/2
    probs = probs + epsilon
    return probs

def graph_decay1_eqSize():
    # sizes = [1000]*10
    sizes = [5000]*10
    r = len(sizes)
    probs = decayingOffDiagonal(r, 0.5, [0.05 for i in range(r)])
    return sizes, probs

def graph_decay1_monoSize():
    # sizes = [i*200 for i in range(1,11)]
    sizes = [i*2000 for i in range(1,11)]
    r = len(sizes)
    probs = decayingOffDiagonal(r, 0.5, [0.05 for i in range(r)])
    return sizes, probs

def graph_decay1_imbSize1():
    # sizes = [300, 400, 500, 100, 200, 300, 400, 500, 400, 2000]
    sizes = list(np.array([300, 400, 500, 100, 200, 300, 400, 500, 400, 2000]) * 10)
    r = len(sizes)
    probs = decayingOffDiagonal(r, 0.5, [0.05 for i in range(r)])
    return sizes, probs

def graph_decay1_imbSize2():
    # sizes = [2000, 100, 200, 300, 400, 400, 300, 200, 100, 2000]
    sizes = list(np.array([2000, 100, 200, 300, 400, 400, 300, 200, 100, 2000]) * 10)
    r = len(sizes)
    probs = decayingOffDiagonal(r, 0.5, [0.05 for i in range(r)])
    return sizes, probs

def graph_decay1_imbSize3():
    # sizes = [2000, 100, 100, 200, 200, 200, 300, 300, 300, 400, 400, 400, 100, 100, 100, 200, 200, 200, 300, 300, 300, 400, 400, 2000]
    sizes = list(np.array([2000, 100, 100, 200, 200, 200, 300, 300, 300, 400, 400, 400, 100, 100, 100, 200, 200, 200, 300, 300, 300, 400, 400, 2000]) * 5 )
    r = len(sizes)
    probs = decayingOffDiagonal(r, 0.5, [0.05 for i in range(r)])
    return sizes, probs

def graph_decay1_imbSize4():
    # sizes = [1000, 200, 100, 300, 400, 300, 2000, 300, 400, 300, 100, 200, 1000]
    sizes = list(np.array([1000, 200, 100, 300, 400, 300, 2000, 300, 400, 300, 100, 200, 1000]) * 10 )
    r = len(sizes)
    probs = decayingOffDiagonal(r, 0.5, [0.05 for i in range(r)])
    return sizes, probs





def graph_decay2_eqSize():
    # sizes = [1000]*10
    sizes = [5000]*10
    r = len(sizes)
    probs = decayingOffDiagonal(r, 0.6, [0.05 for i in range(r)])
    return sizes, probs

def graph_decay2_monoSize():
    # sizes = [i*200 for i in range(1,11)]
    sizes = [i*2000 for i in range(1,11)]
    r = len(sizes)
    probs = decayingOffDiagonal(r, 0.6, [0.05 for i in range(r)])
    return sizes, probs

def graph_decay2_imbSize1():
    # sizes = [300, 400, 500, 100, 200, 300, 400, 500, 400, 2000]
    sizes = list(np.array([300, 400, 500, 100, 200, 300, 400, 500, 400, 2000]) * 10)
    r = len(sizes)
    probs = decayingOffDiagonal(r, 0.6, [0.05 for i in range(r)])
    return sizes, probs

def graph_decay2_imbSize2():
    # sizes = [2000, 100, 200, 300, 400, 400, 300, 200, 100, 2000]
    sizes = list(np.array([2000, 100, 200, 300, 400, 400, 300, 200, 100, 2000]) * 10)
    r = len(sizes)
    probs = decayingOffDiagonal(r, 0.6, [0.05 for i in range(r)])
    return sizes, probs

def graph_decay2_imbSize3():
    # sizes = [2000, 100, 100, 200, 200, 200, 300, 300, 300, 400, 400, 400, 100, 100, 100, 200, 200, 200, 300, 300, 300, 400, 400, 2000]
    sizes = list(np.array([2000, 100, 100, 200, 200, 200, 300, 300, 300, 400, 400, 400, 100, 100, 100, 200, 200, 200, 300, 300, 300, 400, 400, 2000]) * 5 )
    r = len(sizes)
    probs = decayingOffDiagonal(r, 0.6, [0.05 for i in range(r)])
    return sizes, probs

def graph_decay2_imbSize4():
    # sizes = [1000, 200, 100, 300, 400, 300, 2000, 300, 400, 300, 100, 200, 1000]
    sizes = list(np.array([1000, 200, 100, 300, 400, 300, 2000, 300, 400, 300, 100, 200, 1000]) * 10 )
    r = len(sizes)
    probs = decayingOffDiagonal(r, 0.6, [0.05 for i in range(r)])
    return sizes, probs






def graph_decay3_eqSize():
    # sizes = [1000]*10
    sizes = [5000]*10
    r = len(sizes)
    probs = decayingOffDiagonal3(r, 0.5, [0.05 for i in range(r)])
    return sizes, probs

def graph_decay3_monoSize():
    # sizes = [i*200 for i in range(1,11)]
    sizes = [i*2000 for i in range(1,11)]
    r = len(sizes)
    probs = decayingOffDiagonal3(r, 0.5, [0.05 for i in range(r)])
    return sizes, probs

def graph_decay3_imbSize1():
    # sizes = [300, 400, 500, 100, 200, 300, 400, 500, 400, 2000]
    sizes = list(np.array([300, 400, 500, 100, 200, 300, 400, 500, 400, 2000]) * 10)
    r = len(sizes)
    probs = decayingOffDiagonal3(r, 0.5, [0.05 for i in range(r)])
    return sizes, probs

def graph_decay3_imbSize2():
    # sizes = [2000, 100, 200, 300, 400, 400, 300, 200, 100, 2000]
    sizes = list(np.array([2000, 100, 200, 300, 400, 400, 300, 200, 100, 2000]) * 10)
    r = len(sizes)
    probs = decayingOffDiagonal3(r, 0.5, [0.05 for i in range(r)])
    return sizes, probs

def graph_decay3_imbSize3():
    # sizes = [2000, 100, 100, 200, 200, 200, 300, 300, 300, 400, 400, 400, 100, 100, 100, 200, 200, 200, 300, 300, 300, 400, 400, 2000]
    sizes = list(np.array([2000, 100, 100, 200, 200, 200, 300, 300, 300, 400, 400, 400, 100, 100, 100, 200, 200, 200, 300, 300, 300, 400, 400, 2000]) * 5 )
    r = len(sizes)
    probs = decayingOffDiagonal3(r, 0.5, [0.05 for i in range(r)])
    return sizes, probs

def graph_decay3_imbSize4():
    # sizes = [1000, 200, 100, 300, 400, 300, 2000, 300, 400, 300, 100, 200, 1000]
    sizes = list(np.array([1000, 200, 100, 300, 400, 300, 2000, 300, 400, 300, 100, 200, 1000]) * 10 )
    r = len(sizes)
    probs = decayingOffDiagonal3(r, 0.5, [0.05 for i in range(r)])
    return sizes, probs




def graph_decay4_eqSize():
    # sizes = [1000]*10
    sizes = [5000]*10
    r = len(sizes)
    probs = decayingOffDiagonal4(r, 0.5, [0.05 for i in range(r)])
    return sizes, probs

def graph_decay4_monoSize():
    # sizes = [i*200 for i in range(1,11)]
    sizes = [i*2000 for i in range(1,11)]
    r = len(sizes)
    probs = decayingOffDiagonal4(r, 0.5, [0.05 for i in range(r)])
    return sizes, probs

def graph_decay4_imbSize1():
    # sizes = [300, 400, 500, 100, 200, 300, 400, 500, 400, 2000]
    sizes = list(np.array([300, 400, 500, 100, 200, 300, 400, 500, 400, 2000]) * 10)
    r = len(sizes)
    probs = decayingOffDiagonal4(r, 0.5, [0.05 for i in range(r)])
    return sizes, probs

def graph_decay4_imbSize2():
    # sizes = [2000, 100, 200, 300, 400, 400, 300, 200, 100, 2000]
    sizes = list(np.array([2000, 100, 200, 300, 400, 400, 300, 200, 100, 2000]) * 10)
    r = len(sizes)
    probs = decayingOffDiagonal4(r, 0.5, [0.05 for i in range(r)])
    return sizes, probs

def graph_decay4_imbSize3():
    # sizes = [2000, 100, 100, 200, 200, 200, 300, 300, 300, 400, 400, 400, 100, 100, 100, 200, 200, 200, 300, 300, 300, 400, 400, 2000]
    sizes = list(np.array([2000, 100, 100, 200, 200, 200, 300, 300, 300, 400, 400, 400, 100, 100, 100, 200, 200, 200, 300, 300, 300, 400, 400, 2000]) * 5 )
    r = len(sizes)
    probs = decayingOffDiagonal4(r, 0.5, [0.05 for i in range(r)])
    return sizes, probs

def graph_decay4_imbSize4():
    # sizes = [1000, 200, 100, 300, 400, 300, 2000, 300, 400, 300, 100, 200, 1000]
    sizes = list(np.array([1000, 200, 100, 300, 400, 300, 2000, 300, 400, 300, 100, 200, 1000]) * 10 )
    r = len(sizes)
    probs = decayingOffDiagonal4(r, 0.5, [0.05 for i in range(r)])
    return sizes, probs




def graph_decay5_eqSize():
    # sizes = [1000]*10
    sizes = [5000]*10
    r = len(sizes)
    probs = decayingOffDiagonal4(r, 0.5, [0.05 for i in range(r)])
    probs[0,0], probs[r-1,r-1] = 0.08, 0.08
    return sizes, probs

def graph_decay5_monoSize():
    # sizes = [i*200 for i in range(1,11)]
    sizes = [i*2000 for i in range(1,11)]
    r = len(sizes)
    probs = decayingOffDiagonal4(r, 0.5, [0.05 for i in range(r)])
    probs[0,0], probs[1,1] = 0.08, 0.08
    return sizes, probs

def graph_decay5_imbSize1():
    # sizes = [300, 400, 500, 100, 200, 300, 400, 500, 400, 2000]
    sizes = list(np.array([300, 400, 500, 100, 200, 300, 400, 500, 400, 2000]) * 10)
    r = len(sizes)
    probs = decayingOffDiagonal4(r, 0.5, [0.05 for i in range(r)])
    probs[0,0], probs[1,1] = 0.08, 0.08
    return sizes, probs

def graph_decay5_imbSize2():
    # sizes = [2000, 100, 200, 300, 400, 400, 300, 200, 100, 2000]
    sizes = list(np.array([2000, 100, 200, 300, 400, 400, 300, 200, 100, 2000]) * 10)
    r = len(sizes)
    probs = decayingOffDiagonal4(r, 0.5, [0.05 for i in range(r)])
    probs[0,0], probs[1,1] = 0.08, 0.08
    return sizes, probs

def graph_decay5_imbSize3():
    # sizes = [2000, 100, 100, 200, 200, 200, 300, 300, 300, 400, 400, 400, 100, 100, 100, 200, 200, 200, 300, 300, 300, 400, 400, 2000]
    sizes = list(np.array([2000, 100, 100, 200, 200, 200, 300, 300, 300, 400, 400, 400, 100, 100, 100, 200, 200, 200, 300, 300, 300, 400, 400, 2000]) * 5 )
    r = len(sizes)
    probs = decayingOffDiagonal4(r, 0.5, [0.05 for i in range(r)])
    probs[0,0] = 0.08
    return sizes, probs

def graph_decay5_imbSize4():
    # sizes = [1000, 200, 100, 300, 400, 300, 2000, 300, 400, 300, 100, 200, 1000]
    sizes = list(np.array([1000, 200, 100, 300, 400, 300, 2000, 300, 400, 300, 100, 200, 1000]) * 10 )
    r = len(sizes)
    probs = decayingOffDiagonal4(r, 0.5, [0.05 for i in range(r)])
    probs[0,0], probs[r-1,r-1] = 0.08, 0.08
    return sizes, probs




def graph_rand1_eqSize():
    # sizes = [1000]*10
    sizes = [5000]*10
    r = len(sizes)
    probs = rand1Probs(r, 1, 0.01)
    return sizes, probs

def graph_rand1_monoSize():
    # sizes = [i*200 for i in range(1,11)]
    sizes = [i*2000 for i in range(1,11)]
    r = len(sizes)
    probs = rand1Probs(r, 1, 0.01)
    return sizes, probs

def graph_rand1_imbSize1():
    # sizes = [300, 400, 500, 100, 200, 300, 400, 500, 400, 2000]
    sizes = list(np.array([300, 400, 500, 100, 200, 300, 400, 500, 400, 2000]) * 10)
    r = len(sizes)
    probs = rand1Probs(r, 1, 0.01)
    return sizes, probs

def graph_rand1_imbSize2():
    # sizes = [2000, 100, 200, 300, 400, 400, 300, 200, 100, 2000]
    sizes = list(np.array([2000, 100, 200, 300, 400, 400, 300, 200, 100, 2000]) * 10)
    r = len(sizes)
    probs = rand1Probs(r, 1, 0.01)
    return sizes, probs

def graph_rand1_imbSize3():
    # sizes = [2000, 100, 100, 200, 200, 200, 300, 300, 300, 400, 400, 400, 100, 100, 100, 200, 200, 200, 300, 300, 300, 400, 400, 2000]
    sizes = list(np.array([2000, 100, 100, 200, 200, 200, 300, 300, 300, 400, 400, 400, 100, 100, 100, 200, 200, 200, 300, 300, 300, 400, 400, 2000]) * 5 )
    r = len(sizes)
    probs = rand1Probs(r, 1, 0.01)
    return sizes, probs

def graph_rand1_imbSize4():
    # sizes = [1000, 200, 100, 300, 400, 300, 2000, 300, 400, 300, 100, 200, 1000]
    sizes = list(np.array([1000, 200, 100, 300, 400, 300, 2000, 300, 400, 300, 100, 200, 1000]) * 10 )
    r = len(sizes)
    probs = rand1Probs(r, 1, 0.01)
    return sizes, probs



def graph_rand55_eqSize():
    # sizes = [1000]*10
    sizes = [5000]*10
    r = len(sizes)
    probs = rand1Probs(r, 55, 0.015)
    probs[0,0], probs[r-1,r-1] = 0.08, 0.08
    return sizes, probs

def graph_rand55_monoSize():
    # sizes = [i*200 for i in range(1,11)]
    sizes = [i*2000 for i in range(1,11)]
    r = len(sizes)
    probs = rand1Probs(r, 55, 0.015)
    probs[0,0], probs[1,1] = 0.08, 0.08
    return sizes, probs

def graph_rand55_imbSize1():
    # sizes = [300, 400, 500, 100, 200, 300, 400, 500, 400, 2000]
    sizes = list(np.array([300, 400, 500, 100, 200, 300, 400, 500, 400, 2000]) * 10)
    r = len(sizes)
    probs = rand1Probs(r, 55, 0.015)
    probs[0,0], probs[1,1] = 0.08, 0.08
    return sizes, probs

def graph_rand55_imbSize2():
    # sizes = [2000, 100, 200, 300, 400, 400, 300, 200, 100, 2000]
    sizes = list(np.array([2000, 100, 200, 300, 400, 400, 300, 200, 100, 2000]) * 10)
    r = len(sizes)
    probs = rand1Probs(r, 55, 0.015)
    probs[0,0], probs[1,1] = 0.08, 0.08
    return sizes, probs

def graph_rand55_imbSize3():
    # sizes = [2000, 100, 100, 200, 200, 200, 300, 300, 300, 400, 400, 400, 100, 100, 100, 200, 200, 200, 300, 300, 300, 400, 400, 2000]
    sizes = list(np.array([2000, 100, 100, 200, 200, 200, 300, 300, 300, 400, 400, 400, 100, 100, 100, 200, 200, 200, 300, 300, 300, 400, 400, 2000]) * 5 )
    r = len(sizes)
    probs = rand1Probs(r, 55, 0.015)
    probs[0,0] = 0.08
    return sizes, probs

def graph_rand55_imbSize4():
    # sizes = [1000, 200, 100, 300, 400, 300, 2000, 300, 400, 300, 100, 200, 1000]
    sizes = list(np.array([1000, 200, 100, 300, 400, 300, 2000, 300, 400, 300, 100, 200, 1000]) * 10 )
    r = len(sizes)
    probs = rand1Probs(r, 55, 0.015)
    probs[0,0], probs[r-1,r-1] = 0.08, 0.08
    return sizes, probs



def graph_rand99_eqSize():
    # sizes = [1000]*10
    sizes = [5000]*10
    r = len(sizes)
    probs = rand1Probs(r, 99, 0.015)
    return sizes, probs

def graph_rand99_monoSize():
    # sizes = [i*200 for i in range(1,11)]
    sizes = [i*2000 for i in range(1,11)]
    r = len(sizes)
    probs = rand1Probs(r, 99, 0.015)
    return sizes, probs

def graph_rand99_imbSize1():
    # sizes = [300, 400, 500, 100, 200, 300, 400, 500, 400, 2000]
    sizes = list(np.array([300, 400, 500, 100, 200, 300, 400, 500, 400, 2000]) * 10)
    r = len(sizes)
    probs = rand1Probs(r, 99, 0.015)
    return sizes, probs

def graph_rand99_imbSize2():
    # sizes = [2000, 100, 200, 300, 400, 400, 300, 200, 100, 2000]
    sizes = list(np.array([2000, 100, 200, 300, 400, 400, 300, 200, 100, 2000]) * 10)
    r = len(sizes)
    probs = rand1Probs(r, 99, 0.015)
    return sizes, probs

def graph_rand99_imbSize3():
    # sizes = [2000, 100, 100, 200, 200, 200, 300, 300, 300, 400, 400, 400, 100, 100, 100, 200, 200, 200, 300, 300, 300, 400, 400, 2000]
    sizes = list(np.array([2000, 100, 100, 200, 200, 200, 300, 300, 300, 400, 400, 400, 100, 100, 100, 200, 200, 200, 300, 300, 300, 400, 400, 2000]) * 5 )
    r = len(sizes)
    probs = rand1Probs(r, 99, 0.015)
    return sizes, probs

def graph_rand99_imbSize4():
    # sizes = [1000, 200, 100, 300, 400, 300, 2000, 300, 400, 300, 100, 200, 1000]
    sizes = list(np.array([1000, 200, 100, 300, 400, 300, 2000, 300, 400, 300, 100, 200, 1000]) * 10 )
    r = len(sizes)
    probs = rand1Probs(r, 99, 0.015)
    return sizes, probs


if __name__ == "__main__":

    # types = ["graph_decay1_eqSize", "graph_decay1_monoSize", "graph_decay1_imbSize1", "graph_decay1_imbSize2", "graph_decay1_imbSize3", "graph_decay1_imbSize4", \
        # "graph_decay2_eqSize", "graph_decay2_monoSize", "graph_decay2_imbSize1", "graph_decay2_imbSize2", "graph_decay2_imbSize3", "graph_decay2_imbSize4"]
    # types = ["graph_decay3_eqSize", "graph_decay3_monoSize", "graph_decay3_imbSize1", "graph_decay3_imbSize2", "graph_decay3_imbSize3", "graph_decay3_imbSize4", \
                # "graph_decay4_eqSize", "graph_decay4_monoSize", "graph_decay4_imbSize1", "graph_decay4_imbSize2", "graph_decay4_imbSize3", "graph_decay4_imbSize4"]
    # types = ["graph_rand1_eqSize", "graph_rand1_monoSize", "graph_rand1_imbSize1", "graph_rand1_imbSize2", "graph_rand1_imbSize3", "graph_rand1_imbSize4", \
            # "graph_rand99_eqSize", "graph_rand99_monoSize", "graph_rand99_imbSize1", "graph_rand99_imbSize2", "graph_rand99_imbSize3", "graph_rand99_imbSize4"]
    types = ["graph_decay5_eqSize", "graph_decay5_monoSize", "graph_decay5_imbSize1", "graph_decay5_imbSize2", "graph_decay5_imbSize3", "graph_decay5_imbSize4", \
            "graph_rand55_eqSize", "graph_rand55_monoSize", "graph_rand55_imbSize1", "graph_rand55_imbSize2", "graph_rand55_imbSize3", "graph_rand55_imbSize4"]

    for t in types:
        if t == "graph_rand1_eqSize":
            sizes, probs = graph_rand1_eqSize()
        elif t == "graph_decay1_eqSize":
            sizes, probs = graph_decay1_eqSize()
        elif t == "graph_rand1_monoSize":
            sizes, probs = graph_rand1_monoSize()
        elif t == "graph_decay1_monoSize":
            sizes, probs = graph_decay1_monoSize()
        elif t == "graph_rand1_imbSize1":
            sizes, probs = graph_rand1_imbSize1()
        elif t == "graph_decay1_imbSize1":
            sizes, probs = graph_decay1_imbSize1()
        elif t == "graph_rand1_imbSize2":
            sizes, probs = graph_rand1_imbSize2()
        elif t == "graph_decay1_imbSize2":
            sizes, probs = graph_decay1_imbSize2()
        elif t == "graph_rand1_imbSize3":
            sizes, probs = graph_rand1_imbSize3()
        elif t == "graph_decay1_imbSize3":
            sizes, probs = graph_decay1_imbSize3()
        elif t == "graph_rand1_imbSize4":
            sizes, probs = graph_rand1_imbSize4()
        elif t == "graph_decay1_imbSize4":
            sizes, probs = graph_decay1_imbSize4()

        elif t == "graph_decay2_eqSize":
            sizes, probs = graph_decay2_eqSize()
        elif t == "graph_decay2_monoSize":
            sizes, probs = graph_decay2_monoSize()
        elif t == "graph_decay2_imbSize1":
            sizes, probs = graph_decay2_imbSize1()
        elif t == "graph_decay2_imbSize2":
            sizes, probs = graph_decay2_imbSize2()
        elif t == "graph_decay2_imbSize3":
            sizes, probs = graph_decay2_imbSize3()
        elif t == "graph_decay2_imbSize4":
            sizes, probs = graph_decay2_imbSize4()

        
        elif t == "graph_decay3_eqSize":
            sizes, probs = graph_decay3_eqSize()
        elif t == "graph_decay3_monoSize":
            sizes, probs = graph_decay3_monoSize()
        elif t == "graph_decay3_imbSize1":
            sizes, probs = graph_decay3_imbSize1()
        elif t == "graph_decay3_imbSize2":
            sizes, probs = graph_decay3_imbSize2()
        elif t == "graph_decay3_imbSize3":
            sizes, probs = graph_decay3_imbSize3()
        elif t == "graph_decay3_imbSize4":
            sizes, probs = graph_decay3_imbSize4()

        elif t == "graph_decay4_eqSize":
            sizes, probs = graph_decay4_eqSize()
        elif t == "graph_decay4_monoSize":
            sizes, probs = graph_decay4_monoSize()
        elif t == "graph_decay4_imbSize1":
            sizes, probs = graph_decay4_imbSize1()
        elif t == "graph_decay4_imbSize2":
            sizes, probs = graph_decay4_imbSize2()
        elif t == "graph_decay4_imbSize3":
            sizes, probs = graph_decay4_imbSize3()
        elif t == "graph_decay4_imbSize4":
            sizes, probs = graph_decay4_imbSize4()

        
        elif t == "graph_decay5_eqSize":
            sizes, probs = graph_decay5_eqSize()
        elif t == "graph_decay5_monoSize":
            sizes, probs = graph_decay5_monoSize()
        elif t == "graph_decay5_imbSize1":
            sizes, probs = graph_decay5_imbSize1()
        elif t == "graph_decay5_imbSize2":
            sizes, probs = graph_decay5_imbSize2()
        elif t == "graph_decay5_imbSize3":
            sizes, probs = graph_decay5_imbSize3()
        elif t == "graph_decay5_imbSize4":
            sizes, probs = graph_decay5_imbSize4()

        
        elif t == "graph_rand1_eqSize":
            sizes, probs = graph_rand1_eqSize()
        elif t == "graph_rand1_monoSize":
            sizes, probs = graph_rand1_monoSize()
        elif t == "graph_rand1_imbSize1":
            sizes, probs = graph_rand1_imbSize1()
        elif t == "graph_rand1_imbSize2":
            sizes, probs = graph_rand1_imbSize2()
        elif t == "graph_rand1_imbSize3":
            sizes, probs = graph_rand1_imbSize3()
        elif t == "graph_rand1_imbSize4":
            sizes, probs = graph_rand1_imbSize4()

        
        elif t == "graph_rand55_eqSize":
            sizes, probs = graph_rand55_eqSize()
        elif t == "graph_rand55_monoSize":
            sizes, probs = graph_rand55_monoSize()
        elif t == "graph_rand55_imbSize1":
            sizes, probs = graph_rand55_imbSize1()
        elif t == "graph_rand55_imbSize2":
            sizes, probs = graph_rand55_imbSize2()
        elif t == "graph_rand55_imbSize3":
            sizes, probs = graph_rand55_imbSize3()
        elif t == "graph_rand55_imbSize4":
            sizes, probs = graph_rand55_imbSize4()

        
        elif t == "graph_rand99_eqSize":
            sizes, probs = graph_rand99_eqSize()
        elif t == "graph_rand99_monoSize":
            sizes, probs = graph_rand99_monoSize()
        elif t == "graph_rand99_imbSize1":
            sizes, probs = graph_rand99_imbSize1()
        elif t == "graph_rand99_imbSize2":
            sizes, probs = graph_rand99_imbSize2()
        elif t == "graph_rand99_imbSize3":
            sizes, probs = graph_rand99_imbSize3()
        elif t == "graph_rand99_imbSize4":
            sizes, probs = graph_rand99_imbSize4()

        # print(probs)
        G = nx.stochastic_block_model(sizes, probs, seed=0)
        data = json_graph.adjacency_data(G)

        nodes = data["nodes"]
        adjacency = data["adjacency"]

        n = sum(sizes)
        assert n == len(nodes)
        m = 0
        for i in range(n):
            m += len(adjacency[i])

        I = np.zeros(m, dtype=int)
        J = np.zeros(m, dtype=int)
        V = np.ones(m, dtype=float)

        count = 0
        for i in range(n):
            adj_list = adjacency[i]
            for pair in adj_list:
                j = pair["id"]
                I[count] = i
                J[count] = j
                count += 1
        
        curpath = os.getcwd()
        path = curpath+"/synthetic/sbm/"+str(n)
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path) 
        filename = path+"/sbm_"+t+"_"+str(n)+"_nodes"
        tosaveA = filename + "_A.mat"
        tosaveL = filename + "_L.mat"
        tosaveP = filename + "_truePartition.mat"
        
        A = sparse.coo_matrix((V, (I, J)), shape=(n, n))
        A = (A + A.transpose()) / 2
        M = {"data": A}
        io.savemat(tosaveA, M, do_compression=True)
        print(tosaveA, " saved!!")

        D = np.squeeze(np.sqrt(np.array(np.sum(A, axis=1))))
        D2 = np.zeros(n)
        for i in range(n):
            if abs(D[i]) > 0:
                D2[i] = 1/D[i]
        I2, J2 = np.array([i for i in range(n)]), np.array([i for i in range(n)])
        D2inv = sparse.coo_matrix((D2, (I2, J2)), shape=(n, n))
        Identity = sparse.coo_matrix((D*D2, (I2, J2)), shape=(n, n))
        L = Identity - D2inv.dot(A.dot(D2inv))
        L = (L + L.transpose()) / 2

        M = {"data": L}
        io.savemat(tosaveL, M, do_compression=True)
        print(tosaveL, " saved!!")

        I3, J3 = L.nonzero()
        V3 = L.data

        tosaveI = filename + "_I.npy"
        tosaveJ = filename + "_J.npy"
        tosaveV = filename + "_V.npy"
        np.save(tosaveI, I3)
        print(tosaveI, " saved!!")
        np.save(tosaveJ, J3)
        print(tosaveJ, " saved!!")
        np.save(tosaveV, V3)
        print(tosaveV, " saved!!")


        I4 = np.zeros(n, dtype=int)
        J4 = np.array([0 for _ in range(n)])
        V4 = np.zeros(n, dtype=int)
        for i in range(n):
            node = nodes[i]
            I4[i] = node["id"]
            V4[i] = node["block"]+1
        partitions = sparse.coo_matrix((V4, (I4, J4)), shape=(n, 1))
        M = {"data": partitions}
        io.savemat(tosaveP, M, do_compression=True)
        print(tosaveP, " saved!!")
        

        path = curpath+"/synthetic/sbm/thumbnails"
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path) 
        filename = path+"/sbm_"+t+"_"+str(n)+"_nodes"
        tosavePng = filename + "_thumbnail.png"
        plt.figure(dpi=200)
        cmap = ListedColormap(["#F2F2F2", "#00008B"])
        subn = 10000
        sub = range(0,n,n//subn)
        Ar = A.tocsr()[sub,:]
        Ac = Ar.tocsr()[:,sub]
        plt.matshow(Ac.todense(), cmap=cmap)
        plt.axis('off')
        plt.savefig(tosavePng)
        plt.close()
        print(tosavePng, " saved!!")


