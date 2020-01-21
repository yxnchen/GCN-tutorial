# -*- encoding: utf-8 -*-
"""
@Comment  : 
@Time     : 2020/1/5 21:26
@Author   : yxnchen
"""

import numpy as np

# simple directed graph
# 0 -> 1
# 1 -> 2, 1 -> 3
# 2 -> 1
# 3 -> 0, 3 -> 2

# adjacency matrix
A = np.array([
    [0, 1, 0, 0],
    [0, 0, 1, 1],
    [0, 1, 0, 0],
    [1, 0, 1, 0]
], dtype=float)

# feature matrix
X = np.array([
    [i, -i] for i in range(A.shape[0])
], dtype=float)

# propagation rule f(X,A) = AX
# The representation of each node (each row) is now a sum of its neighbors features!
# Note: a node n is a neighbor of node v if there exists an edge from v to n.
print(np.matmul(A, X))

# Problems:
# 1. The aggregated representation of a node does not include its own features!
#    only nodes that has a self-loop will include their own features in the aggregate
# 2. Nodes with large degrees will have large values in their feature representation otherwise the opposite.
#    cause vanishing or exploding gradients

# 1. Adding Self-Loops
I = np.eye(A.shape[0])
A_s = A + I
print(np.matmul(A_s, X))

# 2. Normalizing the Feature Representations
#    transform the adjacency matrix A by multiplying it with the inverse degree matrix D
#    f(X,A) = D^{-1}AX
D = np.array(np.sum(A, axis=0))
D = np.array(np.diag(D))
print(D)
print(np.matmul(np.linalg.inv(D), A))
print(np.matmul(np.matmul(np.linalg.inv(D), A), X))

