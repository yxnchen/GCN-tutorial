# -*- encoding: utf-8 -*-
"""
@Comment  : 
@Time     : 2020/1/13 11:56
@Author   : yxnchen
"""

#%% load Karate Club data

from networkx import read_edgelist,set_node_attributes, to_numpy_matrix
from pandas import read_csv, Series
from numpy import array


def loadKarateClub():
    nw = read_edgelist('karate.edgelist', nodetype=int)
    attributes = read_csv('karate.attributes.csv', index_col=['node'])
    for attr in attributes.columns.values:
        set_node_attributes(nw, values=Series(attributes[attr], index=attributes.index).to_dict(), name=attr)

    X_train, y_train = map(array, zip(*[
        ([node], data['role'] == 'Administrator') for node, data in nw.nodes(data=True)
        if data['role'] in {'Administrator', 'Instructor'}
    ]))
    X_test, y_test = map(array, zip(*[
        ([node], data['community'] == 'Administrator') for node, data in nw.nodes(data=True)
        if data['role'] == 'Member'
    ]))

    return X_train, y_train, X_test, y_test, to_numpy_matrix(nw)


X_train, y_train, X_test, y_test, A = loadKarateClub()


#%% process data

import numpy as np


def preprocess_adj(adj, normalization='spectral'):
    """
    :param adj: adjacency matrix
    :param normalization: 'spectral' or 'simple'
    :return: normalized adjacency matrix
    """
    adj = adj + np.eye(adj.shape[0])
    if normalization == 'spectral':
        d = np.diag(np.array(np.power(np.sum(adj, axis=0), -0.5))[0])
        adj_norm = np.matmul(np.matmul(d, adj), d)
        return adj_norm
    elif normalization == 'simple':
        d = np.diag(np.array(np.power(np.sum(adj, axis=0), -1))[0])
        adj_norm = np.matmul(d, adj)
        return adj_norm
    else:
        return adj


A_norm = preprocess_adj(A, normalization='spectral')


#%% build model

from keras.models import Model
from keras.layers import Input, Dropout
from keras.optimizers import Adam
from GCN import GraphConv

# use identity matrix as features
X_1 = np.eye(A_norm.shape[0])
#


featureInput = Input(shape=(X_1.shape[1],))
adjInput = Input(shape=(None, None), sparse=True)
H = GraphConv(4, activation='relu')([featureInput, adjInput])
H = GraphConv(2, activation='sigmoid')([H, adjInput])

model1 = Model(inputs=[featureInput, adjInput], outputs=H)
model1.compile(optimizer=Adam(lr=0.01), loss='binary_crossentropy')




