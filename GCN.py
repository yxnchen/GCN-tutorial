# -*- encoding: utf-8 -*-
"""
@Comment  : 
@Time     : 2020/1/6 22:03
@Author   : yxnchen
"""

from keras import backend as K
from keras.layers import Layer
from keras import activations, initializers, regularizers, constraints


class GraphConv(Layer):
    """
    inputs = [feature_input, adj_input]
    """
    def __init__(self, units, activation=None, use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(GraphConv, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

    def build(self, input_shape):
        feature_shape = input_shape[0]
        self.kernel = self.add_weight(name='kernel',
                                      shape=(feature_shape[1], self.units),
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint,
                                      trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(name='bias',
                                        shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint,
                                        trainable=True)
        else:
            self.bias = None
        super(GraphConv, self).build(input_shape)

    def call(self, inputs, **kwargs):
        feature = inputs[0]
        adj_norm = inputs[1]
        aggregate = K.dot(adj_norm, feature)
        propagate = K.dot(aggregate, self.kernel)

        if self.use_bias:
            propagate += self.bias

        if self.activation is not None:
            propagate = self.activation(propagate)

        return propagate

    def compute_output_shape(self, input_shape):
        feature_shape = input_shape[0]
        return (feature_shape[0], self.units)
