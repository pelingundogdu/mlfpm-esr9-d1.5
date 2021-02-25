# -*- coding: utf-8 -*-
#from __future__ import absolute_import
#from __future__ import division

import numpy as np

import copy
import inspect
import types as python_types
import marshal
import sys
import warnings
from scipy.sparse import csr_matrix
from keras import backend as K
from keras import activations, initializations, regularizers, constraints
from keras.engine import InputSpec, Layer, Merge
from keras.regularizers import ActivityRegularizer
from keras import initializations
from theano import sparse
import theano
'''
def uniform(shape, scale=0.05, name=None):
    return variable(np.random.uniform(low=-scale, high=scale, size=shape),
                      name=name)
def glorot_uniform(shape, name=None, dim_ordering='th'):
    fan_in, fan_out = get_fans(shape, dim_ordering=dim_ordering)
    s = np.sqrt(6. / (fan_in + fan_out))
    return uniform(shape, s, name=name)
def variable(value, dtype=_FLOATX, name=None):
    value = np.asarray(value, dtype=dtype)
    return theano.shared(value=value, name=name, strict=False)
'''
class MyLayer(Layer):
    '''Just your regular fully connected NN layer.

    # Example

    ```python
        # as first layer in a sequential model:
        model = Sequential()
        model.add(Dense(32, input_dim=16))
        # now the model will take as input arrays of shape (*, 16)
        # and output arrays of shape (*, 32)

        # this is equivalent to the above:
        model = Sequential()
        model.add(Dense(32, input_shape=(16,)))

        # after the first layer, you don't need to specify
        # the size of the input anymore:
        model.add(Dense(32))
    ```

    # Arguments
        output_dim: int > 0.
        init: name of initialization function for the weights of the layer
            (see [initializations](../initializations.md)),
            or alternatively, Theano function to use for weights
            initialization. This parameter is only relevant
            if you don't pass a `weights` argument.
        activation: name of activation function to use
            (see [activations](../activations.md)),
            or alternatively, elementwise Theano function.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: a(x) = x).
        weights: list of numpy arrays to set as initial weights.
            The list should have 2 elements, of shape `(input_dim, output_dim)`
            and (output_dim,) for weights and biases respectively.
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the main weights matrix.
        b_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        activity_regularizer: instance of [ActivityRegularizer](../regularizers.md),
            applied to the network output.
        W_constraint: instance of the [constraints](../constraints.md) module
            (eg. maxnorm, nonneg), applied to the main weights matrix.
        b_constraint: instance of the [constraints](../constraints.md) module,
            applied to the bias.
        bias: whether to include a bias (i.e. make the layer affine rather than linear).
        input_dim: dimensionality of the input (integer).
            This argument (or alternatively, the keyword argument `input_shape`)
            is required when using this layer as the first layer in a model.

    # Input shape
        2D tensor with shape: `(nb_samples, input_dim)`.

    # Output shape
        2D tensor with shape: `(nb_samples, output_dim)`.
    '''
    def __init__(self, output_dim, init='glorot_uniform', activation='linear', weights=None,
                 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 input_output_mat=None,
                 group_gene_dict=None,
                 bias=True, input_dim=None, **kwargs):
        #self.init = init
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.input_output_mat=input_output_mat
        self.group_gene_dict=group_gene_dict
        self.init_str = init
        #print self.input_output_mat
        if self.input_output_mat is not None:
                self.output_dim=self.input_output_mat.shape[1]
        #print 'input_dim: ',self.input_dim
        #print 'output_dim: ',self.output_dim
        self.bias = bias
        self.initial_weights = weights
        self.input_spec = [InputSpec(ndim=2)]
        
        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        
        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        #remember to modify here
        #print 'build'
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = [InputSpec(dtype=K.floatx(),
                                     shape=(None, input_dim))]
        #self.W
        #print(sparse.all_dtypes)
        
        #self.W = self.init((input_dim, self.output_dim),name='{}_W'.format(self.name))
        
        temp_W = np.asarray(self.input_output_mat, dtype=K.floatx())

        #if self.input_output_mat is not None:
            #temp_W=self.W.get_value()
        #    temp_W=temp1.get_value()
        if self.input_output_mat is not None:
            fan_in, fan_out = initializations.get_fans((input_dim, self.output_dim), dim_ordering='th')
            print (fan_in, fan_out)
            for i in range(self.input_output_mat.shape[0]):
                for j in range(self.input_output_mat.shape[1]):
                    if  self.input_output_mat[i,j] == 1.:
                        if self.init_str == 'glorot_uniform': 
                            #print 'PPITF init: ', self.init_str
                            scale = np.sqrt(6. / (fan_in + fan_out))
                            temp_W[i,j]=np.random.uniform(low=-scale, high=scale)
                        if self.init_str == 'glorot_normal':
                            #print 'PPITF init: ', self.init_str
                            scale = np.sqrt(2. / (fan_in + fan_out))
                            temp_W[i,j]=np.random.normal(0,scale)
                        if self.init_str == 'he_uniform':
                            #print 'PPITF init: ', self.init_str
                            scale = np.sqrt(6. / (fan_in))
                            temp_W[i,j]=np.random.uniform(low=-scale, high=scale)
                        if self.init_str == 'he_normal':
                            #print 'PPITF init: ', self.init_str
                            scale = np.sqrt(2. / (fan_in))
                            temp_W[i,j]=np.random.normal(0,scale)


            #temp_W=csr_matrix(temp_W)
        #self.W=temp_W
        #print 'self.W: ',self.W
            
        
        temp_W=csr_matrix(temp_W)
        #print(temp_W)
        #print temp_W.nnz
        #print temp_W.indices
        #print temp_W.indptr
        #print 'temp_W:',temp_W
        self.W=theano.shared(value=temp_W, name='{}_W'.format(self.name), strict=False)
        
        #print 'self.W.get_value():',self.W.get_value()

        #print 'self.W:',self.W
        #print 'self.W.get_value():',self.W.get_value()
        #print 'self.W[1,1]:',self.W[1,1]
        #print 'self.W[1,:]:',self.W[1,:]
        
        
        #value = np.asarray(value, dtype=dtype)
        #return theano.shared(value=value, name=name, strict=False)
        
        #print 'self.W: ',self.W.get_value()
        #print 'self.W: ',self.W.get_value()[1:]

        if self.bias:
            self.b = K.zeros((self.output_dim,),
                             name='{}_b'.format(self.name))
            self.trainable_weights = [self.W, self.b]
        else:
            self.trainable_weights = [self.W]
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        
        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)

        if self.bias and self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        self.constraints = {}
        if self.W_constraint:
            self.constraints[self.W] = self.W_constraint
        if self.bias and self.b_constraint:
            self.constraints[self.b] = self.b_constraint

    def call(self, x, mask=None):
        #output = K.dot(x, self.W)
        output = sparse.structured_dot(x, self.W)
        if self.bias:
            output += self.b
        return self.activation(output)

    def get_output_shape_for(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return (input_shape[0], self.output_dim)
    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'init': self.init.__name__,
                  'activation': self.activation.__name__,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'b_constraint': self.b_constraint.get_config() if self.b_constraint else None,
                  'bias': self.bias,
                  'input_dim': self.input_dim}
        base_config = super(MyLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    
    def set_weights(self, weights):
        '''Sets the weights of the layer, from Numpy arrays.

        # Arguments
            weights: a list of Numpy arrays. The number
                of arrays and their shape must match
                number of the dimensions of the weights
                of the layer (i.e. it should match the
                output of `get_weights`).
        '''
        params = self.trainable_weights + self.non_trainable_weights
        if len(params) != len(weights):
            raise Exception('You called `set_weights(weights)` on layer "' + self.name +
                            '" with a  weight list of length ' + str(len(weights)) +
                            ', but the layer was expecting ' + str(len(params)) +
                            ' weights. Provided weights: ' + str(weights))
        if not params:
            return
        weight_value_tuples = []
        #param_values = K.batch_get_value(params)
        for  p, w in zip(params, weights):
#            if pv.shape != w.shape:
#                raise Exception('Layer weight shape ' +
#                                str(pv.shape) +
#                                ' not compatible with '
#                                'provided weight shape ' + str(w.shape))
            weight_value_tuples.append((p, w))
        tuples=weight_value_tuples
        #K.batch_set_value(weight_value_tuples)
        #def batch_set_value(tuples):
        for x, value in tuples:
            #x.set_value(np.asarray(value, dtype=x.dtype))
            x.set_value(value)
