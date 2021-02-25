#!/usr/bin/env python
# coding: utf-8

# Created python codes
import tfm_neural_network as tfm_NN

# Required libraries
import os
# import glob
import numpy as np
import pandas as pd
import datetime as dt
# import warnings
# warnings.filterwarnings('ignore')

# import statistics as statistics

# from sklearn.model_selection import train_test_split, KFold
# from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
# from sklearn.metrics import accuracy_score, homogeneity_score, completeness_score, v_measure_score, adjusted_rand_score, adjusted_mutual_info_score, fowlkes_mallows_score

from tensorflow import keras
# import tensorflow.keras.backend as K
# # from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Dropout, Input

from sklearn.cluster import KMeans

########################################################################################################################################################################
# design with one hidden layer as middle layer
def AE_one_hidden_layer(input_dim, hidden_layers):
    '''
    one hidden layer autoencoder design
    
    '''
    keras.backend.clear_session()

    #encoder
    autoencoder_inputs = Input(shape=(input_dim, ))
#     middle_layer = Dense(hidden_layers[0], activation='tanh')(autoencoder_inputs)
    middle_layer = Dense(hidden_layers, activation='tanh')(autoencoder_inputs)


    #decoder
    autoencoder_outputs = Dense(input_dim, activation='softmax')(middle_layer)
    autoencoder = Model(autoencoder_inputs, autoencoder_outputs, name="autoencoder")
#     print(autoencoder.summary())

    # this model maps an input to its encoded representation
    encoder = Model(autoencoder_inputs, middle_layer, name='encoder')
#     print(encoder.summary())

    # create a placeholder for an encoded (32-dimensional) input
#     middle_input = Input(shape=(hidden_layers[0],))
    middle_input = Input(shape=(hidden_layers,))
    
    # retrieve the last layer of the autoencoder model
    decoder_1 = autoencoder.layers[-1]
    decoder_1 = decoder_1(middle_input)
    # create the decoder model
    decoder = Model(middle_input, decoder_1, name='decoder')
#     print(decoder.summary())
    
    return autoencoder, encoder, decoder, hidden_layers

########################################################################################################################################################################
# design with multi hidden layer
# def AE_multi_hidden_layer(input_dim, hidden_layers, dropout_ratio):
    
#     keras.backend.clear_session()

#     #encoder
#     autoencoder_inputs = Input(shape=(input_dim, ))
#     hidden_e = Dense(hidden_layers[0], activation='tanh')(autoencoder_inputs)
#     for i_hidden in range(len(hidden_layers)-1):
#         hidden_e = Dropout(dropout_ratio)(hidden_e)
#         hidden_e = Dense(hidden_layers[i_hidden+1], activation='tanh')(hidden_e)

#     #decoder
#     hidden_d = Dense(hidden_layers[-2], activation='tanh')(hidden_e)
#     for i_hidden_d in range(len(hidden_layers)-2):
#         hidden_d = Dense(hidden_layers[-3-i_hidden_d], activation='tanh')(hidden_d)

#     autoencoder_outputs = Dense(input_dim, activation='softmax')(hidden_d)
#     autoencoder = Model(autoencoder_inputs, autoencoder_outputs, name="autoencoder")
# #     print(autoencoder.summary())

#     # this model maps an input to its encoded representation
#     encoder = keras.Model(autoencoder_inputs, hidden_e, name='encoder')
#     # print(encoder.summary())

#     # create a placeholder for an encoded (32-dimensional) input
#     decoder_inputs = Input(shape=(hidden_layers[-1],))
#     decoder_last_layer = autoencoder.layers[-1]

#     decoder_layer = Dense(hidden_layers[-2], activation='tanh')(decoder_inputs)
#     for i_hidden_d in range(len(hidden_layers)-2):
#         decoder_layer = Dense(hidden_layers[-3-i_hidden_d], activation='tanh')(decoder_layer)

#     decoder_last_layer = decoder_last_layer(decoder_layer)
#     decoder = keras.Model(decoder_inputs, decoder_last_layer, name='decoder')
#     # print(decoder.summary())
#     return autoencoder, encoder, decoder, hidden_layers

########################################################################################################################################################################

# def AE_compile_checkpoint(train_X, test_X, epochs_, batch_size_, hidden_layer_, autoencoder, encoder, decoder, optimizer_, loss_, path_):
#     print('AE network started to training!!')
#     train_X_encoded, test_X_encoded = [], []
#     keras.backend.clear_session()
    
#     for i in range(len(train_X)):
# #         print('Experiment -- ', i)
#         keras.backend.clear_session()
# #         print(autoencoder.summary())
        
#         autoencoder.compile(optimizer=optimizer_, loss=loss_)
        
#         checkpoint_path = os.path.join(path_+'/experiment_'+str(i)+'/cp-{epoch:04d}.ckpt')
#         callbacks_ = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path
#                                          , save_weights_only=True
#                                          , verbose=1
#                                          , period=1)
        
#         history = autoencoder.fit(train_X[i], train_X[i], epochs=epochs_, validation_split=0.1, batch_size=batch_size_, verbose=2, callbacks=callbacks_)

# #         train_X = encoder.predict(train_X[i]) #?? burasini sor
# #         test_X = encoder.predict(test_X[i])
        
#         train_X_encoded.append(encoder.predict(train_X[i]))
#         test_X_encoded.append(encoder.predict(test_X[i]))
    
#     print('AE executed for all experiments!!')
#     return train_X_encoded, test_X_encoded, autoencoder



########################################################################################################################################################################

def AE_compile(train_X, test_X, epochs_, batch_size_, hidden_layer_, autoencoder, encoder, decoder, optimizer_, loss_, callbacks_='', path_=''):
    print('AE network started to training!!')
    train_X_encoded, test_X_encoded = [], []
    keras.backend.clear_session()
    
    for i in range(len(train_X)):
        print('Experiment -- ', i)
        keras.backend.clear_session()
#         print(autoencoder.summary())

        autoencoder.compile(optimizer=optimizer_, loss=loss_)
        if callbacks_ == '':
            history = autoencoder.fit(train_X[i], test_X[i], epochs=epochs_, validation_split=0.1, batch_size=batch_size_, verbose=2)
        elif (callbacks_=='checkpoint'):
            if path_=='':
                print('path_ feature is empty!! Autoencoder not executed!!')
            else:
                checkpoint_path = os.path.join(path_+'/experiment_'+str(i)+'/cp-{epoch:04d}.ckpt')

                callbacks_ = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path
                                             , save_weights_only=True
                                             , verbose=1
                                             , period=1)
                history = autoencoder.fit(train_X[i], test_X[i], epochs=epochs_, validation_split=0.1, batch_size=batch_size_, verbose=2, callbacks=callbacks_)
        else:
            history = autoencoder.fit(train_X[i], test_X[i], epochs=epochs_, validation_split=0.2, batch_size=batch_size_, verbose=2, callbacks=callbacks_)

#         train_X = encoder.predict(train_X[i]) #?? burasini sor
#         test_X = encoder.predict(test_X[i])
        
        train_X_encoded.append(encoder.predict(train_X[i]))
        test_X_encoded.append(encoder.predict(test_X[i]))
    
    print('AE executed for all experiments!!')
    return train_X_encoded, test_X_encoded, autoencoder

########################################################################################################################################################################
# # Clustering experiment,
# def AE_kmeans(train_encoded, test_encoded, y_test, cluster_, bio_):
#     print('kmeans started!!')
#     y_pred_all = []
#     keras.backend.clear_session()
#     for experiment_ in range(len(train_encoded)):
# #         print('experiment --', experiment_)
#         keras.backend.clear_session()
#         kmeans = KMeans(n_clusters=cluster_).fit(train_encoded[experiment_])
#         kmeans_predict = kmeans.predict(test_encoded[experiment_])
#         y_pred_all.append(kmeans_predict)

#     metric = tfm_NN.def_network_run_get_metrics(y_pred_=y_pred_all
#                                                     , y_test_=y_test
#                                                     , bio_info_=bio_
#                                                     , name_of_design_='cell_out_'+str(cluster_))
#     return(metric.values[0])