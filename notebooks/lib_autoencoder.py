# Required libraries
import os
import numpy as np
import pandas as pd
import datetime as dt
from tensorflow import keras
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
        
        train_X_encoded.append(encoder.predict(train_X[i]))
        test_X_encoded.append(encoder.predict(test_X[i]))
    
    print('AE executed for all experiments!!')
    return train_X_encoded, test_X_encoded, autoencoder
