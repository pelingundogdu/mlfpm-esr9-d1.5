#!/usr/bin/env python
# coding: utf-8

# Created python codes
import tfm_neural_network as tfm_NN

# Required libraries
import os
# import glob
import numpy as np
import pandas as pd
import datetime as dt # for date information

# from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Input

import tensorflow as tf # for clear the output after each training step
import kerastuner as kt # for hypermodel
import IPython # for clear the output after each training step

import shutil

########################################################################################################################################################################
class TFM_HyperModel(kt.HyperModel):
    """
    Model is getting pathway table and the tuning is appling into this model
    
    """
    
    def __init__(self, input_dim_, units_, df_weight_, second_layer_, output_classes_=16):
        self.input_dim_ = input_dim_
        self.units_ = units_
        self.df_weight_ = df_weight_
        self.second_layer_ = second_layer_
        self.output_classes_ = output_classes_

    def build(self, hp):
        model = Sequential()

        model.add(Dense(units = self.units_, input_dim=self.input_dim_, kernel_initializer='glorot_uniform', bias_initializer='zeros', activation='tanh', name='layer1'))
        if (len(self.df_weight_)>0):
            #print('set_weight applied!!')
            model.set_weights([model.get_weights()[0] * np.array(self.df_weight_),  np.zeros((self.units_,)) ])
        if (self.second_layer_==True):
            #print('second layer applied!!')
            for i in range(hp.Int('n_layers', 1, 1)):  # adding variation of layers.
                model.add(Dense(hp.Int(f'layer_{i}_units',
                                        min_value=0,
                                        max_value=200,
                                        step=50)))
                model.add(Activation('tanh'))
        
        model.add(Dense(self.output_classes_, activation='softmax', name='layer3'))

        hp_learning_rate = hp.Choice('learning_rate', values = [0.0001, 0.001, 0.01, 0.1, 0.2] )
        hp_momentum = hp.Choice('momentum', values=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0] )
        hp_decay = hp.Choice('decay', values=[1e-4, 1e-5, 1e-6, 1e-7, 1e-8] )

        model.compile(optimizer = optimizers.SGD(learning_rate = hp_learning_rate
                                                        , momentum=hp_momentum
                                                        , decay=hp_decay)
                      , loss = 'categorical_crossentropy'
                      , metrics = ['accuracy'])
        return model

########################################################################################################################################################################
class ClearTrainingOutput(tf.keras.callbacks.Callback):
    """
    Define a callback to clear the training outputs at the end of every training step.
    """
    def on_train_end(*args, **kwargs):
        IPython.display.clear_output(wait = True)
        
############################################################ APPLYING THE BEST MODEL CREATED VIA KERAS TUNER ###########################################################
HYPERBAND_MAX_EPOCHS = 100
# MAX_TRIALS = 20

class TFM_KerasTunerExperiment(object):
    def __init__(self, X_train_, y_train_, X_test_, y_test_, df_w_, bio_, design_name_, directory_, project_name_, second_layer, path_, epochs_, batch_size_):
        self.X_train_ = X_train_
        self.y_train_ = y_train_
        self.X_test_ = X_test_
        self.y_test_ = y_test_
        self.df_w_ = df_w_
        self.bio_ = bio_
        if (len(self.df_w_) > 0):
            self.design_name_ = design_name_+str(len(df_w_.columns))
        else:
            self.design_name_ = design_name_
        self.directory_ = directory_
        self.project_name_ = project_name_
        self.second_layer = second_layer
        self.path_ = path_
        self.epochs_ = epochs_
        self.batch_size_ = batch_size_
        
    def build(self):
    
        if self.path_ != '':
            if (os.path.exists(os.path.join(self.path_+'/'+self.directory_))==False):
                os.mkdir(os.path.join(self.path_+'/'+self.directory_))

        y_pred_all = []
        best_hps_all = []
        unit_size = 100

        if(len(self.df_w_) > 0):
            unit_size = len(self.df_w_.columns)

        hypermodel = TFM_HyperModel(input_dim_=len(self.X_train_[0][0])
                                       , units_=unit_size
                                       , df_weight_=self.df_w_
                                       , second_layer_=self.second_layer)

        time_start = dt.datetime.now().time().strftime('%H:%M:%S') # = time.time()
        print('started!!!     ', time_start)


        print("\ninput layer {:}, hidden layer {:}, epoch {:}, batch_size {:}".format(len(self.X_train_[0][0]), unit_size, self.epochs_, self.batch_size_))

        for i in range(len(self.X_train_)):
            print('Experiment -- ', i)
#             print(os.path.join(self.path_+'/'+self.directory_+'/'+self.project_name_+str(i)))
#             if self.path_ != '':
            if (os.path.exists(os.path.join(self.path_+'/'+self.directory_+'/'+self.project_name_+str(i)))==False):
                os.mkdir(os.path.join(self.path_+'/'+self.directory_+'/'+self.project_name_+str(i)))
                print(os.path.join(self.path_+'/'+self.directory_+'/'+self.project_name_+str(i)))
        #     if((i+1)%int(n_splits_)==0):
        #         print('  Experiment {:} / {:}'.format(int(i/(int(1/n_splits_)))+1, n_experiment_))

            tuner = kt.Hyperband(hypermodel
                                 , objective = 'val_accuracy'
                                 , max_epochs = HYPERBAND_MAX_EPOCHS
                                 , overwrite = True
                                 , directory = os.path.join(self.path_+'/'+self.directory_)
                                 , project_name = self.project_name_+str(i) 
                                )
            
            tuner.search(self.X_train_[i]
                         , np.asarray(self.y_train_[i])
                         , epochs=self.epochs_
                         , validation_split=0.1
                         , callbacks = [ClearTrainingOutput()]
                         , verbose=0)

            # Get the optimal hyperparameters
            best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]
            print(best_hps.values)

            model_tuner = tuner.hypermodel.build(best_hps)
            model_tuner.fit(self.X_train_[i]
                            , np.asarray(self.y_train_[i])
                            , epochs = self.epochs_
                            , batch_size = self.batch_size_
                            , verbose=0)
            y_pred = model_tuner.predict(self.X_test_[i])
            best_hps_all.append(best_hps)
            y_pred_all.append(y_pred)
            K.clear_session()
        
            shutil.rmtree(os.path.join(self.path_+'/'+self.directory_+'/'+self.project_name_+str(i)))
            
            print('removed directory !! -- ', os.path.join(self.path_+'/'+self.directory_+'/'+self.project_name_+str(i)))
        
        shutil.rmtree(os.path.join(self.path_+'/'+self.directory_))
        print('removed directory !! -- ', os.path.join(self.path_+'/'+self.directory_))
        
        time_end  = dt.datetime.now().time().strftime('%H:%M:%S') # = time.time()
        print('\nELAPSED TIME, ', (dt.datetime.strptime(time_end,'%H:%M:%S') - dt.datetime.strptime(time_start,'%H:%M:%S')))
        print('\n')

        pred, test = tfm_NN.def_convert_one_hot_encoder_to_list(y_pred_all, self.y_test_)
        
        metric = tfm_NN.def_network_run_get_metrics(y_pred_=pred
                                                        , y_test_=test
                                                        , bio_info_=self.bio_
                                                        , name_of_design_=self.design_name_)

        return(model_tuner, metric, best_hps_all)

############################################################ APPLYING THE BEST MODEL CREATED VIA KERAS TUNER ###########################################################

def def_hp(hp_, name):
    list_append = []
    for i in range(len(hp_)):
        list_append.append(hp_[i].values)

    df_hp = pd.DataFrame(list_append)
    df_hp['hp'] = name
#     for k,v in locals().items():
#         print(v)

    return df_hp