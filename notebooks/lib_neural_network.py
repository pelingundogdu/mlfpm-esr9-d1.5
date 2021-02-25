#!/usr/bin/env python
# coding: utf-8



# Required libraries
import os
# import glob
import numpy as np
import pandas as pd
import datetime as dt
# import warnings
# warnings.filterwarnings('ignore')

import statistics as statistics

# from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, homogeneity_score, completeness_score, v_measure_score, adjusted_rand_score, adjusted_mutual_info_score, fowlkes_mallows_score

# from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Dropout, Input


########################################################################################################################################################################
# Converting OneHotEncoder array as list
# After this process, the scores can be found
def def_convert_one_hot_encoder_to_list(y_pred_, y_test_):
    list_pred = list()
    list_test = list()
    # Converting predictions to label
    # and converting one hot encoded test label to label
    for i in range(len(y_pred_)):
        list_pred_temp = list()
        list_test_temp = list()
        for j in range(len(y_pred_[i])):
            list_pred_temp.append(np.argmax(y_pred_[i][j]))
            list_test_temp.append(np.argmax(y_test_[i][j]))
        
        list_pred.append(list_pred_temp)
        list_test.append(list_test_temp)
        
    return(list_pred, list_test)

########################################################################################################################################################################
# Creating score matrix for  original paper scores
def def_network_run_get_metrics(y_pred_, y_test_, bio_info_, name_of_design_):
    list_homo, list_comp, list_vmes, list_ari, list_ami, list_fm = [],[],[],[],[],[]

#     pred, test = def_convert_one_hot_encoder_to_list(y_pred_, y_test_)

    for i in range(len(y_pred_)):
        list_homo.append(homogeneity_score(y_test_[i], y_pred_[i]))
        list_comp.append(completeness_score(y_test_[i], y_pred_[i]))
        list_vmes.append(v_measure_score(y_test_[i], y_pred_[i]))
        list_ari.append(adjusted_rand_score(y_test_[i], y_pred_[i]))
        list_ami.append(adjusted_mutual_info_score(y_test_[i], y_pred_[i]))
        list_fm.append(fowlkes_mallows_score(y_test_[i], y_pred_[i]))
    
    print('The mean of homogeneity score (with {0})       ; {1:.2f}'.format(bio_info_, statistics.mean(list_homo)))
    print('The mean of completeness score (with {0})      ; {1:.2f}'.format(bio_info_, statistics.mean(list_comp)))
    print('The mean of v-measure score (with {0})         ; {1:.2f}'.format(bio_info_, statistics.mean(list_vmes)))
    print('The mean of ari score (with {0})               ; {1:.2f}'.format(bio_info_, statistics.mean(list_ari)))
    print('The mean of ami score (with {0})               ; {1:.2f}'.format(bio_info_, statistics.mean(list_ami)))
    print('The mean of fowlkes-mallows score (with {0})   ; {1:.2f}'.format(bio_info_, statistics.mean(list_fm)))
    print('The mean (with {0})                            ; {1:.2f}'.format(bio_info_, statistics.mean([statistics.mean(list_homo)
                                                                       , statistics.mean(list_comp)
                                                                       , statistics.mean(list_vmes)
                                                                       , statistics.mean(list_ari)
                                                                       , statistics.mean(list_ami)
                                                                       , statistics.mean(list_fm)]
                                                                      )))   
    
    df_metric = pd.DataFrame(list([bio_info_+name_of_design_, statistics.mean(list_homo), statistics.mean(list_comp), statistics.mean(list_vmes)
                                                , statistics.mean(list_ari), statistics.mean(list_ami), statistics.mean(list_fm)
                                                , statistics.mean( [statistics.mean(list_homo)
                                                               , statistics.mean(list_comp)
                                                               , statistics.mean(list_vmes)
                                                               , statistics.mean(list_ari)
                                                               , statistics.mean(list_ami)
                                                               , statistics.mean(list_fm)]
                                                              ) ])).T
    return(df_metric)

############################################################################## NN DESIGN ##############################################################################
def def_NN_default_design(X_train_, y_train_, X_test_, df_weight_, units_, epochs_, batch_size_, pathway_layer_=False, second_layer_=False):

    K.clear_session()
    model_default = Sequential()
    model_default.add(Dense(units = units_, input_dim=len(X_train_[0]), kernel_initializer='glorot_uniform', bias_initializer='zeros', activation='tanh', name='layer1'))
    if (pathway_layer_==True):
        #print('set_weight applied!!')
        model_default.set_weights([model_default.get_weights()[0] * np.array(df_weight_),  np.zeros((units_,)) ])

    if (second_layer_==True):
        #print('second layer applied!!')
        model_default.add(Dense(100, activation='tanh', name='layer2'))
        
    model_default.add(Dense(16, activation='softmax', name='layer3'))
    
    sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True) # the parameter from paper 
    model_default.compile(optimizer=sgd
                          , loss='categorical_crossentropy'
                          , metrics=['accuracy']
                         )
    model_default.fit(X_train_, y_train_, epochs=epochs_, batch_size=batch_size_, verbose=0)
    y_pred_ = model_default.predict(X_test_)
    K.clear_session()

    return(model_default, y_pred_)
############################################################################## NN DESIGN ##############################################################################

########################################################################################################################################################################
# This is the executed experiment
# If new design is wanted to use then def_NN_default_design function should change
class TFM_NNExperiment(object):
    def __init__(self, X_train_, y_train_, X_test_, y_test_, df_w_, bio_, design_name_, pathway_layer_, second_layer_, epochs_, batch_size_, unit_size_):
        self.X_train_ = X_train_
        self.y_train_ = y_train_
        self.X_test_ = X_test_
        self.y_test_ = y_test_
        self.df_w_ = df_w_
        self.bio_ = bio_
        self.design_name_ = design_name_
#         self.directory_ = directory_
#         self.project_name_ = project_name_
        self.pathway_layer_ = pathway_layer_
        self.second_layer_ = second_layer_
#         self.path_ = path_
        self.epochs_ = epochs_
        self.batch_size_ = batch_size_
        self.unit_size_ = unit_size_
        
    def build(self):
        y_pred_all = []

        time_start = dt.datetime.now().time().strftime('%H:%M:%S') # = time.time()
        print('started!!!     ', time_start)

        print("\ninput layer {:}, hidden layer {:}, epoch {:}, batch_size {:}".format(len(self.X_train_[0][0]), self.unit_size_, self.epochs_, self.batch_size_))
#         print('X_TRAIN UZUNLUGU, ', len(self.X_train_))

        for i in range(len(self.X_train_)):
            print('Experiment -- ',i)
            model_NN, y_pred = def_NN_default_design(X_train_=self.X_train_[i]
                                                     , y_train_=self.y_train_[i]
                                                     , X_test_=self.X_test_[i]
                                                     , df_weight_=self.df_w_
                                                     , units_=self.unit_size_
                                                     , epochs_=self.epochs_
                                                     , batch_size_=self.batch_size_
                                                     , pathway_layer_=self.pathway_layer_
                                                     , second_layer_=self.second_layer_)
            y_pred_all.append(y_pred)

        pred, test = def_convert_one_hot_encoder_to_list(y_pred_all, self.y_test_)
            
        time_end  = dt.datetime.now().time().strftime('%H:%M:%S') # = time.time()
        print('\nELAPSED TIME, ', (dt.datetime.strptime(time_end,'%H:%M:%S') - dt.datetime.strptime(time_start,'%H:%M:%S')))
        print('\n')

        metric = def_network_run_get_metrics(y_pred_=pred
                                             , y_test_=test
                                             , bio_info_=self.bio_
                                             , name_of_design_=self.design_name_)
        
        return(model_NN, metric)




########################################################################################################################################################################
# Creating score matrix for  original paper scores
# def def_network_run_get_metrics_org(y_pred_, y_test_, bio_info_, name_of_design_):
#     list_homo, list_comp, list_vmes, list_ari, list_ami, list_fm = [],[],[],[],[],[]

#     pred, test = def_convert_one_hot_encoder_to_list(y_pred_, y_test_)

#     list_homo.append(homogeneity_score(pred,test)*100)
#     list_comp.append(completeness_score(pred,test)*100)
#     list_vmes.append(v_measure_score(pred,test)*100)
#     list_ari.append(adjusted_rand_score(pred,test)*100)
#     list_ami.append(adjusted_mutual_info_score(pred,test)*100)
#     list_fm.append(fowlkes_mallows_score(pred,test)*100)
    
#     print('The mean of homogeneity score (with {0})       ; {1:.2f}'.format(bio_info_, statistics.mean(list_homo)))
#     print('The mean of completeness score (with {0})      ; {1:.2f}'.format(bio_info_, statistics.mean(list_comp)))
#     print('The mean of v-measure score (with {0})         ; {1:.2f}'.format(bio_info_, statistics.mean(list_vmes)))
#     print('The mean of ari score (with {0})               ; {1:.2f}'.format(bio_info_, statistics.mean(list_ari)))
#     print('The mean of ami score (with {0})               ; {1:.2f}'.format(bio_info_, statistics.mean(list_ami)))
#     print('The mean of fowlkes-mallows score (with {0})   ; {1:.2f}'.format(bio_info_, statistics.mean(list_fm)))
#     print('The mean (with {0})                            ; {1:.2f}'.format(bio_info_, statistics.mean([statistics.mean(list_homo)
#                                                                        , statistics.mean(list_comp)
#                                                                        , statistics.mean(list_vmes)
#                                                                        , statistics.mean(list_ari)
#                                                                        , statistics.mean(list_ami)
#                                                                        , statistics.mean(list_fm)]
#                                                                       )))   
    
#     df_metric = pd.DataFrame(list([bio_info_+name_of_design_, statistics.mean(list_homo), statistics.mean(list_comp), statistics.mean(list_vmes)
#                                                 , statistics.mean(list_ari), statistics.mean(list_ami), statistics.mean(list_fm)
#                                                 , statistics.mean( [statistics.mean(list_homo)
#                                                                , statistics.mean(list_comp)
#                                                                , statistics.mean(list_vmes)
#                                                                , statistics.mean(list_ari)
#                                                                , statistics.mean(list_ami)
#                                                                , statistics.mean(list_fm)]
#                                                               ) ])).T
#     return(df_metric)