#!/usr/bin/env python
# coding: utf-8

# Creating left out cell samples (2,4,6,8) for each experiment which are default, signaling and signaling/metabolic

import tfm_data_operation as tfm_data
# import tfm_neural_network as tfm_NN
# import tfm_keras_tuner as tfm_kt

# Required libraries
import os
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import KFold

# IMPORTING REQUIRED DATASETS
df_weight_signaling, df_weight_metabolic_signaling = tfm_data.def_load_weight_pathways()

df_paper, df_signaling, df_metabolic_signaling = tfm_data.def_load_dataset(['cell_type']+list(df_weight_signaling.index.values)
                                                                           , ['cell_type']+list(df_weight_metabolic_signaling.index.values)
                                                                           , row_scaling=False)

# DELETE UNUSED DATASET
del(df_weight_signaling)
del(df_weight_metabolic_signaling)

# DEFAULT VALUES
# n_cells_out_list=[2, 4, 6, 8]
n_experiment_=20

# TARGET VARIABLE NAME
target_='cell_type'

def def_export_index(name_of_dataset, df_, target_value, number_of_experiment):
    list_export = []

    X_split = df_.loc[:,~df_.columns.isin([target_])]
    y_split = df_.loc[:,df_.columns.isin([target_])]
    path_output = tfm_data.def_check_create_path('EXPERIMENTS_kfold', '')
    kfold = KFold(n_splits=n_experiment_, shuffle=True)
    
    for train, test in kfold.split(df_):
#         list_export.append([tuple(train), tuple(test)])
        list_export.append([train, test])
        
        
    pd.Series(list_export).to_pickle(os.path.join(path_output+str(name_of_dataset)+'.pkl'))
    print(os.path.join(path_output+str(name_of_dataset)+'.pkl'))

####################### CREATING EXPERIMENTS - START #######################
def_export_index(name_of_dataset='signaling'
                 , df_=df_signaling
                 , target_value=target_
                 , number_of_experiment=n_experiment_
                )

def_export_index(name_of_dataset='metabolic_signaling'
                 , df_=df_metabolic_signaling
                 , target_value=target_
                 , number_of_experiment=n_experiment_
                )

def_export_index(name_of_dataset='default'
                 , df_=df_paper
                 , target_value=target_
                 , number_of_experiment=n_experiment_
                )
#######################  CREATING EXPERIMENTS - END  #######################