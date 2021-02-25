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

from sklearn.model_selection import LeavePGroupsOut

# IMPORTING REQUIRED DATASETS
df_weight_signaling, df_weight_metabolic_signaling = tfm_data.def_load_weight_pathways()

df_paper_9437, df_signaling, df_metabolic_signaling = tfm_data.def_load_dataset(['cell_type']+list(df_weight_signaling.index.values)
                                                                                , ['cell_type']+list(df_weight_metabolic_signaling.index.values))

# DELETE UNUSED DATASET
del(df_weight_signaling)
del(df_weight_metabolic_signaling)

# DEFAULT VALUES
n_cells_out_list=[2, 4, 6, 8]
n_experiment_=20

# TARGET VARIABLE NAME
target_='cell_type'

def def_get_n_psplits(X, y, groups, p, n):
    splitter = LeavePGroupsOut(n_groups=p)
    splits = list(splitter.split(X, y, groups))
    ids = np.random.choice(len(splits), n).tolist()
    list_random_selected = [splits[i] for i in ids]
    return(list_random_selected)

def def_export_index(name_of_dataset, df_, target_value, list_left_out, number_of_experiment):
    X_split = df_.loc[:,~df_.columns.isin([target_value])]
    y_split = df_.loc[:,df_.columns.isin([target_value])]
    path_output = tfm_data.def_check_create_path('EXPERIMENTS', '')
    for n_cells_out in list_left_out:
        print('n_cells_out,', n_cells_out)
        list_export = def_get_n_psplits(X_split, y_split, y_split['cell_type'], n_cells_out, number_of_experiment)
        pd.Series(list_export).to_pickle(os.path.join(path_output+str(name_of_dataset)+'_cell_out_'+str(n_cells_out)+'.pkl'))
        print(os.path.join(path_output+str(name_of_dataset)+'_cell_out_'+str(n_cells_out)+'.pkl'))

####################### CREATING EXPERIMENTS - START #######################
def_export_index(name_of_dataset='signaling'
                 , df_=df_signaling
                 , target_value=target_
                 , list_left_out=n_cells_out_list
                 , number_of_experiment=n_experiment_
                )

def_export_index(name_of_dataset='metabolic_signaling'
                 , df_=df_metabolic_signaling
                 , target_value=target_
                 , list_left_out=n_cells_out_list
                 , number_of_experiment=n_experiment_
                )

def_export_index(name_of_dataset='default'
                 , df_=df_paper_9437
                 , target_value=target_
                 , list_left_out=n_cells_out_list
                 , number_of_experiment=n_experiment_
                )
#######################  CREATING EXPERIMENTS - END  #######################