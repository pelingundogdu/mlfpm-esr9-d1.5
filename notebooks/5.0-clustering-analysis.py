import lib_clustering as tfm_cluster
import lib_data_operation as tfm_data

import re
import os
import pandas as pd
import numpy as np
import datetime as dt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from tensorflow import keras
import tensorflow.keras.backend as K

import warnings
warnings.filterwarnings('ignore')

time_start = dt.datetime.now().time().strftime('%H:%M:%S') # = time.time()

# Clustering is designed with 20 experiments. 
# Testing size is %20 (it means that Kfold split is 5)
# and I am applying 4 times to reach the 20 experiments.

# THE LOCATION of THE RESULT
path_output = tfm_data.def_check_create_path(main_folder='clustering_result', sub_folder='')

# Loading required data
df_weight_pathway_signaling, df_weight_pathway_metabolic_signaling = tfm_data.def_load_weight_pathways()
df_paper, df_paper_signaling, df_paper_metabolic_signaling = tfm_data.def_load_dataset(['cell_type']+list(df_weight_pathway_signaling.index.values)
                                                                                       , ['cell_type']+list(df_weight_pathway_metabolic_signaling.index.values)
                                                                                       , row_scaling=False
                                                                                       , retrieval=False)

print('Normalization paper data')
df_scaler_ss = tfm_data.def_dataframe_normalize(df_paper, StandardScaler(), 'cell_type')
print('Normalization signaling data')
df_scaler_ss_signaling = tfm_data.def_dataframe_normalize(df_paper_signaling, StandardScaler(), 'cell_type')
print('Normalization metabolic and signaling data')
df_scaler_ss_metabolic_signaling = tfm_data.def_dataframe_normalize(df_paper_metabolic_signaling, StandardScaler(), 'cell_type')

# DELETE UNUSED DATASET
del(df_paper)
del(df_paper_signaling)
del(df_paper_metabolic_signaling)
del(df_weight_pathway_signaling)
del(df_weight_pathway_metabolic_signaling)

df_scaler_ss['cell_type'] = df_scaler_ss['cell_type'].astype('category')
df_scaler_ss['cell_type_cat'] = df_scaler_ss['cell_type'].cat.codes

df_scaler_ss_signaling['cell_type'] = df_scaler_ss_signaling['cell_type'].astype('category')
df_scaler_ss_signaling['cell_type_cat'] = df_scaler_ss_signaling['cell_type'].cat.codes

df_scaler_ss_metabolic_signaling['cell_type'] = df_scaler_ss_metabolic_signaling['cell_type'].astype('category')
df_scaler_ss_metabolic_signaling['cell_type_cat'] = df_scaler_ss_metabolic_signaling['cell_type'].cat.codes

# IMPORT EXPERIMENT INDEX
path_experiments = os.path.join(os.path.dirname(os.getcwd())+'/data/EXPERIMENTS/')
list_all_experiments = sorted(os.listdir(path_experiments))

# THE TRAINED MODELS
# path_model = os.path.join(os.path.dirname(os.getcwd())+'/data/NN_result/models_epoch_100/')
# path_model = os.path.join(os.path.dirname(os.getcwd())+'/data/NN_result/models_paper_default/')
path_model = os.path.join(os.path.dirname(os.getcwd())+'/data/NN_result/models_False/')
list_all_model = sorted(os.listdir(path_model))


for i_co in list_all_model:
    list_experiments_p = [string for string in list_all_experiments if re.match(re.compile('default_'+str(i_co)), string)]
    list_experiments_s = [string for string in list_all_experiments if re.match(re.compile('signaling_'+str(i_co)), string)]
    list_experiments_ms = [string for string in list_all_experiments if re.match(re.compile('metabolic_signaling_'+str(i_co)), string)]    

    
    list_model_paper_ss = [string for string in os.listdir(os.path.join(path_model+i_co)) if (re.search('ss_dense_p(\w*)', string) or 'ss_p' in string) ]
    list_model_signaling_ss = [string for string in os.listdir(os.path.join(path_model+i_co)) if (re.search('ss_(\w*)_signaling', string) and '_p' not in string) ]
    list_model_met_sig_ss = [string for string in os.listdir(os.path.join(path_model+i_co)) if (re.search('ss_(\w*)_met_sig', string) and '_p' not in string) ]
    
    experiment_pickle_paper = pd.read_pickle(os.path.join(path_experiments+list_experiments_p[0]))
    experiment_pickle_signaling = pd.read_pickle(os.path.join(path_experiments+list_experiments_s[0]))
    experiment_pickle_metabolic_signaling = pd.read_pickle(os.path.join(path_experiments+list_experiments_ms[0]))

    # The design of reference paper
    print(i_co, list_experiments_p)
    df_metric_paper_ss = tfm_cluster.def_clustering_experiment(df_ = df_scaler_ss
                                                          , experiment_index_=experiment_pickle_paper
                                                          , feature_exclude_=['cell_type','cell_type_cat']
                                                          , target_value=['cell_type_cat']
                                                          , model_list_=list_model_paper_ss
                                                          , model_path_=os.path.join(path_model+i_co)
                                                          , cluster_ =int(i_co[-1])
                                                          , bio_= '')
    
    # Proposed model with SIGNALING pathways
    print(i_co, list_experiments_s)
    df_metric_sig_ss = tfm_cluster.def_clustering_experiment(df_ = df_scaler_ss_signaling
                                                          , experiment_index_=experiment_pickle_signaling
                                                          , feature_exclude_=['cell_type','cell_type_cat']
                                                          , target_value=['cell_type_cat']
                                                          , model_list_=list_model_signaling_ss
                                                          , model_path_=os.path.join(path_model+i_co)
                                                          , cluster_ =int(i_co[-1])
                                                          , bio_= '')

    
    # Proposed model with METABOLIC and SIGNALING pathways
    print(i_co, list_experiments_ms)
    df_metric_met_sig_ss = tfm_cluster.def_clustering_experiment(df_ = df_scaler_ss_metabolic_signaling
                                                          , experiment_index_=experiment_pickle_metabolic_signaling
                                                          , feature_exclude_=['cell_type','cell_type_cat']
                                                          , target_value=['cell_type_cat']
                                                          , model_list_=list_model_met_sig_ss
                                                          , model_path_=os.path.join(path_model+i_co)
                                                          , cluster_ =int(i_co[-1])
                                                          , bio_= '' )
    
    df_metric = pd.concat([df_metric_paper_ss, df_metric_sig_ss, df_metric_met_sig_ss
                          ] , axis=0)
    df_metric.columns = ['Feature', 'Homogeneity','Completeness','V-measure','ARI','AMI','Fowlkes-Mallows', 'Average']
    df_metric.sort_values(by='Feature')
    
    df_metric.to_csv(os.path.join(path_output+'/'+str(i_co)+'_clustering_score.txt'), sep=';')

    print('RESULT EXPORTED to "{}"'.format(path_output))

# PRINTING THE DURATION
time_end  = dt.datetime.now().time().strftime('%H:%M:%S') # = time.time()
print('started  !!!     ', time_start)
print('finished !!!     ', time_end)
print('\nELAPSED TIME, ', (dt.datetime.strptime(time_end,'%H:%M:%S') - dt.datetime.strptime(time_start,'%H:%M:%S')))

# RESETTING GPU
from numba import cuda
cuda.select_device(0)
cuda.close()