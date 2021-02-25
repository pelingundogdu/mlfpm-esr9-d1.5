import lib_data_operation as tfm_data
import lib_neural_network as tfm_NN

# Required libraries
import os
import re
import numpy as np
import pandas as pd
import datetime as dt
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import KFold

import tensorflow as tf
import tensorflow.keras.backend as K

# from numba import cuda
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
print('CONTROL!!', tf.config.list_physical_devices('GPU'))

# OPTIMIZATION GPU USAGE
config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.15
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

time_start = dt.datetime.now().time().strftime('%H:%M:%S') # = time.time()

# DEFAULT VALUES for PAPER DESIGN
epochs_default=100
batch_size_default=10
dense_layer=100
# TARGET VARIABLE NAME
target_='cell_type'
TYPE_OF_SCALING = [False, True]

for i_row_scaling in TYPE_OF_SCALING:
    TYPE_OF_EXPERIMENT = 'no_co_'+str(i_row_scaling)
    # THE LOCATION of THE RESULT of SCORE and MODEL
#     path_output = tfm_data.def_check_create_path('NN_result', 'design_'+TYPE_OF_EXPERIMENT)
    path_model = tfm_data.def_check_create_path('NN_result', 'models_'+TYPE_OF_EXPERIMENT)

    # LOADING REQUIRED DATASETS
    df_weight_signaling, df_weight_metabolic_signaling = tfm_data.def_load_weight_pathways()
    df_paper, df_signaling, df_metabolic_signaling = tfm_data.def_load_dataset(['cell_type']+list(df_weight_signaling.index.values)
                                                                               , ['cell_type']+list(df_weight_metabolic_signaling.index.values)
                                                                               , row_scaling=i_row_scaling)

    del(df_signaling)
    del(df_metabolic_signaling)

    df_weight_dense = pd.DataFrame(df_paper.columns[1:]).set_index('Sample')

    for i in range(dense_layer):
        df_weight_dense['dense'+str(i)] = 1

    df_weight_paper_signaling_dense_pathway = df_weight_dense.merge(pd.DataFrame(df_paper.columns[1:]).set_index('Sample').merge(df_weight_signaling
                                                                                                                                     , left_index=True
                                                                                                                                     , right_index=True
                                                                                                                                     , how='left').fillna(0)
                                                                   , left_index=True
                                                                   , right_index=True, how='inner')
    print('df_weight_paper_signaling_dense_pathway shape            , ',df_weight_paper_signaling_dense_pathway.shape)

    df_weight_paper_metabolic_signaling_dense_pathway = df_weight_dense.merge(pd.DataFrame(df_paper.columns[1:]).set_index('Sample').merge(df_weight_metabolic_signaling
                                                                                                                                     , left_index=True
                                                                                                                                     , right_index=True
                                                                                                                                     , how='left').fillna(0)
                                                                   , left_index=True
                                                                   , right_index=True, how='inner')
    print('df_weight_paper_metabolic_signaling_dense_pathway shape  , ',df_weight_paper_metabolic_signaling_dense_pathway.shape)

    print('Normalization paper data - 9437 genes')
    df_ss = tfm_data.def_dataframe_normalize(df_paper, StandardScaler(), 'cell_type')
#     df_mms = tfm_data.def_dataframe_normalize(df_paper, MinMaxScaler(), 'cell_type')

    # # ORIGINAL DATASET (9437 genes)

    # EXPERIMENT DATASETS
    array_train_X_ss, array_train_y_ss = [], []
    array_test_X_ss, array_test_y_ss = [], []

    X_train_ss, X_test_ss, y_train_ss, y_test_ss = tfm_data.def_split_train_test_by_index(dataframe_=df_ss
                                                                                          , train_index_=df_ss.index
                                                                                          , test_index_=[0]
                                                                                          , target_feature_=target_)

    array_train_X_ss.append(np.array(X_train_ss))
    array_test_X_ss.append(np.array(X_test_ss))
    array_train_y_ss.append(np.array(y_train_ss))
    array_test_y_ss.append(np.array(y_test_ss))

    # # DESIGN P1 with 1-LAYER

    # ### StandardScaler normalization - Fully connected - 100 dense
    print('Paper dataset with StandardScaler normalization - '+str(df_paper.shape[1]-1)+' gene - dense100')
    ss_p1_model, _ = tfm_NN.TFM_NNExperiment(X_train_=array_train_X_ss
                                                        , y_train_=array_train_y_ss
                                                        , X_test_=array_test_X_ss
                                                        , y_test_=array_test_y_ss
                                                        , df_w_=pd.DataFrame()
                                                        , bio_='dense_'
                                                        , design_name_='gene_'+str(df_paper.shape[1]-1)+'_SScaler_1_layer_100'
                                                        , pathway_layer_=False
                                                        , second_layer_=False
                                                        , epochs_=epochs_default
                                                        , batch_size_=batch_size_default
                                                        , unit_size_=100).build()

    # # DESIGN P2 with 1-LAYER - signaling
    unit_size=len(df_weight_paper_signaling_dense_pathway.columns)

    # ### StandardScaler normalization - Fully (100 dense) + Partially (92 signaling pathway) connected - dense+pathway192
    print('Paper dataset with StandardScaler normalization - '+str(df_paper.shape[1]-1)+' gene - dense+pathway'+str(unit_size))
    ss_p2_sig_model, _ = tfm_NN.TFM_NNExperiment(X_train_=array_train_X_ss
                                                                , y_train_=array_train_y_ss
                                                                , X_test_=array_test_X_ss
                                                                , y_test_=array_test_y_ss
                                                                , df_w_=df_weight_paper_signaling_dense_pathway
                                                                , bio_='dense_and_pathway_'
                                                                , design_name_='signaling_SScaler_1_layer_'+str(unit_size)
                                                                , pathway_layer_=True
                                                                , second_layer_=False
                                                                , epochs_=epochs_default
                                                                , batch_size_=batch_size_default
                                                                , unit_size_=unit_size).build()

    # # DESIGN P2 with 1-LAYER - metabolic and signaling
    unit_size=len(df_weight_paper_metabolic_signaling_dense_pathway.columns)

    # ### StandardScaler normalization - Fully (100 dense) + Partially (250 signaling metabolic pathway) connected - dense+pathway350
    print('Paper dataset with StandardScaler normalization - '+str(df_paper.shape[1]-1)+' gene - dense+pathway'+str(unit_size))
    ss_p2_met_sig_model, _ = tfm_NN.TFM_NNExperiment(X_train_=array_train_X_ss
                                                                        , y_train_=array_train_y_ss
                                                                        , X_test_=array_test_X_ss
                                                                        , y_test_=array_test_y_ss
                                                                        , df_w_=df_weight_paper_metabolic_signaling_dense_pathway
                                                                        , bio_='dense_and_pathway_'
                                                                        , design_name_='metsig_SScaler_1_layer_'+str(unit_size)
                                                                        , pathway_layer_=True
                                                                        , second_layer_=False
                                                                        , epochs_=epochs_default
                                                                        , batch_size_=batch_size_default
                                                                        , unit_size_=unit_size).build()

    # EXPORTING FINAL MODELS
    ss_p1_model.save(os.path.join(path_model+str('/ss_dense_p1_model')))

    ss_p2_sig_model.save(os.path.join(path_model+str('/ss_p2_model_signaling')))
    ss_p2_met_sig_model.save(os.path.join(path_model+str('/ss_p2_model_met_sig')))
    print('MODELS EXPORTED to "{}"'.format(path_model))

# PRINTING NETWORK INTO CONSOLE
print('design only dense\n')
print(ss_p1_model.summary())
print('\n\ndesign signaling 1-layer dense and pathways\n')
print(ss_p2_sig_model.summary())
print('\n\ndesign signaling and metabolic 1-layer dense and pathways\n')
print(ss_p2_met_sig_model.summary())

# PRINTING THE DURATION
time_end  = dt.datetime.now().time().strftime('%H:%M:%S') # = time.time()
print('started  !!!     ', time_start)
print('finished !!!     ', time_end)
print('\nELAPSED TIME, ', (dt.datetime.strptime(time_end,'%H:%M:%S') - dt.datetime.strptime(time_start,'%H:%M:%S')))