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

import tensorflow.keras.backend as K
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

# OPTIMIZATION GPU USAGE
config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.15
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

time_start = dt.datetime.now().time().strftime('%H:%M:%S') # = time.time()

# DEFAULT VALUES for PAPER DESIGN
epochs_default=100
batch_size_default=10

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

    df_weight_ppi_tf_signaling, df_weight_ppi_tf_metabolic_signaling = tfm_data.def_load_weight_ppi_tf(list(df_weight_signaling.index.values)
                                                                                                       , list(df_weight_metabolic_signaling.index.values))
    df_weight_both = pd.concat([df_weight_ppi_tf_signaling, df_weight_signaling], axis=1)
    print('df_weight_both shape , ', df_weight_both.shape)

    print('Normalization signaling data - 1646 genes')
    df_ss = tfm_data.def_dataframe_normalize(df_signaling, StandardScaler(), 'cell_type')
#     df_mms = tfm_data.def_dataframe_normalize(df_signaling, MinMaxScaler(), 'cell_type')

    # DELETE UNUSED DATASET
    del(df_paper)
    del(df_metabolic_signaling)
    del(df_weight_metabolic_signaling)
    del(df_weight_ppi_tf_metabolic_signaling)

    del(df_signaling)
    del(df_weight_ppi_tf_signaling)

    # # SIGNALING PATHWAY (1843 genes)

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

    # # DESIGN P with dense

    # ### StandarScaler normalization - Fully connected - 100 dense
    print('Signaling with StandardScaler normalization - '+str(df_weight_both.shape[0])+' gene - dense100')
    ss_dense_model, _ = tfm_NN.TFM_NNExperiment(X_train_=array_train_X_ss
                                                              , y_train_=array_train_y_ss
                                                              , X_test_=array_test_X_ss
                                                              , y_test_=array_test_y_ss
                                                              , df_w_=pd.DataFrame()
                                                              , bio_='dense_'
                                                              , design_name_='signaling_SScaler_1_layer_100'
                                                              , pathway_layer_=False
                                                              , second_layer_=False
                                                              , epochs_=epochs_default
                                                              , batch_size_=batch_size_default
                                                              , unit_size_=100).build()

    # # DESIGN A with 1-LAYER and 2-LAYER
    unit_size=len(df_weight_signaling.columns)

    # ## with 1-LAYER
    # ### StandardScaler normalization - Pathways connection - 92 signaling
    print('Signaling with StandardScaler normalization - '+str(df_weight_both.shape[0])+' gene - pathways'+str(unit_size))
    ss_a1_model, _ = tfm_NN.TFM_NNExperiment(X_train_=array_train_X_ss
                                                        , y_train_=array_train_y_ss
                                                        , X_test_=array_test_X_ss
                                                        , y_test_=array_test_y_ss
                                                        , df_w_=df_weight_signaling
                                                        , bio_='pathway_'
                                                        , design_name_='signaling_SScaler_1_layer_'+str(unit_size)
                                                        , pathway_layer_=True
                                                        , second_layer_=False
                                                        , epochs_=epochs_default
                                                        , batch_size_=batch_size_default
                                                        , unit_size_=unit_size).build()

    # ## with 2-LAYER
    # ### StandardScaler normalization - Pathways connection + Fully connected - 92 signaling + 100 dense
    print('Signaling with StandardScaler normalization - '+str(df_weight_both.shape[0])+' gene - pathways'+str(unit_size)+' + dense100')
    ss_a2_model, ss_a2_metric = tfm_NN.TFM_NNExperiment(X_train_=array_train_X_ss
                                                        , y_train_=array_train_y_ss
                                                        , X_test_=array_test_X_ss
                                                        , y_test_=array_test_y_ss
                                                        , df_w_=df_weight_signaling
                                                        , bio_='pathway_'
                                                        , design_name_='signaling_SScaler_2_layer_'+str(unit_size)
                                                        , pathway_layer_=True
                                                        , second_layer_=True
                                                        , epochs_=epochs_default
                                                        , batch_size_=batch_size_default
                                                        , unit_size_=unit_size).build()

    # # DESIGN B with 1-LAYER and 2-LAYER
    unit_size=len(df_weight_both.columns)

    # ## with 1-LAYER
    # ### StandardScaler normalization - Pathways and PPI/TF connection - 769 nodes
    print('Signaling with StandardScaler normalization - '+str(df_weight_both.shape[0])+' gene - pathways_ppi_tf'+str(df_weight_both.shape[1]))
    ss_b1_model, _ = tfm_NN.TFM_NNExperiment(X_train_=array_train_X_ss
                                                        , y_train_=array_train_y_ss
                                                        , X_test_=array_test_X_ss
                                                        , y_test_=array_test_y_ss
                                                        , df_w_=df_weight_both
                                                        , bio_='ppi_tf_pathway_'
                                                        , design_name_='signaling_SScaler_1_layer_'+str(unit_size)
                                                        , pathway_layer_=True
                                                        , second_layer_=False
                                                        , epochs_=epochs_default
                                                        , batch_size_=batch_size_default
                                                        , unit_size_=unit_size).build()

    # ## with 2-LAYER
    # ### StandardScaler normalization - Pathways and PPI/TF connection + Fully connected - 769 nodes + 100 dense
    print('Signaling with StandardScaler normalization - '+str(df_weight_both.shape[0])+' gene - pathways_ppi_tf'+str(df_weight_both.shape[1])+' + dense100')
    ss_b2_model, _ = tfm_NN.TFM_NNExperiment(X_train_=array_train_X_ss
                                                        , y_train_=array_train_y_ss
                                                        , X_test_=array_test_X_ss
                                                        , y_test_=array_test_y_ss
                                                        , df_w_=df_weight_both
                                                        , bio_='ppi_tf_pathway_'
                                                        , design_name_='signaling_SScaler_2_layer_'+str(unit_size)
                                                        , pathway_layer_=True
                                                        , second_layer_=True
                                                        , epochs_=epochs_default
                                                        , batch_size_=batch_size_default
                                                        , unit_size_=unit_size).build()

    # EXPORTING FINAL MODELS
    ss_dense_model.save(os.path.join(path_model+str('/ss_dense_model_signaling')))

    ss_a1_model.save(os.path.join(path_model+str('/ss_a1_model_signaling')))
    ss_a2_model.save(os.path.join(path_model+str('/ss_a2_model_signaling')))

    ss_b1_model.save(os.path.join(path_model+str('/ss_b1_model_signaling')))
    ss_b2_model.save(os.path.join(path_model+str('/ss_b2_model_signaling')))
    print('MODELS EXPORTED to "{}"'.format(path_model))

# PRINTING NETWORK INTO CONSOLE
print("\n\nDesign P dense\n")
print(ss_dense_model.summary())
print("\nDesign A \n")
print(ss_a1_model.summary())
print(ss_a2_model.summary())
print("\nDesign B \n")
print(ss_b1_model.summary())
print(ss_b2_model.summary())

# PRINTING THE DURATION
time_end  = dt.datetime.now().time().strftime('%H:%M:%S') # = time.time()
print('started  !!!     ', time_start)
print('finished !!!     ', time_end)
print('\nELAPSED TIME, ', (dt.datetime.strptime(time_end,'%H:%M:%S') - dt.datetime.strptime(time_start,'%H:%M:%S')))
