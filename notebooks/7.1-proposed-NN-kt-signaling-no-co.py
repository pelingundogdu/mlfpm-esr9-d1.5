# 150 minutes

import lib_data_operation as tfm_data
import lib_keras_tuner as tfm_kt

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
config.gpu_options.per_process_gpu_memory_fraction = 0.45
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

time_start = dt.datetime.now().time().strftime('%H:%M:%S') # = time.time()

# DEFAULT VALUES for PAPER DESIGN
epochs_default=100
batch_size_default=10

# TARGET VARIABLE NAME
target_='cell_type'
TYPE_OF_SCALING = [False, True]

for i_scaling in TYPE_OF_SCALING:
    
    # THE LOCATION of THE RESULT of SCORE and MODEL
    # path_hyperband_ = tfm_data.def_check_create_path('kt_result', '')
    path_hyperband_ = tfm_data.def_check_create_path('kt_result', 'delete')
    path_output_result = tfm_data.def_check_create_path('kt_result', 'design_no_co_'+str(i_scaling))
    path_model = tfm_data.def_check_create_path('kt_result', 'models_no_co_'+str(i_scaling))

    # LOADING REQUIRED DATASETS
    df_weight_signaling, df_weight_metabolic_signaling = tfm_data.def_load_weight_pathways()
    df_paper, df_signaling, df_metabolic_signaling = tfm_data.def_load_dataset(['cell_type']+list(df_weight_signaling.index.values)
                                                                               , ['cell_type']+list(df_weight_metabolic_signaling.index.values)
                                                                               , row_scaling = i_scaling
                                                                               , retrieval = False)
    df_weight_ppi_tf_signaling, df_weight_ppi_tf_metabolic_signaling = tfm_data.def_load_weight_ppi_tf(list(df_weight_signaling.index.values)
                                                                                                       , list(df_weight_metabolic_signaling.index.values))
    df_weight_both = pd.concat([df_weight_ppi_tf_signaling, df_weight_signaling], axis=1)
    print('df_weight_both shape , ', df_weight_both.shape)

    print('Normalization signaling data')
    df_ss = tfm_data.def_dataframe_normalize(df_signaling, StandardScaler(), 'cell_type')

    # DELETE UNUSED DATASET
    del(df_paper)
    del(df_metabolic_signaling)
    del(df_weight_ppi_tf_metabolic_signaling)
    del(df_weight_metabolic_signaling)

    del(df_signaling)
    del(df_weight_ppi_tf_signaling)

    # SIGNALING PATHWAY

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

    # DESIGN P with dense

    # ### StandarScaler normalization - Fully connected - 100 dense
    print('Signaling with StandardScaler normalization - '+str(df_weight_both.shape[0])+' gene - dense100')
    ss_dense_model, _, ss_dense_hp = tfm_kt.TFM_KerasTunerExperiment(X_train_=array_train_X_ss
                                                                          , y_train_=array_train_y_ss
                                                                          , X_test_=array_test_X_ss
                                                                          , y_test_=array_test_y_ss
                                                                          , df_w_=pd.DataFrame()
                                                                          , bio_='dense_'
                                                                          , design_name_='signaling_SScaler_1_layer_100'
                                                                          , directory_='signaling_kt_ss_dense100'
                                                                          , project_name_='kt_ss_dense_100_experiment_'
                                                                          , second_layer=False
                                                                          , path_=path_hyperband_
                                                                          , epochs_=epochs_default
                                                                          , batch_size_=batch_size_default).build()

    # DESIGN A with 1-LAYER and 2-LAYER

    # ## with 1-LAYER
    # ### StandardScaler normalization - Pathways connection
    print('Signaling with StandardScaler normalization - '+str(df_weight_both.shape[0])+' gene - pathways'+str(len(df_weight_signaling.columns)))
    ss_a1_model, _, ss_a1_hp = tfm_kt.TFM_KerasTunerExperiment(X_train_=array_train_X_ss
                                                                          , y_train_=array_train_y_ss
                                                                          , X_test_=array_test_X_ss
                                                                          , y_test_=array_test_y_ss
                                                                          , df_w_=df_weight_signaling
                                                                          , bio_='pathway_'
                                                                          , design_name_='signaling_SScaler_1_layer_'
                                                                          , directory_='signaling_kt_ss_a1_pathway'
                                                                          , project_name_='kt_ss_a1_experiment_'
                                                                          , second_layer=False
                                                                          , path_=path_hyperband_
                                                                          , epochs_=epochs_default
                                                                          , batch_size_=batch_size_default).build()

    # ## with 2-LAYER
    # ### StandardScaler normalization - Pathways connection + Fully connected - signaling + ???? dense
    print('Signaling with StandardScaler normalization - '+str(df_weight_both.shape[0])+' gene - pathways'+str(len(df_weight_signaling.columns))+' + dense ????')
    ss_a2_model, _, ss_a2_hp = tfm_kt.TFM_KerasTunerExperiment(X_train_=array_train_X_ss
                                                                          , y_train_=array_train_y_ss
                                                                          , X_test_=array_test_X_ss
                                                                          , y_test_=array_test_y_ss
                                                                          , df_w_=df_weight_signaling
                                                                          , bio_='pathway_'
                                                                          , design_name_='signaling_SScaler_2_layer_'
                                                                          , directory_='signaling_kt_ss_a2_pathway'
                                                                          , project_name_='kt_ss_a2_experiment_'
                                                                          , second_layer=True
                                                                          , path_=path_hyperband_
                                                                          , epochs_=epochs_default
                                                                          , batch_size_=batch_size_default).build()

    # DESIGN B with 1-LAYER and 2-LAYER

    # ## with 1-LAYER
    # ### StandardScaler normalization - Pathways and PPI/TF connection
    print('Signaling with StandardScaler normalization - '+str(df_weight_both.shape[0])+' gene - pathways_ppi_tf'+str(df_weight_both.shape[1]))
    ss_b1_model, _, ss_b1_hp = tfm_kt.TFM_KerasTunerExperiment(X_train_=array_train_X_ss
                                                                          , y_train_=array_train_y_ss
                                                                          , X_test_=array_test_X_ss
                                                                          , y_test_=array_test_y_ss
                                                                          , df_w_=df_weight_both
                                                                          , bio_='ppi_tf_pathway_'
                                                                          , design_name_='signaling_SScaler_1_layer_'
                                                                          , directory_='signaling_kt_ss_b1_pathway_ppi'
                                                                          , project_name_='kt_ss_b1_experiment_'
                                                                          , second_layer=False
                                                                          , path_=path_hyperband_
                                                                          , epochs_=epochs_default
                                                                          , batch_size_=batch_size_default).build()

    # ## with 2-LAYER
    # ### StandardScaler normalization - Pathways and PPI/TF connection + Fully connected - nodes + ???? dense
    print('Signaling with StandardScaler normalization - '+str(df_weight_both.shape[0])+' gene - pathways_ppi_tf'+str(df_weight_both.shape[1])+' + dense ????')
    ss_b2_model, _, ss_b2_hp = tfm_kt.TFM_KerasTunerExperiment(X_train_=array_train_X_ss
                                                                          , y_train_=array_train_y_ss
                                                                          , X_test_=array_test_X_ss
                                                                          , y_test_=array_test_y_ss
                                                                          , df_w_=df_weight_both
                                                                          , bio_='ppi_tf_pathway_'
                                                                          , design_name_='signaling_SScaler_2_layer_'#+str(len(df_W.columns))
                                                                          , directory_='signaling_kt_ss_b2_pathway_ppi'
                                                                          , project_name_='kt_ss_b2_experiment_'
                                                                          , second_layer=True
                                                                          , path_=path_hyperband_
                                                                          , epochs_=epochs_default
                                                                          , batch_size_=batch_size_default).build()

    ss_dense_model.save(os.path.join(path_model+str('/ss_dense_model_signaling')))

    ss_a1_model.save(os.path.join(path_model+str('/ss_a1_model_signaling')))
    ss_a2_model.save(os.path.join(path_model+str('/ss_a2_model_signaling')))

    ss_b1_model.save(os.path.join(path_model+str('/ss_b1_model_signaling')))
    ss_b2_model.save(os.path.join(path_model+str('/ss_b2_model_signaling')))
    print('MODELS EXPORTED to "{}"'.format(path_model))

    ss_dense_hp = tfm_kt.def_hp(ss_dense_hp, 'ss_dense_hp')

    ss_a1_hp = tfm_kt.def_hp(ss_a1_hp, 'ss_a1_hp')
    ss_a2_hp = tfm_kt.def_hp(ss_a2_hp, 'ss_a2_hp')

    ss_b1_hp = tfm_kt.def_hp(ss_b1_hp, 'ss_b1_hp')
    ss_b2_hp = tfm_kt.def_hp(ss_b2_hp, 'ss_b2_hp')

    df_hp = pd.concat([ss_dense_hp
                       , ss_a1_hp, ss_a2_hp
                       , ss_b1_hp, ss_b2_hp ])

    df_hp = df_hp.set_index('hp')
    df_hp.to_csv(os.path.join(path_output_result+'/kt_hyperparameters_signaling_no_co_'+str(i_scaling)+'.txt'), sep=';')
    print('RESULT EXPORTED to "{}"'.format(path_output_result))

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