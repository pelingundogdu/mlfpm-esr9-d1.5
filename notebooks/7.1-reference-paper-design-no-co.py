# ~70 minutes

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
dense_layer=100

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

    df_weight_dense = pd.DataFrame(df_paper.columns[1:]).set_index('Sample')

    for i in range(dense_layer):
        df_weight_dense['dense'+str(i)] = 1.0

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


    print('Normalization signaling data')
    df_ss = tfm_data.def_dataframe_normalize(df_paper, StandardScaler(), 'cell_type')
    # df_mms = tfm_data.def_dataframe_normalize(df_signaling, MinMaxScaler(), 'cell_type')

    # DELETE UNUSED DATASET
    # del(df_paper)
    del(df_signaling)
    del(df_metabolic_signaling)

    del(df_weight_metabolic_signaling)
    del(df_weight_signaling)

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

    # ORIGINAL DATASET (9437 genes)

    # DESIGN P1 with 1-LAYER

    ### StandardScaler normalization - Fully connected - 100 dense
    print('Default with StandardScaler normalization - '+str(df_paper.shape[1]-1)+' gene - dense100')
    ss_p1_model, _, ss_p1_hp = tfm_kt.TFM_KerasTunerExperiment(X_train_=array_train_X_ss
                                                                          , y_train_=array_train_y_ss
                                                                          , X_test_=array_test_X_ss
                                                                          , y_test_=array_test_y_ss
                                                                          , df_w_=pd.DataFrame()
                                                                          , bio_='dense_'
                                                                          , design_name_='default_SScaler_1_layer_100_p1'
                                                                          , directory_='kt_ss_p1_default'
                                                                          , project_name_='kt_ss_p1_dense100_experiment_'
                                                                          , second_layer=False
                                                                          , path_=path_hyperband_
                                                                          , epochs_=epochs_default
                                                                          , batch_size_=batch_size_default).build()

    # DESIGN P2 with 1-LAYER - signaling

    ### StandardScaler normalization - Fully (100 dense) + Partially (92 signaling pathway) connected - dense+pathway192
    print('Signaling with StandardScaler normalization - '+str(df_paper.shape[1]-1)+' gene - pathways'+str(len(df_weight_paper_signaling_dense_pathway.columns)))
    ss_p2_sig_model, _, ss_p2_sig_hp = tfm_kt.TFM_KerasTunerExperiment(X_train_=array_train_X_ss
                                                                          , y_train_=array_train_y_ss
                                                                          , X_test_=array_test_X_ss
                                                                          , y_test_=array_test_y_ss
                                                                          , df_w_=df_weight_paper_signaling_dense_pathway
                                                                          , bio_='dense_and_pathway_'
                                                                          , design_name_='signaling_SScaler_1_layer_'+str(len(df_weight_paper_signaling_dense_pathway.columns))
                                                                          , directory_='kt_ss_p2_signaling'
                                                                          , project_name_='kt_ss_p2_signaling_experiment_'
                                                                          , second_layer=False
                                                                          , path_=path_hyperband_
                                                                          , epochs_=epochs_default
                                                                          , batch_size_=batch_size_default).build()

    # DESIGN P2 with 1-LAYER - metabolic and signaling

    ### StandardScaler normalization - Fully (100 dense) + Partially (250 signaling metabolic pathway) connected - dense+pathway350
    print('Signaling with StandardScaler normalization - '+str(df_paper.shape[1]-1)+' gene - pathways'+str(len(df_weight_paper_metabolic_signaling_dense_pathway.columns)))
    ss_p2_met_sig_model, _, ss_p2_met_sig_hp = tfm_kt.TFM_KerasTunerExperiment(X_train_=array_train_X_ss
                                                                          , y_train_=array_train_y_ss
                                                                          , X_test_=array_test_X_ss
                                                                          , y_test_=array_test_y_ss
                                                                          , df_w_=df_weight_paper_metabolic_signaling_dense_pathway
                                                                          , bio_='dense_and_pathway_'
                                                                          , design_name_='metsig_SScaler_1_layer_'+str(len(df_weight_paper_metabolic_signaling_dense_pathway.columns))
                                                                          , directory_='kt_ss_p2_metabolic_signaling'
                                                                          , project_name_='kt_ss_p2_metabolic_signaling_experiment_'
                                                                          , second_layer=False
                                                                          , path_=path_hyperband_
                                                                          , epochs_=epochs_default
                                                                          , batch_size_=batch_size_default).build()


    ss_p1_model.save(os.path.join(path_model+str('/ss_p1_model_default')))

    ss_p2_sig_model.save(os.path.join(path_model+str('/ss_p2_model_signaling')))
    ss_p2_met_sig_model.save(os.path.join(path_model+str('/ss_p2_model_metabolic_signaling')))

    print('MODELS EXPORTED to "{}"'.format(path_model))

    ss_p1_hp = tfm_kt.def_hp(ss_p1_hp, 'ss_p1_hp')

    ss_p2_sig_hp = tfm_kt.def_hp(ss_p2_sig_hp, 'ss_p2_sig_hp')
    ss_p2_met_sig_hp = tfm_kt.def_hp(ss_p2_met_sig_hp, 'ss_p2_met_sig_hp')


    df_hp = pd.concat([ss_p1_hp
                       , ss_p2_sig_hp
                       , ss_p2_met_sig_hp ])

    df_hp = df_hp.set_index('hp')
    df_hp.to_csv(os.path.join(path_output_result+'/kt_hyperparameters_design_p_no_co_'+str(i_scaling)+'.txt'), sep=';')
    print('RESULT EXPORTED to "{}"'.format(path_output_result))

    print("\n\nDesign P dense\n")
    print(ss_p1_model.summary())

    print("\nDesign A \n")
    print(ss_p2_sig_model.summary())
    print(ss_p2_met_sig_model.summary())

# PRINTING THE DURATION
time_end  = dt.datetime.now().time().strftime('%H:%M:%S') # = time.time()
print('started  !!!     ', time_start)
print('finished !!!     ', time_end)
print('\nELAPSED TIME, ', (dt.datetime.strptime(time_end,'%H:%M:%S') - dt.datetime.strptime(time_start,'%H:%M:%S')))

session.close()