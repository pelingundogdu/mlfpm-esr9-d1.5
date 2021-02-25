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

    # THE LOCATION of THE RESULT of SCORE and MODEL
    path_output = tfm_data.def_check_create_path('NN_result_2', 'design_'+str(i_row_scaling))
    path_model = tfm_data.def_check_create_path('NN_result_2', 'models_'+str(i_row_scaling))

    # IMPORT EXPERIMENT INDEX
    path_experiments = os.path.join(os.path.dirname(os.getcwd())+'/data/EXPERIMENTS/')
    list_all_model = sorted(os.listdir(path_experiments))
    list_experiments = [string for string in list_all_model if re.match(re.compile('default_cell_out'), string)]

    # LOADING REQUIRED DATASETS
    df_weight_signaling, df_weight_metabolic_signaling = tfm_data.def_load_weight_pathways()
    df_paper, df_signaling, df_metabolic_signaling = tfm_data.def_load_dataset(['cell_type']+list(df_weight_signaling.index.values)
                                                                                    , ['cell_type']+list(df_weight_metabolic_signaling.index.values)
                                                                                    , row_scaling=True)

    del(df_signaling)
    del(df_metabolic_signaling)

    df_weight_dense = pd.DataFrame(df_paper.columns[1:]).set_index('Sample')

    for i in range(dense_layer):
        df_weight_dense['dense'+str(i)] = 1

    df_weight_9347_signaling_dense_pathway = df_weight_dense.merge(pd.DataFrame(df_paper.columns[1:]).set_index('Sample').merge(df_weight_signaling
                                                                                                                                     , left_index=True
                                                                                                                                     , right_index=True
                                                                                                                                     , how='left').fillna(0)
                                                                   , left_index=True
                                                                   , right_index=True, how='inner')
    print('df_weight_9347_signaling_dense_pathway shape            , ',df_weight_9347_signaling_dense_pathway.shape)

    df_weight_9347_metabolic_signaling_dense_pathway = df_weight_dense.merge(pd.DataFrame(df_paper.columns[1:]).set_index('Sample').merge(df_weight_metabolic_signaling
                                                                                                                                     , left_index=True
                                                                                                                                     , right_index=True
                                                                                                                                     , how='left').fillna(0)
                                                                   , left_index=True
                                                                   , right_index=True, how='inner')
    print('df_weight_9347_metabolic_signaling_dense_pathway shape  , ',df_weight_9347_metabolic_signaling_dense_pathway.shape)

    print('Normalization paper data - 9437 genes')
    df_ss = tfm_data.def_dataframe_normalize(df_paper, StandardScaler(), 'cell_type')
    df_mms = tfm_data.def_dataframe_normalize(df_paper, MinMaxScaler(), 'cell_type')

    # # ORIGINAL DATASET (9437 genes)

    # EXPERIMENT DATASETS
    print('the index of experiment dataset, ' ,list_experiments)
    for i_experiment in list_experiments:

        array_train_X_ss, array_train_y_ss = [], []
        array_test_X_ss, array_test_y_ss = [], []

        array_train_X_mms, array_train_y_mms = [], []
        array_test_X_mms, array_test_y_mms = [], []

        experiment_pickle = pd.read_pickle(os.path.join(path_experiments+i_experiment))
        print('Readed pickle file, ', os.path.join(path_experiments+i_experiment))

        for experiment_ in experiment_pickle:
            train_index = experiment_[0]
            test_index = experiment_[1]

            X_train_ss, X_test_ss, y_train_ss, y_test_ss = tfm_data.def_split_train_test_by_index(dataframe_=df_ss
                                                                                                  , train_index_=train_index
                                                                                                  , test_index_=test_index
                                                                                                  , target_feature_=target_)

            array_train_X_ss.append(np.array(X_train_ss))
            array_test_X_ss.append(np.array(X_test_ss))
            array_train_y_ss.append(np.array(y_train_ss))
            array_test_y_ss.append(np.array(y_test_ss))

            X_train_mms, X_test_mms, y_train_mms, y_test_mms = tfm_data.def_split_train_test_by_index(dataframe_=df_mms
                                                                                                      , train_index_=train_index
                                                                                                      , test_index_=test_index
                                                                                                      , target_feature_=target_)

            array_train_X_mms.append(np.array(X_train_mms))
            array_test_X_mms.append(np.array(X_test_mms))
            array_train_y_mms.append(np.array(y_train_mms))
            array_test_y_mms.append(np.array(y_test_mms))

        n_out_cell_type = 'cell_out_'+(re.search(r'(\d+)',i_experiment).group(1))
        path_model_cell_out = tfm_data.def_check_create_path(path_model, n_out_cell_type)
        print(n_out_cell_type)

        # # DESIGN P1 with 1-LAYER

        # ### StandardScaler normalization - Fully connected - 100 dense
        print('Paper dataset with StandardScaler normalization - '+str(df_paper.shape[1]-1)+' gene - dense100')
        #     design01_dense_model, design01_dense_metric = tfm_NN.TFM_NNExperiment(X_train_=array_train_X_ss
        ss_p1_model, ss_p1_metric = tfm_NN.TFM_NNExperiment(X_train_=array_train_X_ss
                                                            , y_train_=array_train_y_ss
                                                            , X_test_=array_test_X_ss
                                                            , y_test_=array_test_y_ss
                                                            , df_w_=pd.DataFrame()
                                                            , bio_='dense_'
                                                            , design_name_='gene_9437_SScaler_1_layer_100'
                                                            , pathway_layer_=False
                                                            , second_layer_=False
                                                            , epochs_=epochs_default
                                                            , batch_size_=batch_size_default
                                                            , unit_size_=100).build()

        # ### MinMaxScaler normalization - Fully connected - 100 dense
        print('Paper dataset with MinMaxScaler normalization - '+str(df_paper.shape[1]-1)+' gene - dense100')
        #     design02_dense_model, design02_dense_metric = tfm_NN.TFM_NNExperiment(X_train_=array_train_X_mms
        mms_p1_model, mms_p1_metric = tfm_NN.TFM_NNExperiment(X_train_=array_train_X_mms
                                                            , y_train_=array_train_y_mms
                                                            , X_test_=array_test_X_mms
                                                            , y_test_=array_test_y_mms
                                                            , df_w_=pd.DataFrame()
                                                            , bio_='dense_'
                                                            , design_name_='gene_9437_MMScaler_1_layer_100'
                                                            , pathway_layer_=False
                                                            , second_layer_=False
                                                            , epochs_=epochs_default
                                                            , batch_size_=batch_size_default
                                                            , unit_size_=100).build()


        # # DESIGN P2 with 1-LAYER - signaling
        unit_size=len(df_weight_9347_signaling_dense_pathway.columns)

        # ### StandardScaler normalization - Fully (100 dense) + Partially (92 signaling pathway) connected - dense+pathway192
        print('Paper dataset with StandardScaler normalization - '+str(df_paper.shape[1]-1)+' gene - dense+pathway'+str(unit_size))
        #     design11_dense_and_pathway_model, design11_dense_and_pathway_metric = tfm_NN.TFM_NNExperiment(X_train_=array_train_X_ss
        ss_p2_sig_model, ss_p2_sig_metric = tfm_NN.TFM_NNExperiment(X_train_=array_train_X_ss
                                                                    , y_train_=array_train_y_ss
                                                                    , X_test_=array_test_X_ss
                                                                    , y_test_=array_test_y_ss
                                                                    , df_w_=df_weight_9347_signaling_dense_pathway
                                                                    , bio_='dense_and_pathway_'
                                                                    , design_name_='signaling_SScaler_1_layer_'+str(unit_size)
                                                                    , pathway_layer_=True
                                                                    , second_layer_=False
                                                                    , epochs_=epochs_default
                                                                    , batch_size_=batch_size_default
                                                                    , unit_size_=unit_size).build()

        # ### MinMaxScaler normalization - Fully (100 dense) + Partially (92 signaling pathway) connected - dense+pathway192
        print('Paper dataset with MinMaxScaler normalization - '+str(df_paper.shape[1]-1)+' gene - dense+pathway'+str(unit_size))
        #     design12_dense_and_pathway_model, design12_dense_and_pathway_metric = tfm_NN.TFM_NNExperiment(X_train_=array_train_X_mms

        mms_p2_sig_model, mms_p2_sig_metric = tfm_NN.TFM_NNExperiment(X_train_=array_train_X_mms
                                                                    , y_train_=array_train_y_mms
                                                                    , X_test_=array_test_X_mms
                                                                    , y_test_=array_test_y_mms
                                                                    , df_w_=df_weight_9347_signaling_dense_pathway
                                                                    , bio_='dense_and_pathway_'
                                                                    , design_name_='signaling_MMScaler_1_layer_'+str(unit_size)
                                                                    , pathway_layer_=True
                                                                    , second_layer_=False
                                                                    , epochs_=epochs_default
                                                                    , batch_size_=batch_size_default
                                                                    , unit_size_=unit_size).build()

        # # DESIGN P2 with 1-LAYER - metabolic and signaling
        unit_size=len(df_weight_9347_metabolic_signaling_dense_pathway.columns)

        # ### StandardScaler normalization - Fully (100 dense) + Partially (250 signaling metabolic pathway) connected - dense+pathway350
        print('Paper dataset with StandardScaler normalization - '+str(df_paper.shape[1]-1)+' gene - dense+pathway'+str(unit_size))
        #     design13_dense_and_pathway_model, design13_dense_and_pathway_metric = tfm_NN.TFM_NNExperiment(X_train_=array_train_X_ss

        ss_p2_met_sig_model, ss_p2_met_sig_metric = tfm_NN.TFM_NNExperiment(X_train_=array_train_X_ss
                                                                            , y_train_=array_train_y_ss
                                                                            , X_test_=array_test_X_ss
                                                                            , y_test_=array_test_y_ss
                                                                            , df_w_=df_weight_9347_metabolic_signaling_dense_pathway
                                                                            , bio_='dense_and_pathway_'
                                                                            , design_name_='metsig_SScaler_1_layer_'+str(unit_size)
                                                                            , pathway_layer_=True
                                                                            , second_layer_=False
                                                                            , epochs_=epochs_default
                                                                            , batch_size_=batch_size_default
                                                                            , unit_size_=unit_size).build()

        # ### MinMaxScaler normalization - Fully (100 dense) + Partially (250 signaling metabolic pathway) connected - dense+pathway350
        print('Paper dataset with MinMaxScaler normalization - '+str(df_paper.shape[1]-1)+' gene - dense+pathway'+str(unit_size))
        #     design14_dense_and_pathway_model, design14_dense_and_pathway_metric = tfm_NN.TFM_NNExperiment(X_train_=array_train_X_mms
        mms_p2_met_sig_model, mms_p2_met_sig_metric = tfm_NN.TFM_NNExperiment(X_train_=array_train_X_mms
                                                                            , y_train_=array_train_y_mms
                                                                            , X_test_=array_test_X_mms
                                                                            , y_test_=array_test_y_mms
                                                                            , df_w_=df_weight_9347_metabolic_signaling_dense_pathway
                                                                            , bio_='dense_and_pathway_'
                                                                            , design_name_='metsig_MMScaler_1_layer_'+str(unit_size)
                                                                            , pathway_layer_=True
                                                                            , second_layer_=False
                                                                            , epochs_=epochs_default
                                                                            , batch_size_=batch_size_default
                                                                            , unit_size_=unit_size).build()

        # ## CREATING TABLE2 SCORING MATRIX

        # ORIGINAL DATASET (9437 genes)
        ### StandardScaler normalization - Fully connected - 100 dense
        ss_p1_metric.columns = ['Feature', 'Homogeneity','Completeness','V-measure','ARI','AMI','Fowlkes-Mallows', 'Average']
        ### MinMaxScaler normalization - Fully connected - 100 dense
        mms_p1_metric.columns = ['Feature', 'Homogeneity','Completeness','V-measure','ARI','AMI','Fowlkes-Mallows', 'Average']

        ### StandardScaler normalization - Fully (100 dense) + Partially (92 signaling pathway) connected - dense+pathway192
        ss_p2_sig_metric.columns = ['Feature', 'Homogeneity','Completeness','V-measure','ARI','AMI','Fowlkes-Mallows', 'Average']
        ### MinMaxScaler normalization - Fully (100 dense) + Partially (92 signaling pathway) connected - dense+pathway192
        mms_p2_sig_metric.columns = ['Feature', 'Homogeneity','Completeness','V-measure','ARI','AMI','Fowlkes-Mallows', 'Average']

        ### StandardScaler normalization - Fully (100 dense) + Partially (250 signaling metabolic pathway) connected - dense+pathway350
        ss_p2_met_sig_metric.columns = ['Feature', 'Homogeneity','Completeness','V-measure','ARI','AMI','Fowlkes-Mallows', 'Average']
        ### MinMaxScaler normalization - Fully (100 dense) + Partially (250 signaling metabolic pathway) connected - dense+pathway350
        mms_p2_met_sig_metric.columns = ['Feature', 'Homogeneity','Completeness','V-measure','ARI','AMI','Fowlkes-Mallows', 'Average']

        df_metrics = pd.concat([ss_p1_metric
                                , ss_p2_sig_metric, mms_p2_sig_metric
                                , mms_p1_metric 
                                , ss_p2_met_sig_metric, mms_p2_met_sig_metric ])
        df_metrics = df_metrics.set_index('Feature')
        print(df_metrics)

        # EXPORTING RESULT SCORE
        df_metrics.to_csv(os.path.join(path_output+'/design_paper_'+n_out_cell_type+'.txt'), sep=';')
        print('RESULT EXPORTED to "{}"'.format(path_output))

        # EXPORTING FINAL MODELS
        ss_p1_model.save(os.path.join(path_model_cell_out+str('/ss_dense_p1_model')))
        mms_p1_model.save(os.path.join(path_model_cell_out+str('/mms_dense_p1_model')))

        ss_p2_sig_model.save(os.path.join(path_model_cell_out+str('/ss_p2_model_signaling')))
        mms_p2_sig_model.save(os.path.join(path_model_cell_out+str('/mms_p2_model_signaling')))
        ss_p2_met_sig_model.save(os.path.join(path_model_cell_out+str('/ss_p2_model_met_sig')))
        mms_p2_met_sig_model.save(os.path.join(path_model_cell_out+str('/mms_p2_model_met_sig')))
        print('MODELS EXPORTED to "{}"'.format(path_model_cell_out))

    # PRINTING NETWORK INTO CONSOLE
    print('design only dense\n')
    print(ss_p1_model.summary())
    print(mms_p1_model.summary())
    print('\n\ndesign signaling 1-layer dense and pathways\n')
    print(ss_p2_sig_model.summary())
    print(mms_p2_sig_model.summary())
    print('\n\ndesign signaling and metabolic 1-layer dense and pathways\n')
    print(ss_p2_met_sig_model.summary())
    print(mms_p2_met_sig_model.summary())

    # PRINTING THE DURATION
    time_end  = dt.datetime.now().time().strftime('%H:%M:%S') # = time.time()
    print('started  !!!     ', time_start)
    print('finished !!!     ', time_end)
    print('\nELAPSED TIME, ', (dt.datetime.strptime(time_end,'%H:%M:%S') - dt.datetime.strptime(time_start,'%H:%M:%S')))
