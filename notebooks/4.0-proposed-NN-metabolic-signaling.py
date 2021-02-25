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

    # THE LOCATION of THE RESULT of SCORE and MODEL
    path_output = tfm_data.def_check_create_path('NN_result_2', 'design_'+str(i_row_scaling))
    path_model = tfm_data.def_check_create_path('NN_result_2', 'models_'+str(i_row_scaling))

    # IMPORT EXPERIMENT INDEX
    path_experiments = os.path.join(os.path.dirname(os.getcwd())+'/data/EXPERIMENTS/')
    list_all_model = sorted(os.listdir(path_experiments))
    list_experiments = [string for string in list_all_model if re.match(re.compile('metabolic_signaling_cell_out_'), string)]

    # LOADING REQUIRED DATASETS
    df_weight_signaling, df_weight_metabolic_signaling = tfm_data.def_load_weight_pathways()
    df_paper, df_signaling, df_metabolic_signaling = tfm_data.def_load_dataset(['cell_type']+list(df_weight_signaling.index.values)
                                                                                    , ['cell_type']+list(df_weight_metabolic_signaling.index.values)
                                                                                    , row_scaling=True)
    df_weight_ppi_tf_signaling, df_weight_ppi_tf_metabolic_signaling = tfm_data.def_load_weight_ppi_tf(list(df_weight_signaling.index.values)
                                                                                                       , list(df_weight_metabolic_signaling.index.values))
    df_weight_both = pd.concat([df_weight_ppi_tf_metabolic_signaling, df_weight_metabolic_signaling], axis=1)
    print('df_weight_both shape , ', df_weight_both.shape)

    print('Normalization metabolic and signaling data - 3737 genes')
    df_ss = tfm_data.def_dataframe_normalize(df_metabolic_signaling, StandardScaler(), 'cell_type')
    df_mms = tfm_data.def_dataframe_normalize(df_metabolic_signaling, MinMaxScaler(), 'cell_type')

    # DELETE UNUSED DATASET
    del(df_paper)
    del(df_signaling)
    del(df_weight_signaling)
    del(df_weight_ppi_tf_signaling)

    del(df_metabolic_signaling)
    del(df_weight_ppi_tf_metabolic_signaling)

    # # METABOLIC and SIGNALING PATHWAY (3922 genes)

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

        # # DESIGN P with dense

        # ### StandarScaler normalization - Fully connected - 100 dense
        print('Metabolic and signaling with StandardScaler normalization - '+str(df_weight_both.shape[0])+' gene - dense100')
        #     design31_dense_model, design31_dense_metric = tfm_NN.TFM_NNExperiment(X_train_=array_train_X_ss
        ss_dense_model, ss_dense_metric = tfm_NN.TFM_NNExperiment(X_train_=array_train_X_ss
                                                                  , y_train_=array_train_y_ss
                                                                  , X_test_=array_test_X_ss
                                                                  , y_test_=array_test_y_ss
                                                                  , df_w_=pd.DataFrame()
                                                                  , bio_='dense_'
                                                                  , design_name_='metsig_SScaler_1_layer_100'
                                                                  , pathway_layer_=False
                                                                  , second_layer_=False
                                                                  , epochs_=epochs_default
                                                                  , batch_size_=batch_size_default
                                                                  , unit_size_=100).build()

        # ### MinMaxScaler normalization - Fully connected - 100 dense
        print('Metabolic and signaling with StandardScaler normalization - '+str(df_weight_both.shape[0])+' gene - dense100')
        #     design32_dense_model, design32_dense_metric = tfm_NN.TFM_NNExperiment(X_train_=array_train_X_mms
        mms_dense_model, mms_dense_metric = tfm_NN.TFM_NNExperiment(X_train_=array_train_X_mms
                                                                  , y_train_=array_train_y_mms
                                                                  , X_test_=array_test_X_mms
                                                                  , y_test_=array_test_y_mms
                                                                  , df_w_=pd.DataFrame()
                                                                  , bio_='dense_'
                                                                  , design_name_='metsig_MMScaler_1_layer_100'
                                                                  , pathway_layer_=False
                                                                  , second_layer_=False
                                                                  , epochs_=epochs_default
                                                                  , batch_size_=batch_size_default
                                                                  , unit_size_=100).build()

        # # DESIGN A with 1-LAYER and 2-LAYER
        unit_size=len(df_weight_metabolic_signaling.columns)

        # ## with 1-LAYER

        # ### StandardScaler normalization - Pathways connection - 250 nodes
        print('Metabolic and signaling with StandardScaler normalization - '+str(df_weight_both.shape[0])+' gene - pathways'+str(unit_size))
        #     design34_A1layer_pathway_model, design34_A1layer_pathway_metric = tfm_NN.TFM_NNExperiment(X_train_=array_train_X_ss
        ss_a1_model, ss_a1_metric = tfm_NN.TFM_NNExperiment(X_train_=array_train_X_ss
                                                            , y_train_=array_train_y_ss
                                                            , X_test_=array_test_X_ss
                                                            , y_test_=array_test_y_ss
                                                            , df_w_=df_weight_metabolic_signaling
                                                            , bio_='pathway_'
                                                            , design_name_='metsig_SScaler_1_layer_'+str(unit_size)
                                                            , pathway_layer_=True
                                                            , second_layer_=False
                                                            , epochs_=epochs_default
                                                            , batch_size_=batch_size_default
                                                            , unit_size_=unit_size).build()

        # ### MinMaxScaler normalization - Pathways connection - 250 nodes
        print('Metabolic and signaling with MinMaxScaler normalization - '+str(df_weight_both.shape[0])+' gene - pathways'+str(unit_size))
        #     design35_A1layer_pathway_model, design35_A1layer_pathway_metric = tfm_NN.TFM_NNExperiment(X_train_=array_train_X_mms
        mms_a1_model, mms_a1_metric = tfm_NN.TFM_NNExperiment(X_train_=array_train_X_mms
                                                            , y_train_=array_train_y_mms
                                                            , X_test_=array_test_X_mms
                                                            , y_test_=array_test_y_mms
                                                            , df_w_=df_weight_metabolic_signaling
                                                            , bio_='pathway_'
                                                            , design_name_='metsig_MMScaler_1_layer_'+str(unit_size)
                                                            , pathway_layer_=True
                                                            , second_layer_=False
                                                            , epochs_=epochs_default
                                                            , batch_size_=batch_size_default
                                                            , unit_size_=unit_size).build()

        # ## with 2-LAYER

        # ### StandardScaler normalization - Pathways connection + Fully connected - 250 nodes + 100 dense
        print('Metabolic and signaling with StandardScaler normalization - '+str(df_weight_both.shape[0])+' gene - pathways'+str(unit_size)+' + dense100' )
        #     design34_A2layer_pathway_model, design34_A2layer_pathway_metric = tfm_NN.TFM_NNExperiment(X_train_=array_train_X_ss
        ss_a2_model, ss_a2_metric = tfm_NN.TFM_NNExperiment(X_train_=array_train_X_ss
                                                            , y_train_=array_train_y_ss
                                                            , X_test_=array_test_X_ss
                                                            , y_test_=array_test_y_ss
                                                            , df_w_=df_weight_metabolic_signaling
                                                            , bio_='pathway_'
                                                            , design_name_='metsig_SScaler_2_layer_'+str(unit_size)
                                                            , pathway_layer_=True
                                                            , second_layer_=True
                                                            , epochs_=epochs_default
                                                            , batch_size_=batch_size_default
                                                            , unit_size_=unit_size).build()

        # ### MinMaxScaler normalization - Pathways connection + Fully connected - 250 nodes + 100 dense
        print('Metabolic and signaling with MinMaxScaler normalization - '+str(df_weight_both.shape[0])+' gene - pathways'+str(unit_size)+' + dense100')
        #     design35_A2layer_pathway_model, design35_A2layer_pathway_metric = tfm_NN.TFM_NNExperiment(X_train_=array_train_X_mms
        mms_a2_model, mms_a2_metric = tfm_NN.TFM_NNExperiment(X_train_=array_train_X_mms
                                                            , y_train_=array_train_y_mms
                                                            , X_test_=array_test_X_mms
                                                            , y_test_=array_test_y_mms
                                                            , df_w_=df_weight_metabolic_signaling
                                                            , bio_='pathway_'
                                                            , design_name_='metsig_MMScaler_2_layer_'+str(unit_size)
                                                            , pathway_layer_=True
                                                            , second_layer_=True
                                                            , epochs_=epochs_default
                                                            , batch_size_=batch_size_default
                                                            , unit_size_=unit_size).build()

        # # DESIGN B with 1-LAYER and 2-LAYER
        unit_size=len(df_weight_both.columns)

        # ## with 1-LAYER

        # ### StandardScaler normalization - Pathways and PPI/TF connection - 941 nodes
        print('Metabolic and signaling with StandardScaler normalization - '+str(df_weight_both.shape[0])+' gene - pathways_ppi_tf'+str(df_weight_both.shape[1]) )
        #     design34_B1layer_pathway_ppi_model, design34_B1layer_pathway_ppi_metric = tfm_NN.TFM_NNExperiment(X_train_=array_train_X_ss
        ss_b1_model, ss_b1_metric = tfm_NN.TFM_NNExperiment(X_train_=array_train_X_ss
                                                            , y_train_=array_train_y_ss
                                                            , X_test_=array_test_X_ss
                                                            , y_test_=array_test_y_ss
                                                            , df_w_=df_weight_both
                                                            , bio_='ppi_tf_pathway_'
                                                            , design_name_='metsig_SScaler_1_layer_'+str(unit_size)
                                                            , pathway_layer_=True
                                                            , second_layer_=False
                                                            , epochs_=epochs_default
                                                            , batch_size_=batch_size_default
                                                            , unit_size_=unit_size).build()

        # ### MinMaxScaler normalization - Pathways and PPI/TF connection - 941 nodes
        print('Metabolic and signaling with MinMaxScaler normalization - '+str(df_weight_both.shape[0])+' gene - pathways_ppi_tf'+str(df_weight_both.shape[1]))
        #     design35_B1layer_pathway_ppi_model, design35_B1layer_pathway_ppi_metric = tfm_NN.TFM_NNExperiment(X_train_=array_train_X_mms
        mms_b1_model, mms_b1_metric = tfm_NN.TFM_NNExperiment(X_train_=array_train_X_mms
                                                            , y_train_=array_train_y_mms
                                                            , X_test_=array_test_X_mms
                                                            , y_test_=array_test_y_mms
                                                            , df_w_=df_weight_both
                                                            , bio_='ppi_tf_pathway_'
                                                            , design_name_='metsig_MMScaler_1_layer_'+str(unit_size)
                                                            , pathway_layer_=True
                                                            , second_layer_=False
                                                            , epochs_=epochs_default
                                                            , batch_size_=batch_size_default
                                                            , unit_size_=unit_size).build()

        # ## with 2-LAYER

        # ### StandardScaler normalization - Pathways and PPI/TF connection + Fully connected - 941 nodes + 100 dense
        print('Metabolic and signaling with StandardScaler normalization - '+str(df_weight_both.shape[0])+' gene - pathways_ppi_tf'+str(df_weight_both.shape[1])+' + dense100')
        #     design34_B2layer_pathway_ppi_model, design34_B2layer_pathway_ppi_metric = tfm_NN.TFM_NNExperiment(X_train_=array_train_X_ss
        ss_b2_model, ss_b2_metric = tfm_NN.TFM_NNExperiment(X_train_=array_train_X_ss
                                                            , y_train_=array_train_y_ss
                                                            , X_test_=array_test_X_ss
                                                            , y_test_=array_test_y_ss
                                                            , df_w_=df_weight_both
                                                            , bio_='ppi_tf_pathway_'
                                                            , design_name_='metsig_SScaler_2_layer_'+str(unit_size)
                                                            , pathway_layer_=True
                                                            , second_layer_=True
                                                            , epochs_=epochs_default
                                                            , batch_size_=batch_size_default
                                                            , unit_size_=unit_size).build()

        # ### MinMaxScaler normalization - Pathways and PPI/TF connection + Fully connected - 941 nodes + 100 dense
        print('Metabolic and signaling with MinMaxScaler normalization - '+str(df_weight_both.shape[0])+' gene - pathways_ppi_tf'+str(df_weight_both.shape[1])+' + dense100')
        #     design35_B2layer_pathway_ppi_model, design35_B2layer_pathway_ppi_metric = tfm_NN.TFM_NNExperiment(X_train_=array_train_X_mms
        mms_b2_model, mms_b2_metric = tfm_NN.TFM_NNExperiment(X_train_=array_train_X_mms
                                                            , y_train_=array_train_y_mms
                                                            , X_test_=array_test_X_mms
                                                            , y_test_=array_test_y_mms
                                                            , df_w_=df_weight_both
                                                            , bio_='ppi_tf_pathway_'
                                                            , design_name_='metsig_MMScaler_2_layer_'+str(unit_size)
                                                            , pathway_layer_=True
                                                            , second_layer_=True
                                                            , epochs_=epochs_default
                                                            , batch_size_=batch_size_default
                                                            , unit_size_=unit_size).build()

        # METABOLIC and SIGNALING PATHWAY
        ### StandardScaler normalization - Fully connected - 100 dense
        ss_dense_metric.columns = ['Feature', 'Homogeneity','Completeness','V-measure','ARI','AMI','Fowlkes-Mallows', 'Average']
        ### MinMaxScaler normalization - Fully connected - 100 dense
        mms_dense_metric.columns = ['Feature', 'Homogeneity','Completeness','V-measure','ARI','AMI','Fowlkes-Mallows', 'Average']

        ### DESING A
        ### StandardScaler normalization - Pathways connection
        ss_a1_metric.columns = ['Feature', 'Homogeneity','Completeness','V-measure','ARI','AMI','Fowlkes-Mallows', 'Average']
        ### MinMaxScaler normalization - Pathways connection
        mms_a1_metric.columns = ['Feature', 'Homogeneity','Completeness','V-measure','ARI','AMI','Fowlkes-Mallows', 'Average']
        ### StandardScaler normalization - Pathways connection + Fully connection
        ss_a2_metric.columns = ['Feature', 'Homogeneity','Completeness','V-measure','ARI','AMI','Fowlkes-Mallows', 'Average']
        ### MinMaxScaler normalization - Pathways connection + Fully connection
        mms_a2_metric.columns = ['Feature', 'Homogeneity','Completeness','V-measure','ARI','AMI','Fowlkes-Mallows', 'Average']

        ### DESING B
        ### StandardScaler normalization - Pathways and PPI/TF connection
        ss_b1_metric.columns = ['Feature', 'Homogeneity','Completeness','V-measure','ARI','AMI','Fowlkes-Mallows', 'Average']
        ### MinMaxScaler normalization - Pathways and PPI/TF connection
        mms_b1_metric.columns = ['Feature', 'Homogeneity','Completeness','V-measure','ARI','AMI','Fowlkes-Mallows', 'Average']
        ### StandardScaler normalization - Pathways and PPI/TF connection + Fully connected
        ss_b2_metric.columns = ['Feature', 'Homogeneity','Completeness','V-measure','ARI','AMI','Fowlkes-Mallows', 'Average']
        ### MinMaxScaler normalization - Pathways and PPI/TF connection + Fully connected
        mms_b2_metric.columns = ['Feature', 'Homogeneity','Completeness','V-measure','ARI','AMI','Fowlkes-Mallows', 'Average']

        df_metrics = pd.concat([ss_dense_metric
                                , ss_a1_metric, ss_a2_metric
                                , ss_b1_metric, ss_b2_metric
                                , mms_dense_metric
                                , mms_a1_metric, mms_a2_metric
                                , mms_b1_metric, mms_b2_metric
                               ])
        df_metrics = df_metrics.set_index('Feature')
        print(df_metrics)

        # EXPORTING RESULT SCORE
        df_metrics.to_csv(os.path.join(path_output+'/design_A_B_metabolic_and_signaling_'+n_out_cell_type+'.txt'), sep=';')
        print('RESULT EXPORTED to "{}"'.format(path_output))

        # EXPORTING FINAL MODELS
        ss_dense_model.save(os.path.join(path_model_cell_out+str('/ss_dense_model_met_sig')))
        mms_dense_model.save(os.path.join(path_model_cell_out+str('/mms_dense_model_met_sig')))

        ss_a1_model.save(os.path.join(path_model_cell_out+str('/ss_a1_model_met_sig')))
        mms_a1_model.save(os.path.join(path_model_cell_out+str('/mms_a1_model_met_sig')))
        ss_a2_model.save(os.path.join(path_model_cell_out+str('/ss_a2_model_met_sig')))
        mms_a2_model.save(os.path.join(path_model_cell_out+str('/mms_a2_model_met_sig')))

        ss_b1_model.save(os.path.join(path_model_cell_out+str('/ss_b1_model_met_sig')))
        mms_b1_model.save(os.path.join(path_model_cell_out+str('/mms_b1_model_met_sig')))
        ss_b2_model.save(os.path.join(path_model_cell_out+str('/ss_b2_model_met_sig')))
        mms_b2_model.save(os.path.join(path_model_cell_out+str('/mms_b2_model_met_sig')))
        print('MODELS EXPORTED to "{}"'.format(path_model_cell_out))

    # PRINTING NETWORK INTO CONSOLE
    print("\n\nDesign P dense\n")
    print(ss_dense_model.summary())
    print(mms_dense_model.summary())
    print("\nDesign A \n")
    print(ss_a1_model.summary())
    print(mms_a1_model.summary())
    print(ss_a2_model.summary())
    print(mms_a2_model.summary())
    print("\nDesign B \n")
    print(ss_b1_model.summary())
    print(mms_b1_model.summary())
    print(ss_b2_model.summary())
    print(mms_b2_model.summary())

    # PRINTING THE DURATION
    time_end  = dt.datetime.now().time().strftime('%H:%M:%S') # = time.time()
    print('started  !!!     ', time_start)
    print('finished !!!     ', time_end)
    print('\nELAPSED TIME, ', (dt.datetime.strptime(time_end,'%H:%M:%S') - dt.datetime.strptime(time_start,'%H:%M:%S')))
    