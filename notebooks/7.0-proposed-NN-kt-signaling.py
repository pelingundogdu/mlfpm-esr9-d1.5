import lib_data_operation as tfm_data
import lib_keras_tuner as tfm_kt

# Required libraries
import shutil
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
TYPE_OF_SCALING = False

# THE LOCATION of THE RESULT of SCORE and MODEL
path_hyperband = tfm_data.def_check_create_path('kt_result', 'DELETE_hyperband')
path_output_result = tfm_data.def_check_create_path('kt_result', 'design_'+str(TYPE_OF_SCALING))
path_model = tfm_data.def_check_create_path('kt_result', 'models_'+str(TYPE_OF_SCALING))

# IMPORT EXPERIMENT INDEX
path_experiments = os.path.join(os.path.dirname(os.getcwd())+'/data/EXPERIMENTS/')
list_all_model = sorted(os.listdir(path_experiments))
list_experiments = [string for string in list_all_model if re.match(re.compile('signaling_cell_out_'), string)]

# LOADING REQUIRED DATASETS
df_weight_signaling, df_weight_metabolic_signaling = tfm_data.def_load_weight_pathways()
df_paper_9437, df_signaling, df_metabolic_signaling = tfm_data.def_load_dataset(['cell_type']+list(df_weight_signaling.index.values)
                                                                                , ['cell_type']+list(df_weight_metabolic_signaling.index.values)
                                                                                , row_scaling = TYPE_OF_SCALING)
df_weight_ppi_tf_signaling, df_weight_ppi_tf_metabolic_signaling = tfm_data.def_load_weight_ppi_tf(list(df_weight_signaling.index.values)
                                                                                                   , list(df_weight_metabolic_signaling.index.values))
df_weight_both = pd.concat([df_weight_ppi_tf_signaling, df_weight_signaling], axis=1)
print('df_weight_both shape , ', df_weight_both.shape)

print('Normalization signaling data')
df_ss = tfm_data.def_dataframe_normalize(df_signaling, StandardScaler(), 'cell_type')
# df_mms = tfm_data.def_dataframe_normalize(df_signaling, MinMaxScaler(), 'cell_type')

# DELETE UNUSED DATASET
del(df_paper_9437)
del(df_metabolic_signaling)
del(df_weight_ppi_tf_metabolic_signaling)
del(df_weight_metabolic_signaling)

del(df_signaling)
del(df_weight_ppi_tf_signaling)

# SIGNALING PATHWAY

# EXPERIMENT DATASETS
print('the index of experiment dataset, ' ,list_experiments)
for i_experiment in list_experiments:

    array_train_X_ss, array_train_y_ss = [], []
    array_test_X_ss, array_test_y_ss = [], []

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

    n_out_cell_type = 'cell_out_'+(re.search(r'(\d+)',i_experiment).group(1))
    path_hyperband_ = tfm_data.def_check_create_path(path_hyperband, n_out_cell_type)
#     path_hyperband_ = ''
    path_model_cell_out = tfm_data.def_check_create_path(path_model, n_out_cell_type)
    print(n_out_cell_type)
    
    # DESIGN P with dense

    # ### StandarScaler normalization - Fully connected - 100 dense
    print('Signaling with StandardScaler normalization - '+str(df_weight_both.shape[0])+' gene - dense100')
    ss_dense_model, ss_dense_metric, ss_dense_hp = tfm_kt.TFM_KerasTunerExperiment(X_train_=array_train_X_ss
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
    ss_a1_model, ss_a1_metric, ss_a1_hp = tfm_kt.TFM_KerasTunerExperiment(X_train_=array_train_X_ss
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
    ss_a2_model, ss_a2_metric, ss_a2_hp = tfm_kt.TFM_KerasTunerExperiment(X_train_=array_train_X_ss
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
    ss_b1_model, ss_b1_metric, ss_b1_hp = tfm_kt.TFM_KerasTunerExperiment(X_train_=array_train_X_ss
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
    ss_b2_model, ss_b2_metric, ss_b2_hp = tfm_kt.TFM_KerasTunerExperiment(X_train_=array_train_X_ss
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

    # SIGNALING PATHWAY 
    ### StandardScaler normalization - Fully connected - 100 dense
    ss_dense_metric.columns = ['Feature', 'Homogeneity','Completeness','V-measure','ARI','AMI','Fowlkes-Mallows', 'Average']

    ### DESING A
    ### StandardScaler normalization - Pathways connection
    ss_a1_metric.columns = ['Feature', 'Homogeneity','Completeness','V-measure','ARI','AMI','Fowlkes-Mallows', 'Average']
    ### StandardScaler normalization - Pathways connection + Fully connection
    ss_a2_metric.columns = ['Feature', 'Homogeneity','Completeness','V-measure','ARI','AMI','Fowlkes-Mallows', 'Average']

    ### DESING B
    ### StandardScaler normalization - Pathways and PPI/TF connection
    ss_b1_metric.columns = ['Feature', 'Homogeneity','Completeness','V-measure','ARI','AMI','Fowlkes-Mallows', 'Average']
    ### StandardScaler normalization - Pathways and PPI/TF connection + Fully connected
    ss_b2_metric.columns = ['Feature', 'Homogeneity','Completeness','V-measure','ARI','AMI','Fowlkes-Mallows', 'Average']

    df_metrics = pd.concat([ss_dense_metric
                            , ss_a1_metric, ss_a2_metric
                            , ss_b1_metric, ss_b2_metric
                           ])

    df_metrics = df_metrics.set_index('Feature')
    print(df_metrics)

    # EXPORTING RESULT SCORE
    df_metrics.to_csv(os.path.join(path_output_result+'/kt_score_signaling_'+n_out_cell_type+'.txt'), sep=';')
    print('RESULT EXPORTED to "{}"'.format(path_output_result))

    ss_dense_model.save(os.path.join(path_model_cell_out+str('/ss_dense_model_signaling')))

    ss_a1_model.save(os.path.join(path_model_cell_out+str('/ss_a1_model_signaling')))
    ss_a2_model.save(os.path.join(path_model_cell_out+str('/ss_a2_model_signaling')))

    ss_b1_model.save(os.path.join(path_model_cell_out+str('/ss_b1_model_signaling')))
    ss_b2_model.save(os.path.join(path_model_cell_out+str('/ss_b2_model_signaling')))
    print('MODELS EXPORTED to "{}"'.format(path_model_cell_out))

    ss_dense_hp = tfm_kt.def_hp(ss_dense_hp, 'ss_dense_hp')

    ss_a1_hp = tfm_kt.def_hp(ss_a1_hp, 'ss_a1_hp')
    ss_a2_hp = tfm_kt.def_hp(ss_a2_hp, 'ss_a2_hp')

    ss_b1_hp = tfm_kt.def_hp(ss_b1_hp, 'ss_b1_hp')
    ss_b2_hp = tfm_kt.def_hp(ss_b2_hp, 'ss_b2_hp')

    df_hp = pd.concat([ss_dense_hp
                       , ss_a1_hp, ss_a2_hp
                       , ss_b1_hp, ss_b2_hp
                           ])

    df_hp = df_hp.set_index('hp')
    df_hp.to_csv(os.path.join(path_output_result+'/kt_hyperparameters_signaling_'+n_out_cell_type+'.txt'), sep=';')
    print('RESULT EXPORTED to "{}"'.format(path_output_result))

# shutil.rmtree(path_hyperband)
    
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
