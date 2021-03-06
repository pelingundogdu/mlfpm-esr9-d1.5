# Created python codes
import lib_neural_network as tfm_NN

# Required libraries
import os
import numpy as np
import pandas as pd
import datetime as dt

from sklearn.model_selection import train_test_split, KFold

from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential, Model
from sklearn.cluster import KMeans

########################################################################################################################################################################
# Creating experiment datasets
# These datasets will be using in training and testing process in NN and KerasTuner
def def_create_experiment(df_, n_splits_, experiment_, feature_exclude_, target_ ):
    
    kf = KFold(n_splits=n_splits_, shuffle=True)
    
    df_X = df_.iloc[:, ~df_.columns.isin(feature_exclude_)]
    df_y = df_.iloc[:, df_.columns.isin(target_)]

    array_train_X, array_train_y = [], []
    array_test_X, array_test_y = [], []

    for e_ in range(0,experiment_):
        kf_count = 0
        for train_index, test_index in kf.split(df_):
            kf_count = kf_count+1
            array_train_X.append(np.asarray(df_X.iloc[train_index]))
            array_test_X.append(np.asarray(df_X.iloc[test_index]))
            array_train_y.append([item for sublist in np.array(df_y.iloc[train_index]) for item in sublist])
            array_test_y.append([item for sublist in np.array(df_y.iloc[test_index]) for item in sublist])
            
    return array_train_X, array_train_y, array_test_X, array_test_y

########################################################################################################################################################################
# Clustering experiment,
def def_clustering_experiment(df_, experiment_index_, feature_exclude_, target_value, model_list_, model_path_, cluster_, bio_ ):
    df_X = df_.iloc[:, ~df_.columns.isin(feature_exclude_)]
    df_y = df_.iloc[:, df_.columns.isin(target_value)]
    
    metric_all = []
    for i_, m_ in enumerate(model_list_):
        print(m_)
        K.clear_session()
        model_loc = os.path.join(model_path_+'/'+m_)
        model_load = keras.models.load_model(model_loc)
        model_= Model(inputs=model_load.layers[0].input # first layer
                      , outputs=model_load.layers[-2].output) # the last layer before output
        y_pred_all, array_y_test = [], []

        for experiment_ in experiment_index_:
            train_index = experiment_[0]
            test_index = experiment_[1]

            X_train = df_X.iloc[train_index]
            X_test = df_X.iloc[test_index]
            array_y_test.append([item for sublist in np.array(df_y.iloc[test_index]) for item in sublist])
            
            nn_last_layer_training = model_.predict(X_train)
            nn_last_layer_testing = model_.predict(X_test)
            kmeans = KMeans(n_clusters=cluster_).fit(nn_last_layer_training)
            kmeans_predict = kmeans.predict(nn_last_layer_testing)

            K.clear_session()
            y_pred_all.append(kmeans_predict)
#         print(len(y_pred_all))
            
        metric = tfm_NN.def_network_run_get_metrics(y_pred_=y_pred_all
                                                    , y_test_=array_y_test
                                                    , bio_info_=bio_
                                                    , name_of_design_=m_)
        metric_all.append(metric.values[0])

    return(pd.DataFrame(metric_all))

########################################################################################################################################################################
