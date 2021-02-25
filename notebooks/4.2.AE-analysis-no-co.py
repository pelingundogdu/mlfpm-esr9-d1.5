# ~2 hours
import tfm_data_operation as tfm_data
import tfm_autoencoder as tfm_ae
import tfm_keras_tuner as tfm_kt

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

from tensorflow.keras import callbacks
import tensorflow.keras.backend as K
from tensorflow import keras

# from numba import cuda
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

# # OPTIMIZATION GPU USAGE
config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

time_start = dt.datetime.now().time().strftime('%H:%M:%S')

# DEFAULT VALUES for PAPER DESIGN
epochs_default=100
batch_size_default=500

# TARGET VARIABLE NAME
target_='cell_type'
TYPE_OF_SCALING = [True, False]

# sgd_ = keras.optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True) # the parameter from paper
earlystopping_ = keras.callbacks.EarlyStopping( monitor='val_loss', min_delta=0.02, patience=5, verbose=2)


for i_scaling in TYPE_OF_SCALING:
    # THE LOCATION of THE RESULT of SCORE and MODEL
#     path_model_checkpoint = tfm_data.def_check_create_path('AE_result','models_no_co_'+str(i_scaling)+'_checkpoint')
    path_model = tfm_data.def_check_create_path('AE_result','models_no_co_'+str(i_scaling))

    # LOADING REQUIRED DATASETS
    df_weight_signaling, df_weight_metabolic_signaling = tfm_data.def_load_weight_pathways()
    df_paper, df_signaling, df_metabolic_signaling = tfm_data.def_load_dataset(['cell_type']+list(df_weight_signaling.index.values)
                                                                               , ['cell_type']+list(df_weight_metabolic_signaling.index.values)
                                                                               , row_scaling=i_scaling
                                                                               , retrieval=True)

    # DELETE UNUSED DATASET
    del(df_weight_metabolic_signaling)
    del(df_weight_signaling)

    metric=[]

    h_layer = [100, 796]
    model_vanilla = ['ae_vanilla_model_default', 'ae_vanilla_model_signaling', 'ae_vanilla_model_met_sig']
    model_denoising = ['ae_denoising_model_default', 'ae_denoising_model_signaling', 'ae_denoising_model_met_sig']
    
    for i, i_df in enumerate([df_paper, df_signaling, df_metabolic_signaling]):
        df_ss = tfm_data.def_dataframe_normalize(i_df, StandardScaler(), 'cell_type')
        print(df_ss.shape)
        del(i_df)

        # EXPERIMENT DATASETS   
        array_train_X = []
        array_train_X.append(np.array(df_ss.iloc[:, ~df_ss.columns.isin([target_])]).astype(np.float))
        trainNoise = np.random.normal(loc=0, scale=0.1, size=len(array_train_X[0][0]))
        
        # Autoencoder 
        for i_hidden_layer in h_layer:
            ae_ohl, ae_e_ohl, ae_d_ohl, layers_ohl = tfm_ae.AE_one_hidden_layer(input_dim=len(array_train_X[0][0]), hidden_layers=int(i_hidden_layer))

            # Denoising Autoencoder
            _, _, model_denoising_ohl_sgd = tfm_ae.AE_compile(train_X=(array_train_X + trainNoise)
                                                        , test_X=array_train_X
                                                        , epochs_=epochs_default
                                                        , batch_size_=batch_size_default
                                                        , hidden_layer_=layers_ohl
                                                        , autoencoder=ae_ohl
                                                        , encoder=ae_e_ohl
                                                        , decoder=ae_d_ohl
                                                        , optimizer_=keras.optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True) # the parameter from paper
                                                        , loss_=keras.losses.MeanSquaredError()
                                                        , callbacks_=[earlystopping_]
                                                        )

            print(ae_ohl.summary())
            model_denoising_ohl_sgd.save(os.path.join(path_model+str('/'+model_denoising[i]+'_ohl_sgd_'+str(i_hidden_layer))))
            print('MODELS EXPORTED!! -- ', os.path.join(path_model) ,  str(i_hidden_layer), ' - ', model_denoising[i], ' - ', str(i_hidden_layer))
                                                       

# PRINTING THE DURATION
time_end  = dt.datetime.now().time().strftime('%H:%M:%S') # = time.time()
print('started  !!!     ', time_start)
print('finished !!!     ', time_end)
print('\nELAPSED TIME, ', (dt.datetime.strptime(time_end,'%H:%M:%S') - dt.datetime.strptime(time_start,'%H:%M:%S')))
