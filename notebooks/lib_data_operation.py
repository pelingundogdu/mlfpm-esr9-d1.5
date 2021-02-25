#!/usr/bin/env python
# coding: utf-8

# Required libraries
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder

########################################################################################################################################################################
# Checking location for specified folder
def def_check_create_path(main_folder, sub_folder):
    if (os.path.exists(os.path.join('../data/'+main_folder))==False):
        os.mkdir(os.path.join('../data/'+main_folder))

    if (os.path.exists(os.path.join('../data/'+main_folder+'/'+sub_folder))==False):
        os.mkdir(os.path.join('../data/'+main_folder+'/'+sub_folder))
    
    print('../data/'+main_folder+'/'+sub_folder)
    return ('../data/'+main_folder+'/'+sub_folder)

########################################################################################################################################################################
# Dropping empty nodes
def def_remove_empty_nodes(dataframe_):   
#    list_empty_nodes = dataframe_.sum()[dataframe_.sum() == 0].index
#    print('Deleted nodes, ', dataframe_.sum()[dataframe_.sum() == 0].index)
    dataframe_.drop(columns=(dataframe_.sum()[dataframe_.sum() == 0].index), inplace=True)
    return dataframe_

########################################################################################################################################################################
# Importing pathways weight table
# def def_load_required_weight_pathways():
def def_load_weight_pathways():
    print('IMPORTING WEIGHT(PATHWAYS) TABLE')
    
    # Importing weight table (signaling pathway)
    df_weight_sig = pd.read_csv('../data/weights/pathway_signaling_weight.txt' ).set_index('symbol')
    print('df_weight_signaling shape           , ', df_weight_sig.shape)
    
    # Importing weight table (metabolic pathway)
    df_weight_metsig = pd.read_csv('../data/weights/pathway_metabolic_weight.txt').set_index('symbol')
    print('df_weight_metabolic_signaling shape , ', df_weight_metsig.shape)
    
    return ( df_weight_sig , df_weight_metsig )

########################################################################################################################################################################
# Importing PPI and TF weight table
# def def_load_required_weight_ppi_tf(list_signaling, list_metabolic, list_common):
# def def_load_required_weight_ppi_tf(list_signaling, list_metabolic_signaling):
def def_load_weight_ppi_tf(list_signaling, list_metabolic_signaling):    
    print('IMPORTING WEIGHT(PPI and TF) TABLE')
    
    # Importing weight table (ppi node)
    df_weight_ppi = pd.read_csv('../data/weights/ppi_weight.txt' )
    #print('df_weight_ppi shape              , ', df_weight_ppi.shape)
    
    # Importing weight table (tf node)
    df_weight_tf = pd.read_csv('../data/weights/tf_weight.txt')
    #print('df_weight_tf shape               , ', df_weight_tf.shape)
    
    # Combining weight tables (ppi_tf genes)
    df_weight_ppi_tf = pd.merge(left=df_weight_ppi, right=df_weight_tf, on='symbol', how='outer').set_index('symbol')
    df_weight_ppi_tf = df_weight_ppi_tf.fillna(0)
    #print('df_weight_ppi_tf shape           , ', df_weight_ppi_tf.shape)
    
    # Weight tables (ppi_tf genes) with signaling pathway genes
    df_weight_ppi_tf_sig = def_remove_empty_nodes(df_weight_ppi_tf.loc[df_weight_ppi_tf.index.isin(list_signaling)])
    print('df_weight_ppi_tf_signaling shape           , ', df_weight_ppi_tf_sig.shape)

    # Weight tables (ppi_tf genes) with metabolic pathway genes
    df_weight_ppi_tf_metsig = def_remove_empty_nodes(df_weight_ppi_tf.loc[df_weight_ppi_tf.index.isin(list_metabolic_signaling)])
    print('df_weight_ppi_tf_metabolic_signaling shape , ', df_weight_ppi_tf_metsig.shape)

    return( df_weight_ppi_tf_sig , df_weight_ppi_tf_metsig )

########################################################################################################################################################################
# Importing dataset
# original dataset                         --> df_paper_9437
# signaling pathways dataset               --> df_paper_signaling
# metabolic and signaling pathways dataset --> df_paper_metabolic_signaling
def def_load_dataset(list_signaling, list_metabolic_signaling, row_scaling=False, retrieval=False):
    print('IMPORTING DATASETS')
        
    if retrieval == True:
        
        df_ret = pd.read_csv('../third_party/PMC5737331/NN_code_release/important_file/integrate_imputing_dataset_kNN10_simgene.txt', sep='\t', header=None)
        array_ret = np.delete(np.array(df_ret), 2, axis=0).T
        df_paper = pd.DataFrame(data=array_ret[1:,1:]        # values
#                                 , index=array_ret[1:,0]    # 1st column as index
                                , columns=array_ret[0,1:])   # 1st row as the column names


#         df_ret = pd.read_csv('../third_party/PMC5737331/NN_code_release/important_file/integrate_imputing_dataset_kNN10_simgene.txt', sep='\t', index_col=0)
#         df_paper = pd.DataFrame(np.array(df_ret).T).drop(columns='Dataset')
        df_paper = df_paper.rename(columns={'Label':'cell_type'})
    else:
        df_paper = pd.read_csv('../third_party/PMC5737331/NN_code_release/important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt', sep='\t', index_col=0).T.drop(columns='Weight')
        df_paper = df_paper.rename(columns={'Label':'cell_type'})

    if row_scaling == True:
        # Original(author) dataset
        df_paper.iloc[:, 1:] = df_paper.iloc[:, 1:].astype(np.float32)

        row_scaling = df_paper.values[:, 1:] / df_paper.values[:, 1:].sum(axis=1).reshape(-1,1) * 1_000_000
#         row_scaling = df_paper.values[:, 1:] = df_paper.values[:, 1:] / df_paper.values[:, 1:].sum(axis=1).reshape(-1,1) * 1_000_000
        
        array_concat = np.concatenate([df_paper.values[:,0].reshape(-1,1), row_scaling], axis=1)
        df_paper = pd.DataFrame(array_concat, columns=df_paper.columns)
        print('row scaler implemented!')
    
    print('df_paper shape               , ', df_paper.shape)

    # Original(author) dataset with signaling pathway genes
    df_paper_signaling = df_paper.iloc[:, df_paper.columns.isin(list_signaling)]
    print('df_signaling shape           , ', df_paper_signaling.shape)
    
    # Original(author) dataset with metabolic pathway genes
    df_paper_metabolic_signaling = df_paper.iloc[:, df_paper.columns.isin(list_metabolic_signaling)]
    print('df_metabolic_signaling shape , ', df_paper_metabolic_signaling.shape)
        
    return ( df_paper, df_paper_signaling, df_paper_metabolic_signaling )

########################################################################################################################################################################
# Normalizing dataset with specified scaler_
def def_dataframe_normalize(dataframe, scaler_, target_feature_):
    scaler = scaler_
    df_scaler = pd.DataFrame()
    df_scaler['cell_type'] = dataframe[target_feature_].reset_index(drop=True)
    # df_scaler = df_scaler.reset_index(drop=True)
    df_scaler=pd.concat([df_scaler, pd.DataFrame(scaler.fit_transform(dataframe.iloc[: , 1:]), columns=dataframe.columns[1:]).reset_index(drop=True)], axis=1)
#     print(df_scaler.shape)
    return(df_scaler)

########################################################################################################################################################################
# Splitting dataset via index
def def_split_train_test_by_index(dataframe_, train_index_, test_index_, target_feature_):
    ohe = OneHotEncoder()
    df_X = dataframe_.loc[:, ~dataframe_.columns.isin([target_feature_])]
    df_y = ohe.fit_transform(pd.DataFrame(dataframe_[target_feature_].values)).toarray()

    df_X_train = df_X.iloc[train_index_]
    df_X_test = df_X.iloc[test_index_]
    df_y_train = df_y[[train_index_]]
    df_y_test = df_y[[test_index_]]
    
    return ( df_X_train, df_X_test, df_y_train, df_y_test )

