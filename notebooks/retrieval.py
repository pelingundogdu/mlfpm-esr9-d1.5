# C. Lin, S. Jain, H. Kim, and Z. Bar-Joseph, “Using neural networks for reducing 
# the dimensions of single-cell RNA-Seq data,” Nucleic Acids Res, vol. 45, no. 17, 
# pp. e156–e156, Sep. 2017, doi: 10.1093/nar/gkx681.

# Code modified for this project.

import os
import re
import sys
import dotenv
import numpy as np
import itertools
from collections import defaultdict

from scipy.spatial import distance
from sklearn.preprocessing import MinMaxScaler

dotenv.load_dotenv()
important_folder = os.getenv("important_folder")
important_folder = os.path.abspath(important_folder)

data_file_name = "integrate_imputing_dataset_kNN10_simgene.txt"
# data_file_name = "TPM_mouse_7_8_10_PPITF_gene_9437.txt"
retrieval_topcell = 100
retrieval_map = 1

project_dir = os.path.abspath(dotenv.find_dotenv())
project_dir = os.path.dirname(project_dir)
data_dir = os.path.join(project_dir, "data")
### # data_file_path = os.path.join(data_dir, "important_data", data_file_name)

if not os.path.exists(os.path.join(project_dir, "retrieval_analysis")):
    os.mkdir(os.path.join(project_dir, "retrieval_analysis"))

output_dir = os.path.join(project_dir, "retrieval_analysis")

data_file_path = os.path.join(important_folder, data_file_name)
print('\nFILE for USING RETRIEVAL ANALYSIS IS ',data_file_path)

def AvgPrecision(ans, pred):
    # ans is a single integer denoting the class
    # pred is a vector of the retrieved items
    correct = 0
    total = 0
    ret = 0.0
    for p in pred:
        total += 1
        if p == ans:
            correct += 1
            ret += correct / float(total)
    if correct > 0:
        ret /= float(correct)
    return ret

def MeanAvgPrecision(anss, preds):
    ret = 0.0
    for ans, pred in zip(anss, preds):
        ret += AvgPrecision(ans, pred)
    ret /= float(len(anss))
    return ret


def load_integrated_data(
    filename,
    landmark=False,
    sample_normalize=True,
    gene_normalize=False,
    ref_gene_file=None,
    log_trans=False,
):
    all_data = []
    all_sample_ID = []
    labeled_sample_ID = []
    labeled_data = []
    unlabeled_data = []
    gene_names = []
    all_label = []
    labeled_label = []
    
    sample_normalize = bool(int(sample_normalize))

    if ref_gene_file != "all":
        lines = open(ref_gene_file).readlines()
        group_genes = []
        if len(lines) == 1:
            group_genes = lines[0].split("\t")
        else:
            for line in lines:
                group_genes.append(line.replace("\n", "").split("\t")[0].lower())

    lines = open(filename).readlines()
    Sample_ID = lines[0].replace("\n", "").split("\t")[1:]
    labels = lines[1].replace("\n", "").split("\t")[1:]
    all_weights = lines[2].replace("\n", "").split("\t")[1:]
    all_sample_ID = Sample_ID

    label_index = [i for i, val in enumerate(labels) if val != "None"]
    unlabeled_index = [i for i, val in enumerate(labels) if val == "None"]
    labeled_sample_ID = [
        all_sample_ID[i] for i, val in enumerate(labels) if val != "None"
    ]
    unlabeled_sample_ID = [
        all_sample_ID[i] for i, val in enumerate(labels) if val == "None"
    ]
    unlabeled_weights = [
        all_weights[i] for i, val in enumerate(labels) if val == "None"
    ]
    labeled_weights = [all_weights[i] for i, val in enumerate(labels) if val != "None"]

    label_unique_list = ["None"] + list(
        set([val for i, val in enumerate(labels) if val != "None"])
    )

    for lab in labels:
        all_label.append(label_unique_list.index(lab))
        if lab != "None":
            labeled_label.append(label_unique_list.index(lab))

    sum_all_data=[]
    for line in lines[3:]:
        splits = line.replace("\n", "").split("\t")
        gene = splits[0]
        # print gene
        #   if landmark and gene not in landmark_genes.keys():
        #       continue
        
        sum_all_data.append(list(itertools.chain(np.array(splits[1:], dtype="float") ) ))
        if ref_gene_file != "all" and gene not in group_genes:
            continue
        gene_names.append(gene)
        # print splits[1:]
        all_data.append(splits[1:])
        # print len(splits)
    all_data = np.array(all_data, dtype="float32")
    # print all_data.shape
    
    if sample_normalize:
        s = np.array(sum_all_data)[:, :].astype('float').sum(axis=0)
        all_data = all_data / s * 1000000
        print('SAMPLE WISE NORMALIZATION APPLIED!!')
        
    ####################################################################################################
    print('    Lenght of gene list, ', len(all_data))
    print('    all_data',type(all_data))
    print('    sum_all_data',type(sum_all_data))
    print('    all_data shape, ', all_data.shape)
    print('    sum_all_data shape, ', np.asarray(sum_all_data).shape)
    if sample_normalize:
        print('    sum shape, ',s.shape)
#     print('    new_all_data shape, ', new_all_data.shape)
    ####################################################################################################
#  ORIGINAL LOCATION of sample_normalization opertation    
#     if sample_normalize:
#         for j in range(all_data.shape[1]):
#             s = np.sum(all_data[:, j])
#             if s == 0:
#                 s = 0
#                 # print 'normalize sum==0: sample',j
#             else:
#                 all_data[:, j] = all_data[:, j] / s * 1000000
                
    if log_trans:
        all_data = np.log(all_data + 1)
    if gene_normalize:
        for j in range(all_data.shape[0]):
            mean = np.mean(all_data[j, :])
            std = np.std(all_data[j, :])
            if std == 0:
                std = 0
                # print 'gene_normalize: std==0 data: ',j,mean,std
            else:
                all_data[j, :] = (all_data[j, :] - mean) / std
            # print all_data[j,:]
    labeled_data = np.zeros((all_data.shape[0], len(label_index)), dtype="float32")
    unlabeled_data = np.zeros(
        (all_data.shape[0], len(unlabeled_index)), dtype="float32"
    )
    count = 0
    print(len(label_index))
    print(all_data.shape)
    for i in label_index:
        labeled_data[:, count] = all_data[:, i]
        count += 1
    # print count
    count = 0
    for i in unlabeled_index:
        unlabeled_data[:, count] = all_data[:, i]
        count += 1
    all_label = np.array(all_label)
    labeled_label = np.array(labeled_label)

    all_data = np.transpose(all_data)
    labeled_data = np.transpose(labeled_data)
    unlabeled_data = np.transpose(unlabeled_data)

    all_weights = np.transpose(all_weights)
    labeled_weights = np.transpose(labeled_weights)
    unlabeled_weights = np.transpose(unlabeled_weights)

    return (
        all_data,
        labeled_data,
        unlabeled_data,
        label_unique_list,
        all_label,
        labeled_label,
        all_weights,
        labeled_weights,
        unlabeled_weights,
        all_sample_ID,
        labeled_sample_ID,
        unlabeled_sample_ID,
        gene_names,
    )


def compute_retrieval_scores(model_path, n_epochs, modality, snorm, ref_gene_file, sub_output_dir, out_file):
    # nearest neighbor retrieval things
    # data_file_name='hannah_mouse_data/TPM_6_8_9_15_25_41_44_45_46_.txt'
    # model_name='3layer_SN1_GN1_BS32_hls100_mls696_seed0_classifier_merge0_tanh'
    # nn_iteration=100
    (
        all_data,
        labeled_data,
        unlabeled_data,
        label_unique_list,
        all_label,
        labeled_label,
        all_weights,
        labeled_weights,
        unlabeled_weights,
        all_sample_ID,
        labeled_sample_ID,
        unlabeled_sample_ID,
        gene_names,
    ) = load_integrated_data(
        data_file_path,
        sample_normalize=snorm,
        gene_normalize=0,
        log_trans=0,
        ref_gene_file=ref_gene_file,
    )

    code = encode_data(model_path, n_epochs, modality, all_data)

    print("all_data.shape: ", all_data.shape)
    # code = get_nn_code(model_name,nn_iteration,all_data)
    # code=transform_data
    print("code.shape: ", code.shape)
    print("all_weights: ", all_weights)
#     print("\n".join(label_unique_list))
    verify_lab = [
        "2cell",
        "4cell",
        "ICM",
        "zygote",
        "8cell",
        "ESC",
        "lung",
        "TE",
        "thymus",
        "spleen",
        "HSC",
        "neuron",
    ]

    for index, lab in enumerate(label_unique_list):
        if "cortex" in lab:
            label_unique_list[index] = "neuron"
        if "CNS" in lab:
            label_unique_list[index] = "neuron"
        if "brain" in lab:
            label_unique_list[index] = "neuron"
        for vl in verify_lab:
            if vl in lab:
                label_unique_list[index] = vl
    dataset_sets = map(str, sorted(map(int, list(set(all_weights)))))
    ntop = 10
    # vtop=100

#     out_file = os.path.basename(model_path)
#     out_file = os.path.splitext(out_file)[0]
    out_file_1 = os.path.join(sub_output_dir, out_file + "_retrieval.csv")
    out_file_1 = os.path.abspath(out_file_1)
    out_ret = open(out_file_1, "w")
    out_ret.write(
        "dataset,sample,celltype,retrieval #cell in top100, total #cell,retrieval_ratio\n"
    )
    out_file_2 = os.path.join(sub_output_dir, out_file + "_retrieval_summary.csv")
    out_file_2 = os.path.abspath(out_file_2)
    out_ret2 = open(out_file_2, "w")
    out_ret2.write("dataset,celltype,#cell,mean retrieval ratio\n")
    for ds in dataset_sets:
        meindex = np.where(all_weights == ds)
        nmeindex = np.where(all_weights != ds)
        # isme= all_data[meindex]
        # isnme= all_data[nmeindex]
        isme = code[meindex]
        ####################################################################################################
#         print("isme")
#         print(isme)
#         print("isme.shape")
#         print(isme.shape)
        isnme = code[nmeindex]
        dismat = distance.cdist(isme, isnme, "euclidean")
#         print(dismat.shape)
        ####################################################################################################
        cell_dict = defaultdict(lambda: [])
        for index, row in enumerate(dismat):
            # print index, row
            now_label = label_unique_list[all_label[meindex[0][index]]]
            if now_label not in verify_lab:
                continue
            now_sample = all_sample_ID[meindex[0][index]]
#             aprint("now dataset: ", ds)
#             aprint("now_sample: ", now_sample)
#             aprint("now_label: ", now_label)
            sort_index = np.argsort(row)
            temp_lab = []
            temp_dist = []
            temp_set = []
            total_vl = len(
                [
                    label_unique_list[all_label[nmeindex[0][x]]]
                    for x in range(len(row))
                    if label_unique_list[all_label[nmeindex[0][x]]] == now_label
                ]
            )
            if retrieval_topcell > 0:
                total_vl = retrieval_topcell
            for si in sort_index[:total_vl]:
                temp_set.append(all_weights[nmeindex[0][si]])
                temp_dist.append(row[si])
                temp_lab.append(label_unique_list[all_label[nmeindex[0][si]]])
            vtop_vl = len(
                [temp_lab[x] for x in range(len(temp_lab)) if temp_lab[x] == now_label]
            )
#             print("# of hit in top 100 cells: ", vtop_vl)
#             print("total # of cell in reference cells:", total_vl)
            ratio = vtop_vl / float(total_vl)
#             print("ratio: ", ratio)
            if retrieval_map:
                AP = AvgPrecision(now_label, temp_lab)
#                 aprint("AP=", AP)
                ratio = AP
            cell_dict[now_label].append(ratio)
            out_ret.write(
                str(ds)
                + ","
                + now_sample
                + ","
                + now_label
                + ","
                + str(vtop_vl)
                + ","
                + str(total_vl)
                + ","
                + str(ratio)
                + "\n"
            )
#             print("top ", ntop, " neighbor distance, label, and dataset: ")
#             print(temp_dist[:ntop])
#             print(temp_lab[:ntop])
#             print(temp_set[:ntop])
        for key, val in cell_dict.items():
#             print(key, np.mean(val))
            out_ret2.write(
                str(ds)
                + ","
                + key
                + ","
                + str(len(val))
                + ","
                + str(np.mean(val))
                + "\n"
            )

def encode_from_saved_model(model_path, data):
    """Load model from `model_path` and extract
    the n-1 hidden layer, then encode `data`

    Parameters
    ----------
    model_path : [type]
        [description]
    data : [type]
        [description]
    target : [type]
        [description]

    Returns
    -------
    Numpy array
        Encoded data.
    """

    from sklearn.preprocessing import StandardScaler
    from tensorflow import keras
    import tensorflow.keras.backend as K

    data_scl = StandardScaler().fit_transform(data)

    K.clear_session()

    model_load=keras.models.load_model(model_path)
    model= keras.models.Model(
        inputs=model_load.layers[0].input,      # first layer
        outputs=model_load.layers[-2].output )  #  the last layer before output

    code=model.predict(data_scl)

    K.clear_session()

    return code

def encode_pca(data, n_epochs, modality):
    """Encode using l1-driven PCA.

    Parameters
    ----------
    data : np.array
        Samples per genes dataframe

    Returns
    -------
    np.array
        Encoded dataframe
    """

    from sklearn.decomposition import SparsePCA, PCA, FastICA
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    
    if n_epochs != "mle":
        n_epochs = int(n_epochs)

    data_trans = StandardScaler().fit_transform(data)

    code = PCA(n_components=n_epochs).fit_transform(data_trans)
    # code = SparsePCA(method="cd", n_jobs=-1).fit_transform(data)
    if modality=='pca_kmeans':
        print('pca_kmeans')
        code = KMeans(n_clusters=cluster_).fit(data_trans)
#         code = kmeans.predict(nn_last_layer_testing)

    return code

def encode_data(model_path, n_epochs, modality, data):
    if modality == "saved_model":
#         print('\nRETRIEVAL ANALYSIS IS PERFORMING for SAVED MODELS in ',model_path)
#         code = encode_from_saved_model(model_path, n_epochs, data)
        code = encode_from_saved_model(model_path, data)
#     elif ((modality == "pca") or (modality=="pca_kmeans")):
    elif re.search("pca", modality):
#     elif model_path=='pca':
#         print('\nRETRIEVAL ANALYSIS IS PERFORMING for PCA')
        code = encode_pca(data, n_epochs, modality)
    else:
        raise NotImplementedError("Unknown learning modality.")

    return code


def main(model_path, n_epochs, modality, snorm, ref_gene_file, experiment_gene):
    
    print('OUTPUT MAIN FOLDER,', output_dir)
    
    list_experiments=[]
    name_folder = 'unknown_model'
    if re.search('\w+/AE_result/\w+', model_path):
        name_folder = 'AE_retrieval_'+str(bool(int(snorm)))
        list_experiments = [string for string in os.listdir(model_path) if re.search('ae_\w+', string)]
#         print(list_experiments)
        out_file = ''
    elif re.search('\w+/kt_result/\w+', model_path):
        name_folder = 'kt_retrieval_'+str(bool(int(snorm)))
        list_experiments = [string for string in os.listdir(model_path) if re.search('ss_\w+', string)]
#         print(list_experiments)
        out_file = 'kt_'
    elif re.search('\w+/NN_result/\w+', model_path):
        name_folder = 'NN_retrieval_'+str(bool(int(snorm)))
        list_experiments = [string for string in os.listdir(model_path) if re.search('ss_\w+', string)]
        out_file = 'nn_'
#         print(list_experiments)
    elif re.search('pca', model_path):
        name_folder = 'pca_retrieval_'+str(bool(int(snorm)))
        model_path = ''
        out_file = modality+'_'+str(n_epochs)
        
#     print('****************out_file, ',out_file)
    if not os.path.exists(os.path.join(output_dir, name_folder)):
        os.mkdir(os.path.join(output_dir, name_folder))

    sub_output_dir = os.path.join(output_dir, name_folder)
    
    if experiment_gene == 'default' :#and len(list_experiments)>0 :
        list_experiments = [string for string in list_experiments if (re.search('\w+_default_\w+', string) or re.search('\w+_p1_\w+', string) or re.search('\w+_p2_\w+', string) )]
    elif experiment_gene =='signaling':
        list_experiments = [string for string in list_experiments if (re.search('\w+_model_signaling', string) and not (re.search('\w+_default_\w+', string) or re.search('\w+_p1_\w+', string) or re.search('\w+_p2_\w+', string) ) ) ]
    elif experiment_gene =='metabolic_signaling':
        list_experiments = [string for string in list_experiments if (re.search('\w+_model_met\w+', string) and not ( (re.search('\w+_default_\w+', string) or re.search('\w+_p1_\w+', string) or re.search('\w+_p2_\w+', string) ) )) ]
    
    print('RETRIEVAL ANALYSIS WILL PERFORM for ', list_experiments, '\n')        
    
    if len(list_experiments)==0:
        compute_retrieval_scores(model_path, n_epochs, modality, snorm, ref_gene_file, sub_output_dir, out_file)
        pass
    
    else:
        for i_model in list_experiments:
            print('  PERFORMING... MODEL IS USING FROM ',os.path.join(model_path, i_model))
            save_out_file = out_file + i_model
            compute_retrieval_scores(os.path.join(model_path, i_model), n_epochs, modality, snorm, ref_gene_file, sub_output_dir, save_out_file)
            print('    RESULTS EXPOERTED! -- ', save_out_file)
        
    print('\nRETRIEVAL ANALYSIS FINISHED!! - {}'.format(sub_output_dir))
    
# def main(model_path, n_epochs, modality, snorm, ref_gene_file, res_path):
#     compute_retrieval_scores(model_path, n_epochs, modality, snorm, ref_gene_file, res_path)


if __name__ == "__main__":
#     _, model_path, n_epochs, modality, snorm, ref_gene_file, res_path = sys.argv
    _, model_path, n_epochs, modality, snorm, ref_gene_file, experiment_gene = sys.argv
    

#     main(model_path, n_epochs, modality, snorm, ref_gene_file, res_path)
    main(model_path, n_epochs, modality, snorm, ref_gene_file, experiment_gene)
    

