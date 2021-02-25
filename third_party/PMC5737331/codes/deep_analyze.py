from collections import defaultdict

import numpy as np

from deep_net import *
from keras.models import Sequential,model_from_json
import argparse
import parse_data 
from myKerasLayer_new import MyLayer
#import sys
if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-p',"--PPI_TF_cluster_file", help="specify the cluster file for layer connection, if not specified then Dense layer will be used")
    parser.add_argument('-r',"--reference_gene_file", help="specify the file that contatins the genes to be kept, if not specified then all the genes in training data will be used")
    parser.add_argument('-d',"--data_file", help="specify the data file, if not specified then a default training data file will be used")
    parser.add_argument('-mn',"--model_name", help="specify the model_name, if not specified then a default model_name will be used") 
    parser.add_argument('-merge',"--merge_size", help="specify the merge size, default is 0", type=int,default=0)
    parser.add_argument('-sn',"--sample_normalize", help="specify whether to normalize each sample (divide by total reads), default is 1 (means True)",choices=[0,1], type=int,default=1)
    parser.add_argument('-gn',"--gene_normalize", help="specify whether to normalize each gene (mean 0 std 1), default is 1 (means True)",choices=[0,1], type=int,default=1)
    parser.add_argument('-si',"--starting_iteration", help="specify the starting iteration (resume from the last training, -sm is need to store model), default is 0", type=int,default=100)
    parser.add_argument('-c',"--cutoff", help="specify the cutoff for deciding important groups, default is 0.03", type=float,default=0.03)
    
    args=parser.parse_args()
    data_file_name='important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt'
    #data_file_name='../data/TPM_mouse_1_4_6_7_8_10_16.txt'
    if args.data_file is not None:
        data_file_name=args.data_file
    
    all_data, labeled_data,unlabeled_data,label_unique_list,all_label, labeled_label, all_weights, labeled_weights, unlabeled_weights,all_sample_ID,labeled_sample_ID,unlabeled_sample_ID,gene_names=parse_data.load_integrated_data(data_file_name,sample_normalize=args.sample_normalize,gene_normalize=args.gene_normalize,ref_gene_file=args.reference_gene_file)#'cluster_genes.txt'   
    PPI_TF_cluster_file='ppi_tf_merge_cluster.txt'
    if args.PPI_TF_cluster_file is not None:
        PPI_TF_cluster_file=args.PPI_TF_cluster_file
    group_gene_index_dict, sorted_group_names, group_gene_mat= parse_data.load_group_gene_index_dict(gene_names,PPI_TF_cluster_file)#"PPI_TF_merge_cluster.txt"
    #model_name='1layer_SN1_GN1_BS32_hls696_mls696_seed0_PPITF_classifier_merge100_tanh_100.pickle'
    #model_name='1layer_SN1_GN1_BS32_hls696_mls696_seed0_PPITF_classifier_merge100_tanh'
    model_name='1layer_SN1_GN1_LOG0_BS10_hls100_mls696_seed0_PPITF_classifier_merge100_tanh'
    if args.model_name is not None:
        model_name=args.model_name
    custom_obj={}
    custom_obj['MyLayer']=MyLayer
    model = model_from_json(open('model/'+model_name+'.json').read(),custom_objects=custom_obj)
    weight_list=load_model_weight_from_pickle(model,'model/'+model_name+'_'+str(args.starting_iteration)+'.pickle')
    #print weight_list
    #for w in weight_list:
    #    print w[0]
    #    print w[1][0]
    ws=weight_list[1][1][0]
    ws=np.transpose(ws)
    ws=np.absolute(ws)
    #print weight_list[0][1][0][0].shape
    #print weight_list[0][1][1][0].shape
    ws_d=weight_list[0][1][1][0]
    ws_d=np.transpose(ws_d)
    ws_d=np.absolute(ws_d)
    print ws_d.shape
    #print weight_list[0][2][0]
    #ws_d=weight_list[0][1]
    #ws_0=np.transpose(ws_0)
    #ws_0=np.absolute(ws_0)
    #print ws_0.shape
    #print ws.shape
    group_type=defaultdict(lambda:[])
    topXGroup=10
    head=['Cell Type','NO.']
    output_lines=[]
    group_gene=defaultdict(lambda:[])
    #cutoff=0.03
    #cutoff=args.cutoff
    cutoff = 0.08
    for cell_type,w in enumerate(ws):
        if cell_type==0:
            continue
        print'\ncell_type: ', cell_type, label_unique_list[cell_type]
        sort_index = np.argsort(-1*w)
        #print sort_index
        print 'top '+str(topXGroup)+' weighted groups: '
        output_line=[label_unique_list[cell_type],str(cell_type)]
        for index,s in enumerate(sort_index[:topXGroup]):
            if cell_type==1:
                head.append('Group '+str(index))
            if s <696:
                gn= sorted_group_names[s]
                ggis= group_gene_index_dict[gn]
                ggns=[]
                for gi in ggis:
                    ggns.append(gene_names[gi])
                #print gn 
                #print '\t'.join(ggns)
                group_gene[gn]=ggns
            else:
                s-=696
                sort_index_d = np.argsort(-1*ws_d[s])
                gs=[]
                for sd in sort_index_d:
                    if ws_d[s,sd]>cutoff:
                        gs.append(gene_names[sd])
                print 'dense: ',s
                print len(gs)
                #print ws_d[s,sort_index_d[:50]]
                #print '\t'.join(gs)
                gn='dense_'+str(s)
                group_gene[gn]=gs
            group_type[gn].append(cell_type)
            output_line.append(gn)
        output_lines.append(output_line)
        
    output_csv=open('deep_ana'+str(cutoff)+'.csv','w')
    output_csv.write(','.join(head)+'\n')
    for line in output_lines:
        output_csv.write(','.join(line)+'\n')
        #print 'min_index: ',np.min(sort_index)
        #print 'max_index: ',np.max(sort_index)

        #print w[sort_index[0:5]]
    group_count=[]
    for key,val in group_type.items():
        group_count.append((len(val),key))
    print sorted(group_count)
    ngene=0
    nd=0
    for count,gn in sorted(group_count,reverse=True):
        output_csv.write(gn+','+str(count)+','+' '.join(group_gene[gn])+'\n')
        if 'dense' in gn:
            ngene+=len(group_gene[gn])
            nd+=1
    print 'mean ngene: ',ngene/nd
