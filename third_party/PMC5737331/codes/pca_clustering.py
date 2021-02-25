
import matplotlib
matplotlib.use('Agg')
import argparse
#import brewer2mpl
from palettable.colorbrewer.qualitative import Dark2_7
#from palettable.colorbrewer.sequential import Plasma_20
import palettable
import numpy as np
np.random.seed(0)
from sklearn.decomposition import PCA,KernelPCA,FastICA,NMF
from sklearn.manifold import TSNE
import parse_data
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans,SpectralClustering
import clustering
import random
from keras.models import Sequential,model_from_json
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils, generic_utils
from keras.optimizers import SGD, Adadelta, Adagrad
from keras import models
from collections import defaultdict
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from ZIFA import ZIFA
from ZIFA import block_ZIFA
import sys

from scipy.spatial import distance
import numpy as np
import parse_data
import time
import math
import pickle
from deep_net import *
import matplotlib.pyplot as plt

import os
import scipy.io as sio
import SIMLR
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
from sklearn.metrics.cluster import adjusted_rand_score as ari
from scipy.sparse import csr_matrix,csc_matrix
def AvgPrecision(ans,pred):
    #ans is a single integer denoting the class
    #pred is a vector of the retrieved items
    correct=0
    total=0
    ret=0.
    for p in pred:
        total+=1
        if p == ans:
            correct+=1
            ret+=correct/float(total)
    if correct>0:
        ret/=float(correct)
    return ret
def MeanAvgPrecision(anss,preds):
    ret=0.
    for ans, pred in zip(anss,preds):
        ret+=AvgPrecision(ans,pred)
    ret/=float(len(anss))
    return ret

def get_nn_code(model_name,iteration,transform_data):
    custom_obj={}
    custom_obj['MyLayer']=MyLayer
    #model_name=args.neural_network
    #iteration=args.nn_iteration
    model = model_from_json(open('model/'+model_name+'.json').read(),custom_objects=custom_obj)
    load_model_weight_from_pickle(model,'model/'+model_name+'_'+str(iteration)+'.pickle')
    model.compile(optimizer='sgd', loss='mse')
    
    #print model.predict([transform_data,transform_data])
    nlayer= len(model.layers)
    print 'nlayer: ',nlayer
    if 'class' in model_name :
        layer=(nlayer+1)/2-1
    else:
        layer=nlayer/2-1 
    
    #if args.validation_data_classifier:
    #    layer=nlayer-1
    #else:
    #    layer=nlayer/2-1 
    if 'merge100' in model_name:
        #transform_data=[transform_data,transform_data]
        code=get_output(model,layer,transform_data,merge=True)
    else:
        code=get_output(model,layer,transform_data)
    print 'layer: ',layer
    print 'code shape: ', code.shape
    return code

def plot_code(code,labels,label_unique_list,meth,dim_name=""):
    print code.shape
    print labels.shape
    print meth
    print label_unique_list
    if code.shape[1]>2:
        if meth == 'tsne':
            method=TSNE(2)
        else:
            method=PCA(2)
        xy=method.fit_transform(code)
    else:
        xy=code
    print xy.shape
    print xy
    #set2 = brewer2mpl.get_map('Set2', 'qualitative',len(label_unique_list)+1).mpl_colors
    #set2 = brewer2mpl.get_map('qualitative',len(label_unique_list)+1).mpl_colors

    # Set the random seed for consistency
    np.random.seed(12)
    fig, ax = plt.subplots(1)
    #ax.set_color_cycle(palettable.colorbrewer.qualitative.Dark2_8.mpl_colors)
    #ax.set_color_cycle(palettable.cmocean.sequential.Algae_20.mpl_colors)
    ax.set_color_cycle(palettable.mycarta.Cube1_16.mpl_colors)
    #ax.set_color_cycle(palettable.colorbrewer.sequential.Plasma_20.mpl_colors)
    #plt.scatter(xy[:,0],xy[:,1],c=labels,cmap=plt.cm.jet)
    for i in range(len(label_unique_list)):
        if i ==0 :
            continue
        print i, label_unique_list[i]
        xy_lab = xy[labels==i]
        print xy_lab.shape
        #color = set2[i]
        #c=[i]*xy_lab.shape[0]
        #print c

        #ax.scatter(xy_lab[:,0],xy_lab[:,1],label=label_unique_list[i],c=[i]*xy_lab.shape[0],cmap=plt.cm.get_cmap('RdYlBu'))
        ax.scatter(xy_lab[:,0],xy_lab[:,1],label=label_unique_list[i])
        #ax.scatter(xy_lab[:,0],xy_lab[:,1],label=label_unique_list[i],cmap=Blues_8.mpl_colormap)
    #ax.legend(loc='upper center',ncol=3,fancybox=True,shadow=True)
    #ax.set_ylim(np.min(xy[:,1])-5,np.min(np.max(xy[:,1])*1.5+30,np.max(xy[:,1]*3)))
    #fig.savefig('plot_'+dim_name+'_reduced_dimension_'+str(meth)+'.png')

    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5,-0.1),ncol=3,fancybox=True,shadow=True)
    ax.grid('on')
    #fig.savefig('samplefigure', bbox_extra_artists=(lgd,), bbox_inches='tight')
    #fig.savefig('plot_'+dim_name+'_reduced_dimension_'+str(meth)+'.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
    fig.savefig('plot_'+dim_name+'_reduced_dimension_'+str(meth)+'.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')

    #for i in range(labels.shape[0]):
    #    lab = labels[i]
    #    lab_name = label_unique_list[lab]
    #    print lab, lab_name

if __name__=='__main__':
    start=time.clock()
    parser = argparse.ArgumentParser()
    parser.add_argument('-plot',"--plot_reduced_dimension", help="specify how to plot the reduced dimension, default is None", choices=['pca','tsne','None'],default=None)
    parser.add_argument('-plot_name',"--plot_dim_name", help="specify the name of the reduced dimension, default is empty string",default='')
    
    parser.add_argument('-t',"--transform", help="specify the transform set, default is all", choices=['all','labeled','unlabeled'],default='all')
    parser.add_argument('-ori',"--original", help="specify whether to do clustering on original data, default is 0", choices=[0,1],type=int,default=0)
    parser.add_argument('-z',"--zifa", help="specify whether to do clustering on zifa-ed data, default is 0", choices=[0,1],type=int,default=0)
    parser.add_argument('-tsne',"--use_tsne", help="specify whether to do clustering on tsne-ed data, default is 0", choices=[0,1],type=int,default=0)
    parser.add_argument('-ica',"--use_ica", help="specify whether to do clustering on ica-ed data, default is 0", choices=[0,1],type=int,default=0)
    parser.add_argument('-SIMLR',"--use_SIMLR", help="specify whether to do clustering on SIMLR-ed data, default is 0", choices=[0,1],type=int,default=0)
    parser.add_argument('-nmf',"--use_nmf", help="specify whether to do clustering on nmf-ed data, default is 0", choices=[0,1],type=int,default=0)
    parser.add_argument('-nn',"--neural_network", help="specify whether to do clustering on nn data, and please provide nn model")
    parser.add_argument('-f',"--fit", help="specify the fitting set, default is all", choices=['all','labeled','unlabeled'],default='all')
    parser.add_argument('-n',"--n_component", help="specify the number of components, default is 0, means all", type=int,default=0)
    parser.add_argument('-ni',"--nn_iteration", help="specify the number of nn iterations, default is 10", type=int,default=10)
    parser.add_argument('-c',"--n_cluster", help="specify the number of clusters, default is the number of labels", type=int,default=0)
    parser.add_argument('-i',"--n_iteration", help="specify the number of iteration, default is 10", type=int,default=10)
    parser.add_argument('-l',"--loss", help="specify whether to show loss or not, default is 1",choices=[0,1], type=int,default=1)
    parser.add_argument('-vdc',"--validation_data_classifier", help="specify whether to use validation data from pickle, default is 0",choices=[0,1], type=int,default=0)
    parser.add_argument('-vct',"--validation_cell_types", help="specify the number of validation cell types, default is 0", type=int,default=0)
    parser.add_argument('-seed',"--random_seed", help="specify the random seed, default is 0", type=int,default=0)
    parser.add_argument('-fs',"--fitself", help="specify whether to use validation data from pickle and fit the test set, default is 0",choices=[0,1], type=int,default=0)
    parser.add_argument('-cut',"--clusterUnlabeledAndTest", help="specify whether to cluster unlabeled data and test set, default is 0",choices=[0,1], type=int,default=0)
    parser.add_argument('-retrieval',"--retrieval_analysis", help="specify whether to perform retrieval analysis, default is 0",choices=[0,1], type=int,default=0)
    parser.add_argument('-ret_top',"--retrieval_topcell", help="specify the number of returned cell in retrieval analysis, default is 0(all the desired type)", type=int,default=0)
    parser.add_argument('-ret_map',"--retrieval_map", help="specify whether to use map score in retrieval analysis, default is 0",choices=[0,1], type=int,default=0)
    
    parser.add_argument('-r',"--reference_gene_file", help="specify the file that contatins the genes to be kept, if not specified then all the genes in training data will be used")
    parser.add_argument('-rl',"--reference_gene_file_lines", help="specify the number of the genes in the file that contatins the genes to be kept, if not specified then all the genes in training data will be used", type=int,default=-1)
    parser.add_argument('-d',"--data_file", help="specify the data file, if not specified then a default training data file will be used")
    parser.add_argument('-o',"--out_file", help="specify the output analysis file, default is None", default=None)
    parser.add_argument('-sn',"--sample_normalize", help="specify whether to normalize each sample (divide by total reads), default is 1 (means True)",choices=[0,1], type=int,default=1)
    parser.add_argument('-gn',"--gene_normalize", help="specify whether to normalize each gene (mean 0 std 1), default is 1 (means True)",choices=[0,1], type=int,default=1)
    parser.add_argument('-log',"--log_trans", help="specify whether to do log transform",choices=[0,1], type=int,default=0)
    parser.add_argument('-ID',"--identification", help="specify the additional identification append to model name or output file name, default is None",default=None)
    
    args=parser.parse_args()
    '''
    print args.transform
    print args.fit
    print args.n_component
    print args.n_cluster
    print args.n_iteration
    print args.loss
    '''
    #data_file_name='../data/TPM_mouse_1_4_6_7_8_10_16.txt'
    data_file_name='important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt'
    if args.data_file is not None:
        data_file_name=args.data_file
    #if 'boot' in args.neural_network:
    #    append_ID='_'+args.neural_network.split('_')[-1].split('.')[0]
    #else:
    #    append_ID=''
    if args.identification is not None:
        append_ID='_'+args.identification
    else:
        append_ID=''
    all_data, labeled_data,unlabeled_data,label_unique_list,all_label, labeled_label, all_weights, labeled_weights, unlabeled_weights,all_sample_ID,labeled_sample_ID,unlabeled_sample_ID,gene_names=parse_data.load_integrated_data(data_file_name,sample_normalize=args.sample_normalize,gene_normalize=args.gene_normalize,log_trans=args.log_trans,ref_gene_file=args.reference_gene_file)#'cluster_genes.txt'   
    all_data_sn1_gn0, _,_,_,_, _, _, _, _,_,_,_,_=parse_data.load_integrated_data(data_file_name,sample_normalize=1,gene_normalize=0,ref_gene_file=args.reference_gene_file)#'cluster_genes.txt'   
    fit_data=all_data
    transform_data=all_data

    #print 'all_data.shape: ', all_data.shape	
    
    #all_data=all_data[:100,:200]


    if args.fit=='labeled':
        fit_data=labeled_data
    elif args.fit=='unlabeled':
        fit_data=unlabeled_data
    if args.transform=='labeled':
        transform_data=labeled_data
    elif args.transform=='unlabeled':
        transform_data=unlabeled_data
    if (args.validation_data_classifier==1 or args.validation_cell_types>0) and args.zifa !=1 :
    #if (args.validation_data_classifier==1 or args.validation_cell_types>0)  :
        if args.validation_data_classifier==1:
            with open('seed'+str(args.random_seed)+'_vp'+str(args.validation_percent)+append_ID+'.pickle', 'rb') as handle:
                output_dict = pickle.load(handle)
        if args.validation_cell_types>0:
            with open('seed'+str(args.random_seed)+'_vct'+str(args.validation_cell_types)+append_ID+'.pickle', 'rb') as handle:
                output_dict = pickle.load(handle)
            #fit_data=output_dict['train_X']
        transform_data=output_dict['test_X']
        fit_data=output_dict['train_X']
        if args.fitself:
            fit_data=output_dict['test_X']
        valid_Y=output_dict['test_Y_array']
    print 'fit_data.shape: ', fit_data.shape	
    print 'transform_data.shape: ', transform_data.shape	
    if args.clusterUnlabeledAndTest==1:
        transform_data=np.concatenate((transform_data,unlabeled_data))
        #valid_Y.extend([0]*unlabeled_data.shape[0])
        valid_Y=np.concatenate((valid_Y,np.zeros(unlabeled_data.shape[0])))
    if args.original==1:
        code=transform_data
    elif args.neural_network is not None:
        code=get_nn_code(args.neural_network, args.nn_iteration,transform_data)
    elif args.zifa==1:
        #Y=transform_data
        #valid_Y=labeled_label[valid_set_index]
        #Y=Y[:, np.sum(Y >1e-6, axis=0)/float(Y.shape[0])>0.9] # filter genes that are zero in
        #code, model_params = block_ZIFA.fitModel(Y, 100)
        #data_file_name='../data/TPM_mouse_1_4_6_7_8_10_16.txt'
        all_data, labeled_data,unlabeled_data,label_unique_list,all_label, labeled_label, all_weights, labeled_weights, unlabeled_weights,all_sample_ID,labeled_sample_ID,unlabeled_sample_ID,gene_names=parse_data.load_integrated_data(data_file_name,sample_normalize=1,gene_normalize=0)#'cluster_genes.txt'   
        #print labeled_data
        train_data=labeled_data
        train_valid=labeled_label
        valid_cell_types=np.random.choice(np.arange(1,max(labeled_label)+1),size=args.validation_cell_types,replace=False)
        print 'valid cell types:'
        for ct in valid_cell_types:
            print ct, label_unique_list[ct]
        valid_set_index=[]
        print labeled_label
        for index,lab in enumerate(labeled_label):
            if lab in valid_cell_types:
                valid_set_index.append(index)
        print valid_set_index
        #valid_set_index=np.random.choice(train_data.shape[0],size=s,replace=False)
        train_set_index=[x for x in range(train_data.shape[0]) if x not in valid_set_index]
        valid_data=train_data[valid_set_index,:] 
        #valid_valid=train_valid[valid_set_index,:]
        #train_data=train_data[train_set_index,:]
        #train_valid=train_valid[train_set_index,:]
        Y=valid_data
        valid_Y=labeled_label[valid_set_index]
        print 'before shape: ',Y.shape
        #Y=Y[:, np.sum(Y >1e-6, axis=0)/float(Y.shape[0])>0.9]#keep genes that are expressed in 90% samples
        print 'after shape: ',Y.shape
        code, model_params = block_ZIFA.fitModel(Y, 100)
        print code.shape
        #print Z
        
    else:
        #if args.n_component==0:
        #    pca=PCA()
        #else:
        print 'fitting data:' + args.fit
        if args.use_nmf==1:
            #transform_data = 
            if args.validation_cell_types>0:
                transform_data=output_dict['test_X_TPMgn0']
                #print 'vct!'
            else:
                #print 'not vct'
                transform_data=all_data_sn1_gn0
                fit_data = all_data_sn1_gn0
            
            if args.fitself:
                fit_data = transform_data
            print 'NMF, using only sample normalized data'
            print np.max(transform_data)
            print np.min(transform_data)
        if args.n_component > transform_data.shape[0]:
            args.n_component=transform_data.shape[0]
        if args.use_ica:
            if args.n_component:
                ica=FastICA(args.n_component)
            else:
                ica=FastICA()
            print 'check ica NAN: '
            print np.any(np.isnan(fit_data))
            print 'check ica INF: '
            print np.any(np.isinf(fit_data))
            ica.fit(fit_data)
            code=ica.transform(transform_data)
        elif args.use_nmf:
            if args.n_component:
                nmf=NMF(args.n_component)
            else:
                nmf=NMF()
            nmf.fit(fit_data)
            code=nmf.transform(transform_data)
            #print 'tshape:',transform_data.shape
            #print 'cshape: ',code.shape
        elif args.use_tsne:
            if args.n_component:
                tsne=TSNE(args.n_component)
            else:
                tsne=TSNE()
            
            code=tsne.fit_transform(transform_data)
        elif args.use_SIMLR:
            print 'not implemented for experiment'
            X=transform_data
            n_dim=min([X.shape[0],X.shape[1]])
            if X.shape[1]>500:
                #X = SIMLR.helper.fast_pca(X,max([X.shape[0],X.shape[1]]))
                X = SIMLR.helper.fast_pca(X,n_dim)
            else:
                X = X.todense()
            c=len(set(labeled_label))
            simlr = SIMLR.SIMLR_LARGE(c, min(30,X.shape[0]/2), 0); ###This is how we initialize an object for SIMLR. the first input is number of rank (clusters) and the second input is number of neighbors. The third one is an binary indicator whether to use memory-saving mode. you can turn it on when the number of cells are extremely large to save some memory but with the cost of efficiency.
            S, F, val, ind = simlr.fit(X)
            print F.shape
            print val.shape
            print S.shape
            print ind.shape
            code =val
        else:
            if args.n_component:
                pca=PCA(args.n_component)
            else:
                pca=PCA()
            pca.fit(fit_data)
            code=pca.transform(transform_data)

        print 'transforming data: '+args.transform
        
        #print 'inverse transforming'
        #recovered_data=pca.inverse_transform(code)

        #print 'pca recover MSE:'
        #print mean_squared_error(transform_data, recovered_data)
    if args.retrieval_analysis:
        # nearest neighbor retrieval things
        #data_file_name='hannah_mouse_data/TPM_6_8_9_15_25_41_44_45_46_.txt'
        #model_name='3layer_SN1_GN1_BS32_hls100_mls696_seed0_classifier_merge0_tanh'
        #nn_iteration=100
        all_data, labeled_data,unlabeled_data,label_unique_list,all_label, labeled_label, all_weights, labeled_weights, unlabeled_weights,all_sample_ID,labeled_sample_ID,unlabeled_sample_ID,gene_names=parse_data.load_integrated_data(data_file_name,sample_normalize=1,gene_normalize=1,log_trans=args.log_trans,ref_gene_file=args.reference_gene_file)
        print 'all_data.shape: ',all_data.shape
        #code = get_nn_code(model_name,nn_iteration,all_data)
        #code=transform_data
        print 'code.shape: ',code.shape
        print 'all_weights: ',all_weights
        print '\n'.join(label_unique_list)
        verify_lab=['2cell','4cell','ICM','zygote','8cell','ESC','lung','TE','thymus','spleen','HSC','neuron']
        verify_lab_count=[]

        for index, lab in enumerate(label_unique_list):
            if 'cortex' in lab:
                label_unique_list[index]='neuron'
            if 'CNS' in lab:
                label_unique_list[index]='neuron'
            if 'brain' in lab:
                label_unique_list[index]='neuron'
            for vl in verify_lab:
                if vl in lab:
                    label_unique_list[index]=vl
        dataset_sets=map(str,sorted(map(int,list(set(all_weights)))))
        ntop=10
        #vtop=100
        out_ret=open(args.out_file,'w')
        out_ret.write('dataset,sample,celltype,retrieval #cell in top100, total #cell,retrieval_ratio\n')
        out_ret2=open(args.out_file+'_summary.csv','w')
        out_ret2.write('dataset,celltype,#cell,mean retrieval ratio\n')
        for ds in dataset_sets:
            meindex=np.where(all_weights==ds)
            nmeindex=np.where(all_weights!=ds)
            #isme= all_data[meindex]
            #isnme= all_data[nmeindex]
            isme= code[meindex]
            print 'isme'
            print isme
            print 'isme.shape'
            print isme.shape
            isnme= code[nmeindex]
            dismat=distance.cdist(isme,isnme, 'euclidean')
            print dismat.shape
            cell_dict=defaultdict(lambda:[])
            for index,row in enumerate(dismat):
                #print index, row
                now_label=label_unique_list[all_label[meindex[0][index]]]
                if now_label not in verify_lab:
                    continue
                now_sample=all_sample_ID[meindex[0][index]]
                print 'now dataset: ',ds
                print 'now_sample: ',now_sample
                print 'now_label: ',now_label
                sort_index = np.argsort(row)
                temp_lab=[]
                temp_dist=[]
                temp_set=[]
                total_vl=len([ label_unique_list[all_label[nmeindex[0][x]]] for x in range(len(row)) if label_unique_list[all_label[nmeindex[0][x]]]==now_label])
                if args.retrieval_topcell>0:
                    total_vl=args.retrieval_topcell
                for si in sort_index[:total_vl]:
                    temp_set.append(all_weights[nmeindex[0][si]])
                    temp_dist.append(row[si])
                    temp_lab.append(label_unique_list[all_label[nmeindex[0][si]]])
                vtop_vl=len([ temp_lab[x] for x in range(len(temp_lab)) if temp_lab[x]==now_label])
                print '# of hit in top 100 cells: ', vtop_vl
                print 'total # of cell in reference cells:', total_vl
                ratio=vtop_vl/float(total_vl)
                print 'ratio: ',ratio
                if args.retrieval_map:
                    AP=AvgPrecision(now_label,temp_lab)
                    print 'AP=',AP
                    ratio=AP
                cell_dict[now_label].append(ratio)
                out_ret.write(str(ds)+','+now_sample+','+now_label+','+str(vtop_vl)+','+str(total_vl)+','+str(ratio)+'\n')
                print 'top ',ntop,' neighbor distance, label, and dataset: '
                print temp_dist[:ntop]
                print temp_lab[:ntop]
                print temp_set[:ntop]
            for key, val in cell_dict.items():
                print key, np.mean(val)
                out_ret2.write(str(ds)+','+key+','+str(len(val))+','+str(np.mean(val))+'\n') 
    else:
        #clustering
        print 'code.shape:',code.shape
        useAllData=False
        if args.validation_data_classifier or args.validation_cell_types>0:
            Label=valid_Y
        else:
            Label=labeled_label
        print 'Label:'
        print Label
        n_cluster=args.n_cluster
        if n_cluster==0:
            #n_cluster=max(Label)
            n_cluster=len(set(Label))
        print 'n_cluster: ',n_cluster

        est=KMeans(init='k-means++', n_clusters=n_cluster, n_init=args.n_iteration)
        if args.transform=='all' and args.validation_data_classifier==0 and args.validation_cell_types==0 or args.clusterUnlabeledAndTest:
            useAllData=True
            Label=all_label
        print useAllData
        print 'time\t ARI'
        #print 'tshape:',transform_data.shape
        #print 'cshape: ',code.shape
        #print Label.shape
        pred_label,cluster_center,score=clustering.bench_clustering(est,name='kmeans++', data=code, labels=Label,useAllData=useAllData)
        score['silhouette_score'] = metrics.silhouette_score(transform_data, pred_label)
        print 'scores:'
        for key,val in score.items():
            print key+'\t'+str(val)
            #print pred_label,cluster_center
        if args.out_file is not None:
            outfile=open(args.out_file,'w')
            outfile.write('\t'.join(map(str,label_unique_list))+'\n')
            outfile.write('\t'.join(map(str,labeled_label))+'\n')
            outfile.write('\t'.join(map(str,pred_label))+"\n")
            for c in cluster_center:
               outfile.write('\t'.join(map(str,c))+'\n')
            out_dict={}
            out_dict['pred_label']=pred_label
            out_dict['cluster_center']=cluster_center
            out_dict['score']=score
            with open(args.out_file+'.pickle', 'wb') as handle:
                print 'saving clustering results'
                pickle.dump(out_dict, handle)
    sys.stderr.write( str(time.clock() - start) +" seconds\n" )
    if args.plot_reduced_dimension:
        plot_code(code,labeled_label,label_unique_list,args.plot_reduced_dimension,dim_name=args.plot_dim_name)

