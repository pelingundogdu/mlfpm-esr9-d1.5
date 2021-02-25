from  handle_other_algo_data import *
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from statistics import mean,stdev
import sys
from plot_error import plot_error
def file_name_to_ARI(file_name,metrics):
    print file_name
    lines=open(file_name).readlines()[-7:]
    print lines
    #print lines[-1]
    for line in lines:
        splits=line.split('\t')
        if splits[0]==metrics:
            return float(splits[1])
    #return float(lines[-1].split('\t')[-2])

ppi_tf_cluster_file='ppi_tf_merge_cluster.txt'
times=20
metricss=[]
metricss.append( 'homogeneity_score')
metricss.append('completeness_score')
metricss.append( 'v_measure_score')
metricss.append( 'adjusted_rand_score')
metricss.append( 'adjusted_mutual_info_score')
metricss.append( 'fowlkes_mallows_score')
metricss.append( 'silhouette_score')
#if len(sys.argv)>2:
#    metrics = (sys.argv[2])
'''
score['homogeneity_score'] = metrics.homogeneity_score(labels, estimator.labels_)
score['completeness_score'] = metrics.completeness_score(labels, estimator.labels_)
score['v_measure_score'] = metrics.v_measure_score(labels, estimator.labels_)
score['adjusted_rand_score'] = metrics.adjusted_rand_score(labels, estimator.labels_)
score['adjusted_mutual_info_score'] = metrics.adjusted_mutual_info_score(labels,  estimator.labels_)
#score['silhouette_score'] = metrics.silhouette_score(data, estimator.labels_,metric='euclidean',sample_size=len(labels))))
'''
#times=20
#times=3
act='tanh'
max_iteration=100
step_size=100
sm=1
sn=1
#sn=0
gn=1
stc=1
#vcts=[4,6,8]
vcts=[2,4,6,8]
#vcts=[4]
pts=[0,1]
#pts=[0]
seed=-1
output_all=['all:']
output_make=[]

#data_file= 'important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt'
params=[]# archi,fp,hiddenlayersize,middle layer size, merge
#1layer, dense, hidden layer size100,no middle layer, no merge
params.append(('Dense 1layer 100',['1layer','',100,0,0]))
#1layer, dense, hidden layer size 796,no middle layer, no merge
params.append(('Dense 1layer 796',['1layer','',796,0,0]))
#2layer, dense, hidden layer size100,middle layer size 796, no merge
params.append(('Dense 2layer 796/100',['3layer','',100,796,0]))
#1layer, PPITF, hidden layer size 0,no middle layer, merge 100
params.append(('PPITF 1layer 696+100',['1layer',ppi_tf_cluster_file,0,0,100]))
#2layer, PPITF, hidden layer size 100 ,no middle layer, merge 100
params.append(('PPITF 2layer 696+100/100',['3layer',ppi_tf_cluster_file,100,0,100]))

#out=[]
#head=['#cell type','pca_n100','pca_n796','original']
append_str='boot'
#append_str='boot_DeNorm'
st_output_mat_head=['model']
st_output_mat=[]
#for time in range(times):
each_time_size=len(vcts)*len(params)*len(pts)
for metrics in metricss:
    models=['original']
    vct_mn_ARI=defaultdict(lambda:[])
    count=0
    for time in range(times):
        seed = time
        for vct in vcts:
            max=(0,0)
            log_files=[]
            for pt in pts:
                for model_name,param in params:
                    archi=param[0]
                    fp=param[1]
                    hls=param[2]
                    mls=param[3]
                    mer=param[4]
                    #append_ID='boot'+str(time)+'_'+str(count)
                    #append_ID='boot'+str(time)+'_'+str(count%each_time_size)
                    append_ID=append_str+str(time)+'_'+str(count%each_time_size)
                    #print append_ID
                    count+=1
                    mn=archi
                    mn+='_SN'+str(sn)
                    mn+='_GN'+str(gn)
                    mn+='_LOG0'
                    mn+='_BS10'
                    mn+='_hls'+str(hls)
                    mn+='_mls'+str(mls)
                    mn+='_seed'+ str(seed)
                    if pt:
                        mn+='_pt'
                    if fp!='':
                        mn+='_PPITF'
                    mn+='_classifier'
                    mn+='_vct'+ str(vct)
                    mn+='_merge'+str(mer)
                    mn+='_'+act
                    mn+='_'+append_ID
                    log_file='clustering_log/'+mn+'_labeled_c0_cut0_100.log'
                    ARI=file_name_to_ARI(log_file,metrics)
                    if pt:
                        vct_mn_ARI[(vct,model_name+' pt'+str(pt))].append(ARI)
                    else:
                        vct_mn_ARI[(vct,model_name)].append(ARI)

            
            seedFileName='seed'+str(seed)+'_vct'+str(vct)+'_'+append_ID+'.pickle'
            log_file=  'clustering_log/'+seedFileName+"_SIMLR_ARI.txt"
            #ARI=float(open(log_file).readlines()[0])
            ARI=file_name_to_ARI(log_file,metrics)
            vct_mn_ARI[(vct,'SIMLR')].append(ARI)
            
                
            log_file='clustering_log/pca_vct'+str(vct)+'_seed'+str(seed)+'_ori_c0_'+append_ID+'.log'
            ARI=file_name_to_ARI(log_file,metrics)
            vct_mn_ARI[(vct,'original')].append(ARI)
               
            '''
            log_file='clustering_log/pca_vct'+str(vct)+'_fit_test_seed'+str(seed)+'_c0_n100_'+append_ID+'.log' 
            ARI=file_name_to_ARI(log_file,metrics)
            vct_mn_ARI[(vct,'PCA 100')].append(ARI)
            
            log_file='clustering_log/pca_vct'+str(vct)+'_fit_test_seed'+str(seed)+'_c0_n796_'+append_ID+'.log'
            ARI=file_name_to_ARI(log_file,metrics)
            vct_mn_ARI[(vct,'PCA 796')].append(ARI)
            '''
            n_components=[2,5,10,50,100]
            methods=['pca','tsne','ica','nmf']
            for nc in n_components:
                for meth in methods:
                    if meth == 'tsne' and nc >=50:
                        continue
                    if meth == 'ica' and nc >=50:
                        continue
                    if meth != 'pca':
                        output_make.append('\t-'+meth+' 1 \\')
                    log_file='clustering_log/'+meth+'_vct'+str(vct)+'_fit_test_seed'+str(seed)+'_c0_n'+str(nc)+'_'+append_ID+'.log'
                    ARI=file_name_to_ARI(log_file,metrics)
                    vct_mn_ARI[(vct,meth+' '+str(nc))].append(ARI)
                    #print len(vct_mn_ARI[(vct,meth+' '+str(nc))])
                    if vct ==2 and len(vct_mn_ARI[(vct,meth+' '+str(nc))])==1:
                        models.append(meth+' '+str(nc))
            #print 'after pcas: ',models 
            seedFileName='seed'+str(seed)+'_vct'+str(vct)+'_'+append_ID+'.pickle'
            SNNinputfileName=seedFileName+'_SNNinput.txt'
            if vct>2:
                log_file='clustering_log/'+SNNinputfileName+"_ARI.txt"
                #ARI=float(open(log_file).readlines()[0])
                ARI=file_name_to_ARI(log_file,metrics)
                vct_mn_ARI[(vct,'SNN-Cliq')].append(ARI)
            
            log_file='clustering_log/'+seedFileName+'_pcaReduce_ARI.txt'
            #ARI=float(open(log_file).readlines()[0])
            ARI=file_name_to_ARI(log_file,metrics)
            vct_mn_ARI[(vct,'pcaReduce')].append(ARI)
            
            #sincera_cluster_algs=['hc','consensus','tight']
            sincera_cluster_algs=['hc']
            for alg in sincera_cluster_algs:
                #sinceraOutfile=seedFileName+'_sincera_'+alg+'_clus.txt'
                sinceraARIfile=seedFileName+'_sincera_'+alg+'.txt'
                log_file='clustering_log/'+sinceraARIfile
                #ARI=float(open(log_file).readlines()[0])
                ARI=file_name_to_ARI(log_file,metrics)
                vct_mn_ARI[(vct,'sincera_'+alg)].append(ARI)
                #output_make.append('\tRscript sincera_clustering.R '+str(vct)+" "+alg+' '+SNNinputfileName+' '+sinceraOutfile)
                #output_make.append('\tpython handle_other_algo_data.py 2 '+sinceraOutfile+' '+SNNinputfileName+'_Answer.txt clustering_log/'+sinceraARIfile)

    #models=['original','PCA 100', 'PCA 796']
    print 'before output: ',models
    models+=['pcaReduce','SIMLR', 'SNN-Cliq','sincera_hc', 'Dense 1layer 100', 'Dense 1layer 796', 'Dense 2layer 796/100','PPITF 1layer 696+100', 'PPITF 2layer 696+100/100', 'Dense 1layer 100 pt1', 'Dense 1layer 796 pt1', 'Dense 2layer 796/100 pt1','PPITF 1layer 696+100 pt1', 'PPITF 2layer 696+100/100 pt1']
    #models=['original','PCA 100', 'PCA 796','pcaReduce','SIMLR', 'SNN-Cliq','sincera_hc', 'Dense 1layer 100', 'Dense 1layer 796', 'Dense 2layer 796/100','PPITF 1layer 696+100', 'PPITF 2layer 696+100/100']
    #models=['original','PCA 100', 'PCA 796','pcaReduce', 'SNN-Cliq','sincera_hc','sincera_consensus','sincera-tight', 'Dense 1layer 100', 'Dense 1layer 796', 'Dense 2layer 796/100','PPITF 1layer 696+100', 'PPITF 2layer 696+100/100']
    #models=['original','PCA 100', 'PCA 796', 'Dense 1layer 100', 'Dense 1layer 796', 'Dense 2layer 796/100','PPITF 1layer 696+100', 'PPITF 2layer 696+100/100']
    output_mat=np.zeros((len(vcts)+1,len(models)))

    error_mat=np.zeros((len(vcts),len(models)))
    for key,val in vct_mn_ARI.items():
        print key ,val
        vct=key[0]
        model=key[1]
        r= vcts.index(vct)
        c= models.index(model)
        output_mat[r,c]=mean(val)
        if 0:
            error_mat[r,c]=stdev(val)
        output_mat[-1,c]+=output_mat[r,c]
    output_mat[-1,:]/=float(len(vcts))
    if 1 or metrics != 'silhouette_score':
        st_output_mat.append(output_mat[-1,:].tolist())
        st_output_mat_head.append(metrics)
    if len(vcts)>1:
        output_mat[-1,models.index('SNN-Cliq')]*=float(len(vcts))/float(len(vcts)-1)
    #output_file=open('clustering_result20170321.csv','w')
    output_file=open('clustering_result20170405_'+metrics+'_'+str(times)+'.csv','w')
    output_file.write('#cell type, '+','.join(models)+'\n')
    for i in range(output_mat.shape[0]):
        if i != len(vcts):
            output_file.write(str(vcts[i])+','+','.join(map(str,output_mat[i,:]))+'\n')
        else:
            output_file.write('mean'+','+','.join(map(str,output_mat[i,:]))+'\n')
    print output_mat.shape
    print error_mat.shape

    #st_output_mat[metrics] = output_mat
    print output_mat
    if 0:
        means={}
        errors={}
        for i in range(1,output_mat.shape[0]):
            means[i]=output_mat[i-1,:]
            errors[i]=error_mat[i-1,:]
        plot_error('clustering_fig_result_20170405.pdf',models,means,errors)
    #for j in range(output_mat.shape[1]):
        #    tmp[i].append()
    #print means
    #print errors
    #print output_mat
means=[]
#print st_output_mat
row,col= np.array(st_output_mat).shape
#print st_output_mat_head
output_all_metrics=open('clustering_result_all_metric_mean_wsil.csv','w')
st_output_mat_head.append('mean')
for j in range(col):
    su=0
    for i in range(row):
        su+=st_output_mat[i][j]
        #print su
    #print su/row
    means.append(su/row)
st_output_mat.append(means)
st_output_mat=np.array(st_output_mat)
output_all_metrics.write(','.join(st_output_mat_head)+'\n')
row,col= st_output_mat.shape
for j in range(col):
    output_all_metrics.write(models[j]+','+','.join(map(str,st_output_mat[:,j]))+'\n')
    #for i in range(row):
        
