from sklearn import svm
import pickle
from sklearn.cross_validation import StratifiedKFold, StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
import random
import parse_data
#print(__doc__)

from time import time
import numpy as np
import sys
from sklearn import metrics
from sklearn.cluster import KMeans,SpectralClustering
from sklearn.preprocessing import scale
from sklearn.metrics.cluster import adjusted_rand_score



def bench_clustering(estimator, name, data,labels,useAllData=False):
    t0 = time()
    estimator.fit(data)
    if not useAllData:
        print '%.2fs\t%.3f\t%s' %((time() - t0),metrics.adjusted_rand_score(labels, estimator.labels_),name)
    else:
        all_data_label=estimator.predict(data)
        eval_data_label=[]
        eval_true_label=[]
        for index, lab in enumerate(labels):
            if lab!=0:
                eval_true_label.append(lab)
                eval_data_label.append(all_data_label[index])
        #print eval_true_label
        #print eval_data_label
            #if lab==0:
                #print 'skip None type'
        
        print '%.2fs\t%.3f\t%s' %((time() - t0),metrics.adjusted_rand_score(eval_true_label,eval_data_label),name)
        
    score={}
    score['homogeneity_score'] = metrics.homogeneity_score(labels, estimator.labels_)
    score['completeness_score'] = metrics.completeness_score(labels, estimator.labels_)
    score['v_measure_score'] = metrics.v_measure_score(labels, estimator.labels_)
    score['adjusted_rand_score'] = metrics.adjusted_rand_score(labels, estimator.labels_)
    score['adjusted_mutual_info_score'] = metrics.adjusted_mutual_info_score(labels,  estimator.labels_)
    #score['silhouette_score'] = metrics.silhouette_score(data, estimator.labels_,metric='euclidean',sample_size=len(labels))))
    score['fowlkes_mallows_score'] = metrics.fowlkes_mallows_score(labels, estimator.labels_)
    #fowlkes_mallows_score
    #return estimator.labels_, estimator.cluster_centers_,score
    return estimator.labels_, estimator.cluster_centers_,score
    '''print('% 9s   %.2fs    %i   %.3f   %.3f   %.3f   %.3f   %.3f    %.3f'
          % (name, (time() - t0), estimator.inertia_,
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_),
             metrics.v_measure_score(labels, estimator.labels_),
             metrics.adjusted_rand_score(labels, estimator.labels_),
             metrics.adjusted_mutual_info_score(labels,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean',
                                      sample_size=len(labels))))
    '''
if __name__=='__main__':
    n_cluster=0
    if len(sys.argv)>1:
        n_cluster=int(sys.argv[1])
    print 'loading data'

    with open('deep_feature.pickle', 'rb') as handle:
        NNdict = pickle.load(handle)
    with open('pca_feature.pickle', 'rb') as handle:
        PCAdict = pickle.load(handle)

    #all_data, labeled_data,unlabeled_data,label_unique_list,all_label, labeled_label,all_sample_ID,labeled_sample_ID,unlabeled_sample_ID,gene_names=parse_data.load_integrated_data('data/TPM_mouse_1_4_6_7_8_10_16.txt', whitening=True)
    #all_data, labeled_data,unlabeled_data,label_unique_list,all_label, labeled_label, all_weights, labeled_weights, unlabeled_weights,all_sample_ID,labeled_sample_ID,unlabeled_sample_ID,gene_names=parse_data.load_integrated_data('data/TPM_mouse_1_4_6_7_8_10_16.txt',sample_normalize=True,gene_normalize=True)   
    all_data, labeled_data,unlabeled_data,label_unique_list,all_label, labeled_label, all_weights, labeled_weights, unlabeled_weights,all_sample_ID,labeled_sample_ID,unlabeled_sample_ID,gene_names=parse_data.load_integrated_data('data/TPM_mouse_1_4_6_7_8_10_16.txt',sample_normalize=True,gene_normalize=True)   
    #all_data, labeled_data,unlabeled_data,label_unique_list,all_label, labeled_label, all_weights, labeled_weights, unlabeled_weights,all_sample_ID,labeled_sample_ID,unlabeled_sample_ID,gene_names=parse_data.load_integrated_data('data/TPM_mouse_1_4_6_7_8_10_16.txt',whitening=False)   
    #Label=labeled_label
    Label=labeled_label

    #bench_k_means(KMeans(init='k-means++', n_clusters=n_digits, n_init=10),

    if n_cluster==0:
        n_cluster=max(Label)
    print 'n_cluster=',n_cluster
    #nlabels=[]
    #nlabels.append(max(Label))
    n_iter=10
    estimators=[]
    est_name=[]
    est_name.append('kmeans++')

    estimators.append(KMeans(init='k-means++', n_clusters=n_cluster, n_init=n_iter))
    #est_name.append('spectral')
    #estimators.append(SpectralClustering(n_clusters=15,eigen_solver='arpack',affinity='nearest_neighbors'))
    #estimators.append(SpectralClustering(n_clusters=15))
    #estimators.append(SpectralClustering(n_clusters=15,eigen_solver='arpack'))
    #nlabels.append(100)

    #print NNdict.keys()
    pca=True
    NN=False
    ori=False
    for index,est in enumerate(estimators):
        #for lab in nlabels:
            #print 'nlabel: ',lab
            #print('% 9s' % 'init'
            #  '    time  inertia    homo   compl  v-meas     ARI AMI  silhouette')
        print 'time\t ARI'
        print(79 * '_')
        if pca:    
            print 'pca feature:'# 100 component'
            for key,val in sorted(PCAdict.items()):
                useAllData=False
                if key == 'pca_all_data': 
                    useAllData=True
                    Label=all_label
                else:
                    Label=labeled_label
                print 'data: ', key
                print "data.shape: ", val.shape
                bench_clustering(est,name=est_name[index], data=val[:,:100], labels=Label,useAllData=useAllData)
        if NN: 
            print 'NN feature: '
            
            for key,val in sorted(NNdict.items()):
                useAllData=False
                if 'all_data' in key:
                    useAllData=True
                    Label=all_label
                else:
                    Label=labeled_label
                #if 'all_data' in key or 'all_label' in key or 'labeled_label' in key or 'labeled_data' == key:
                    #print 'skip!'
                #    continue
                print 'data: ', key       
                print "data.shape: ", val.shape
                bench_clustering(est,name=est_name[index], data=val, labels=Label,useAllData=useAllData)
        if ori: 
            print 'original_data: '
            print 'data: all_data'
            print "data.shape: ", all_data.shape
            bench_clustering(est,name=est_name[index], data=all_data, labels=all_label,useAllData=True)
            print 'data: labeled_data'
            print "data.shape: ", labeled_data.shape
            bench_clustering(est,name=est_name[index], data=labeled_data, labels=labeled_label,useAllData=False)
            


