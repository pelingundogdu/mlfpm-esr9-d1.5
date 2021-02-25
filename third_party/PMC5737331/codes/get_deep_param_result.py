import matplotlib
matplotlib.use('Agg')
from  handle_other_algo_data import *
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from statistics import mean,stdev
import sys
#from plot_error import plot_error
def file_name_to_loss(file_name,maxit=100):
    print file_name
    lines=open(file_name).readlines()
    #print lines
    #print lines[-1]
    train_loss=[]
    train_acc=[]
    test_loss=[]
    test_acc=[]
    count=0
    for line in lines:
        if 'loss' in line and count<maxit:
            #print line
            count+=1
            splits=line.split(' ')
            #print splits
            tal=float(splits[3])
            taa=float(splits[6])
            tel=float(splits[9])
            tea=float(splits[12])
            train_loss.append(tal)
            train_acc.append(taa)
            test_loss.append(tel)
            test_acc.append(tea)
    #print 'max test acc: ',np.max(test_acc)
    max_acc = np.max(test_acc)
    #print 'min test loss: ',np.min(test_loss)
    min_loss = np.min(test_loss)
    #print 'index: ',test_acc.index(max_acc)
    #print 'index: ',test_loss.index(min_loss)
    max_acc_index=test_acc.index(max_acc)
    min_loss_index=test_loss.index(min_loss)
    output_dict={}
    output_dict['train_loss']=train_loss
    output_dict['train_acc']=train_acc
    output_dict['test_loss']=test_loss
    output_dict['test_acc']=test_acc
    output_dict['max_acc']=max_acc
    output_dict['min_loss']=min_loss
    output_dict['max_acc_index']=max_acc_index
    output_dict['min_loss_index']=min_loss_index
    return output_dict
#output_dict=file_name_to_loss('log_deep_param/1layer_dense_vp0.1_hls100_maxit1000_0.log')

hlss=[2,5,10,30,50,80,100,150,200]
times = 10
#vps=[0.1]
vps=[0.1,0.2]
max_iteration=100
stc=1
seed=-1
count=0
data_file= 'important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt'
archis=['PPITF','dense']
inits=['glorot_uniform','glorot_normal','he_normal','he_uniform']
for archi in archis:
    for init in inits:
        vp_hls_performance_dict=defaultdict(lambda:{})
        for vp in vps:
            min_loss=100
            max_acc=0
            best_acc_model=''
            best_loss_model=''
            for hls in hlss:
                max_accs=[]
                min_losss=[]
                max_acc_indexs=[]
                min_loss_indexs=[]
                for _ in range(times):
                    count+=1
                    log_file='1layer_'+archi
                    log_file+='_init_'+init
                    log_file+='_vp'+str(vp)
                    log_file+='_hls'+str(hls)
                    if archi=='PPITF':
                        log_file+='_maxit'+str(100)
                    if archi=='dense':
                        log_file+='_maxit'+str(100)
                        #log_file+='_maxit'+str(max_iteration)
                    log_file+='_'+str(_)
                    log_file+='.log'
                    output_dict=file_name_to_loss('log_deep_param/'+log_file)
                    print log_file
                    #print output_dict['max_acc_index'],output_dict['max_acc']
                    #print output_dict['min_loss_index'],output_dict['min_loss']
                    max_accs.append(output_dict['max_acc'])
                    min_losss.append(output_dict['min_loss'])
                    min_loss_indexs.append(output_dict['min_loss_index'])
                    max_acc_indexs.append(output_dict['max_acc_index'])
                if np.mean(max_accs)> max_acc:
                    max_acc=np.mean(max_accs)
                    best_acc_model=hls
                if np.mean(min_losss)< min_loss:
                    min_loss=np.mean(min_losss)
                    best_loss_model=hls
                vp_hls_performance_dict[(vp,hls)]['max_acc']=np.mean(max_accs)
                vp_hls_performance_dict[(vp,hls)]['min_loss']=np.mean(min_losss)
                vp_hls_performance_dict[(vp,hls)]['max_acc_index']=np.mean(max_acc_indexs)
                vp_hls_performance_dict[(vp,hls)]['min_loss_index']=np.mean(min_loss_indexs)
            print max_acc, best_acc_model
            print min_loss, best_loss_model

        y_accs=defaultdict(lambda:[])
        y_losss=defaultdict(lambda:[])
        for vp in vps:
            for hls in hlss:
                y_accs[vp].append(vp_hls_performance_dict[(vp,hls)]['max_acc'])
                y_losss[vp].append(vp_hls_performance_dict[(vp,hls)]['min_loss'])
        print y_accs
        print y_losss
        plt.clf()
        for vp in vps:
            plt.plot(hlss,y_accs[vp],label= 'vp_'+str(vp))
        plt.legend(loc=4)
        plt.xlabel('hidden layer size')
        plt.ylabel('accuracy')
        #plt.savefig('deep_param_exp_acc_'+archi+'.png')
        plt.savefig('deep_param_exp_acc_'+archi+'_'+init+'.pdf')
        plt.savefig('deep_param_exp_acc_'+archi+'_'+init+'.png')

        plt.clf()
        for vp in vps:
            plt.plot(hlss,y_losss[vp],label= 'vp_'+str(vp))
        plt.legend(loc=1)
        plt.xlabel('hidden layer size')
        plt.ylabel('cross entropy')
        #plt.savefig('deep_param_exp_loss_'+archi+'.png')
        plt.savefig('deep_param_exp_loss_'+archi+'_'+init+'.pdf')
        plt.savefig('deep_param_exp_loss_'+archi+'_'+init+'.png')
    
    #plt.clf()
    #plt.plot(hlss,y_losss[vp])
    #plt.savefig('deep_param_exp_'+str(vp)+'_loss.png')
#for vp in vps:
#    for hls in hlss:

#for key,val in vp_hls_performance_dict.items():
#    print key
#    print mean(val)
