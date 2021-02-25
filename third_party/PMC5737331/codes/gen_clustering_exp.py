from  handle_other_algo_data import *

ppi_tf_cluster_file='ppi_tf_merge_cluster.txt'
#times=100
times=20
#times=1
#times=50
act='tanh'
max_iteration=100
step_size=100
sm=1
sn=1
#sn=0
gn=1
stc=1
#vcts=[4,6,8]
#vcts=[2,4,6,8]
vcts=[2,4,6,8]
#vcts=[2]
#vcts=[4]
#vcts=[6]
#vcts=[8]
seed=-1
output_all=['all:']
output_make=[]
pts=[0,1]
#pts=[0,1]
data_file= 'important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt'
#data_file= 'important_file/TPM_mouse_7_8_10_PPITF_gene_9437_DeNorm.txt'
params=[]# archi,fp,hiddenlayersize,middle layer size, merge
#1layer, dense, hidden layer size100,no middle layer, no merge
params.append(['1layer','',100,0,0])
#1layer, dense, hidden layer size 796,no middle layer, no merge
params.append(['1layer','',796,0,0])
#2layer, dense, hidden layer size100,middle layer size 796, no merge
params.append(['3layer','',100,796,0])
#1layer, PPITF, hidden layer size 0,no middle layer, merge 100
params.append(['1layer',ppi_tf_cluster_file,0,0,100])
#2layer, PPITF, hidden layer size 100 ,no middle layer, merge 100
params.append(['3layer',ppi_tf_cluster_file,100,0,100])

n_components=[2,5,10,50,100]

append_str='boot'
#append_str='boot_DeNorm'

count=0
each_time_size=len(vcts)*len(params)*len(pts)
for time in range(times):
    seed = time
    for vct in vcts:
        for pt in pts:
            for param in params:
                archi=param[0]
                fp=param[1]
                hls=param[2]
                mls=param[3]
                mer=param[4]

                #if archi=='3layer':
                #    if hidden_layer_size!=100:
                #        continue
                #elif fp!='' and hidden_layer_size!=696:
                #    continue
                append_ID=append_str+str(time)+'_'+str(count%each_time_size)
                #if count >1:
                #    exit(0)
                #hls=hidden_layer_size
                #if 1 and archi=='1layer' and fp=='':
                    #print 'Dense 1layer 696!!!!!!!!!!!'
                #    hls=696
                log_file='Dense'
                if fp!='':
                    log_file='PPITF'
                #elif mer>0:
                #    continue
                count+=1
                output_all.append('t'+str(count))
                log_file+='_'+archi
                if stc==1:
                    log_file+='_classifier'
                else:
                    log_file+='_autoencoder'
                log_file+='_hls'+str(hls)
                log_file+='_mls'+str(mls)
                log_file+='_SN'+str(sn)
                log_file+='_GN'+str(gn)
                log_file+='_sm'+str(sm)
                log_file+='_seed'+str(seed)
                log_file+='_pt'+str(pt)
                log_file+='_vct'+str(vct)
                #log_file+='_VP'+str(validation_percent)
                log_file+='_merge'+str(mer)
                log_file+='_'+str(act)
                log_file+='_'+append_ID
                log_file+='.log'
                output_make.append('.PHONY:t'+str(count))
                output_make.append('t'+str(count)+':')
               


                #print 'echo '+str(count)

                #output_make.append('\t echo '+str(count))
               #print 'python deep_net.py \\'
                output_make.append('\tpython deep_net.py \\')
                #output_make.append()
                output_make.append('\t-d '+data_file+" \\" )
                if fp!='':
                    #print '-p '+fp+' -r '+ref_gene_file+' \\'
                    output_make.append('\t-p '+fp+' \\')
                #else:
                    #print ' -r '+ref_gene_file+' \\'

                #print '-m '+ str(max_iteration)+' -hls '+str(hls)+" -mls " + str(middle_layer_size)+' -s '+str(step_size)+ ' \\'
                output_make.append('\t-m '+ str(max_iteration)+' -hls '+str(hls)+" -mls " + str(mls)+' -s '+str(step_size)+ ' \\' )
                #print '-a '+archi+" \\"
                output_make.append('\t-a '+archi+" \\" )
                #print '-vp '+ str(validation_percent)+" \\"
                #output_make.append('\t-vp '+ str(validation_percent)+" \\" )
                #print '-vct '+ str(vct)+" \\"
                output_make.append('\t-vct '+ str(vct)+" \\" )
                #print '-seed '+ str(seed)+" \\"
                output_make.append('\t-seed '+ str(seed)+" \\" )
                #print '-act '+act + ' \\'
                output_make.append('\t-act '+act + ' \\' )
                #print '-stc '+str(stc)+' \\'
                output_make.append('\t-stc '+str(stc)+' \\' )
                output_make.append('\t-merge '+str(mer)+' \\' )
                #print '-sm '+ str(sm)+' \\'
                output_make.append('\t-sm '+ str(sm)+' \\' )
                #print '-sn '+ str(sn)+' \\'
                output_make.append('\t-sn '+ str(sn)+' \\' )
                output_make.append('\t-pt '+ str(pt)+' \\' )
                #print '-gn '+ str(gn)+' \\'
                output_make.append('\t-gn '+ str(gn)+' \\' )
                #print '-ID '+ append_ID+' \\'
                output_make.append('\t-ID '+ append_ID+' \\' )
                #print '> log/'+log_file
                output_make.append('\t> log/'+log_file )
                mn=archi
                mn+='_SN'+str(sn)
                mn+='_GN'+str(gn)
                mn+='_LOG0'
                #mn+='_BS32'
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
                output_make.append('\tpython pca_clustering.py -nn '+mn+' -t labeled \\')
                output_make.append('\t-ni 100 \\')
                output_make.append('\t-d '+ data_file+' \\' )
                output_make.append('\t-seed '+str(seed)+' \\')
                output_make.append('\t-vct '+str(vct)+' \\')
                output_make.append('\t-ID '+ append_ID+' \\' )
                output_make.append('\t> clustering_log/'+mn+'_labeled_c0_cut0_100.log')
                if not (param==params[-1] and pt==pts[-1]):
                
                    seedFileName='seed'+str(seed)+'_vct'+str(vct)+'_'+append_ID+'.pickle'
                    output_make.append('\t rm '+seedFileName)
                    output_make.append('\t rm model/'+mn+'.*')

        output_make.append('\tpython pca_clustering.py \\')
        output_make.append('\t-ori 1 \\')
        output_make.append('\t-d '+ data_file+' \\' )
        output_make.append('\t-seed '+str(seed)+' \\')
        output_make.append('\t-vct '+str(vct)+' \\')
        output_make.append('\t-ID '+ append_ID+' \\' )
        output_make.append('\t> clustering_log/pca_vct'+str(vct)+'_seed'+str(seed)+'_ori_c0_'+append_ID+'.log')
        
        methods=['pca','tsne','ica','nmf']
        for nc in n_components:
            for meth in methods:
                if meth == 'tsne' and nc >=50:
                    continue
                if meth == 'ica' and nc >=50:
                    continue
                output_make.append('\tpython pca_clustering.py \\')
                output_make.append('\t-fs 1 \\')
                if meth != 'pca':
                    output_make.append('\t-'+meth+' 1 \\')
                output_make.append('\t-d '+ data_file+' \\' )
                output_make.append('\t-n '+str(nc)+' \\' )
                output_make.append('\t-seed '+str(seed)+' \\')
                output_make.append('\t-vct '+str(vct)+' \\')
                output_make.append('\t-ID '+ append_ID+' \\' )
                output_make.append('\t> clustering_log/'+meth+'_vct'+str(vct)+'_fit_test_seed'+str(seed)+'_c0_n'+str(nc)+'_'+append_ID+'.log')

        '''
        output_make.append('\tpython pca_clustering.py \\')
        output_make.append('\t-fs 1 \\')
        output_make.append('\t-d '+ data_file+' \\' )
        output_make.append('\t-n 100 \\' )
        output_make.append('\t-seed '+str(seed)+' \\')
        output_make.append('\t-vct '+str(vct)+' \\')
        output_make.append('\t-ID '+ append_ID+' \\' )
        output_make.append('\t> clustering_log/pca_vct'+str(vct)+'_fit_test_seed'+str(seed)+'_c0_n100_'+append_ID+'.log')
        
        output_make.append('\tpython pca_clustering.py \\')
        output_make.append('\t-fs 1 \\')
        output_make.append('\t-d '+ data_file+' \\' )
        output_make.append('\t-n 796 \\' )
        output_make.append('\t-seed '+str(seed)+' \\')
        output_make.append('\t-vct '+str(vct)+' \\')
        output_make.append('\t-ID '+ append_ID+' \\' )
        output_make.append('\t> clustering_log/pca_vct'+str(vct)+'_fit_test_seed'+str(seed)+'_c0_n796_'+append_ID+'.log')
        '''
        seedFileName='seed'+str(seed)+'_vct'+str(vct)+'_'+append_ID+'.pickle'
        output_make.append('\tpython SIMLR/SIMLR_largescale.py '+seedFileName+' \\')
        output_make.append('\t> clustering_log/'+seedFileName+"_SIMLR_ARI.txt")



        seedFileName='seed'+str(seed)+'_vct'+str(vct)+'_'+append_ID+'.pickle'
        SNNinputfileName=seedFileName+'_SNNinput.txt'
        SNNedgefileName=seedFileName+'_SNNedge.txt'
        SNNIDXfileName=seedFileName+'_SNNIDX.txt'
        
        output_make.append('\tpython handle_other_algo_data.py 1 '+seedFileName+' '+SNNinputfileName)
        if vct>2:
            output_make.append('\tRscript run_SNN-Cliq.R '+SNNinputfileName+' '+SNNedgefileName+' '+str(vct))
            output_make.append('\tpython Cliq.py -i '+SNNedgefileName+' -o '+SNNIDXfileName)
            output_make.append('\tpython handle_other_algo_data.py 2 '+SNNIDXfileName+' '+SNNinputfileName+'_Answer.txt clustering_log/'+SNNinputfileName+"_ARI.txt"+' '+seedFileName)
        pcaReduceOutfile=seedFileName+'_pcaReduce_clus.txt'
        pcaReduceARIfile=seedFileName+'_pcaReduce_ARI.txt'
        output_make.append('\tRscript pcaReduce.R '+str(vct)+" "+SNNinputfileName+' '+SNNinputfileName+'_Answer.txt '+pcaReduceOutfile)
        output_make.append('\tpython handle_other_algo_data.py 2 '+pcaReduceOutfile+' '+SNNinputfileName+'_Answer.txt clustering_log/'+pcaReduceARIfile+' '+seedFileName)
        
        #sincera_cluster_algs=['hc','consensus','tightclust']
        sincera_cluster_algs=['hc']
        #sincera_cluster_algs=['consensus']
        #sincera_cluster_algs=['tightclust']
        for alg in sincera_cluster_algs:
            sinceraOutfile=seedFileName+'_sincera_'+alg+'_clus.txt'
            sinceraARIfile=seedFileName+'_sincera_'+alg+'.txt'
            output_make.append('\tRscript sincera_clustering.R '+str(vct)+" "+alg+' '+SNNinputfileName+' '+sinceraOutfile)
            output_make.append('\tpython handle_other_algo_data.py 2 '+sinceraOutfile+' '+SNNinputfileName+'_Answer.txt clustering_log/'+sinceraARIfile+' '+seedFileName)
        output_make.append('\t rm '+SNNinputfileName)
        output_make.append('\t rm '+seedFileName)
            
print ' '.join(output_all)
print '\n'.join(output_make)
