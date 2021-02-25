
hlss=[2,5,10,30,50,80,100,150,200]
#hlss=[2,5,10]
times = 10
vps=[0.1,0.2]
#vps=[0.1]
max_iteration=100
stc=1
seed=-1
output_all=['all:']
output_make=[]
count=0
archis=['dense','PPITF']
#archis=['PPITF']
inits=['glorot_uniform','glorot_normal','he_normal','he_uniform']
data_file= 'important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt'
PPITF_file='ppi_tf_merge_cluster.txt'
for archi in archis:
    for init in inits:
        for vp in vps:
            for hls in hlss:
                for _ in range(times):
                    count+=1
                    log_file='1layer_'+archi
                    log_file+='_init_'+init
                    log_file+='_vp'+str(vp)
                    log_file+='_hls'+str(hls)
                    log_file+='_maxit'+str(max_iteration)
                    log_file+='_'+str(_)
                    log_file+='.log'

                    output_all.append('t'+str(count))
                    output_make.append('.PHONY:t'+str(count))
                    output_make.append('t'+str(count)+':')
                    output_make.append('\tpython deep_net.py \\')
                    if archi=='PPITF':
                        output_make.append('\t-p '+PPITF_file+" \\" )
                        output_make.append('\t-merge '+ str(hls)+ ' \\' )
                    output_make.append('\t-d '+data_file+" \\" )
                    output_make.append('\t-init '+init+" \\" )
                    output_make.append('\t-m '+ str(max_iteration)+ ' \\' )
                    output_make.append('\t-hls '+ str(hls)+ ' \\' )
                    output_make.append('\t-stc '+ str(stc)+ ' \\' )
                    output_make.append('\t-vp '+ str(vp)+ ' \\' )
                    output_make.append('\t> log_deep_param/'+log_file )        
print ' '.join(output_all)
print '\n'.join(output_make)
