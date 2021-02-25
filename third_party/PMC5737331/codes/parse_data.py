from os import listdir
from os.path import isfile, join
from collections import defaultdict
import numpy as np
def zero_one_normalize(row):
    upper=max(row)
    lower=min(row)
    range=upper-lower
    if range==0:
        return None
    data=[(x-lower)/range for x in row]
    #print upper
    #print lower
    #print range
    #print data
    return data

def load_data(sample_label_dict,gene_sample_expression_dict,data_file_name,label_file_name, landmark=True):
    #data=[]
    
    #labels=[]
    #IDs=[]
    #genes=[]
    
    data_file_lines=open(data_file_name).readlines()
    label_file_lines=open(label_file_name).readlines()
    landmark_genes=load_landmark_gene_names()
    #landmark_genes.tolower()
    
    #print data_file_lines
    #print label_file_lines
    IDs=label_file_lines[0].split('\t')[1:-1]
    sample_label=label_file_lines[1].split('\t')[1:]
    #print len(IDs)
    #print len(sample_label)
    #print IDs[-1]
    #print sample_label[-1]
    for index,ID in enumerate(IDs):
        if ID not in sample_label_dict.keys():
            sample_label_dict[ID]=sample_label[index]
        else:
            print 'sample ID conflict: '+ID
    #for i,x in enumerate(label_sample):
    #   if x not in labels:
    #      labels.append(x)
    #   label_sample[i]=labels.index(x)
        
    #label=[IDs.index(x) for x in label]
    #print label
    #landmark_count=0
    IDs_expr=data_file_lines[0].replace('\n','').split('\t')[1:]
    for ID in IDs:
        if ID not in IDs_expr:
            del sample_label_dict[ID]
    
    #print IDs
    for line in data_file_lines[1:]:
        splits = line.replace('\n','').split('\t')
        gene=splits[0].lower()
        #print gene
        if not landmark or gene in landmark_genes.keys():
            if landmark:
                gene=landmark_genes[gene]
        #   print 'landmark: '+ gene
            #landmark_count+=1
            #print splits
            row=map(float,splits[1:])
            row=zero_one_normalize(row)
            if row != None:
                for index,ID in enumerate(IDs_expr):
                    if ID in IDs and row[index]>0:
                        gene_sample_expression_dict[(gene,ID)]=row[index]
                #data.append(row)
                #for index
            #print row
            #genes.append(gene)
        #else:
        #   print 'not landmark: '+gene
    #print landmark_count
    #data= np.array(data)
    #data=np.transpose(data)
    #label_sample= np.array(label_sample)
    return sample_label_dict,gene_sample_expression_dict
def load_landmark_gene_names():
    gene_names={}
    landmark_gene_file_lines=open('data/Landmark Genes n978.csv').readlines()[1:]
    for line in landmark_gene_file_lines:
        genes=line.split('\t')[:3]
        genes=[x.lower() for x in genes]
        #gene_names.append(line.split('\t')[:3])
        gene_names[genes[0]]=genes[1]
        gene_names[genes[1]]=genes[1]
        gene_names[genes[2]]=genes[1]
    return gene_names
def gen_nn_data(sample_label_dict,gene_sample_expression_dict):
    samples=sample_label_dict.keys()
    labels=list(set(sample_label_dict.values()))
    #print len(labels)
    #print labels
    genes=list(set([x[0] for x in gene_sample_expression_dict.keys()]))
    n_col = len(genes)
    n_row = len(samples)
    #samples_expr=list(set([x[1] for x in gene_sample_expression_dict.keys()]))
    #n_label= len(labels)
    data = np.zeros((n_row,n_col),dtype="float32")
    label = np.zeros((n_row,),dtype="uint8")
    for key,val in gene_sample_expression_dict.items():
        gene_index=genes.index(key[0])
        sample_index=samples.index(key[1])
        label_ID=labels.index(sample_label_dict[key[1]])
        #if label_ID > len(labels):
        #   print key[1]
            
        data[sample_index,gene_index]=val
        label[sample_index]=label_ID
    #for index in range(len(label)):
    #   if label[index]==0:
    #      print samples[index]
        
    return data,label

def load_integrated_data(filename,landmark=False,sample_normalize=True,gene_normalize=False,ref_gene_file=None,log_trans=False):
    all_data=[]
    all_sample_ID=[]
    labeled_sample_ID=[]
    labeled_data=[]
    unlabeled_data=[]
    label=[]
    gene_names=[]
    all_label=[]
    labeled_label=[]
    #print filename
    if ref_gene_file is not None:
        lines=open(ref_gene_file).readlines()
        group_genes=[]
        if len(lines)==1:
            group_genes=lines[0].split('\t')
        else:
            for line in lines:
                group_genes.append(line.replace('\n','').split('\t')[0].lower())
            #print group_genes
            #print len(group_genes)
    #print len(group_genes)
    #print group_genes
    lines=open(filename).readlines()
    Sample_ID=lines[0].replace('\n','').split('\t')[1:]
    labels=lines[1].replace('\n','').split('\t')[1:]
    all_weights=lines[2].replace('\n','').split('\t')[1:]
    all_sample_ID=Sample_ID
    #print Sample_ID.index('')
    #print labels.index('')
    #print len(Sample_ID)
    #print len(labels)
    #print len(all_weights)
    #print len(lines[4].split('\t')[1:])
    label_index=[i for i,val in enumerate(labels) if val!='None']
    unlabeled_index=[i for i,val in enumerate(labels) if val=='None']
    labeled_sample_ID=[all_sample_ID[i] for i,val in enumerate(labels) if val!='None']
    unlabeled_sample_ID=[all_sample_ID[i] for i,val in enumerate(labels) if val=='None']
    unlabeled_weights=[all_weights[i] for i,val in enumerate(labels) if val=='None']
    labeled_weights=[all_weights[i] for i,val in enumerate(labels) if val!='None']
    
    #print no_label_index
    label_unique_list=['None']+list(set([val for i,val in enumerate(labels) if val!='None']))
    #print labels
    #print label_unique_list
    for lab in labels:
        all_label.append(label_unique_list.index(lab))
        if lab!="None":
            labeled_label.append(label_unique_list.index(lab))
    #print label
    #print labels
    #print Sample_ID
    #for line in lines:
    #landmark_genes=load_landmark_gene_names()
    for line in lines[3:]:
        splits=line.replace('\n','').split('\t')
        gene=splits[0]
        #print gene
    #   if landmark and gene not in landmark_genes.keys():
    #       continue
        if ref_gene_file is not None and gene not in group_genes:
            continue
        gene_names.append(gene)
        #print splits[1:]
        all_data.append(splits[1:])
        #print len(splits)
    all_data=np.array(all_data,dtype='float32')
    #print all_data.shape
    if sample_normalize:
        for j in range(all_data.shape[1]):
            s=np.sum(all_data[:,j])
            if s ==0:
                s=0
                #print 'normalize sum==0: sample',j
            else:
                all_data[:,j]=all_data[:,j]/s*1000000
    if log_trans:
        all_data=np.log(all_data+1)
    if gene_normalize:
        for j in range(all_data.shape[0]):
            mean=np.mean(all_data[j,:])
            std=np.std(all_data[j,:])
            if std==0:
                std=0
                #print 'gene_normalize: std==0 data: ',j,mean,std
            else:
                all_data[j,:]=(all_data[j,:]-mean)/std
            #print all_data[j,:]
    labeled_data=np.zeros((all_data.shape[0],len(label_index)),dtype="float32")
    unlabeled_data=np.zeros((all_data.shape[0],len(unlabeled_index)),dtype="float32")
    count=0
    print len(label_index)
    print all_data.shape
    for i in label_index:
        labeled_data[:,count]=all_data[:,i]
        count+=1
    #print count
    count=0
    for i in unlabeled_index:
        unlabeled_data[:,count]=all_data[:,i]
        count+=1
    all_label=np.array(all_label)
    labeled_label=np.array(labeled_label)
    
    all_data=np.transpose(all_data)
    labeled_data=np.transpose(labeled_data)
    unlabeled_data=np.transpose(unlabeled_data)
    
    all_weights=np.transpose(all_weights)
    labeled_weights=np.transpose(labeled_weights)
    unlabeled_weights=np.transpose(unlabeled_weights)
    
    #print all_label.shape
    #print labeled_label.shape
    #print count
    #print labeled_data
    #print labeled_data.shape
    #print unlabeled_data.shape
    #print all_data
    #print all_data.shape
    #print gene_names
    return all_data, labeled_data,unlabeled_data,label_unique_list,all_label, labeled_label, all_weights, labeled_weights, unlabeled_weights,all_sample_ID,labeled_sample_ID,unlabeled_sample_ID,gene_names
def load_group_gene_index_dict(gene_names,group_file):
    group_lines=open(group_file).readlines()
    #print group_lines
    #print gene_names
    group_gene_index_dict=defaultdict(lambda:[])
    for line in group_lines:
        #print line
        splits=line.replace('\n','').replace('\r','').split('\t')
        #print splits
        gn=splits[0]
        for sp in splits[1:]:
            group_gene_index_dict[gn].append(gene_names.index(sp))
    m=max(map(max,group_gene_index_dict.values()))+1
    #print m
    sorted_group_names=sorted(group_gene_index_dict.keys())
    output_mat=np.zeros((len(sorted_group_names),m),dtype="float32")
    for index,key in enumerate(sorted_group_names):
    #   print key
        for d in group_gene_index_dict[key]:
            output_mat[index,d]=1
    #print output_mat
    return group_gene_index_dict, sorted_group_names, output_mat
if __name__=="__main__":
    '''
    sample_label_dict=defaultdict(lambda: None)
    gene_sample_expression_dict=defaultdict(lambda: 0)
    sample_label_dict,gene_sample_expression_dict=load_data(sample_label_dict,gene_sample_expression_dict,'mouse_10_expr_RPKM.txt','mouse_10_label.txt')    
    data,label=gen_nn_data(sample_label_dict,gene_sample_expression_dict)
    print data.shape
    print label.shape
    print label[:-1]
    print max(label)
    '''
    #all_data, labeled_data,unlabeled_data,label_unique_list,all_label, labeled_label,all_sample_ID,labeled_sample_ID,unlabeled_sample_ID,gene_names=load_integrated_data('data/mouse_1_4_6_7_10_16.txt')
    
    #all_data, labeled_data,unlabeled_data,label_unique_list,all_label, labeled_label, all_weights, labeled_weights, unlabeled_weights,all_sample_ID,labeled_sample_ID,unlabeled_sample_ID,gene_names=load_integrated_data('data/TPM_mouse_1_4_6_7_8_10_16.txt',whitening=True)   
    #all_data, labeled_data,unlabeled_data,label_unique_list,all_label, labeled_label, all_weights, labeled_weights, unlabeled_weights,all_sample_ID,labeled_sample_ID,unlabeled_sample_ID,gene_names=load_integrated_data('data/TPM_mouse_1_4_6_7_8_10_16.txt',sample_normalize=True,gene_normalize=True)   
    all_data, labeled_data,unlabeled_data,label_unique_list,all_label, labeled_label, all_weights, labeled_weights, unlabeled_weights,all_sample_ID,labeled_sample_ID,unlabeled_sample_ID,gene_names=load_integrated_data('../data/TPM_mouse_1_4_6_7_8_10_16.txt',sample_normalize=True,gene_normalize=True,ref_gene_file='../cluster_genes.txt')   
    print 'all_data.shape:',all_data.shape
    print 'labeled_data.shape:',labeled_data.shape
    count_dict=defaultdict(lambda:0)
    for lab in labeled_label:
        count_dict[lab]+=1
    print count_dict
    for key,val in count_dict.items():
        print label_unique_list[key]
        print val
    #print all_data[:5,:5]
    #group_gene_index_dict, sorted_group_names, output_mat= load_group_gene_index_dict(gene_names,'ppi_tf_merge_cluster.txt')
    #print group_gene_index_dict.values()
    #print 'len(sorted_group_names): ',len(sorted_group_names)
    #print 'output_mat.shape:',output_mat.shape
    #print all_weights
    #print labeled_weights
    #print unlabeled_weights
    #load_integrated_data('data/mouse_7_10.txt')
    
    #load_integrated_data('data/mouse_10_normalized_expr.txt')
    #load_integrated_data('data/mouse_4_normalized_expr.txt')
    #load_integrated_data('data/mouse_6_normalized_expr.txt')
    #load_integrated_data('data/mouse_7_normalized_expr.txt')
