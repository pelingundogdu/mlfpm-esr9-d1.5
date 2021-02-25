
import pickle
import numpy as np

from sklearn.metrics.cluster import adjusted_rand_score
from sklearn import metrics
import sys

def gen_SNNCliq_data_file(fileName,outFileName):

    with open(fileName, 'rb') as handle:
        output_dict = pickle.load(handle)
    #transform_data=output_dict['test_X_TPMgn0']
    transform_data=output_dict['test_X']
    valid_Y=output_dict['test_Y_array']
    data=np.transpose(transform_data)
    nrow=data.shape[0]
    ncol=data.shape[1]
    #outC=open(outFileName+'_cell.txt','w')
    #outC.write('Cell\tSample\n')
    outF=open(outFileName,'w')
    outF.write('Sample')
    for i in range(ncol):
        outF.write('\tS'+str(i))
        #outC.write('S'+str(i)+'\t'+str(i)+'\n')
    outF.write('\n')
    for i in range(nrow):
        out=map(str,data[i])
        outF.write('Gene'+str(i)+'\t'+'\t'.join(out)+'\n')
    outA=open(outFileName+'_Answer.txt','w')
    outA.write('\t'.join(map(str,valid_Y))+"\n")
def gen_SNNCliq_ARI_file(SNNIDXfile,answerfile,outFileName,dataFileName=None):
    SNN_lines=open(SNNIDXfile).readlines()
    SNNs=[]
    for line in SNN_lines:
        SNNs.append(line.replace('\n',''))
    print SNNs
    print len(SNNs)
    answer=open(answerfile).readlines()[0].replace('\n','').split('\t')
    print answer
    print len(answer)
    #ARI= adjusted_rand_score(SNNs,answer)
    if dataFileName:
        with open(dataFileName, 'rb') as handle:
            output_dict = pickle.load(handle)
        #transform_data=output_dict['test_X_TPMgn0']
        data=output_dict['test_X']
    score={}
    score['homogeneity_score'] = metrics.homogeneity_score(answer,SNNs)
    score['completeness_score'] = metrics.completeness_score(answer,SNNs)
    score['v_measure_score'] = metrics.v_measure_score(answer,SNNs)
    score['adjusted_rand_score'] = metrics.adjusted_rand_score(answer,SNNs)
    score['adjusted_mutual_info_score'] = metrics.adjusted_mutual_info_score(answer,SNNs)
    score['silhouette_score'] = metrics.silhouette_score(data, SNNs)
    score['fowlkes_mallows_score'] = metrics.fowlkes_mallows_score(answer,SNNs)
    #print ARI
    print score
    outfile = open(outFileName,'w')
    for key,val in score.items():
        outfile.write(key+'\t'+str(val)+'\n')

if __name__=='__main__':
    print sys.argv 
    if sys.argv[1]=='1':
        #gen_SNNCliq_data_file('seed-1_vct8_boot0_1.pickle','test.output')
        gen_SNNCliq_data_file(sys.argv[2],sys.argv[3])
    else:
        #gen_SNNCliq_ARI_file('IDX.txt','test.output_Answer.txt','ARI.txt')
        gen_SNNCliq_ARI_file(sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5])
    
