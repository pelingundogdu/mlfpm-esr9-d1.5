import sys
import os
from collections import defaultdict

filenames=[]
filenames.append('deep_ana0.08.csv')
dir1='GO_analysis_gprofiler_results'
os.system('mkdir '+dir1)
for filename in filenames:
    file_lines=open(filename).readlines()
    cell_type_group_dict={}
    group_cell_type_dict=defaultdict(lambda:[])
    dir2=dir1+'/'+filename
    os.system('mkdir '+dir2)
    dir3=dir2+'/all_groups'
    os.system('mkdir '+dir3)
    for line in file_lines[1:17]:
        line = line.replace('\n','')
        splits = line.split(',')
        cell_type = splits[0]
        cell_type_dir=dir2+'/'+cell_type
        os.system('mkdir '+cell_type_dir)
        groups = splits[2:]
        cell_type_group_dict[cell_type]=groups
        for gp in groups:
            group_cell_type_dict[gp].append(cell_type)
    for line in file_lines[18:]:
        line = line.replace('\n','')
        splits = line.split(',')
        group = splits[0]
        count = int(splits[1])
        genes = splits[2:]
        if group.startswith('TF'):
            continue
        head=["query", "significant", "p_value", "T", "Q", "Q&T", "precision", "recall", "term_id","domain", "group", "description", "depth", "intersection", "evcodes"]
        out_file=dir3+'/'+group+'.txt' 
        cmd = 'printf "'+'\\t'.join(head)+'\\n" > '+out_file
        print cmd
        cmd = 'python ~/gprofiler-official-0.2.3/gprofiler.py -o mmusculus "'+ ' '.join(genes) + '" >> '+out_file
        print cmd
        for ct in group_cell_type_dict[group]:
            cell_type_dir=dir2+'/'+ct
            cmd = 'cp '+out_file+' '+cell_type_dir
            print cmd
