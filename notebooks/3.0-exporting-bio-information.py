# #!/usr/bin/env python
# # coding: utf-8

import os

# UPLOAD geneSCF databaser
# !wget -P ../source/ "http://genescf.kandurilab.org/downloads.php/geneSCF-master-v1.1-p2.tar.gz"
# get_ipython().system('wget -P ../source/ "http://genescf.kandurilab.org/downloads.php/geneSCF-master-v1.1-p2.tar.gz"')
# # Unzip folder

# # ~10 minutes
# ! ../source/geneSCF-master-source-v1.1-p2/prepare_database -db=KEGG -org=mmu
# get_ipython().system(' ../source/geneSCF-master-source-v1.1-p2/prepare_database -db=KEGG -org=mmu')

# # !mv ../source/NAME_OF_FOLDER/ ../source/geneSCF
# !mv ../source/geneSCF-master-source-v1.1-p2/ ../source/geneSCF
# get_ipython().system('mv ../source/geneSCF-master-source-v1.1-p2/ ../source/geneSCF')


# TO INSTALL REQUESTED LIBRARIES and TO UPDATE OLD PACKAGES
# If you need to update old package uncomment update.packages(ask = FALSE) line.
# os.system('Rscript install_and_update_R_packages.r')

if (os.path.exists('../data/pathways/')==False):
    os.mkdir('../data/pathways/')

if (os.path.exists('../data/weights/')==False):
    os.mkdir('../data/weights/')

if (os.path.exists('../data/hipathia_pathways/')==False):
    os.mkdir('../data/hipathia_pathways/')
    
if (os.path.exists('../data/hipathia_genes_detail/')==False):
    os.mkdir('../data/hipathia_genes_detail/')

# PATHWAY INFROMATION    
print('EXPORTING PATHWAY INFORMATIONS')
os.system('python pathway_kegg_find_disease.py')

os.system('Rscript pathway_hipathia_export.r')

os.system('python pathway_all_export_final_versions.py')

# GENE INFORMATION
print('EXPORTING GENE INFORMATIONS')
os.system('Rscript gene_hipathia_export.r')

os.system('python gene_kegg_export.py')

os.system('Rscript gene_all_export.r')

print('PATHWAY and GENE INFORMATION EXPORTED!!')