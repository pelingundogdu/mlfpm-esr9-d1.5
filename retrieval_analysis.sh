###############################################        PCA - RETRIEVAL ANALYSIS         ###############################################
date
# with no sample wise normalization
## PCA - 100
## DEFAULT GENE LIST
python notebooks/retrieval.py pca 100 pca 0 all default
## SIGNALING GENE LIST
python notebooks/retrieval.py pca 100 pca_signaling 0 data/weights/gene_signaling_retrieval.txt signaling
## METABOLIC and SIGNALING GENE LIST
python notebooks/retrieval.py pca 100 pca_met_sig 0 data/weights/gene_met_sig_retrieval.txt metabolic_signaling

## PCA - 796
## DEFAULT GENE LIST
python notebooks/retrieval.py pca 796 pca 0 all default
## SIGNALING GENE LIST
python notebooks/retrieval.py pca 796 pca_signaling 0 data/weights/gene_signaling_retrieval.txt signaling
## METABOLIC and SIGNALING GENE LIST
python notebooks/retrieval.py pca 796 pca_met_sig 0 data/weights/gene_met_sig_retrieval.txt metabolic_signaling

# with sample wise normalization
## PCA - 100
## DEFAULT GENE LIST
python notebooks/retrieval.py pca 100 pca 1 all default
## SIGNALING GENE LIST
python notebooks/retrieval.py pca 100 pca_signaling 1 data/weights/gene_signaling_retrieval.txt signaling
## METABOLIC and SIGNALING GENE LIST
python notebooks/retrieval.py pca 100 pca_met_sig 1 data/weights/gene_met_sig_retrieval.txt metabolic_signaling

## PCA - 796
## DEFAULT GENE LIST
python notebooks/retrieval.py pca 796 pca 1 all default
## SIGNALING GENE LIST
python notebooks/retrieval.py pca 796 pca_signaling 1 data/weights/gene_signaling_retrieval.txt signaling
## METABOLIC and SIGNALING GENE LIST
python notebooks/retrieval.py pca 796 pca_met_sig 1 data/weights/gene_met_sig_retrieval.txt metabolic_signaling

############################################### ARCHITECTURE P,A,B - RETRIEVAL ANALYSIS ###############################################

# with no sample wise normalization
## DEFAULT GENE LIST
python notebooks/retrieval.py data/NN_result/models_no_co_False/ 0 saved_model 0 all default
## SIGNALING GENE LIST
python notebooks/retrieval.py data/NN_result/models_no_co_False/ 0 saved_model 0 data/weights/gene_signaling_retrieval.txt signaling
## METABOLIC and SIGNALING GENE LIST
python notebooks/retrieval.py data/NN_result/models_no_co_False/ 0 saved_model 0 data/weights/gene_met_sig_retrieval.txt metabolic_signaling

# with sample wise normalization
## DEFAULT GENE LIST
python notebooks/retrieval.py data/NN_result/models_no_co_True/ 0 saved_model 1 all default
## SIGNALING GENE LIST
python notebooks/retrieval.py data/NN_result/models_no_co_True/ 0 saved_model 1 data/weights/gene_signaling_retrieval.txt signaling
## METABOLIC and SIGNALING GENE LIST
python notebooks/retrieval.py data/NN_result/models_no_co_True/ 0 saved_model 1 data/weights/gene_met_sig_retrieval.txt metabolic_signaling

###############################################    KERAS TUNER - RETRIEVAL ANALYSIS     ###############################################

# with no sample wise normalization
## DEFAULT GENE LIST
python notebooks/retrieval.py data/kt_result/models_no_co_False/ 0 saved_model 0 all default
## SIGNALING GENE LIST
python notebooks/retrieval.py data/kt_result/models_no_co_False/ 0 saved_model 0 data/weights/gene_signaling_retrieval.txt signaling
## METABOLIC and SIGNALING GENE LIST
python notebooks/retrieval.py data/kt_result/models_no_co_False/ 0 saved_model 0 data/weights/gene_met_sig_retrieval.txt metabolic_signaling

# with sample wise normalization
## DEFAULT GENE LIST
python notebooks/retrieval.py data/kt_result/models_no_co_True/ 0 saved_model 1 all default
## SIGNALING GENE LIST
python notebooks/retrieval.py data/kt_result/models_no_co_True/ 0 saved_model 1 data/weights/gene_signaling_retrieval.txt signaling
## METABOLIC and SIGNALING GENE LIST
python notebooks/retrieval.py data/kt_result/models_no_co_True/ 0 saved_model 1 data/weights/gene_met_sig_retrieval.txt metabolic_signaling

###############################################    AUTOENCODER - RETRIEVAL ANALYSIS     ###############################################

# with no sample wise normalization
## DEFAULT GEE LIST
python notebooks/retrieval.py data/AE_result/models_no_co_False/ 0 saved_model 0 all default
## SIGNALING GENE LIST
python notebooks/retrieval.py data/AE_result/models_no_co_False/ 0 saved_model 0 data/weights/gene_signaling_retrieval.txt signaling
## METABOLIC and SIGNALING GENE LIST
python notebooks/retrieval.py data/AE_result/models_no_co_False/ 0 saved_model 0 data/weights/gene_met_sig_retrieval.txt metabolic_signaling

# with sample wise normalization
## DEFAULT GEE LIST
python notebooks/retrieval.py data/AE_result/models_no_co_True/ 0 saved_model 1 all default
## SIGNALING GENE LIST
python notebooks/retrieval.py data/AE_result/models_no_co_True/ 0 saved_model 1 data/weights/gene_signaling_retrieval.txt signaling
## METABOLIC and SIGNALING GENE LIST
python notebooks/retrieval.py data/AE_result/models_no_co_True/ 0 saved_model 1 data/weights/gene_met_sig_retrieval.txt metabolic_signaling

date