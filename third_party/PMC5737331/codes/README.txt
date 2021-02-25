0. download data.zip from the website, decompress it and copy the following file to important_file/ folder and rename it
	1-3_integrated_NNtraining.txt -> TPM_mouse_7_8_10_PPITF_gene_9437.txt
	3-33_integrated_retrieval_set.txt -> integrate_imputing_dataset_kNN10_simgene.txt
1. See requirement.txt for the required python packages and other packages to be installed
2. basically, deep_net.py is used to train and store NN models
	see script_train_autoencoder.sh for training autoencoder
	see script_train_deep_model.sh for training NN
3. pca_clustering.py is used to do dimension reduction and then do clustering or retrieval task, it can also be used to generate visualization of reduced dimension
	see exp_clustering.Makefile for example usage
	see script_retrieval_* for how to do retrieval experiment
	see script_gen_reduced_dim.sh for how to generate visualization of reduced dimension
4. gen_clustering_exp.py is used to generate the experiment Makefile for clustering experiment
	usage: python gen_clustering_exp.py > exp_clustering.Makefile
	to do clustering experiment: make -j #core -f exp_clustering.Makefile
	example: make -j 1 -f exp_clustering.Makefile
5. get_clustering_result.py is used to integrate the output log file of exp_clustering.Makefile	
	usage: python get_clustering_result.py
6. deep_analyze.py is used to generate file for GO analysis of NN
	usage: python deep_analyze.py
7. after running deep_analyze.py, run GO_analysis_gprofiler.py to generate the output
	python GO_analysis_gprofiler.py > GO_analysis.sh
	sh GO_analysis.sh
8. gen_deep_param_exp.py is use to generate the experiment Makefile for NN parameter choosing
	usage: python gen_deep_param_exp.py > exp_deep_parameter.Makefile
	example: make -j 1 -f exp_deep_parameter.Makefile
9. get_deep_param_result.py is used to integrate the output log file of exp_deep_parameter.Makefile	
	usage: python get_deep_param_result.py

