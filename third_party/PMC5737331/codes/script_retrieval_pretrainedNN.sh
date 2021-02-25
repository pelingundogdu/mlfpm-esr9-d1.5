
python pca_clustering.py \
-nn 1layer_SN1_GN1_LOG0_BS32_hls100_mls696_seed0_pt_PPITF_classifier_merge100_tanh \
-retrieval 1 \
-i 100 \
-t labeled \
-ni 100 \
-sn 1 \
-gn 1 \
-ret_top 100 \
-ret_map 1 \
-o retrieval_analysis_PPITF_merge100_kNN10_top100_map_pt.csv \
-d important_file/integrate_imputing_dataset_kNN10_simgene.txt\
> retrieval_NN_PPITF_merge100_kNN10_simgene_top100_map.log

python pca_clustering.py \
-nn 3layer_SN1_GN1_LOG0_BS32_hls100_mls696_seed0_pt_PPITF_classifier_merge100_tanh \
-retrieval 1 \
-i 100 \
-t labeled \
-ni 100 \
-sn 1 \
-gn 1 \
-ret_top 100 \
-ret_map 1 \
-o retrieval_analysis_PPITF_3layer_merge100_kNN10_top100_map_pt.csv \
-d important_file/integrate_imputing_dataset_kNN10_simgene.txt\
> retrieval_NN_PPITF_3layer_merge100_kNN10_simgene_top100_map.log


python pca_clustering.py \
-nn 3layer_SN1_GN1_LOG0_BS32_hls100_mls796_seed0_pt_classifier_merge0_tanh \
-retrieval 1 \
-i 100 \
-t labeled \
-ni 100 \
-sn 1 \
-gn 1 \
-ret_top 100 \
-ret_map 1 \
-o retrieval_analysis_3layer_dense100_kNN10_top100_map_pt.csv \
-d important_file/integrate_imputing_dataset_kNN10_simgene.txt\
> retrieval_NN_3layer_dense100_kNN10_simgene_top100_map.log

python pca_clustering.py \
-nn 1layer_SN1_GN1_LOG0_BS32_hls100_mls796_seed0_pt_classifier_merge0_tanh \
-retrieval 1 \
-i 100 \
-t labeled \
-ni 100 \
-sn 1 \
-gn 1 \
-ret_top 100 \
-ret_map 1 \
-o retrieval_analysis_1layer_dense100_kNN10_top100_map_pt.csv \
-d important_file/integrate_imputing_dataset_kNN10_simgene.txt\
> retrieval_NN_1layer_dense100_kNN10_simgene_top100_map.log

python pca_clustering.py \
-nn 1layer_SN1_GN1_LOG0_BS32_hls796_mls696_seed0_pt_classifier_merge0_tanh \
-retrieval 1 \
-i 100 \
-t labeled \
-ni 100 \
-sn 1 \
-gn 1 \
-ret_top 100 \
-ret_map 1 \
-o retrieval_analysis_1layer_dense796_kNN10_top100_map_pt.csv \
-d important_file/integrate_imputing_dataset_kNN10_simgene.txt\
> retrieval_NN_1layer_dense796_kNN10_simgene_top100_map.log
