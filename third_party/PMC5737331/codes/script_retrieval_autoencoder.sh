
python pca_clustering.py \
-nn  1layer_SN1_GN1_LOG0_BS32_hls796_mls696_seed0_autoencoder_merge0_tanh \
-retrieval 1 \
-i 100 \
-t labeled \
-ni 100 \
-sn 1 \
-gn 1 \
-ret_map 1 \
-ret_top 100 \
-o retrieval_analysis_autoencoder796_knn10_top100_map.csv \
-d important_file/integrate_imputing_dataset_kNN10_simgene.txt \
> retrieval_nn_autoencoder796_knn10_simgene_top100_map.log

python pca_clustering.py \
-nn  1layer_SN1_GN1_LOG0_BS32_hls100_mls696_seed0_autoencoder_merge0_tanh \
-retrieval 1 \
-i 100 \
-t labeled \
-ni 100 \
-sn 1 \
-gn 1 \
-ret_map 1 \
-ret_top 100 \
-o retrieval_analysis_autoencoder100_knn10_top100_map.csv \
-d important_file/integrate_imputing_dataset_kNN10_simgene.txt \
> retrieval_nn_autoencoder100_knn10_simgene_top100_map.log

