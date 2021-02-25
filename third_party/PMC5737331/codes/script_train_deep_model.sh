
#python deep_net.py \
#-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
#-m 100 -hls 100 -mls 796 -s 100 \
#-a 3layer \
#-vp 0 \
#-act tanh \
#-stc 1 \
#-sm 1 \
#-sn 1 \
#-gn 1 \

#python deep_net.py \
#-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
#-m 100 -hls 100 -mls 796 -s 100 \
#-a 1layer \
#-vp 0 \
#-act tanh \
#-stc 1 \
#-sm 1 \
#-sn 1 \
#-gn 1 \

#python deep_net.py \
#-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
#-m 100 -hls 796 -mls 696 -s 100 \
#-a 1layer \
#-vp 0 \
#-act tanh \
#-stc 1 \
#-sm 1 \
#-sn 1 \
#-gn 1 \

python deep_net.py \
-p ppi_tf_merge_cluster.txt \
-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
-m 100 -hls 100 -mls 696 -s 100 \
-a 3layer \
-vp 0 \
-act tanh \
-stc 1 \
-sm 1 \
-sn 1 \
-gn 1 \
-merge 100 \

python deep_net.py \
-p ppi_tf_merge_cluster.txt \
-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
-m 100 -hls 100 -mls 696 -s 100 \
-a 1layer \
-vp 0 \
-act tanh \
-stc 1 \
-sm 1 \
-sn 1 \
-gn 1 \
-merge 100 \

python deep_net.py \
-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
-m 100 -hls 100 -mls 796 -s 100 \
-a 3layer \
-vp 0 \
-act tanh \
-pt 1 \
-stc 1 \
-sm 1 \
-sn 1 \
-gn 1 \

python deep_net.py \
-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
-m 100 -hls 100 -mls 796 -s 100 \
-a 1layer \
-vp 0 \
-act tanh \
-pt 1 \
-stc 1 \
-sm 1 \
-sn 1 \
-gn 1 \

python deep_net.py \
-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
-m 100 -hls 796 -mls 696 -s 100 \
-a 1layer \
-vp 0 \
-act tanh \
-pt 1 \
-stc 1 \
-sm 1 \
-sn 1 \
-gn 1 \

python deep_net.py \
-p ppi_tf_merge_cluster.txt \
-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
-m 100 -hls 100 -mls 696 -s 100 \
-a 3layer \
-vp 0 \
-act tanh \
-pt 1 \
-stc 1 \
-sm 1 \
-sn 1 \
-gn 1 \
-merge 100 \

python deep_net.py \
-p ppi_tf_merge_cluster.txt \
-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
-m 100 -hls 100 -mls 696 -s 100 \
-a 1layer \
-vp 0 \
-act tanh \
-pt 1 \
-stc 1 \
-sm 1 \
-sn 1 \
-gn 1 \
-merge 100 \






