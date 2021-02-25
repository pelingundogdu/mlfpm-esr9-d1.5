#!/bin/bash
#
# -p gpu			  # partition (queue)
#SBATCH -p zbj1-bigmem			  # partition (queue)
# --gres=gpu:2		  # gpu resource
#SBATCH -N 1                      # number of nodes
#SBATCH -n 32                     # number of cores
#SBATCH -o slurm.%N.%j.out        # STDOUT
#SBATCH -e slurm.%N.%j.err        # STDERR
#SBATCH --mail-type=END,FAIL      # notifications for job done & fail
#SBATCH --mail-user=jessica1338@gmail.com # send-to address
#SBATCH --mem 100000

#module load cuda-8.0


python pca_clustering.py -nn 3layer_SN1_GN1_LOG0_BS10_hls100_mls696_seed0_PPITF_classifier_merge100_tanh -plot pca -ni 100 -plot_name nn_PPITF_2layer
python pca_clustering.py -nn 3layer_SN1_GN1_LOG0_BS10_hls100_mls696_seed0_PPITF_classifier_merge100_tanh -plot tsne -ni 100 -plot_name nn_PPITF_2layer
python pca_clustering.py -n 2 -plot pca  -plot_name pca2
python pca_clustering.py -n 2 -plot pca  -plot_name tsne2 -tsne 1
python pca_clustering.py -n 2 -plot pca  -plot_name ica2 -ica 1
python pca_clustering.py -n 2 -plot pca  -plot_name nmf2 -nmf 1
python pca_clustering.py -SIMLR 1 -plot pca -plot_name SIMLR_pca2

