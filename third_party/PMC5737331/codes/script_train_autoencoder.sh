
python deep_net.py \
-d important_file/integrate_imputing_dataset_kNN10_simgene.txt \
-ae 1 \
-hls 100 -s 100 -m 100 \
-a 1layer \
-act tanh \
-b 10 \
-sm 1 \
-sn 1 \
-gn 1 \
> log/ae100_log.txt

python deep_net.py \
-d important_file/integrate_imputing_dataset_kNN10_simgene.txt \
-ae 1 \
-hls 796 -s 100 -m 100 \
-a 1layer \
-act tanh \
-b 10 \
-sm 1 \
-sn 1 \
-gn 1 \
> log/ae796_log.txt

