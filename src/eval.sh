#!/usr/bin/bash

###################################################################################################################
config=../config/config-bertweet.txt
model_dir=./trained_models/bertweet_multitask

predictions_dir=../target-classification/output/targets_bertweet_bad_regex
for seed in 0 112 343
do
    test_data=${predictions_dir}/predictions_Bertweet_seed_${seed}.csv
    echo "Start evaluation on seed ${seed}......"
    python eval.py -s ${seed} -c ${config} -test ${test_data} -mod_dir ${model_dir} -m bertweet
done
