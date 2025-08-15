#!/usr/bin/bash

config=../config/config-bertweet.txt
test_data=../data/test.csv

###################################################################################################################
model_dir=./trained_models/downloaded
for seed in {1..3}
do
    echo "Start evaluation on seed ${seed}......"
    python stance_eval.py -s ${seed} -c ${config} -test ${test_data} -mod_dir ${model_dir} -m bertweet
done
###################################################################################################################

exit 0
###################################################################################################################
model_dir=./trained_models/bertweet_multitask

for seed in 0 112 343
do
    echo "Start evaluation on seed ${seed}......"
    python stance_eval.py -s ${seed} -c ${config} -test ${test_data} -mod_dir ${model_dir} -m bertweet
done
###################################################################################################################




