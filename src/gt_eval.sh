#!/usr/bin/bash

###################################################################################################################
config=../config/config-bertweet.txt
model_dir=./trained_models/downloaded
for seed in {1..3}
do
    test_data=../data/gtpreds_test_all_onecol.csv
    echo "Start evaluation on seed ${seed}......"
    python eval.py -s ${seed} -c ${config} -test ${test_data} -mod_dir ${model_dir} -m bertweet
done

exit 0

###################################################################################################################
config=../config/config-bertweet.txt
model_dir=./trained_models/bertweet_multitask

for seed in 0 112 343
do
    test_data=../data/gtpreds_test_all_onecol.csv
    echo "Start evaluation on seed ${seed}......"
    python eval.py -s ${seed} -c ${config} -test ${test_data} -mod_dir ${model_dir} -m bertweet
done
