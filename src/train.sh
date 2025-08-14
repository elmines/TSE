#!/bin/bash

config=../config/config-bertweet.txt
train_data=../data/train.csv
dev_data=../data/val.csv
test_data=../data/test.csv

# for seed in 0 112 343
# do
#     echo "Start training on seed ${seed}......"
#     python train.py -s ${seed} -c ${config} -train ${train_data} -dev ${dev_data} -test ${test_data} -mod_dir ./trained_models/bertweet_multitask -m bertweet -mul
# done

for seed in 0 112 343
do
    echo "Start training on seed ${seed}......"
    python train.py -s ${seed} -c ${config} -train ${train_data} -dev ${dev_data} -test ${test_data} -mod_dir ./trained_models/bertweet_singletask -m bertweet
done
