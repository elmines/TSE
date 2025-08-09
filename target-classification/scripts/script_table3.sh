#!/usr/bin/bash


for seed in 0 112 342; do
    for model in "Bertweet"; do
        python train_model.py  --dataset=Stance_Merge_Unrelated --model_select=$model --seed=$seed
    done
done

# Get results for individual datasets
python utils/eval_utils_outputs.py