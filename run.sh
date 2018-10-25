#!/bin/bash

data_path=$PWD/data/test_raw.csv
log_path=$PWD/logs/`date +%s`".log"
model_base_path=$PWD/saved_models
lm_model_path=$model_base_path/LM
coh1_model_path=$model_base_path/coh1
coh2_model_path=$model_base_path/coh2
sent_model_path=$model_base_path/sentiment_analysis

# get LM score
cd LM; python infer_scores.py --inference_data_path=$data_path --log_path=$log_path --model_path=$lm_model_path; cd ..
# get sentiment score
cd sentiment_analysis; python infer_scores.py --inference_data_path=$data_path --log_path=$log_path --model_path=coh1_model_path; cd ..
# get coh2 score
cd coh2; python infer_scores.py --inference_data_path=$data_path --log_path=$log_path  --model_path=coh2_model_path; cd ..
# get coh1 score
cd coh1; python infer_scores.py --inference_data_path=$data_path --log_path=$log_path --model_path=coh1_model_path; cd ..
echo "------------------" >> $log_path
