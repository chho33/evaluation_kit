#!/bin/bash

data_path=$PWD/data/test_raw.csv
log_path=$PWD/logs/`date +%s`".log"

cd LM; python infer_scores.py --inference_data_path=$data_path --log_path=$log_path; cd ..
cd sentiment_analysis; python infer_scores.py --inference_data_path=$data_path --log_path=$log_path; cd ..
cd coh2; python infer_scores.py --inference_data_path=$data_path --log_path=$log_path; cd ..
cd coh1; python infer_scores.py --inference_data_path=$data_path --log_path=$log_path; cd ..
echo "------------------" >> $log_path
