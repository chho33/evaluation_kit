#!/bin/bash

data_path=$PWD/data/test_raw.csv
log_path=$PWD/logs/`date +%s`".log"
model_base_path=$PWD/saved_models
lm_model_path=$model_base_path/LM
coh1_model_path=$model_base_path/coh1
coh2_model_path=$model_base_path/coh2
sent_model_path=$model_base_path/sentiment_analysis
sent_char_mapping=$PWD/$data_path"char_mapping"
sent_word_mapping=$PWD/$data_path"word_mapping"

# get LM score
cd LM; python infer_scores.py --inference_data_path=$data_path --log_path=$log_path --model_dir=$lm_model_path; cd ..
# get sentiment score
cd sentiment_analysis; python infer_scores.py --inference_data_path=$data_path --log_path=$log_path --model_dir=$sent_model_path --jieba_dict=$data_path/dict_fasttext.txt --mapping_path=$sent_word_mapping --model_type=rnn_last --sentence_cut_mode=word --vocab_size=50000; cd ..
# get coh2 score
cd coh2; python infer_scores.py --inference_data_path=$data_path --log_path=$log_path  --model_dir=$coh2_model_path; cd ..
# get coh1 score
cd coh1; python infer_scores.py --inference_data_path=$data_path --log_path=$log_path --model_dir=$coh1_model_path; cd ..
echo "------------------" >> $log_path
