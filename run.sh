#!/bin/bash

# default
log_path=$PWD/logs/`date +%s`".log"
data_path=$PWD/data/test_raw.csv

#while getopts d:l: option
#do
#case "${option}"
#in
#d) DATA=${OPTARG};;
#l) LOG=${OPTARG};;
#esac
#done

for i in "$@"
do
case $i in
    -d=*|--data_path=*)
    DATA="${i#*=}"
    shift # past argument=value
    ;;
    -l=*|--log_path=*)
    LOG="${i#*=}"
    shift # past argument=value
    ;;
    *)
          # unknown option
    ;;
esac
done

#if [ -n "$1" ]; then
#    data_path=$1
#fi

if [ -n "$DATA" ]; then
    data_path=$DATA
fi

if [ -n "$LOG" ]; then
    log_path=$LOG
fi

model_base_path=$PWD/saved_models
lm_model_path=$model_base_path/LM
coh1_model_path=$model_base_path/coh1
coh2_model_path=$model_base_path/coh2
#sent_model_path=$model_base_path/sentiment_analysis
sent_model_path=$model_base_path/sentiment_analysis/Model07
sent_char_mapping=$PWD/data/"char_mapping"
sent_word_mapping=$PWD/data/"word_mapping"

# get LM score
cd LM; python infer_scores.py --inference_data_path=$data_path --log_path=$log_path --model_dir=$lm_model_path; cd ..
# get sentiment score
# ---- word base version ----
#cd sentiment_analysis; python infer_scores.py --inference_data_path=$data_path --log_path=$log_path --model_dir=$sent_model_path --jieba_dict=$data_path/dict_fasttext.txt --mapping_path=$sent_word_mapping --model_type=rnn_last --sentence_cut_mode=word --vocab_size=50000; cd ..
# ---- char base version ----
cd sentiment_analysis; python infer_scores.py --inference_data_path=$data_path --log_path=$log_path --model_dir=$sent_model_path --mapping_path=$sent_char_mapping --sentence_cut_mode=char; cd ..
# get coh2 score
cd coh2; python infer_scores.py --inference_data_path=$data_path --log_path=$log_path  --model_dir=$coh2_model_path; cd ..
# get coh1 score
cd coh1; python infer_scores.py --inference_data_path=$data_path --log_path=$log_path --model_dir=$coh1_model_path; cd ..
echo "------------------" >> $log_path
