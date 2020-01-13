#!/bin/bash

# make needed dirs
./make_dirs.sh

# download data and models
data_path="data/"
coh1_source_mapping=$data_path"source.150000.mapping","1I1SiWYmtvSQSmI-JiXoqcb1az1GjeKzl"
coh1_target_mapping=$data_path"target.6185.mapping","16Q-GAJ-JwZRLJKZiWKbgcVvcaLJH8p3Z"
fasttext_npy=$data_path"fasttext.npy.zip","1p3ZpcBeZcpIjMmbx3aD0QWEks-TtrPSP"
dict_fasttext=$data_path"dict_fasttext.txt.gz","1jk6pjs535ujn4SQkK5GTwy_LWaLbyMto"
sentiment_char_mapping=$data_path"char_mapping","1CenvR6GH8qaSn2evOA3_T6rSn0EUDfUN"
sentiment_word_mapping=$data_path"word_mapping","1lqdsfeBRgc4YHO3go4txu9OKXa7cvjBh"
lm_mapping=$data_path"words_char.json.gz","1iMD5KAgH4Ro8JqLo_V3XCGLGwz9nsTaw"

model_path="saved_models/"
coh1_model=$model_path"coh1/coh1.tar.gz","1FkVBItYo4ra9B-yF76o3h3WCQ9lLSs7p"
coh2_model=$model_path"coh2/coh2.tar.gz","1rJUDPJ8nng-vKUNuaSbSGlz1Ye05qOj4"
lm_model=$model_path"LM/LM.tar.gz","16vZJvf5_NqFabcKITOjyVYeAKfryb6Ei"
sent_word_model=$model_path"sentiment_analysis/sentiment_word.zip","1oA9WuYa-jHCimMYRElC7qVuOggrBdaE9"
sent_char_model=$model_path"sentiment_analysis/sentiment_char.tar.gz","112GPe7_tIoqKQwcgiXBgh6FeFPn7-ZK8"


#files_arr=($coh1_source_mapping $coh1_target_mapping $fasttext_npy $dict_fasttext $sentiment_char_mapping $sentiment_word_mapping $lm_mapping $coh1_model $coh2_model $lm_model $sent_word_model $sent_char_model)
files_arr=($coh1_source_mapping $coh1_target_mapping $dict_fasttext $sentiment_char_mapping $sentiment_word_mapping $lm_mapping $coh2_model $sent_word_model $sent_char_model)
for f in ${files_arr[@]}; do
    IFS=',' read filename fileid <<< "${f}"
    download_url="https://drive.google.com/uc?id=${fileid}&export=download"
    echo ${download_url}
    curl -L -o "${filename}" "${download_url}" 
    #download_from_gdrive "$fileid" "$filename"
    echo "${filename}" ":" "${download_url} downloaded!"
    if [[ ${filename} = *"zip"* ]]; then
        unzip ${filename}
        rm ${filename}
    elif [[ ${filename} = *"gz"* ]]; then
        gunzip ${filename}
    elif [[ ${filename} = *".tar.bz"* ]]; then
        tar jxvf ${filename}
        rm ${filename} --strip 1
    elif [[ ${filename} = *".tar.gz"* ]]; then
        tar zxvf ${filename} --strip 1
        rm ${filename}
    fi
done

# download fasttext model
filename=$data_path"cc.zh.300.bin.gz"
wget -O $filename https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.zh.300.bin.gz
gunzip $filename

# build soft link
data_path="data/"
fasttext_model=$PWD/$data_path"cc.zh.300.bin"
dict_fasttext=$PWD/$data_path"dict_fasttext.txt"
test_raw=$PWD/$data_path"test_raw.csv"
coh1_source_mapping=$PWD/$data_path"source.150000.mapping"
coh1_target_mapping=$PWD/$data_path"target.6185.mapping"
fasttext_npy=$PWD/$data_path"fasttext.npy"
sentiment_char_mapping=$PWD/$data_path"char_mapping"
sentiment_word_mapping=$PWD/$data_path"word_mapping"
lm_mapping=$PWD/$data_path"words_char.json"
ln -s $dict_fasttext coh1/;
ln -s $test_raw coh1/corpus/;
ln -s $coh1_source_mapping coh1/corpus/;
ln -s $coh1_target_mapping coh1/corpus/;
ln -s $fasttext_npy coh1/corpus/;
ln -s $dict_fasttext coh2/data/;
ln -s $fasttext_model coh2/data/;
ln -s $test_raw coh2/data/;
ln -s $dict_fasttext LM/data/;
ln -s $test_raw LM/data/;
ln -s $lm_mapping LM/data/;
ln -s $test_raw sentiment_analysis/corpus/;
ln -s $sentiment_char_mapping sentiment_analysis/corpus/;
ln -s $sentiment_word_mapping sentiment_analysis/corpus/;
