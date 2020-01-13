#!/bin/bash

# top dir
dirs=("logs" "saved_models/coh1" "saved_models/coh2" "saved_models/LM" "saved_models/sentiment_analysis" "data")
for d in ${dirs[@]}; do
    if [[ -d $d ]] ;
    then
        echo $d exists.
    else
        mkdir -p $d
        echo $d created.
    fi
done

# coh1
base_dir="coh1"
dirs=("data" "model")
for d in ${dirs[@]}; do
    target_dir=$base_dir/$d
    if [[ -d $target_dir ]] ; 
    then
        echo $target_dir exists.
    else
        mkdir -p $target_dir
        echo $target_dir created.
    fi
done

# coh2
base_dir="coh2"
dirs=("data" "save")
for d in ${dirs[@]}; do
    target_dir=$base_dir/$d
    if [[ -d $target_dir ]] ; 
    then
        echo $target_dir exists.
    else
        mkdir -p $target_dir
        echo $target_dir created.
    fi
done

# LM 
base_dir="LM"
dirs=("data" "save")
for d in ${dirs[@]}; do
    target_dir=$base_dir/$d
    if [[ -d $target_dir ]] ; 
    then
        echo $target_dir exists.
    else
        mkdir -p $target_dir
        echo $target_dir created.
    fi
done

# sentiment 
base_dir="sentiment_analysis"
dirs=("data" "save")
for d in ${dirs[@]}; do
    target_dir=$base_dir/$d
    if [[ -d $target_dir ]] ; 
    then
        echo $target_dir exists.
    else
        mkdir -p $target_dir
        echo $target_dir created.
    fi
done
