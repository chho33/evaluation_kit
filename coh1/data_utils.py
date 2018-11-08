from __future__ import absolute_import
from __future__ import division

import re
import sys
#import nltk
from flags import buckets,split_ratio,SEED,src_vocab_size,_START_VOCAB,SPECIAL_TAGS_COUNT,PAD_ID,GO_ID,EOS_ID,UNK_ID,dict_path
import jieba_fast as jieba
import opencc
import numpy as np

import tensorflow as tf
from tensorflow.python.platform import gfile

import subprocess

WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
#WORD_SPLIT = re.compile(b"")
DIGIT_RE = re.compile(br"\d")
DU_RE = re.compile(b"\!")

# Tokenize a sentence into a word list
def tokenizer(sentence):
  sentence = DU_RE.sub(b'', sentence)
  words = []
  for split_sen in sentence.lower().strip().split():
    words.extend(WORD_SPLIT.split(split_sen))
  return [word for word in words if word]

# Form vocab map (vocab to index) according to maxsize
# Temporary combine source and target vocabulary map together
# mode:[same|diff], to decide source and target share the same mapping or not
def form_vocab_mapping(filename_1, filename_2, max_size_1, max_size_2, nltk_tokenizer=None, mode='diff'):
  
  output_path = filename_1 + '.' + str(max_size_1) + '.mapping' 
  output_path2 = filename_2 + '.' + str(max_size_2) + '.mapping' 
  max_sizes = (max_size_1,max_size_2)
  if gfile.Exists(output_path) and gfile.Exists(output_path2):
    print('Map file has already been formed!')
  else:
    print('Forming mapping file according to %s and %s' % (filename_1, filename_2))  
    print('Source max vocabulary size : %s' % max_size_1)
    print('Target max vocabulary size : %s' % max_size_2)

    vocab = {}
    with gfile.GFile(filename_1, mode = 'rb') as f_1, gfile.GFile(filename_2, mode = 'rb') as f_2:
      f = [f_1, f_2]
      counter = 0
      for i, fil in enumerate(f):
        print('Processing file %s' % i)
        for line in fil:
          counter += 1
          if counter % 100000 == 0:
            print("  Processing to line %s" % counter)

          line = tf.compat.as_bytes(line) 
          tokens = nltk.word_tokenize(line) if nltk_tokenizer else tokenizer(line)
          for w in tokens:
            #word = DIGIT_RE.sub(b"0", w)
            word = w
            if word in vocab:
              vocab[word] += 1
            else:
              vocab[word] = 1
        if mode == 'diff':
          output_path = fil.name + '.' + str(max_sizes[i]) + '.mapping' 
          write_mapping(vocab,max_sizes[i],output_path)
          vocab = {}
      
      if mode == 'same': 
        write_mapping(vocab,max_sizes[0],output_path)
        subprocess.run("ln %s %s"%(output_path,output_path2),shell=True)

def write_mapping(vocab,max_size,output_path):
  vocab_list = _START_VOCAB + sorted(vocab, key = vocab.get, reverse = True)
  if len(vocab_list) > max_size:
    vocab_list = vocab_list[:max_size]

  with gfile.GFile(output_path, 'wb') as vocab_file:
    for w in vocab_list:
      vocab_file.write(w + b'\n')

# Read mapping file from map_path
# Return mapping dictionary
def read_map(map_path):

  if gfile.Exists(map_path):
    vocab_list = []
    with gfile.GFile(map_path, mode = 'rb') as f:
      vocab_list.extend(f.readlines())
    
    vocab_list = [tf.compat.as_bytes(line).strip() for line in vocab_list]
    vocab_dict = dict([(x, y) for (y, x) in enumerate(vocab_list)])
    
    return vocab_dict, vocab_list

  else:
    raise ValueError("Vocabulary file %s not found!", map_path)

def convert_to_token(sentence, vocab_map, nltk_tokenizer):
 
  if nltk_tokenizer:
    words = nltk.word_tokenize(sentence)
  else:
    words = tokenizer(sentence)  
  
  #return [vocab_map.get(DIGIT_RE.sub(b"0", w), UNK_ID) for w in words]
  return [vocab_map.get(w, UNK_ID) for w in words]

def file_to_token(file_path, vocab_map, nltk_tokenizer):
  output_path = file_path + ".token"
  if gfile.Exists(output_path):
    print("Token file %s has already existed!" % output_path)
  else:
    print("Tokenizing data according to %s" % file_path)

    with gfile.GFile(file_path, 'rb') as input_file:
      with gfile.GFile(output_path, 'w') as output_file:
        counter = 0
        for line in input_file:
          counter += 1
          if counter % 100000 == 0:
            print("  Tokenizing line %s" % counter)
          token_ids = convert_to_token(tf.compat.as_bytes(line), vocab_map, nltk_tokenizer)

          output_file.write(" ".join([str(tok) for tok in token_ids]) + '\n')

def prepare_whole_data(input_path_1, input_path_2, max_size_1, max_size_2, nltk_tokenizer = False, skip_to_token = False, mode='diff'):
  form_vocab_mapping(input_path_1, input_path_2, max_size_1, max_size_2, nltk_tokenizer, mode)

  map_src_path = input_path_1 + '.' + str(max_size_1) + '.mapping'  
  map_trg_path = input_path_2 + '.' + str(max_size_2) + '.mapping'  
  vocab_map_src , _ = read_map(map_src_path)
  vocab_map_trg , _ = read_map(map_trg_path)
  files = {input_path_1+'_train': vocab_map_src,
           input_path_1+'_val': vocab_map_src,
           input_path_2+'_train': vocab_map_trg,
           input_path_2+'_val': vocab_map_trg}
  if not skip_to_token:
    for f,vocab_map in files.items():
      file_to_token(f , vocab_map, nltk_tokenizer)

def read_data(source_path, target_path, bucket):

  data_set = [[] for _ in range(len(bucket))]
  with tf.gfile.GFile(source_path, mode="r") as source_file:
    with tf.gfile.GFile(target_path, mode="r") as target_file:
      
      source, target = source_file.readline(), target_file.readline()
      counter = 0     
      while source and target:
        counter += 1
        if counter % 100000 == 0:
          print("  reading data line %d" % counter)
          #print('source: ',source, 'target: ',target)
          #print('bucket: ',bucket)
 
          sys.stdout.flush()
        source_ids = [int(x) for x in source.split()]
        target_ids = [int(x) for x in target.split()]
        target_ids.append(EOS_ID)

        for bucket_id, (source_size, target_size) in enumerate(bucket):
          #if counter % 100000 == 0:
          #    print('bucket_id, (source_size, target_size): ',bucket_id, (source_size, target_size))
          #    print('len(source_ids): ',len(source_ids),
          #          ' len(target_ids): ',len(target_ids))
          if len(source_ids) < source_size and len(target_ids) < target_size:
            data_set[bucket_id].append((source_ids, target_ids))
            break
          #if bucket_id == 3: print('too long=======>',counter)

        source, target = source_file.readline(), target_file.readline()

  return data_set

# Read token data from tokenized data
def read_token_data(file_path):
  token_path = file_path + '.token'
  if gfile.Exists(token_path):
    data_set = []
    print(" Reading from file %s" % file_path)
    with gfile.GFile(token_path, mode = 'r') as t_file:
      counter = 0
      token_file = t_file.readline()
      while token_file:
        counter += 1
        if counter % 100000 == 0:
          print("  Reading data line %s" % counter)
          sys.stdout.flush()
        token_ids = [int(x) for x in token_file.split()]
        data_set.append(token_ids)
        token_file = t_file.readline()

    return data_set

  else:
    raise ValueError("Can not find token file %s" % token_path)

def token_to_text(ids,mapping):
    with open(mapping,'r') as f:
        mapping = [row.strip() for row in f.readlines()]
    mapping = np.array(mapping)
    if isinstance(ids,list):
        ids = np.array(ids)
    elif isinstance(ids,str):
        ids = int(ids)
        ids = np.array([ids])
    elif isinstance(ids,int):
        ids = np.array([ids])
    return mapping[ids]

def sub_words(word):
    for rep in replace_words.keys():
        if rep in word:
            word = re.sub(rep,replace_words[rep],word)
    return word

def word_seg(input_file,output_file,mode):
    if mode == 'word':
        jieba.load_userdict(dict_path)
    
    with open(output_file,'w') as f, open(input_file,'r') as fi:
        for l in fi:
            # remove all whitespace characters
            l = ''.join(l.split())
            if mode == 'char':
                f.write(' '.join(list(l)) + '\n')
            else:
                seg = jieba.cut(l, cut_all=False)
                f.write(' '.join(seg) + '\n')

def split_train_val(source,target,buckets=buckets):
    data = [[] for i in range(len(buckets))]
    with open(source,'r') as src, open(target,'r') as trg:
        src = list(src)
        np.random.seed(SEED)
        np.random.shuffle(src)
        src = iter(src)
        trg = list(trg)
        np.random.seed(SEED)
        np.random.shuffle(trg)
        trg = iter(trg)
        for s,t in zip(src,trg):
            sl, tl = len(s.split()), len(t.split())
            for bucket_id, (source_size, target_size) in enumerate(buckets):         
                if sl < source_size and tl < target_size:
                    data[bucket_id].append((s, t, sl, tl))
                    break

    with open(source+'_train', 'w') as src_train,\
         open(source+'_val', 'w') as src_val,\
         open(target+'_train', 'w') as trg_train,\
         open(target+'_val','w') as trg_val:

        for b, ds in zip(buckets, data):
            dl = len(ds)
            split_index = int(dl*split_ratio)
            print('\n')
            print(b)
            print('data : ' + str(dl))
            for i, d in enumerate(ds):
                (s, t, sl, tl) = d
                if i < split_index:
                    src_train.write(s)
                    trg_train.write(t)
                else:
                    src_val.write(s)
                    trg_val.write(t)

def simple2tradition(text):
    return opencc.convert(text, config='zhs2zht.ini')

def tradition2simple(text):
    return opencc.convert(text,config='zht2zhs.ini')

def load_fasttext_vec(model_path,mapping,hkl_file,t2s=False):
    import hickle as hkl
    from fastText import load_model
    model = load_model(model_path)
    text = []
    with open(mapping, 'r') as f:
        for row in f.readlines():
            row = row.strip()
            if t2s:
                row = tradition2simple(row)
            vec = model.get_word_vector(row)
            text.append(vec)
    text = np.array(text)
    hkl.dump(text,hkl_file)
    
if __name__ == "__main__":
  prepare_whole_data('corpus/source', 'corpus/target', src_vocab_size)
  #data_set_1 = read_token_data('corpus/valid.source')
  #data_set_2 = read_token_data('corpus/valid.target')

