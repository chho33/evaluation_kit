from data_utils import build_word_dict, build_word_list, build_dataset
import json
import os
import tracemalloc
import pdb

data_path = 'data/'
train_data_path = 'ptt_train.txt'
valid_data_path = 'ptt_valid.txt'
word_dict_path  = 'words.json'
train_data_path = os.path.join(data_path,train_data_path)
valid_data_path = os.path.join(data_path,valid_data_path)
word_dict_path  = os.path.join(data_path,word_dict_path)

new_train_data_path = 'train.txt'
new_valid_data_path = 'valid.txt'
new_train_data_path = os.path.join(data_path,new_train_data_path)
new_valid_data_path = os.path.join(data_path,new_valid_data_path)

def dump_word_list(prefix="vocabulary",train_data_path=train_data_path,valid_data_path=valid_data_path):
  c = build_word_list(train_data_path)
  c.update(build_word_list(valid_data_path))
  print('dumping word list...')
  #with open("data/vocabulary.txt","w") as f:
  #  f.write("<pad>\n")
  #  f.write("<bos>\n")
  #  f.write("<eos>\n")
  #  f.write("<unk>\n")
  ks,kvs = [],[]
  for i,(k,v) in enumerate(c.items()):
    if i%10000==0 and i!=0:
      with open("data/%s.txt"%prefix, 'a') as f:
        f.write("".join(ks))
      with open("data/%s_freq.txt"%prefix, 'a') as f:
        f.write("".join(kvs))
      ks,kvs = [],[]
    ks.append('%s\n'%k)
    kvs.append('%s %s\n'%(k,v))
  if len(ks) != 0:  
    with open("data/%s.txt"%prefix, 'a') as f:
      f.write("".join(ks))
    with open("data/%s_freq.txt"%prefix, 'a') as f:
      f.write("".join(kvs))
  print('word list dumped')

def dump_list_to_dict(from_file="data/vocabulary.txt",to_file="data/words.json"):
    with open(from_file,"r") as f:
        rows = []
        for i, row in enumerate(f.readlines()):
            rows.append((row.strip(),i+4))
    dic = dict(rows)
    dic["<pad>"] = 0
    dic["<bos>"] = 1
    dic["<eos>"] = 2
    dic["<unk>"] = 3
    with open(to_file,"w") as f:
        json.dump(dic,f)

def dump_word_dict():
  word_dict = build_word_dict(train_data_path)
  start_index = len(word_dict) + 4 + 1
  word_dict.update(build_word_dict(valid_data_path,start_index))
  word_dict["<pad>"] = 0
  word_dict["<bos>"] = 1
  word_dict["<eos>"] = 2
  word_dict["<unk>"] = 3
  print('dumping word_dict...')
  with open(word_dict_path, 'w') as f:
    json.dump(word_dict,f)
  print('word_dict dumped')

def load_word_dict(word_dict_path=word_dict_path):
  with open(word_dict_path, 'r') as f:
    word_dict = json.load(f)
  return word_dict

def dump_train_data(word_dict,train_data_path=train_data_path,new_train_data_path=new_train_data_path):
  print('building dataset...')
  train_data = build_dataset(train_data_path,word_dict)
  print('building dataset2...')
  train_data = map(lambda x:" ".join(map(lambda y:str(y),x)),train_data)
  print('dumping train...')
  rows = []
  for i,d in enumerate(train_data):
    if i%10000==0 and i!=0: 
      with open(new_train_data_path, 'a') as f:
        f.write("".join(rows))
      rows = []
    rows.append("%s\n"%d)
  if len(rows)!=0:
    with open(new_train_data_path, 'a') as f:
      f.write("".join(rows))

def dump_valid_data(word_dict,valid_data_path=valid_data_path,new_valid_data_path=new_valid_data_path):
  valid_data = build_dataset(valid_data_path,word_dict)
  valid_data = list(map(lambda x:" ".join(map(lambda y:str(y),x)),valid_data))
  print('dumping valid...')
  with open(new_valid_data_path, 'w') as f:
    f.write("\n".join(valid_data))

'''
tracemalloc.start(10)
prev = tracemalloc.take_snapshot()
dump_word_list()
#dump_word_dict()
#word_dict = load_word_dict(word_dict_path=word_dict_path)
snap1 = tracemalloc.take_snapshot()
stats = snap1.compare_to(prev,'lineno')
for stat in stats:
  print(stat)

word_dict = load_word_dict(word_dict_path=word_dict_path)
#pdb.set_trace()
dump_train_data(word_dict)
snap2 = tracemalloc.take_snapshot()
stats = snap2.compare_to(snap1,'lineno')
for stat in stats:
  print(stat)
dump_valid_data(word_dict)
snap3 = tracemalloc.take_snapshot()
stats = snap3.compare_to(snap2,'lineno')
for stat in stats:
  print(stat)
tracemalloc.stop()
'''


# char
#dump_word_list("vocabulary_char","data/ptt_train_char.txt","data/ptt_valid_char.txt")
#dump_list_to_dict("data/vocabulary_char.txt","data/words_char.json")
word_dict = load_word_dict(word_dict_path="data/words_char.json")
dump_train_data(word_dict,train_data_path="data/ptt_train_char.txt",new_train_data_path="data/train_char.txt")
dump_valid_data(word_dict,valid_data_path="data/ptt_valid_char.txt",new_valid_data_path="data/valid_char.txt")

'''
# word
#dump_word_list()
dump_list_to_dict("data/vocabulary.txt")
word_dict = load_word_dict(word_dict_path=word_dict_path)
dump_train_data(word_dict)
dump_valid_data(word_dict)
'''
