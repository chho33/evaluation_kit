import numpy as np
import tensorflow as tf
import json
from collections import Counter
from memory_profiler import profile
max_document_len = 40

def dump_list_to_dict(filename):
    with open(filename,"r") as f:
        rows = [(row,i) for i, row in enumerate(f.readlines())]
    dic = dict(rows)
    with open("data/words.json","w") as f:
        json.dump(dic,f)

#@profile
def build_word_list(filename):
    c = Counter()
    rows = []
    with open(filename, "r") as f:
        for i,row in enumerate(f.readlines()):
            if i%10000==0 and i!=0: 
                print(i)
                c.update(Counter(rows))
                rows = []
            row = row.strip().split()
            rows+=row
        if len(rows)!=0:
            c.update(Counter(rows))
    return c

def build_word_dict(filename,start_index=4):
    words = []
    with open(filename, "r") as f:
        for row in f.readlines():
            row = row.strip().split()
            #print(row)
            words+=list(set(row))
        #words = f.read().replace("\n", "").split()
    words = set(words)
    word_dict = dict(((v,i+start_index) for i, v in enumerate(words)))
    del words
    gc.collect()
    return word_dict

def build_dataset(filename, word_dict, pad=True):
    if isinstance(filename, str): 
        with open(filename, "r") as f:
            lines = f.readlines()
            data = map(lambda s: s.strip().split()[:max_document_len], lines)
    elif isinstance(filename, (list, np.ndarray, np.generic)):
        data = filename
        data = map(lambda s: s.strip().split()[:max_document_len], data)
    data = map(lambda s: ["<bos>"] + s + ["<eos>"], data)
    data = map(lambda s: [word_dict.get(w, word_dict["<unk>"]) for w in s], data)
    if pad: 
        data = map(lambda d: d + (max_document_len +2 - len(d)) * [word_dict["<pad>"]], data)
    return data

def get_dataset(filename):
    with open(filename, "r") as f: 
        rows = [list(map(lambda x:int(x),row.strip().split())) for row in f.readlines()]
    return rows

def get_word_dict(filename):
    with open(filename, "r") as f:
        word_dict = json.load(f)
    return word_dict

def batch_iter(inputs, batch_size, num_epochs):
    #inputs = np.array(inputs)

    num_batches_per_epoch = (len(inputs) - 1) // batch_size + 1
    for epoch in range(num_epochs):
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, len(inputs))
            yield np.array(inputs[start_index:end_index])

def data_loader(filename,batch_size=64,epochs=5,shuffle=True):
    dataset = tf.data.TextLineDataset([filename])
    if shuffle:
        dataset = dataset.shuffle(buffer_size=25000000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(epochs)
    iterator = dataset.make_initializable_iterator()
    return iterator
    next_element = iterator.get_next()
    return next_element 

def map_get_words(txts,kind="char",return_type="str"):
    if isinstance(txts, str):
        with open(txts,"r") as f:
            txts = [row.strip() for row in f.readlines()]
    jieba = None
    if kind == "word":
        import jieba_fast as jieba
        jieba.initialize()
        jieba.load_userdict("dict_fasttext.txt")
    txts = list(map(lambda txt:get_words(txt,kind,return_type,jieba), txts)) 
    return txts  

def get_words(txt,kind="char",return_type="str",jieba=None):
    if not isinstance(txt,str) and np.isnan(txt):
        result = []
        if return_type == "str":
            result = ' '.join(result)
        return result
    if kind == "word":
        result = jieba.lcut(txt)[:max_document_len]
        #result = list(filter(lambda x:len(x.strip())>0,result))
    elif kind == "char":
        result = [t for t in txt][:max_document_len]
    if return_type == "str":
        result = ' '.join(result)
    return result
