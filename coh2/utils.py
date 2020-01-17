import tensorflow as tf
from tensorflow.python.estimator.canned import metric_keys
from flags import FLAGS
import array
import numpy as np
import sys
import os
import re
import csv
from collections import namedtuple, defaultdict

SPECIAL_TAGS_COUNT = 2

def load_vocab(filename):
    with open(filename) as f:
        vocab = f.read().splitlines()
    return vocab

def load_pretrain_vectors(filename, vocab):
    vectors = array.array('d')
    num_vectors = 0
    with open(filename, "r", encoding="utf-8") as f:
        for _, line in enumerate(f):
            tokens = line.split(" ")
            word = tokens[0]
            entries = tokens[1:]
            if not vocab or word in vocab:
                vectors.extend(float(x) for x in entries)
                num_vectors += 1
        word_dim = len(entries)
        tf.logging.info("Found {} out of {} vectors in pretrain vec".format(num_vectors, len(vocab)))
        return np.array(vectors).reshape(num_vectors, word_dim)

def get_embeddings(params=FLAGS):
    vocab_array = load_vocab(params.vocab_path)
    pretrain_vectors = load_pretrain_vectors(params.pretrain_path, vocab=set(vocab_array))
    num_vectors = pretrain_vectors.shape[0]
    pretrain_vectors = pretrain_vectors.astype("float32")
    if params.pretrain_path and params.vocab_path:
        tf.logging.info("Loading pretrain embeddings...")
        special_tags = np.random.uniform(-0.25, 0.25, (SPECIAL_TAGS_COUNT, params.embedding_dim)).astype("float32")
        special_tags = tf.get_variable("special_tag",
                                  initializer=special_tags,
                                  trainable=True)
        pretrain_vectors = tf.get_variable("word_embeddings",
                                       initializer=pretrain_vectors,
                                       trainable=params.pretrain_trainable)
    else:
        tf.logging.info("No pretrain_vec path specificed, starting with random embeddings.")
        special_tags = np.random.uniform(-0.25, 0.25, (SPECIAL_TAGS_COUNT, params.embedding_dim)).astype("float32")
        special_tags = tf.get_variable("special_tag",
                                  initializer=special_tags,
                                  trainable=True)
        pretrain_vectors = np.random.uniform(-0.25, 0.25, (num_vectors, params.embedding_dim)).astype("float32")
        pretrain_vectors = tf.get_variable("word_embeddings",
                                       initializer=pretrain_vectors,
                                       trainable=params.pretrain_trainable)
    pretrain_vectors = tf.concat([special_tags,pretrain_vectors],0)
    return pretrain_vectors

def compare_fn(best_eval_result, current_eval_result, default_key = metric_keys.MetricKeys.LOSS):
    '''
    default_key can be: [recall_at_1 | recall_at_2 | recall_at_5 | metric_keys.MetricKeys.LOSS]
    '''
    print('********* best_eval_result: %s **********'%best_eval_result[default_key])
    print('********* current_eval_result: %s **********'%current_eval_result[default_key])
    print('### metric:%s ###'%default_key)
    if not best_eval_result or default_key not in best_eval_result:
      raise ValueError(
          'best_eval_result cannot be empty or no loss is found in it.')

    if not current_eval_result or default_key not in current_eval_result:
      raise ValueError(
          'current_eval_result cannot be empty or no loss is found in it.')

    if 'loss' in default_key.lower():
        return best_eval_result[default_key] > current_eval_result[default_key]
    else:
        return best_eval_result[default_key] < current_eval_result[default_key]

def np_save(arr, pth):
    with open(pth, 'wb+') as f:
        np.save(f, arr, allow_pickle=False)
        f.flush()
        os.fsync(f.fileno())

def data_loader(filenames,batch_size=64,epochs=5,shuffle=True,mode="train"):
    if mode == "train":
        record_defaults = [tf.string,tf.string,tf.int32]
    elif mode == "infer":
        record_defaults = [tf.string,tf.string]
    dataset = tf.contrib.data.CsvDataset(filenames, record_defaults, header=True)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=400000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(epochs)
    #iterator = dataset.make_one_shot_iterator()
    iterator = dataset.make_initializable_iterator()
    return iterator
