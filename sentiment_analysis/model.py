import dataset

import numpy as np
import random
import tensorflow as tf
import os
from settings import *
dirname = os.path.dirname(os.path.abspath(__file__))
if args.cut:
  cut_mode = args.cut
if args.m:
  mapping_path = args.m
model_type = args.model_type

class discriminator():

  def __init__(self, vocab_size, unit_size, batch_size, max_length, mode):
    self.vocab_size = vocab_size
    self.unit_size = unit_size
    self.batch_size = batch_size
    self.max_length = max_length
    self.mode = mode
    self.model_type = {
      'rnn_last': self.build_model_rnn_last,
      'rnn_ave': self.build_model_rnn_ave,
      'cnn': self.build_model_cnn,
      'xgboost': self.build_model_xgboost,
    }
    self.dropout_keep_prob = 1.0 
    self.seq_length = tf.placeholder(tf.int32, [None])
    self.initializer = tf.random_normal_initializer(seed=1995)
    #self.initializer = tf.glorot_uniform_initializer()
    #self.initializer = tf.glorot_normal_initializer()

    # for cnn
    self.filter_sizes = [3,4]
    self.num_filters = 128 

    self.build_model(typ=model_type)
    self.saver = tf.train.Saver(max_to_keep = 2)

  def build_model(self,typ):
    self.model_type[typ]()

  def build_model_cnn(self):
    print('==== model type: cnn ====')
    params = tf.get_variable('embedding', [self.vocab_size, self.unit_size],initializer=self.initializer)
    self.encoder_input = tf.placeholder(tf.int32, [None, self.max_length])
    embedding = tf.nn.embedding_lookup(params, self.encoder_input)
    embedding_expanded = tf.expand_dims(embedding,-1)
    pooled_outputs = []
    for i, filter_size in enumerate(self.filter_sizes):
      with tf.name_scope('conv-maxpool-%s' % filter_size):
        filter_shape = [filter_size, self.unit_size, 1, self.num_filters]
        W = tf.Variable(
            tf.truncated_normal(filter_shape, stddev=0.1), name='cnn_w')
        b = tf.Variable(
            tf.constant(0.1, shape=[self.num_filters]), name='cnn_b')
        conv = tf.nn.conv2d(
            embedding_expanded,
            W,
            strides=[1, 1, 1, 1],
            padding='VALID',
            name='conv')
        h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
        pooled = tf.nn.max_pool(
            h,
            ksize=[1, self.max_length - filter_size + 1, 1, 1],
            strides=[1, 1, 1, 1],
            padding='VALID',
            name='pool')
        pooled_outputs.append(pooled)

    num_filters_total = self.num_filters * len(self.filter_sizes)
    with tf.name_scope('concat'):
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
    with tf.name_scope('dropout'):
        self.h_drop = tf.nn.dropout(self.h_pool_flat,
                                    self.dropout_keep_prob)
    self.top_layer(self.h_drop)

  def build_model_xgboost(self):
    print('==== model type: xgboost ====')
    pass

  def build_model_rnn_ave(self):
    print('==== model type: rnn ave ====')
    cell = tf.contrib.rnn.GRUCell(self.unit_size)
    params = tf.get_variable('embedding', [self.vocab_size, self.unit_size],initializer=self.initializer)
    self.encoder_input = tf.placeholder(tf.int32, [None, self.max_length])
    embedding = tf.nn.embedding_lookup(params, self.encoder_input)
    
    outputs, hidden_state = tf.nn.dynamic_rnn(cell, embedding, sequence_length = self.seq_length, dtype = tf.float32) 
    outputs = tf.reduce_mean(outputs,axis=1)
    self.top_layer(outputs)

  def build_model_rnn_last(self):
    print('==== model type: rnn last ====')
    cell = tf.contrib.rnn.GRUCell(self.unit_size)
    params = tf.get_variable('embedding', [self.vocab_size, self.unit_size],initializer=self.initializer)
    print('params: ',params,tf.shape(params))
    self.encoder_input = tf.placeholder(tf.int32, [None, self.max_length])
    print('encoder_input: ',self.encoder_input,tf.shape(self.encoder_input))
    embedding = tf.nn.embedding_lookup(params, self.encoder_input)
    print('embedding: ',embedding, tf.shape(embedding))
    
    _, hidden_state = tf.nn.dynamic_rnn(cell, embedding, sequence_length = self.seq_length, dtype = tf.float32) 
    self.top_layer(hidden_state)

  def top_layer(self,outputs):
    w = tf.get_variable('w', [self.unit_size, 1])
    b = tf.get_variable('b', [1])
    output = tf.matmul(outputs, w) + b

    self.logit = tf.nn.sigmoid(output)

    if self.mode != 'test':
      self.target = tf.placeholder(tf.float32, [None, 1])
      self.loss = tf.reduce_mean(tf.square(self.target - self.logit))

      self.opt = tf.train.AdamOptimizer().minimize(self.loss)
    else:
      #self.vocab_map, _ = dataset.read_map('sentiment_analysis/corpus/mapping')
      self.vocab_map, _ = dataset.read_map(mapping_path)

  def step(self, session, encoder_inputs, seq_length, target = None):
    input_feed = {}
    input_feed[self.encoder_input] = encoder_inputs
    input_feed[self.seq_length] = seq_length

    output_feed = []

    if self.mode == 'train':
      input_feed[self.target] = target
      output_feed.append(self.loss)
      output_feed.append(self.opt)
      #output_feed.append(self.encoder_input)
      #output_feed.append(self.target)
      outputs = session.run(output_feed, input_feed)
      #return outputs[0], outputs[2], outputs[3]
      return outputs[0]
    elif self.mode == 'valid':
      input_feed[self.target] = target
      output_feed.append(self.loss)
      outputs = session.run(output_feed, input_feed)
      return outputs[0]
    elif self.mode == 'test':
      output_feed.append(self.logit)
      outputs = session.run(output_feed, input_feed)
      return outputs[0]

  def get_batch(self, data, shuffle=True):
    encoder_inputs = []
    encoder_length = []
    target = []

    for i in range(self.batch_size):
      if shuffle:
          pair = random.choice(data)
      else:
          try:
              pair = data[i]
          except IndexError:
              break
          #print('pair: ',pair)
      #pair = data[i]
      length = len(pair[1])
      target.append([pair[0]])
      if length > self.max_length:
        encoder_inputs.append(pair[1][:self.max_length])
        encoder_length.append(self.max_length)
      else:
        if cut_mode == "char":
          encoder_pad = [dataset.PAD_ID] * (self.max_length - length)
        if cut_mode == "word":
          encoder_pad = [dataset.EOS_ID] * (self.max_length - length)
        encoder_inputs.append(pair[1] + encoder_pad)
        encoder_length.append(length)

    batch_input = np.array(encoder_inputs, dtype = np.int32)
    batch_length = np.array(encoder_length, dtype = np.int32)
    batch_target = np.array(target, dtype = np.float32)

    return batch_input, batch_length, batch_target

if __name__ == '__main__':
  test = discriminator(1000, 100, 32, 1, 50)
