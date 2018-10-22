import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
#from tensorflow.contrib.rnn.python.ops import core_rnn
#from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from six.moves import range
import numpy as np
import random
import copy

import data_utils
import seq2seq

setattr(tf.contrib.rnn.GRUCell, '__deepcopy__', lambda self, _: self)
setattr(tf.contrib.rnn.BasicLSTMCell, '__deepcopy__', lambda self, _: self)
setattr(tf.contrib.rnn.MultiRNNCell, '__deepcopy__', lambda self, _: self)

class Seq2seq():
  
  def __init__(self,
               src_vocab_size,
               trg_vocab_size,
               buckets,
               size,
               num_layers,
               batch_size,
               mode,
               input_keep_prob,
               output_keep_prob,
               state_keep_prob,
               beam_search,
               beam_size,
               schedule_sampling='linear', 
               sampling_decay_rate=0.99,
               sampling_global_step=150000,
               sampling_decay_steps=500,
               pretrain_vec = None,
               pretrain_trainable = False,
               length_penalty = None,
               length_penalty_factor = 0.6, 
               feed_previous = False 
               ):
    
    self.feed_previous = feed_previous
    self.decoder_max_len = tf.placeholder(tf.int32,[None]) 
    self.src_vocab_size = src_vocab_size
    self.trg_vocab_size = trg_vocab_size
    self.buckets = buckets
    # units of rnn cell
    self.size = size
    # dimension of words
    self.num_layers = num_layers
    self.batch_size = batch_size
    self.learning_rate = tf.Variable(0.5, trainable=False)
    self.mode = mode
    self.dummy_reply = ["what ?", "yeah .", "you are welcome ! ! ! !"]

    # learning rate decay
    self.learning_rate_decay = self.learning_rate.assign(self.learning_rate * 0.99) 

    # input for Reinforcement part
    self.loop_or_not = tf.placeholder(tf.bool)
    self.reward = tf.placeholder(tf.float32, [None])
    batch_reward = tf.stop_gradient(self.reward)
    self.RL_index = [None for _ in self.buckets]

    # dropout
    self.input_keep_prob =  input_keep_prob
    self.output_keep_prob = output_keep_prob
    self.state_keep_prob =  state_keep_prob

    # beam search
    self.beam_search = beam_search
    self.beam_size = beam_size
    self.length_penalty = length_penalty
    self.length_penalty_factor = length_penalty_factor

    # if load pretrain word vector
    self.pretrain_vec = pretrain_vec
    self.pretrain_trainable = pretrain_trainable

    # schedule sampling
    self.sampling_probability_clip = None 
    self.schedule_sampling = schedule_sampling
    if self.schedule_sampling == 'False': self.schedule_sampling = False
    self.init_sampling_probability = 1.0
    self.sampling_global_step = sampling_global_step
    self.sampling_decay_steps = sampling_decay_steps 
    self.sampling_decay_rate = sampling_decay_rate 

    if self.schedule_sampling == 'linear':
      self.decay_fixed = self.init_sampling_probability * (self.sampling_decay_steps / self.sampling_global_step)
      with tf.variable_scope('sampling_prob',reuse=tf.AUTO_REUSE):
        self.sampling_probability = tf.get_variable(name=self.schedule_sampling,initializer=tf.constant(self.init_sampling_probability),trainable=False)
      self.sampling_probability_decay = tf.assign_sub(self.sampling_probability, self.decay_fixed)
      self.sampling_probability_clip = tf.clip_by_value(self.sampling_probability,0.0,1.0)
      #self.sampling_probability = tf.maximum(self.sampling_probability,tf.constant(0.0))
    elif self.schedule_sampling == 'exp':
      with tf.variable_scope('sampling_prob',reuse=tf.AUTO_REUSE):
        self.sampling_probability = tf.get_variable(name=self.schedule_sampling,initializer=tf.constant(self.init_sampling_probability),trainable=False)
      #self.sampling_probability = tf.train.exponential_decay(
      self.sampling_probability_decay = tf.assign(
        self.sampling_probability,
        tf.train.natural_exp_decay(
          self.sampling_probability,
          self.sampling_global_step,
          self.sampling_decay_steps,
          self.sampling_decay_rate,
          staircase = True)
      )
      self.sampling_probability_clip = tf.clip_by_value(self.sampling_probability,0.0,1.0)
    elif self.schedule_sampling == 'inverse_sigmoid':
      with tf.variable_scope('sampling_prob',reuse=tf.AUTO_REUSE):
        self.sampling_probability = tf.get_variable(name=self.schedule_sampling,initializer=tf.constant(self.init_sampling_probability),trainable=False)
      self.sampling_probability_decay = tf.assign(
        self.sampling_probability,
        #tf.train.cosine_decay(
        tf.train.linear_cosine_decay(
          self.sampling_probability,
          self.sampling_decay_steps,
          self.sampling_global_step,
        )
      )
      self.sampling_probability_clip = tf.clip_by_value(self.sampling_probability,0.0,1.0)
    elif not self.schedule_sampling:
      pass
    else:
      raise ValueError("schedule_sampling must be one of the following: [linear|exp|inverse_sigmoid|False]")

    w_t = tf.get_variable('proj_w', [self.trg_vocab_size, self.size])
    w = tf.transpose(w_t)
    b = tf.get_variable('proj_b', [self.trg_vocab_size])
    output_projection = (w, b)

    def sample_loss(labels, inputs):
      labels = tf.reshape(labels, [-1, 1])
      local_w_t = tf.cast(w_t, tf.float32)
      local_b = tf.cast(b, tf.float32)
      local_inputs = tf.cast(inputs, tf.float32)
      return tf.cast(tf.nn.sampled_softmax_loss(weights = local_w_t,
                                                biases = local_b,
                                                inputs = local_inputs,
                                                labels = labels,
                                                num_sampled = 512,
                                                num_classes = self.trg_vocab_size),
                                                dtype = tf.float32)
    softmax_loss_function = sample_loss

    #FIXME add RL function
    def seq2seq_multi(encoder_inputs, decoder_inputs, mode, pretrain_vec = None):
      if pretrain_vec is not None: 
        pad_num = self.src_vocab_size - pretrain_vec.shape[0]
        pretrain_vec = np.pad(pretrain_vec, [(0, pad_num), (0, 0)], mode='constant')
        tag_vec = pretrain_vec[:data_utils.SPECIAL_TAGS_COUNT]
        pretrain_vec = pretrain_vec[data_utils.SPECIAL_TAGS_COUNT:]
        special_tags = tf.get_variable(
                name="special_tags",
                initializer = tag_vec,
                trainable = True)
        embedding = tf.get_variable(
                name = "embedding", 
                initializer = pretrain_vec,
                trainable = self.pretrain_trainable)
        embedding = tf.concat([special_tags,embedding],0)
      else:
        embedding = tf.get_variable("embedding", [self.src_vocab_size, self.size])
      loop_function_RL = None
      self.loop_function_RL = loop_function_RL

      return seq2seq.embedding_attention_seq2seq(
             encoder_inputs,
             decoder_inputs,
             cell,
             num_encoder_symbols = self.src_vocab_size,
             num_decoder_symbols = self.trg_vocab_size,
             embedding_size = self.size,
             output_projection = output_projection,
             feed_previous = self.feed_previous,
             dtype = tf.float32,
             embedding = embedding,
             beam_search = self.beam_search,
             beam_size = self.beam_size,
             loop = loop_function_RL,
             schedule_sampling = self.schedule_sampling,
             sampling_probability = self.sampling_probability_clip,
             length_penalty = self.length_penalty,
             length_penalty_factor = self.length_penalty_factor)
    
    # inputs
    self.encoder_inputs = []
    self.decoder_inputs = []
    self.target_weights = []

    for i in range(buckets[-1][0]):
      self.encoder_inputs.append(tf.placeholder(tf.int32, shape = [None],
                                                name = 'encoder{0}'.format(i)))
    for i in range(buckets[-1][1] + 1):
      self.decoder_inputs.append(tf.placeholder(tf.int32, shape = [None],
                                                name = 'decoder{0}'.format(i)))
      self.target_weights.append(tf.placeholder(tf.float32, shape = [None],
                                                name = 'weight{0}'.format(i)))
    targets = [self.decoder_inputs[i + 1] for i in range(len(self.decoder_inputs) - 1)]

    def single_cell():
      return tf.contrib.rnn.GRUCell(self.size)
      #return tf.contrib.rnn.BasicLSTMCell(self.size)
    cell = single_cell()
    if self.num_layers > 1:
      cell = tf.contrib.rnn.MultiRNNCell([single_cell() for _ in range(self.num_layers)])
      cell = rnn.DropoutWrapper(cell,input_keep_prob=self.input_keep_prob,output_keep_prob=self.output_keep_prob,state_keep_prob=self.state_keep_prob)

    #self.buckets = [(10, self.decoder_max_len), (15, self.decoder_max_len), (25, self.decoder_max_len), (50, self.decoder_max_len)]
    self.buckets = [(10, 50), (15, 50), (25, 50), (50, 50)]

    self.outputs, self.losses = seq2seq.model_with_buckets(
         self.encoder_inputs, self.decoder_inputs, targets,
         self.target_weights, self.buckets, lambda x, y: seq2seq_multi(x, y, self.mode, self.pretrain_vec),
         softmax_loss_function = softmax_loss_function
    )
    
    for b in range(len(self.buckets)):
      #print('self.outputs[b]: ',self.outputs[b])
      self.outputs[b] = [tf.nn.log_softmax(tf.matmul(output, output_projection[0]) + output_projection[1])
                         for output in self.outputs[b]]

    self.saver = tf.train.Saver(max_to_keep = 2)

  # a and b are both list of token ids. ex:[1,2,3,4,5...]
  # a--> encoder_input, b--> decoder_input in get_batch
  def prob(self, a, b, X, bucket_id):
    # define softmax
    def softmax(x):
      e_x = np.exp(x)
      return e_x / e_x.sum()

    # function X, not trainable, batch = 1
    temp = self.batch_size
    self.batch_size = 1
    encoder_input, decoder_input, weight = self.get_batch({bucket_id: [(a, b)]}, bucket_id)
    self.batch_size = temp
    outputs = X(encoder_input, decoder_input, weight, bucket_id)
    #print('b: ',b)
    #print('outputs: ',outputs,outputs[0].shape)
    r = 0.0
    # outputs已經project過(6258維)，看decoder_input的tokan_id(b)在output的softmax之機率高不高，越高reward越好。
    for logit, i in zip(outputs, b):
      #print('logit: ',logit,len(logit),logit[0].shape)
      #print('i: ',i)
      #print('r: ',np.log10(softmax(logit[0])[i]))
      r += np.log10(softmax(logit[0])[i])
    return r

  def run(self, sess, encoder_inputs, decoder_inputs, target_weights,
          bucket_id, forward_only = False, X = None, Y = None):
    
    encoder_size = self.buckets[bucket_id][0]
    decoder_size = self.buckets[-1][-1] 
    decoder_inputs = np.reshape(np.repeat(decoder_inputs[0],decoder_size),(-1,1))
    target_weights = np.reshape(np.repeat(target_weights[0],decoder_size),(-1,1))
    
    input_feed = {}
    for l in range(encoder_size):
      input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
    for l in range(decoder_size):
      input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
      input_feed[self.target_weights[l].name] = target_weights[l]

    last_target = self.decoder_inputs[decoder_size].name
    input_feed[last_target] = np.zeros([self.batch_size], dtype = np.int32)

    output_feed = [self.outputs[bucket_id]]
    outputs = sess.run(output_feed, input_feed)
    return outputs[0]

    # r2 = self.prob(r_input, token_ids, X, bucket_id) / float(len(token_ids)) if len(token_ids) != 0 else 0

  def get_batch(self, data, bucket_id, rand = True, initial_id=0):
    # data should be [whole_data_length x (source, target)] 
    # decoder_input should contain "GO" symbol and target should contain "EOS" symbol
    encoder_size, decoder_size = self.buckets[bucket_id]
    encoder_inputs, decoder_inputs = [], []

    # data[bucket_id] == [(incoder_inp_list,decoder_inp_list),...]

    data_len = len(data[bucket_id])
    if data_len >= self.batch_size:
        data_range = self.batch_size
    else:
        data_range = data_len
    for i in range(data_range):
      if rand:
        encoder_input, decoder_input = random.choice(data[bucket_id])
      else:
        encoder_input, decoder_input = data[bucket_id][i+initial_id]

      encoder_pad = [data_utils.PAD_ID] * (encoder_size - len(encoder_input))
      encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))

      decoder_pad = [data_utils.PAD_ID] * (decoder_size - len(decoder_input) - 1)
      decoder_inputs.append([data_utils.GO_ID] + decoder_input + decoder_pad)

    batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

    for length_idx in range(encoder_size):
      batch_encoder_inputs.append(np.array([encoder_inputs[batch_idx][length_idx]
                                  for batch_idx in range(data_range)], dtype = np.int32))

    for length_idx in range(decoder_size):
      batch_decoder_inputs.append(np.array([decoder_inputs[batch_idx][length_idx]
                                  for batch_idx in range(data_range)], dtype = np.int32))

      batch_weight = np.ones(data_range, dtype = np.float32)
      for batch_idx in range(data_range):
        # We set weight to 0 if the corresponding target is a PAD symbol.
        # The corresponding target is decoder_input shifted by 1 forward.
        if length_idx < decoder_size - 1:
          target = decoder_inputs[batch_idx][length_idx + 1]
        if length_idx == decoder_size - 1 or target == data_utils.PAD_ID:
          batch_weight[batch_idx] = 0.0
      batch_weights.append(batch_weight)

    return batch_encoder_inputs, batch_decoder_inputs, batch_weights

if __name__ == '__main__':

  test = Seq2seq(50, 100, 200, 300, 1, 128) 
