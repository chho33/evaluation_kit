import tensorflow as tf
import numpy as np
import pandas as pd
import data_utils
import seq2seq_model
from flags import FLAGS, buckets, source_mapping, target_mapping 

def create_seq2seq(session, mode):

  FLAGS.schedule_sampling = False 
      
  model = seq2seq_model.Seq2seq(src_vocab_size = FLAGS.src_vocab_size,
                                trg_vocab_size = FLAGS.trg_vocab_size,
                                buckets = buckets,
                                size = FLAGS.hidden_size,
                                num_layers = FLAGS.num_layers,
                                batch_size = FLAGS.batch_size,
                                mode = mode,
                                input_keep_prob = FLAGS.input_keep_prob,
                                output_keep_prob = FLAGS.output_keep_prob,
                                state_keep_prob = FLAGS.state_keep_prob,
                                beam_search = FLAGS.beam_search,
                                beam_size = FLAGS.beam_size,
                                schedule_sampling = FLAGS.schedule_sampling,
                                sampling_decay_rate = FLAGS.sampling_decay_rate,
                                sampling_global_step = FLAGS.sampling_global_step,
                                sampling_decay_steps = FLAGS.sampling_decay_steps,
                                pretrain_vec = FLAGS.pretrain_vec,
                                pretrain_trainable = FLAGS.pretrain_trainable,
                                length_penalty = FLAGS.length_penalty,
                                length_penalty_factor = FLAGS.length_penalty_factor,
                                feed_previous = FLAGS.feed_previous
                                )
  
  if len(FLAGS.bind) > 0:
      ckpt = tf.train.get_checkpoint_state(FLAGS.bind)
  else:
      ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)

  if ckpt:
    print("Reading model from %s, mode: %s" % (ckpt.model_checkpoint_path, mode))
    model.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    print("Create model with fresh parameters, mode: %s" % mode)
    session.run(tf.global_variables_initializer())
  
  return model


def test(filename):
  if FLAGS.src_word_seg == 'word':
    import jieba_fast as jieba
    jieba.load_userdict("dict_fasttext.txt")
  sess = tf.Session()
  src_vocab_dict, _ = data_utils.read_map(source_mapping)
  trg_vocab_dict, _ = data_utils.read_map(target_mapping)
  model = create_seq2seq(sess, 'TEST')
  model.batch_size = 1
  #model.decoder_max_len = None
  
  #sources = ["你是誰","你是誰"]
  #targets = ["你是不是想人家","我是說你是我老婆"]
  df = pd.read_csv(filename)
  df = df.fillna('')
  sources = list(df["context"])
  targets = list(df["utterance"])
  scores = []
  for source, target in zip(sources,targets):
    if FLAGS.src_word_seg == 'word':
      source = (' ').join(jieba.lcut(source))
    elif FLAGS.src_word_seg == 'char':
      source = (' ').join([s for s in source])
    if FLAGS.trg_word_seg == 'word':
      target = (' ').join(jieba.lcut(target))
    elif FLAGS.trg_word_seg == 'char':
      target = (' ').join([t for t in target])
    src_token_ids = data_utils.convert_to_token(tf.compat.as_bytes(source), src_vocab_dict, False)
    trg_token_ids = data_utils.convert_to_token(tf.compat.as_bytes(target), trg_vocab_dict, False)
    trg_len = len(trg_token_ids)
    for i, bucket in enumerate(buckets):
      if bucket[0] >= len(src_token_ids):
        bucket_id = i
        break
    encoder_input, decoder_input, weight = model.get_batch({bucket_id: [(src_token_ids, [])]}, bucket_id)
    output = model.run(sess, encoder_input, decoder_input, weight, bucket_id)[:trg_len]
    output = [o[0][t] for t,o in zip(trg_token_ids,output)]
    output = np.mean(output)
    scores.append(output)
  scores = np.mean(scores) 
  return scores

if __name__ == '__main__':
  score = test(FLAGS.inference_data_path)
  print('coh1 score: ',score)
  if FLAGS.log_path:
      with open(FLAGS.log_path,"a") as f:
          f.write("coh1: %s\n"%score)
