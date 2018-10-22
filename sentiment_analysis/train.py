import tensorflow as tf
from tensorflow.python.platform import gfile
import random
import os
import sys
import numpy as np
import dataset
import model
from settings import *

def sentence_cutter(sentence):
    sentence = [s for s in sentence]
    return (' ').join(sentence)

def create_model(session, mode):
  m = model.discriminator(VOCAB_SIZE,
                          UNIT_SIZE,
                          BATCH_SIZE,
                          MAX_LENGTH,
                          mode)
  ckpt = tf.train.get_checkpoint_state(model_dir)
  print('ckpt: ',ckpt)

  if ckpt:
    print("Reading model from %s" % ckpt.model_checkpoint_path)
    m.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    print("Create model with fresh parameters")
    session.run(tf.global_variables_initializer())

  return m

def train(file_path=file_path, token_path=token_path, mapping_path=mapping_path):
   if gfile.Exists(mapping_path) and gfile.Exists(token_path):
     print('Files have already been formed!')
   else:
     dataset.form_vocab_mapping(50000)
     vocab_map, _ = dataset.read_map(mapping_path)
     dataset.file_to_token(file_path, vocab_map)

   d = dataset.read_data(token_path)
   random.seed(SEED)
   random.shuffle(d)    
   
   train_set = d[:int(0.9 * len(d))]
   valid_set = d[int(-0.1 * len(d)):]

   sess = tf.Session()

   Model = create_model(sess, 'train')
   #Model = create_model(sess, 'valid')
   step = 0
   loss = 0

   while(True):
     step += 1
     encoder_input, encoder_length, target = Model.get_batch(train_set)
     '''
     print(encoder_input)
     print(encoder_length)
     print(target)
     exit()
     '''
     loss_train = Model.step(sess, encoder_input, encoder_length, target)
     loss += loss_train/CHECK_STEP
     if step % CHECK_STEP == 0:
       Model.mode = 'valid'
       temp_loss = 0
       for _ in range(100):
         encoder_input, encoder_length, target = Model.get_batch(valid_set)
         loss_valid = Model.step(sess, encoder_input, encoder_length, target)
         temp_loss += loss_valid/100.
       Model.mode = 'train'
       print("Train Loss: %s" % loss)
       print("Valid Loss: %s" % temp_loss)
       checkpoint_path = os.path.join(model_dir, 'dis.ckpt')
       Model.saver.save(sess, checkpoint_path, global_step = step)
       print("Model Saved!")
       loss = 0

def evaluate(mapping_path=mapping_path):
  vocab_map, _ = dataset.read_map(mapping_path)
  sess = tf.Session()
  Model = create_model(sess, 'test')
  Model.batch_size = 1
  
  sys.stdout.write('>')
  sys.stdout.flush()
  sentence = sys.stdin.readline()
  sentence = sentence_cutter(sentence)

  while(sentence):
    print('sentence: ',sentence)
    token_ids = dataset.convert_to_token(sentence, vocab_map)
    print('toekn_ids: ',token_ids)
    encoder_input, encoder_length, _ = Model.get_batch([(0, token_ids)]) 
    print('encoder_input: ',encoder_input, encoder_input.shape)
    print('encoder_length: ',encoder_length)
    score = Model.step(sess, encoder_input, encoder_length)
    print('Score: ' , score[0][0])
    print('>', end = '')
    sys.stdout.flush()
    sentence = sys.stdin.readline()
    sentence = sentence_cutter(sentence)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", default=None, dest="f" ,help="file path")
    parser.add_argument("--token_path", default=None, dest="t" ,help="token path")
    parser.add_argument("--mapping_path", default=None, dest="m" ,help="mapping path")
    args = parser.parse_args()
    if args.file_path: file_path = args.file_path
    if args.token_path: token_path = args.token_path
    if args.mapping_path: mapping_path = args.mapping_path
    #train(file_path, token_path, mapping_path)
    evaluate(mapping_path)
