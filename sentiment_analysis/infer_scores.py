import tensorflow as tf
import pandas as pd
import numpy as np
import argparse
import dataset
import model
from settings import *
from train import create_model, sentence_cutter
batch_size = 100

def inference(infer_file_path,mapping_path=mapping_path):
  vocab_map, _ = dataset.read_map(mapping_path)
  with tf.Session() as sess:
      Model = create_model(sess, 'test')
      Model.batch_size = batch_size 
      sentences = pd.read_csv(infer_file_path) 
      sentences = list(sentences["utterance"])
      sentences = map(lambda s:sentence_cutter(s),sentences)
      token_ids = map(lambda s:dataset.convert_to_token(s,vocab_map), sentences)
      token_ids = list(map(lambda x:(0,x), token_ids))
      corpus_num = len(token_ids) 
      iters = int(np.ceil(corpus_num/batch_size))
      scores = []
      for i in range(0,iters):
          token_id = token_ids[i:i+batch_size] 
          encoder_input, encoder_length, _ = Model.get_batch(token_id,shuffle=False) 
          score = Model.step(sess, encoder_input, encoder_length)
          scores.append(score)
      mean_score = np.mean(scores)
      return mean_score

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_data_path", default=None, dest="f" ,help="file path")
    parser.add_argument("--mapping_path", default=None, dest="m" ,help="mapping path")
    parser.add_argument("--log_path", dest="l" ,help="mean score log path")
    args = parser.parse_args()
    if args.f: infer_file_path = args.f
    if args.m: mapping_path = args.m
    mean_score = inference(infer_file_path,mapping_path)
    print("sentiment mean_score: ",mean_score)
    if args.l:
        with open(args.l,"a") as f:
            f.write("sent: %s\n"%mean_score)
