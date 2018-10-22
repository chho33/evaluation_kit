import argparse
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from data_utils import build_word_dict, build_dataset, get_word_dict, get_dataset, data_loader, batch_iter, map_get_words
from model.rnn_lm import RNNLanguageModel
from model.bi_rnn_lm import BiRNNLanguageModel

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def inference(inference_file, vocabulary_size, args):
    if args.model == "rnn":
        model = RNNLanguageModel(vocabulary_size, args)
    elif args.model == "birnn":
        model = BiRNNLanguageModel(vocabulary_size, args)
    else:
        raise ValueError("Unknown model option {}.".format(args.model))

    # Define training procedure
    global_step = tf.Variable(0, trainable=False)
    params = tf.trainable_variables()
    gradients = tf.gradients(model.loss, params)
    clipped_gradients, _ = tf.clip_by_global_norm(gradients, 10.0)
    #global_step = tf.train.get_global_step()
    learning_rate = tf.train.exponential_decay(args.learning_rate, global_step, args.decay_steps, args.decay_rate, staircase=True)
    #learning_rate = tf.Print(learning_rate,[learning_rate],"learning_rate: ")
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.apply_gradients(zip(clipped_gradients, params), global_step=global_step)

    saver = tf.train.Saver(max_to_keep = 5)
    with tf.Session() as sess:
        def infer_step(batch_x):
            if isinstance(batch_x, tf.Tensor):
                batch_x = sess.run(batch_x)
                batch_x = [row.strip().split() for row in batch_x]
                batch_x = list(map(lambda x: list(map(lambda y:int(y),x)),batch_x))
            feed_dict = {model.x: batch_x, model.keep_prob: args.keep_prob}
            logits = sess.run([model.logits], feed_dict=feed_dict)[0]
            scores = list(map(lambda x:list(map(lambda y:softmax(y),x)),logits))
            return scores 

        # Initialize all variables
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(args.model_dir)
        if ckpt:
            saver.restore(sess, ckpt.model_checkpoint_path)

        scores = []
        if isinstance(inference_file,str): 
            batch_x = data_loader(inference_file,args.batch_size,1,args.shuffle)
            while True: 
                try: 
                    score = infer_step(batch_x)
                    scores+=(score)
                except tf.errors.OutOfRangeError:
                    print('inference finished...')
                    break
        elif isinstance(inference_file,(list,np.ndarray, np.generic)):   
            batchs = batch_iter(inference_file,args.batch_size,1)
            for batch_x in batchs:
                score = infer_step(batch_x)
                scores+=(score)
        #scores = np.mean(scores) 
        return scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="birnn", help="rnn | birnn")
    parser.add_argument("--embedding_size", type=int, default=300, help="embedding size.")
    parser.add_argument("--num_layers", type=int, default=3, help="RNN network depth.")
    parser.add_argument("--num_hidden", type=int, default=300, help="RNN network size.")
    parser.add_argument("--keep_prob", type=float, default=1, help="dropout keep prob.")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="learning rate.")
    parser.add_argument("--decay_rate", type=float, default=0.98, help="Decay learning rate every n steps")
    parser.add_argument("--decay_steps", type=int, default=10000, help="Decay learning rate every n steps")

    parser.add_argument("--batch_size", type=int, default=64, help="batch size.")
    parser.add_argument("--num_epochs", type=int, default=100, help="number of epochs.")
    parser.add_argument("--shuffle", type=bool, default=False, help="if shuffle the dataset?")

    parser.add_argument("--model_dir", type=str, default="save", help="model params dir")
    parser.add_argument("--check_steps", type=int, default=300, help="every n steps check one time")
    parser.add_argument("--inference_data_path", type=str, default="data/test_raw.csv", help="inference data path")
    parser.add_argument("--log_path", type=str, help="mean loss log path")
    args = parser.parse_args()

    word_dict_file = "data/words_char.json"
    word_dict = get_word_dict(word_dict_file)

    # file not yet transformed to one hot
    data = pd.read_csv(args.inference_data_path) 
    data = list(data["utterance"])
    data = map_get_words(data)
    data = list(build_dataset(data,word_dict,True))
    scores = inference(data, len(word_dict), args)

    # file already transformed to one hot
    #inference_file = "data/valid_char1.txt"
    #scores = inference(inference_file, len(word_dict), args)
    #data = get_dataset(inference_file)

    data = list(map(lambda x:x[1:(np.sum(np.sign(x))-1)],data))
    mean_log_scores = []
    for d,s in zip(data,scores):
        mean_log_score = np.mean([-np.log(s[i][n]) for i,n in enumerate(d)])
        #print(mean_log_score)
        mean_log_scores.append(mean_log_score)
    mean_log_score = np.mean(mean_log_scores)
    print("LM:mean_log_score: ",mean_log_score) 
    print("LM:mean_perplexity: ",np.exp(np.mean(mean_log_scores)))
    if args.log_path: 
        with open(args.log_path,"a") as f:
            f.write("LM: %s\n"%mean_log_score)
