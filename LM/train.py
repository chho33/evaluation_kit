import argparse
import os
import numpy as np
import tensorflow as tf
from data_utils import build_word_dict, build_dataset, batch_iter, get_word_dict, get_dataset, data_loader
from model.rnn_lm import RNNLanguageModel
from model.bi_rnn_lm import BiRNNLanguageModel


def train(train_file, test_file, vocabulary_size, args):
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

    # Summary
    loss_summary = tf.summary.scalar("loss", model.loss)
    summary_op = tf.summary.merge([loss_summary])
    saver = tf.train.Saver(max_to_keep = 5)
    with tf.Session() as sess:
    #with tf.train.MonitoredTrainingSession(checkpoint_dir='/tmp/checkpoints') as sess:
        def train_step(batch_x):
            batch_x = sess.run(batch_x)
            batch_x = [row.strip().split() for row in batch_x]
            batch_x = list(map(lambda x: list(map(lambda y:int(y),x)),batch_x))
            feed_dict = {model.x: batch_x, model.keep_prob: args.keep_prob}
            _, step, summaries, loss = sess.run([train_op, global_step, summary_op, model.loss], feed_dict=feed_dict)
            train_summary_writer.add_summary(summaries, step)

            if step % 100 == 1 and step!=1:
                print("step {0}: loss = {1}".format(step, loss))

        def test_perplexity(test_file, step):
            #test_batches = batch_iter(test_file, args.batch_size, 1)
            iterator = data_loader(test_file, args.batch_size, 1)
            sess.run(iterator.initializer)
            losses, iters = 0, 0

            #while not sess.should_stop(): 
            while True: 
                try:
                    test_batch_x = iterator.get_next()
                    batch_x = sess.run(test_batch_x)
                except tf.errors.OutOfRangeError:
                    break
                batch_x = [row.strip().split() for row in batch_x]
                batch_x = list(map(lambda x: list(map(lambda y:int(y),x)),batch_x))
                feed_dict = {model.x: batch_x, model.keep_prob: 1.0}
                summaries, loss = sess.run([summary_op, model.loss], feed_dict=feed_dict)
                test_summary_writer.add_summary(summaries, step)
                losses += loss
                iters += 1

            return np.exp(losses / iters)

        #batches = batch_iter(train_data, args.batch_size, args.num_epochs)
        iterator = data_loader(train_file,args.batch_size, args.num_epochs)
        sess.run(iterator.initializer)
        # Initialize all variables
        sess.run(tf.global_variables_initializer())
        train_summary_writer = tf.summary.FileWriter(args.model + "-train", sess.graph)
        test_summary_writer = tf.summary.FileWriter(args.model + "-test", sess.graph)
        ckpt = tf.train.get_checkpoint_state(args.model_dir)
        if ckpt:
            saver.restore(sess, ckpt.model_checkpoint_path)

        #while not sess.should_stop(): 
        while True: 
            try: 
                batch_x = iterator.get_next()
                train_step(batch_x)
            except tf.errors.OutOfRangeError:
                print('training finish...')
                break
            step = tf.train.global_step(sess, global_step)
            if step % args.check_steps == 1 & step!=1:
                print('step: ', step)
                perplexity = test_perplexity(test_file, step)
                print("\ttest perplexity: {}".format(perplexity))
                print("\tlearning_rate: {}".format(sess.run(learning_rate)))
                checkpoint_path = os.path.join(args.model_dir, "MLE.ckpt")
                #global_step = tf.Print(global_step,[global_step],"global_step: ")
                saver.save(sess, checkpoint_path, global_step = global_step)
                print('Saving model at step %s' % step)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="birnn", help="rnn | birnn")
    parser.add_argument("--embedding_size", type=int, default=300, help="embedding size.")
    parser.add_argument("--num_layers", type=int, default=3, help="RNN network depth.")
    parser.add_argument("--num_hidden", type=int, default=300, help="RNN network size.")
    parser.add_argument("--keep_prob", type=float, default=1.0, help="dropout keep prob.")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="learning rate.")
    parser.add_argument("--decay_rate", type=float, default=0.98, help="Decay learning rate every n steps")
    parser.add_argument("--decay_steps", type=int, default=10000, help="Decay learning rate every n steps")

    parser.add_argument("--batch_size", type=int, default=64, help="batch size.")
    parser.add_argument("--num_epochs", type=int, default=20, help="number of epochs.")
    parser.add_argument("--model_dir", type=str, default="save/20181021", help="model params dir")
    parser.add_argument("--check_steps", type=int, default=300, help="every n steps check one time")
    args = parser.parse_args()

    train_file = "data/train_char.txt"
    test_file = "data/valid_char.txt"
    word_dict_file = "data/words_char.json"
    word_dict = get_word_dict(word_dict_file)
    #train_data = get_dataset(train_file)
    #test_data = get_dataset(test_file)

    train(train_file, test_file, len(word_dict), args)
