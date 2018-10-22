import tensorflow as tf
import numpy as np
import functools
import os
from flags import FLAGS, MAX_LEN
from model import Model
from utils import data_loader 
from fastText import load_model
fasttext_model = load_model("data/cc.zh.300.bin")

def get_data(data):
    data = map(lambda x:x.decode('utf-8') if not isinstance(x,str) else x, data)
    data = map(lambda x:x.strip().split(), data)
    data = map(lambda x:list(map(lambda y:fasttext_model.get_word_vector(y),x))[:MAX_LEN] ,data)
    data = list(data)
    data_len = list(map(lambda x:len(x), data))
    data_padding = map(lambda x:(MAX_LEN-x)*[np.zeros(300)], data_len) 
    data = list(map(lambda x: x[0]+x[1], zip(data,data_padding)))
    return data, data_len

def train_step(sess, model, data, train_op,\
               global_step, summary_op, summary_writer): 
    batch_context, batch_utterance, batch_labels = sess.run(data)
    batch_context, batch_context_len = get_data(batch_context) 
    batch_utterance, batch_utterance_len = get_data(batch_utterance)
    batch_labels = batch_labels.reshape(-1,1)
    #print('batch_context: ',len(batch_context),len(batch_context[0]))
    #print('batch_utterance: ',len(batch_utterance),len(batch_utterance[0]))
    #print('batch_labels: ',len(batch_labels),len(batch_labels[0]))
    #print('=====================')
    feed_dict = {model.context_embedded: batch_context,
                 model.utterance_embedded: batch_utterance, 
                 model.context_len: batch_context_len, 
                 model.utterance_len: batch_utterance_len, 
                 model.labels: batch_labels}
    _, step, summaries, loss = sess.run([train_op, global_step, summary_op, model.loss], feed_dict=feed_dict)
    summary_writer.add_summary(summaries, step)
    return loss

def test_step(sess, model, valid_file, step=None, summary_op=None, summary_writer=None): 
    iterator = data_loader(valid_file, FLAGS.batch_size, 1)
    sess.run(iterator.initializer)
    losses, iters = 0, 0
    elems = iterator.get_next()

    while True:
        try:
            batch_context, batch_utterance, batch_labels = sess.run(elems)
            batch_context, batch_context_len = get_data(batch_context) 
            batch_utterance, batch_utterance_len = get_data(batch_utterance)
            batch_labels = batch_labels.reshape(-1,1)
        except tf.errors.OutOfRangeError:
            break
        feed_dict = {model.context_embedded: batch_context, 
                     model.utterance_embedded: batch_utterance, 
                     model.context_len: batch_context_len, 
                     model.utterance_len: batch_utterance_len, 
                     model.labels: batch_labels}

        if summary_op is not None:
            summaries, loss = sess.run([summary_op, model.loss], feed_dict=feed_dict)
            summary_writer.add_summary(summaries, step)
        else:
            loss = sess.run([model.loss], feed_dict=feed_dict)[0]
        losses += loss
        iters += 1
    return losses/iters 

def infer_step(sess, model, infer_file, jieba): 
    iterator = data_loader(infer_file, FLAGS.batch_size, 1, False, "infer")
    sess.run(iterator.initializer)
    elems = iterator.get_next()

    probs = []
    while True:
        try:
            try:
                batch_context, batch_utterance, _ = sess.run(elems)
            except ValueError:
                batch_context, batch_utterance = sess.run(elems)
            batch_context = list(map(lambda d:(' ').join(jieba.lcut(d.decode())) ,batch_context))
            batch_utterance = list(map(lambda d:(' ').join(jieba.lcut(d.decode())) ,batch_utterance))
            batch_context, batch_context_len = get_data(batch_context) 
            batch_utterance, batch_utterance_len = get_data(batch_utterance)
        except tf.errors.OutOfRangeError:
            break
        feed_dict = {model.context_embedded: batch_context, 
                     model.utterance_embedded: batch_utterance, 
                     model.context_len: batch_context_len, 
                     model.utterance_len: batch_utterance_len} 
        prob = sess.run([model.probs], feed_dict=feed_dict)
        prob = list(np.squeeze(prob))
        probs += prob
    probs = np.mean(probs)
    return probs 

def run(files,mode="train",jieba=None):
    model = Model() 
    global_step = tf.Variable(0, trainable=False)
    trainable_params = tf.trainable_variables()
    gradients = tf.gradients(model.loss, trainable_params)
    clipped_gradients, _ = tf.clip_by_global_norm(gradients, 10.0)
    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, FLAGS.decay_steps, FLAGS.decay_rate, staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.apply_gradients(zip(clipped_gradients, trainable_params), global_step=global_step)
    
    #summary
    loss_summary = tf.summary.scalar("loss", model.loss)
    summary_op = tf.summary.merge([loss_summary])
    saver = tf.train.Saver(max_to_keep = 3)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
        if ckpt:
            saver.restore(sess, ckpt.model_checkpoint_path)

        if mode == "train":
            train_file, valid_file = files
            iterator = data_loader(train_file, FLAGS.batch_size, FLAGS.num_epochs)
            sess.run(iterator.initializer)
            data = iterator.get_next()
            train_summary_writer = tf.summary.FileWriter("%s-train"%FLAGS.model_dir)
            test_summary_writer = tf.summary.FileWriter("%s-test"%FLAGS.model_dir)
            while True:
                try:
                    train_loss = train_step(sess, model, data, train_op, global_step, summary_op, train_summary_writer)
                except tf.errors.OutOfRangeError:
                    print('training finish...')
                    break
                step = tf.train.global_step(sess, global_step)
                if step % FLAGS.save_checkpoints_steps == 1 & step!=1:
                    print('step: ',step)
                    test_loss = test_step(sess, model, valid_file, step, summary_op, test_summary_writer)
                    print("\tloss: {}".format(train_loss))
                    print("\ttest loss: {}".format(test_loss))
                    print("\tlearning_rate: {}".format(sess.run(learning_rate)))
                    checkpoint_path = os.path.join(FLAGS.model_dir, "MLE.ckpt")
                    saver.save(sess, checkpoint_path, global_step = global_step)
                    print("Saving model at step %s"%step)
        elif mode == "infer":
            infer_mean_prob = infer_step(sess, model, files[0], jieba)
            return infer_mean_prob

        elif mode == "test_loss":
            mean_loss = test_step(sess, model, files[0])
            return mean_loss

if __name__ == "__main__":
    train_file = "data/train.csv"
    valid_file = "data/valid.csv"
    run([train_file, valid_file], mode="train")
