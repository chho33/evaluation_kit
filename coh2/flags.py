import tensorflow as tf
import os
dirname = os.path.dirname(os.path.abspath(__file__))

FLAGS = tf.flags.FLAGS

# infer data path
tf.app.flags.DEFINE_string('inference_data_path', 'data/test_raw.csv', 'inference data path')
# log performance
tf.app.flags.DEFINE_string('log_path', None, 'inference score log path')

# Data path
tf.flags.DEFINE_string("input_dir", os.path.join(dirname,"data"), "Directory containing input data files 'train.tfrecords' and 'validation.tfrecords'")
tf.flags.DEFINE_string("training_data_path", os.path.join(tf.flags.FLAGS.input_dir,'train.tfrecords'), "path of training data")
tf.flags.DEFINE_string("valid_data_path", os.path.join(tf.flags.FLAGS.input_dir,'validation.tfrecords'), "path of validation data")
tf.flags.DEFINE_string("vocab_path", os.path.join(tf.flags.FLAGS.input_dir,'vocabulary.txt'), "Path to vocabulary.txt file")

with open(tf.flags.FLAGS.vocab_path,'r') as f:
    vocab_size = len([row for row in f.readlines()]) 
# Model Parameters
tf.flags.DEFINE_integer(
  "vocab_size",
  vocab_size,
  "The size of the vocabulary. Only change this if you changed the preprocessing")
## glove = 100, fasttext = 300
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of the embeddings")
tf.flags.DEFINE_string("rnn_dim", '200,100,100,200', "Dimensionality of the RNN cell")
last_rnn_dim = int(tf.flags.FLAGS.rnn_dim.split(',')[-1])
tf.flags.DEFINE_integer("last_rnn_dim", last_rnn_dim, "Dimensionality of the RNN cell")
tf.flags.DEFINE_integer("max_context_len", 30, "Truncate contexts to this length")
tf.flags.DEFINE_integer("max_utterance_len", 30, "Truncate utterance to this length")
# for rnn dropout
tf.app.flags.DEFINE_float('input_keep_prob', '1.0', 'step input dropout of saving model')
tf.app.flags.DEFINE_float('output_keep_prob', '1.0', 'step output dropout of saving model')
tf.app.flags.DEFINE_float('state_keep_prob', '1.0', 'step state dropout of saving model')

# Training Parameters
tf.flags.DEFINE_float("learning_rate", 0.001, "Learning rate")
tf.flags.DEFINE_float("decay_steps", 5000, "Decay learning rate every n steps")
tf.flags.DEFINE_float("decay_rate", 0.98, "Decay learning rate every n steps")
tf.flags.DEFINE_integer("batch_size", 64, "Batch size during training")
tf.flags.DEFINE_string("optimizer", "Adam", "Optimizer Name (Adam, Adagrad, etc)")

# Training Config
tf.flags.DEFINE_string("model_dir","save/20181021", "Directory to store model checkpoints (defaults to ./runs)")
tf.flags.DEFINE_integer("loglevel", 20, "Tensorflow log level")
tf.flags.DEFINE_integer("num_epochs", 25, "Number of training Epochs. Defaults to indefinite.")
tf.flags.DEFINE_integer("eval_every", 2000, "Evaluate after this many train steps")
tf.flags.DEFINE_integer("save_summary_steps", 100, "save_summary_steps")
tf.flags.DEFINE_integer("log_step_count_steps", 100, "log_step_count_steps")
tf.flags.DEFINE_integer("save_checkpoints_steps", 300, "save_checkpoints_steps")

MAX_LEN = 20
print(FLAGS.log_path)
print(FLAGS.inference_data_path)
