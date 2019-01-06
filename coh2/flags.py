import tensorflow as tf
import os
dirname = os.path.dirname(os.path.abspath(__file__))

flags = tf.app.flags
FLAGS = flags.FLAGS  
# infer data path
flags.DEFINE_string('inference_data_path', 'data/test.csv', 'inference data path')
# log performance
flags.DEFINE_string('log_path', None, 'inference score log path')
# model path
flags.DEFINE_string("model_dir","save", "Directory to store model checkpoints (defaults to ./runs)")

## glove = 100, fasttext = 300
flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of the embeddings")
flags.DEFINE_string("rnn_dim", '200,100,100,200', "Dimensionality of the RNN cell")
flags.DEFINE_integer("max_context_len", 30, "Truncate contexts to this length")
flags.DEFINE_integer("max_utterance_len", 30, "Truncate utterance to this length")
# for rnn dropout
flags.DEFINE_float('input_keep_prob', '1.0', 'step input dropout of saving model')
flags.DEFINE_float('output_keep_prob', '1.0', 'step output dropout of saving model')
flags.DEFINE_float('state_keep_prob', '1.0', 'step state dropout of saving model')

# Training Parameters
flags.DEFINE_boolean("pretrain_trainable", False, "fast2text embedding trainable or not")
flags.DEFINE_float("learning_rate", 0.0005, "Learning rate")
flags.DEFINE_float("decay_steps", 5000, "Decay learning rate every n steps")
flags.DEFINE_float("decay_rate", 0.98, "Decay learning rate every n steps")
flags.DEFINE_integer("batch_size", 64, "Batch size during training")
flags.DEFINE_string("optimizer", "Adam", "Optimizer Name (Adam, Adagrad, etc)")

# Training Config
flags.DEFINE_integer("loglevel", 20, "Tensorflow log level")
flags.DEFINE_integer("num_epochs", 25, "Number of training Epochs. Defaults to indefinite.")
flags.DEFINE_integer("eval_every", 2000, "Evaluate after this many train steps")
flags.DEFINE_integer("save_summary_steps", 100, "save_summary_steps")
flags.DEFINE_integer("log_step_count_steps", 100, "log_step_count_steps")
flags.DEFINE_integer("save_checkpoints_steps", 300, "save_checkpoints_steps")


# dynamic setting
## Data path
flags.DEFINE_string("input_dir", os.path.join(dirname,"data"), "Directory containing input data files 'train.tfrecords' and 'validation.tfrecords'")
flags.DEFINE_string("training_data_path", os.path.join(flags.FLAGS.input_dir,'train.tfrecords'), "path of training data")
flags.DEFINE_string("valid_data_path", os.path.join(flags.FLAGS.input_dir,'validation.tfrecords'), "path of validation data")
flags.DEFINE_string("vocab_path", os.path.join(flags.FLAGS.input_dir,'vocabulary.txt'), "Path to vocabulary.txt file")

with open(flags.FLAGS.vocab_path,'r') as f:
    vocab_size = len([row for row in f.readlines()]) 

## Model Parameters
flags.DEFINE_integer(
  "vocab_size",
  vocab_size,
  "The size of the vocabulary. Only change this if you changed the preprocessing")
last_rnn_dim = int(tf.flags.FLAGS.rnn_dim.split(',')[-1])
flags.DEFINE_integer("last_rnn_dim", last_rnn_dim, "Dimensionality of the RNN cell")

MAX_LEN = 20
