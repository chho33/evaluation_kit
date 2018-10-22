from flags import FLAGS, MAX_LEN
import tensorflow as tf

class Model(object):
    def __init__(self):
        self.labels = tf.placeholder(tf.int32, (None,1))
        #self.context = tf.placeholder(tf.int, [])
        #self.utterance = tf.placeholder(tf.int, [])
        #self.context_len = tf.reduce_sum(tf.sign(self.context), 1)
        #self.utterance_len = tf.reduce_sum(tf.sign(self.utterance), 1)
        self.context_embedded = tf.placeholder(tf.float32, (None,MAX_LEN,FLAGS.embedding_dim))
        self.utterance_embedded = tf.placeholder(tf.float32, (None,MAX_LEN,FLAGS.embedding_dim))
        self.context_len = tf.placeholder(tf.int32, (None,)) 
        self.utterance_len = tf.placeholder(tf.int32, (None,)) 
        #self.context_len = tf.reduce_sum(tf.sign(self.context_embedded), axis=1)[0]
        #self.utterance_len = tf.reduce_sum(tf.sign(self.utterance_embedded), axis=1)[0]

        #embeddings_W = get_embeddings(FLAGS)
        #context_embedded = tf.nn.embedding_lookup(
        #    embeddings_W, self.context, name="embed_context")
        #utterance_embedded = tf.nn.embedding_lookup(
        #    embeddings_W, self.utterance, name="embed_utterance")
  
        with tf.variable_scope("rnn") as vs:
            rnn_dims = FLAGS.rnn_dim.split(',')
            cell = [ tf.nn.rnn_cell.LSTMCell(
                         int(rnn_dim),
                         forget_bias=1.0,
                         use_peepholes=True) for rnn_dim in rnn_dims]
            cell = tf.nn.rnn_cell.MultiRNNCell(cell) 
            cell = tf.nn.rnn_cell.DropoutWrapper(
                cell,
                input_keep_prob=FLAGS.input_keep_prob,
                output_keep_prob=FLAGS.output_keep_prob,
                state_keep_prob=FLAGS.state_keep_prob)

            tmp_concat = tf.concat([self.context_embedded, self.utterance_embedded],0)
            tmp_concat_len = tf.concat([self.context_len, self.utterance_len],0)
            rnn_outputs, rnn_states = tf.nn.dynamic_rnn(
                cell,
                tmp_concat,
                tmp_concat_len,
                dtype=tf.float32)
            if isinstance(rnn_states,list) or isinstance(rnn_states,tuple):
                rnn_states = rnn_states[0]
            encoding_context, encoding_utterance = tf.split(rnn_states.h,2,0)

        with tf.variable_scope("prediction") as vs:
            M = tf.get_variable("M",
              shape=[FLAGS.last_rnn_dim, FLAGS.last_rnn_dim],
              initializer=tf.truncated_normal_initializer())
        
            # "Predict" a  response: c * M
            generated_response = tf.matmul(encoding_context, M)
            generated_response = tf.expand_dims(generated_response, 2)
            encoding_utterance = tf.expand_dims(encoding_utterance, 2)
        
            # Dot product between generated response and actual response
            # (c * M) * r
            #logits = tf.batch_matmul(generated_response, encoding_utterance, True)
            logits = tf.matmul(generated_response, encoding_utterance, True)
            self.logits = logits = tf.squeeze(logits, [2])
            self.probs = probs = tf.sigmoid(logits)
            losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.to_float(self.labels))
            # Mean loss across the batch of examples
            self.loss = tf.reduce_mean(losses, name="mean_loss")
