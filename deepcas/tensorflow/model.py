import sys
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell


def batched_scalar_mul(w, x):
    x_t = tf.transpose(x, [2, 0, 1, 3])
    shape = x_t.get_shape()
    x_t = tf.reshape(x_t, [int(shape[0]), -1])
    wx_t = tf.mul(w, x_t)
    res = tf.reshape(wx_t, [int(shape[0]), -1, int(shape[2]), int(shape[3])])
    res = tf.transpose(res, [1, 2, 0, 3])
    return res


def batched_scalar_mul3(w, x):
    x_t = tf.transpose(x, [1, 0, 2])
    shape = x_t.get_shape()
    x_t = tf.reshape(x_t, [int(shape[0]), -1])
    wx_t = tf.mul(w, x_t)
    res = tf.reshape(wx_t, [int(shape[0]), -1, int(shape[2])])
    res = tf.transpose(res, [1, 0, 2])
    return res


class DeepCas(object):
    def __init__(self, config, sess, node_vec):

        self.n_sequences = config.n_sequences
        self.learning_rate = config.learning_rate
        self.emb_learning_rate = config.emb_learning_rate
        self.training_iters = config.training_iters
        self.sequence_batch_size = config.sequence_batch_size
        self.batch_size = config.batch_size
        self.display_step = config.display_step

        self.embedding_size = config.embedding_size
        self.dropout_prob = config.dropout_prob
        self.vocab_size = config.vocab_size
        self.node_vec = node_vec
        # self.n_input = config.n_input
        self.n_steps = config.n_steps
        self.n_hidden_gru = config.n_hidden_gru
        self.scale1 = config.l1
        self.scale2 = config.l2
        self.scale = config.l1l2
        if config.activation == "tanh":
            self.activation = tf.tanh
        else:
            self.activation = tf.nn.relu
        self.max_grad_norm = config.max_grad_norm
        self.initializer = tf.random_normal_initializer(stddev=config.stddev)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.regularizer = tf.contrib.layers.l1_l2_regularizer(self.scale1, self.scale2)
        self.sess = sess
        self.name = "deepcas"

        self.build_input()
        self.build_var()
        self.loss, self.prob = self.build_model()

        # Define loss and optimizer
        loss = tf.reduce_mean(self.loss)
        cost = loss + self.scale * tf.add_n([self.regularizer(var) for var in tf.trainable_variables()])
        hits_score = tf.reduce_mean(tf.cast(tf.nn.in_top_k(self.prob, tf.squeeze(self.y), 10), tf.float32))

#        optim = self.optimizer
#        gvs = optim.compute_gradients(cost)
#        capped_gvs = [(tf.clip_by_norm(grad, self.max_grad_norm), var)
#                      if not 'embedding' in var.name
#                      else (tf.clip_by_norm(tf.mul(grad, 0.005), self.max_grad_norm), var)
#                      for grad, var in gvs]
#        train_op = optim.apply_gradients(capped_gvs)

        var_list1 = [var for var in tf.trainable_variables() if 'embedding' not in var.name]
        var_list2 = [var for var in tf.trainable_variables() if 'embedding' in var.name]
        opt1 = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        opt2 = tf.train.AdamOptimizer(learning_rate=self.emb_learning_rate)
        grads = tf.gradients(cost, var_list1 + var_list2)
        grads1 = [tf.clip_by_norm(grad, self.max_grad_norm) for grad in grads[:len(var_list1)]]
        grads2 = [tf.clip_by_norm(grad, self.max_grad_norm) for grad in grads[len(var_list1):]]
        train_op1 = opt1.apply_gradients(zip(grads1, var_list1))
        train_op2 = opt2.apply_gradients(zip(grads2, var_list2))
        train_op = tf.group(train_op1, train_op2)

        self.cost = cost
        self.loss = loss
        self.hits_score = hits_score
        self.train_op = train_op

        init_op = tf.initialize_all_variables()
        self.sess.run(init_op)

    def build_input(self):
        self.x = tf.placeholder(tf.int32, [None, self.n_sequences, self.n_steps], name="x")
        self.y = tf.placeholder(tf.int32, [None, 1], name="y")
        self.sz = tf.placeholder(tf.float32, [None, 1], name="sz")

    def build_var(self):
        with tf.variable_scope(self.name):
            with tf.variable_scope('embedding'):
                self.embedding = tf.get_variable('embedding', initializer=tf.constant(self.node_vec, dtype=tf.float32))
            with tf.variable_scope('BiGRU'):
                self.gru_fw_cell = rnn_cell.GRUCell(self.n_hidden_gru)
                self.gru_bw_cell = rnn_cell.GRUCell(self.n_hidden_gru)
            with tf.variable_scope('attention'):
                self.p_step = tf.get_variable('p_step', initializer=self.initializer([1, self.n_steps]), dtype=tf.float32)
                self.a_geo = tf.get_variable('a_geo', initializer=self.initializer([1]))
            with tf.variable_scope('dense'):
                self.weights = {
                    'dense1': tf.get_variable('dense1_weight', initializer=self.initializer([2 * self.n_hidden_gru,
                                                                                             self.vocab_size])),
                }
                self.biases = {
                    'dense1': tf.get_variable('dense1_bias', initializer=self.initializer([self.vocab_size])),
                }

    def build_model(self):
        with tf.device('/gpu:0'):
            with tf.variable_scope('deepcas'):
                with tf.variable_scope('embedding'):
                    x_vector = tf.nn.dropout(tf.nn.embedding_lookup(self.embedding, self.x),
                                             self.dropout_prob)
                    # (batch_size, n_sequences, n_steps, n_input)

                with tf.variable_scope('BiGRU'):
                    x_vector = tf.transpose(x_vector, [1, 0, 2, 3])
                    # (n_sequences, batch_size, n_steps, n_input)
                    x_vector = tf.reshape(x_vector, [-1, self.n_steps, self.embedding_size])
                    # (n_sequences*batch_size, n_steps, n_input)

                    x_vector = tf.transpose(x_vector, [1, 0, 2])
                    # (n_steps, n_sequences*batch_size, n_input)
                    x_vector = tf.reshape(x_vector, [-1, self.embedding_size])
                    # (n_steps*n_sequences*batch_size, n_input)

                    # Split to get a list of 'n_steps' tensors of shape (n_sequences*batch_size, n_input)
                    x_vector = tf.split(0, self.n_steps, x_vector)

                    outputs, _, _ = rnn.bidirectional_rnn(self.gru_fw_cell, self.gru_bw_cell, x_vector,
                                                          dtype=tf.float32)
                    hidden_states = tf.transpose(tf.pack(outputs), [1, 0, 2])
                    # (n_sequences*batch_size, n_steps, 2*n_hidden_gru)
                    hidden_states = tf.transpose(tf.reshape(hidden_states, [self.n_sequences, -1, self.n_steps, 2 * self.n_hidden_gru]), [1, 0, 2, 3])
                    # (batch_size, n_sequences, n_steps, 2*n_hiddent_gru)

                with tf.variable_scope('attention'):
                    # attention over sequence steps
                    attention_step = tf.nn.softmax(self.p_step)
                    attention_step = tf.transpose(attention_step, [1, 0])
                    attention_result = batched_scalar_mul(attention_step, hidden_states)
                    # (batch_size, n_sequences, n_steps, 2*n_hiddent_gru)

                    # attention over sequence batches
                    p_geo = tf.sigmoid(self.a_geo)
                    attention_batch = tf.pow(tf.mul(p_geo, tf.ones_like(self.sz)), tf.div(1.0 + tf.log(self.sz), tf.log(2.0)))

                    attention_batch_seq = tf.tile(attention_batch, [1, self.sequence_batch_size])
                    for i in range(1, self.n_sequences / self.sequence_batch_size):
                        attention_batch_seq = tf.concat(1,
                                                        [attention_batch_seq,
                                                         tf.tile(tf.pow(1 - attention_batch, i) * attention_batch,
                                                                 [1, self.sequence_batch_size])])
                    attention_batch_lin = tf.reshape(attention_batch_seq, [-1, 1])

                    shape = attention_result.get_shape()
                    shape = [-1, int(shape[1]), int(shape[2]), int(shape[3])]
                    attention_result_t = tf.mul(tf.reshape(tf.transpose(attention_result, [2, 3, 0, 1]), [shape[2], shape[3], -1]), tf.squeeze(attention_batch_lin))
                    attention_result = tf.reshape(tf.transpose(attention_result_t, [2, 0, 1]), [-1, shape[1], shape[2], shape[3]])
                    hidden_graph = tf.reduce_sum(attention_result, reduction_indices=[1, 2])

                with tf.variable_scope('dense'):
                    dense1 = self.activation(tf.add(tf.matmul(hidden_graph, self.weights['dense1']), self.biases['dense1']))
                    prob = tf.nn.softmax(dense1)
                    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.squeeze(self.y), logits=dense1)

                return loss, prob

    def train_batch(self, x, y, sz):
        self.sess.run(self.train_op, feed_dict={self.x: x, self.y: y, self.sz: sz})

    def get_loss(self, x, y, sz):
        return self.sess.run(self.loss, feed_dict={self.x: x, self.y: y, self.sz: sz})

    def get_score(self, x, y, sz):
        return self.sess.run(self.hits_score, feed_dict={self.x: x, self.y: y, self.sz: sz})

    def get_prob(self, x, y, sz):
        return self.sess.run(self.prob, feed_dict={self.x: x, self.y: y, self.sz: sz})
