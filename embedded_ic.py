from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import sys
# import time
import random
import os
# import numpy as np
import tensorflow as tf
import numpy as np
# import pdb

from metrics import top_k_accuracy

random.seed(0)

flags = tf.app.flags

flags.DEFINE_string("data_dir", 'data/memes', "data directory.")
flags.DEFINE_integer("max_samples", 300000, "max number of samples.")
flags.DEFINE_integer("emb_dim", 64, "embedding dimension.")
flags.DEFINE_integer("disp_freq", 100, "frequency to output.")
flags.DEFINE_integer("save_freq", 10000, "frequency to save.")
flags.DEFINE_integer("test_freq", 10000, "frequency to evaluate.")
flags.DEFINE_float("lr", 0.001, "initial learning rate.")
flags.DEFINE_boolean("reload_model", 0, "whether to reuse saved model.")

FLAGS = flags.FLAGS


class Options(object):
    """options used by CDK model."""

    def __init__(self):
        # model options.
        self.emb_dim = FLAGS.emb_dim

        self.train_data = os.path.join(FLAGS.data_dir, 'train.txt')
        self.test_data = os.path.join(FLAGS.data_dir, 'test.txt')
        self.save_path = os.path.join(FLAGS.data_dir, 'embedded_ic/embedded_ic.ckpt')

        self.max_samples = FLAGS.max_samples

        self.lr = FLAGS.lr

        self.disp_freq = FLAGS.disp_freq
        self.save_freq = FLAGS.save_freq
        self.test_freq = FLAGS.test_freq

        self.reload_model = FLAGS.reload_model


class Embedded_IC(object):
    """Embedded IC model."""

    def __init__(self, options, session):
        self._maxlen = 30
        self._options = options
        self._session = session
        self._u2idx = {}
        self._idx2u = []
        self._buildIndex()
        self._n_words = len(self._u2idx)
        self._train_cascades = self._readFromFile(options.train_data)
        self._test_cascades = self._readFromFile(options.test_data)
        self._options.train_size = len(self._train_cascades)
        self._options.test_size = len(self._test_cascades)
        self.buildGraph()

        if options.reload_model:
            self.saver.restore(session, options.save_path)

    def _getUsers(self, datafile):
        user_set = set()
        for line in open(datafile, 'rb'):
            if len(line.strip()) == 0:
                continue
            query, cascade = line.strip().split(' ', 1)
            users = [query] + cascade.split()[::2][:self._maxlen]
            user_set.update(users)

        return user_set

    def _buildIndex(self):
        """
        compute an index of the users that appear at least once in the training and testing cascades.
        """
        opts = self._options
        user_set = self._getUsers(opts.train_data) | self._getUsers(opts.test_data)
        self._idx2u = list(user_set)
        self._u2idx = {u: i for i, u in enumerate(self._idx2u)}
        opts.user_size = len(user_set)

    def _readFromFile(self, filename):
        """read all cascade from training or testing files. """
        t_cascades = []
        for line in open(filename):
            if len(line.strip()) == 0:
                continue
            query, cascade = line.strip().split(' ', 1)
            userlist = [query] + cascade.split()[::2][:self._maxlen]
            userlist = [self._u2idx[v] for v in userlist]

            if len(userlist) > 1:
                t_cascades.append(userlist)

        return t_cascades

    def buildGraph(self):
        opts = self._options
        u = tf.placeholder(tf.int32, shape=())
        v = tf.placeholder(tf.int32, shape=())
        p_v_hat = tf.placeholder(tf.float32, shape=())
        p_uv_hat = tf.placeholder(tf.float32, shape=())

        emb_sender = tf.Variable(tf.random_uniform([opts.user_size, opts.emb_dim], -0.1, 0.1),
                                 name='emb_sender')

        emb_receiver = tf.Variable(tf.random_uniform([opts.user_size, opts.emb_dim], -0.1, 0.1),
                                   name='emb_receiver')

        u_emb = tf.nn.embedding_lookup(emb_sender, u)
        v_emb = tf.nn.embedding_lookup(emb_receiver, v)

        u_0 = tf.slice(u_emb, [0], [1])[0]
        v_0 = tf.slice(v_emb, [0], [1])[0]
        v_0_all = tf.squeeze(tf.slice(emb_receiver, [0, 0], [-1, 1]))

        u_1_n = tf.slice(u_emb, [1], [-1])
        v_1_n = tf.slice(v_emb, [1], [-1])
        v_1_n_all = tf.squeeze(tf.slice(emb_receiver, [0, 1], [-1, -1]))

        x = u_0 + v_0 + tf.reduce_sum(tf.square(u_1_n - v_1_n))
        x_all = u_0 + v_0_all + tf.reduce_sum(tf.square(v_1_n_all - u_1_n), axis=1)

        f = tf.sigmoid(-x)
        f_all = tf.sigmoid(-x_all)

        eps = 1e-8
        loss1 = -(1.0 - p_uv_hat / (p_v_hat + eps)) * tf.log(1.0 - f + eps) - (p_uv_hat / (p_v_hat + eps)) * tf.log(f + eps)
        loss2 = -tf.log(1.0 - f + eps)

        # weight_decay = 0.0005 * tf.nn.l2_loss(emb_user)
        # loss1 += weight_decay
        # loss1 += weight_decay

        tvars = tf.trainable_variables()

        grads1, _ = tf.clip_by_global_norm(tf.gradients(loss1, tvars), clip_norm=1.)
        grads2, _ = tf.clip_by_global_norm(tf.gradients(loss2, tvars), clip_norm=1.)

        train1 = tf.train.AdamOptimizer(opts.lr).apply_gradients(zip(grads1, tvars))
        train2 = tf.train.AdamOptimizer(opts.lr).apply_gradients(zip(grads2, tvars))

        self.u = u
        self.v = v
        self.p_uv_hat = p_uv_hat
        self.p_v_hat = p_v_hat
        self.emb_sender = emb_sender
        self.emb_receiver = emb_receiver
        self.p_uv = f
        self.p_u_all = f_all
        self.loss1 = loss1
        self.loss2 = loss2
        self.train1 = train1
        self.train2 = train2

        tf.initialize_all_variables().run()
        self.saver = tf.train.Saver()

    def SampleCascade(self):
        """sample a cascade randomly."""
        opts = self._options
        return random.randint(0, opts.train_size - 1)

    def SampleV(self, cascadeId):
        """
        sample a user V, which can not be the initial user of the given cascade.
        with 0.5 probability V is in the given cascade (positive sample).
        """
        opts = self._options
        c = self._train_cascades[cascadeId]
        if random.random() < 0.5:
            while True:
                idx = random.randint(0, opts.user_size - 1)
                if idx != c[0]:
                    break
        else:
            i = random.randint(1, len(c) - 1)
            idx = c[i]

        v_in_cascade = idx in set(self._train_cascades[cascadeId])
        return v_in_cascade, idx

    def SampleU(self, cascadeId, vId):
        """sample users u given sampled cascade and user v."""
        ulist = []
        for user in self._train_cascades[cascadeId]:
            if user == vId:
                break
            ulist.append(user)

        return ulist

    def computePv(self, v, ul):
        '''computes \hat{P}_v'''
        pv = 1.0
        assert len(ul) > 0, (v, ul)
        for u in ul:
            feed_dict = {self.u: u, self.v: v}
            p_uv = self._session.run(self.p_uv, feed_dict=feed_dict)
            pv = pv * (1.0 - p_uv)
        p_v = 1.0 - pv
        return p_v

    # def compute_ll(self):
    #     opts = self._options
    #     ll = 0.0
    #     for i in xrange(opts.train_size):
    #         cur = self._train_cascades[i]

    #         # nodes in the cascade.
    #         for j in xrange(1, len(cur)):
    #             ll += math.log(self.computePv(cur[j], cur[:j]))

    #         # nodes not in the cascade.
    #         uset = set(self._train_cascades[i])
    #         u_set = set(self._idx2u) - uset
    #         for user in u_set:
    #             ll += math.log(1 - self.computePv(user, cur))

    #     return ll

    def train(self):
        """train the model."""
        opts = self._options
        n_samples = 0
        for _ in xrange(opts.max_samples):
            cascade_id = self.SampleCascade()
            v_in_cascade, v_id = self.SampleV(cascade_id)
            u_id_list = self.SampleU(cascade_id, v_id)
            if v_in_cascade:
                p_v_hat = self.computePv(v_id, u_id_list)
                for u in u_id_list:
                    p_uv_hat = self._session.run(self.p_uv,
                                                 feed_dict={self.u: u, self.v: v_id})
                    loss, _ = self._session.run([self.loss1, self.train1],
                                                feed_dict={self.u: u,
                                                           self.v: v_id,
                                                           self.p_uv_hat:
                                                           p_uv_hat,
                                                           self.p_v_hat: p_v_hat})
            else:
                for u in u_id_list:
                    loss, _ = self._session.run([self.loss2, self.train2],
                                                feed_dict={self.u: u, self.v: v_id})

            n_samples += 1

            if n_samples % opts.disp_freq == 0:
                print('step %d, loss=%f' % (n_samples, loss))
            if n_samples % opts.save_freq == 0:
                self.saver.save(self._session, opts.save_path)
            if n_samples % opts.test_freq == 0:
                print(self.evaluate(k=10))
                print(self.evaluate(k=50))
                print(self.evaluate(k=100))

    def evaluate(self, k=10):
        '''evaluate the model.'''
        acc_list = []
        for c in self._test_cascades:
            p = np.ones(self._n_words)
            for i, u in enumerate(c[:-1]):
                pf = c[:i + 1]
                p_u_all = self._session.run(self.p_u_all, feed_dict={self.u: u})
                p_u_all[pf] = 0.
                p *= (1 - p_u_all)
                y = c[i + 1]
                acc = top_k_accuracy(1 - p, y, k=k)
                acc_list += acc

        return sum(acc_list) / len(acc_list)


def main(_):
    options = Options()
    with tf.Graph().as_default(), tf.Session() as session:
        model = Embedded_IC(options, session)
        model.train()


if __name__ == "__main__":
    tf.app.run()
