#!/usr/bin/env python
# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import sys
# import os
# import time
import math
import logging
import random
import os
# import numpy as np
import tensorflow as tf
import numpy as np
# import pdb

from metrics import top_k_accuracy

flags = tf.app.flags

flags.DEFINE_string("data_dir", 'data/twitter', "data directory.")
flags.DEFINE_integer("max_epoch", 10, "max epochs.")
flags.DEFINE_integer("max_batches", 50000, "max batches.")
flags.DEFINE_integer("emb_dim", 64, "embedding dimension.")
flags.DEFINE_float("lr", 0.001, "initial learning rate.")
flags.DEFINE_integer("disp_freq", 100, "frequency to output.")
flags.DEFINE_integer("save_freq", 1000, "frequency to save model.")
flags.DEFINE_boolean("reload_model", 1, "whether to reuse saved model.")

FLAGS = flags.FLAGS


class Options(object):
    """options used by CDK model."""

    def __init__(self):
        # model options.
        self.emb_dim = FLAGS.emb_dim

        self.train_data = os.path.join(FLAGS.data_dir, 'train.txt')
        self.test_data = os.path.join(FLAGS.data_dir, 'test.txt')
        self.save_path = os.path.join(FLAGS.data_dir, 'embedded_ic/embedded_ic.ckpt')

        self.max_epoch = FLAGS.max_epoch
        self.max_batches = FLAGS.max_batches

        self.lr = FLAGS.lr

        self.disp_freq = FLAGS.disp_freq
        self.save_freq = FLAGS.save_freq

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

        self.saver = tf.train.Saver()
        if options.reload_model:
            self.saver.restore(session, options.save_path)

    def _buildIndex(self):
        # compute an index of the users that appear at least once in the training and testing cascades.
        opts = self._options

        train_user_set = set()
        test_user_set = set()

        for line in open(opts.train_data):
            if len(line.strip()) == 0:
                continue
            query, cascade = line.strip().split(' ', 1)
            users = [query] + cascade.split()[::2][:self._maxlen]
            train_user_set.update(users)

        for line in open(opts.test_data):
            if len(line.strip()) == 0:
                continue
            query, cascade = line.strip().split(' ', 1)
            users = [query] + cascade.split()[::2][:self._maxlen]
            train_user_set.update(users)

        user_set = train_user_set | test_user_set

        pos = 0
        for user in user_set:
            self._u2idx[user] = pos
            pos += 1
            self._idx2u.append(user)
        opts.user_size = len(user_set)
        logging.info("user_size : %d" % (opts.user_size))

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

        emb_user = tf.Variable(tf.random_uniform([opts.user_size, opts.emb_dim], -0.1, 0.1), name="emb_user")
        global_step = tf.Variable(0, trainable=False, name="global_step", dtype=tf.int32)

        u_emb = tf.nn.embedding_lookup(emb_user, u)
        v_emb = tf.nn.embedding_lookup(emb_user, v)

        u_0 = tf.slice(u_emb, [0], [1])[0]
        v_0 = tf.slice(v_emb, [0], [1])[0]
        v_0_all = tf.squeeze(tf.slice(emb_user, [0, 0], [-1, 1]))

        u_1_n = tf.slice(u_emb, [1], [-1])
        v_1_n = tf.slice(v_emb, [1], [-1])
        v_1_n_all = tf.squeeze(tf.slice(emb_user, [0, 1], [-1, -1]))

        x = u_0 + v_0 + tf.reduce_sum(tf.square(u_1_n - v_1_n))
        x_all = u_0 + v_0_all + tf.reduce_sum(tf.square(v_1_n_all - u_1_n), axis=1)

        f = tf.sigmoid(-x)
        f_all = tf.sigmoid(-x_all)
        # one = tf.convert_to_tensor(1.0, dtype = tf.float32)

        loss1 = -(1.0 - p_uv_hat / p_v_hat) * tf.log(1.0 - f) - (p_uv_hat / p_v_hat) * tf.log(f)
        loss2 = -tf.log(1.0 - f)

        lr = tf.train.exponential_decay(opts.lr, global_step, 1000, 0.96, staircase=True)

        train1 = tf.train.AdamOptimizer(lr).minimize(loss1, global_step=global_step)
        train2 = tf.train.AdamOptimizer(lr).minimize(loss2, global_step=global_step)

        self.u = u
        self.v = v
        self.p_uv_hat = p_uv_hat
        self.p_v_hat = p_v_hat
        self.emb_user = emb_user
        self.global_step = global_step
        self.p_uv = f
        self.p_u_all = f_all
        self.lr = lr
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
        """sample a user V, which can not be the initial user of the given cascade."""
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
        """sample all users u for sampled cascade and user v."""
        ulist = []
        for user in self._train_cascades[cascadeId]:
            if user == vId:
                break
            ulist.append(user)

        return ulist

    def computePv(self, v, ul):
        pv = 1.0
        assert len(ul) > 0, (v, ul)
        for u in ul:
            feed_dict = {self.u: u, self.v: v}
            p_uv = self._session.run(self.p_uv, feed_dict=feed_dict)
            pv = pv * (1.0 - p_uv)
        p_v = 1.0 - pv
        return p_v

    def compute_ll(self):
        opts = self._options
        ll = 0.0
        for i in xrange(opts.train_size):
            cur = self._train_cascades[i]

            # nodes in the cascade.
            for j in xrange(1, len(cur)):
                ll += math.log(self.computePv(cur[j], cur[:j]) + 1e-8)

            # nodes not in the cascade.
            uset = set(self._train_cascades[i])
            u_set = set(self._idx2u) - uset
            for user in u_set:
                ll += math.log(1 - self.computePv(user, cur) + 1e-8)

        return ll

    def train(self):
        """train the model."""
        opts = self._options
        # loss_list = []
        # oldL = float("-inf")

        n_pairs = 0
        for _ in xrange(opts.max_batches):
            cascade_id = self.SampleCascade()
            v_in_cascade, v_id = self.SampleV(cascade_id)
            u_id_list = self.SampleU(cascade_id, v_id)
            if v_in_cascade:
                # print(self._train_cascades[cascade_id], v_id, u_id_list)
                p_v_hat = self.computePv(v_id, u_id_list)
                for u in u_id_list:
                    p_uv_hat = self._session.run(self.p_uv, feed_dict={self.u: u, self.v: v_id})
                    feed_dict = {self.u: u, self.v: v_id, self.p_uv_hat: p_uv_hat, self.p_v_hat: p_v_hat}
                    (lr, loss, step, _) = self._session.run([self.lr, self.loss1, self.global_step, self.train1],
                                                            feed_dict=feed_dict)
            else:
                for u in u_id_list:
                    feed_dict = {self.u: u, self.v: v_id}
                    (lr, loss, step, _) = self._session.run([self.lr, self.loss2, self.global_step, self.train2],
                                                            feed_dict=feed_dict)
            # loss_list.append(loss)
            if n_pairs % opts.disp_freq == 0:
                print(n_pairs, loss)
            #     L = self.compute_ll()
            #     print('Step %, ll=%f' % (global_step, L))
            #     if L < oldL:
            #         break
            if n_pairs % opts.save_freq == 0:
                self.saver.save(self._session, opts.save_path)

            n_pairs += 1

    def evaluate(self, k=10):
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
    logging.basicConfig(level=logging.INFO)
    # if not FLAGS.train_data:
    #    logging.error("train file not found.")
    #    sys.exit(1)
    options = Options()
    with tf.Graph().as_default(), tf.Session() as session:
        model = Embedded_IC(options, session)
        model.train()
        print(model.evaluate(k=10))


if __name__ == "__main__":
    tf.app.run()
