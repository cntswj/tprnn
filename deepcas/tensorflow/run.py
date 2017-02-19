# import sys
import six.moves.cPickle as pickle
import time
import pprint
import numpy as np
import tensorflow as tf

from model import DeepCas
import metrics

# import gzip
tf.set_random_seed(0)

NUM_THREADS = 20

tf.flags.DEFINE_float("learning_rate", 0.001, "learning_rate.")
tf.flags.DEFINE_float("emb_learning_rate", 0.00001, "embedding learning_rate.")
tf.flags.DEFINE_integer("sequence_batch_size", 10, "sequence batch size.")
tf.flags.DEFINE_integer("batch_size", 128, "batch size.")
tf.flags.DEFINE_integer("n_hidden_gru", 128, "hidden gru size.")
tf.flags.DEFINE_float("l1", 5e-5, "l1.")
tf.flags.DEFINE_float("l2", 1e-8, "l2.")
tf.flags.DEFINE_float("l1l2", 1.0, "l1l2.")
tf.flags.DEFINE_string("activation", "relu", "activation function.")
tf.flags.DEFINE_integer("n_sequences", 30, "num of sequences.")
tf.flags.DEFINE_integer("training_iters", 20000, "max training iters.")
tf.flags.DEFINE_integer("display_step", 100, "display step.")
tf.flags.DEFINE_integer("eval_step", 1000, "evaluation step.")
tf.flags.DEFINE_integer("embedding_size", 128, "embedding size.")
# tf.flags.DEFINE_integer("n_input", 50, "input size.")
tf.flags.DEFINE_integer("n_steps", 10, "num of step.")
# tf.flags.DEFINE_integer("n_hidden_dense1", 32, "dense1 size.")
# tf.flags.DEFINE_integer("n_hidden_dense2", 16, "dense2 size.")
tf.flags.DEFINE_string("version", "v4", "data version.")
tf.flags.DEFINE_integer("max_grad_norm", 10, "gradient clip.")
tf.flags.DEFINE_float("stddev", 0.01, "initialization stddev.")
tf.flags.DEFINE_float("dropout_prob", 0.7, "dropout probability.")

config = tf.flags.FLAGS

assert config.n_sequences % config.sequence_batch_size == 0


def get_batch(x, y, sz, step, batch_size=128):
    batch_x = np.zeros((batch_size, len(x[0]), len(x[0][0])))
    batch_y = np.zeros((batch_size, 1)).astype(np.int)
    batch_sz = np.zeros((batch_size, 1))
    start = step * batch_size % len(x)
    for i in range(batch_size):
        batch_y[i, 0] = y[(i + start) % len(x)]
        batch_sz[i, 0] = sz[(i + start) % len(x)]
        batch_x[i, :] = np.array(x[(i + start) % len(x)])

    return batch_x, batch_y, batch_sz


def shuffle_data(x, y, sz):
    n_samples = len(x)
    indices = np.random.permutation(n_samples)
    x_ = [x[i] for i in indices]
    y_ = [y[i] for i in indices]
    sz_ = [sz[i] for i in indices]
    return x_, y_, sz_


version = config.version
x_train, y_train, sz_train, vocab_size = pickle.load(open('data/data_train.pkl', 'r'))
x_test, y_test, sz_test, _ = pickle.load(open('data/data_test.pkl', 'r'))
config.vocab_size = vocab_size

node_vec = pickle.load(open('data/node_vec.pkl', 'r'))

x_train, y_train, sz_train = shuffle_data(x_train, y_train, sz_train)

training_iters = config.training_iters
batch_size = config.batch_size
display_step = min(config.display_step, len(sz_train) / batch_size)

np.set_printoptions(precision=2)
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
start = time.time()
model = DeepCas(config, sess, node_vec)
step = 1

# Keep training until reach max iterations or max_try
train_loss = []
while step < training_iters:
    batch_x, batch_y, batch_sz = get_batch(x_train, y_train, sz_train, step, batch_size=batch_size)
    model.train_batch(batch_x, batch_y, batch_sz)
    train_loss.append(model.get_loss(batch_x, batch_y, batch_sz))
    if step % display_step == 0:
        # Calculate batch loss
        test_losses = []
        # test_score = []
        test_probs = []
        test_ys = []
        for test_step in range(len(y_test) / batch_size):
            test_x, test_y, test_sz = get_batch(x_test, y_test, sz_test, test_step, batch_size=batch_size)
            test_losses.append(model.get_loss(test_x, test_y, test_sz))
            # test_score.append(model.get_score(test_x, test_y, test_sz))
            test_probs.append(model.get_prob(test_x, test_y, test_sz))
            test_ys.append(test_y[:, 0].tolist())

        if step % config.eval_step == 0:
            test_probs = np.concatenate(test_probs, axis=0)
            test_ys = np.concatenate(test_ys, axis=0)
            pprint.pprint(metrics.portfolio(test_probs, test_ys, k_list=[10, 50, 100]))

        print("#" + str(step / display_step) +
              ", Training loss= " + "{:.6f}".format(np.mean(train_loss)) +
              ", Test loss= " + "{:.6f}".format(np.mean(test_losses))
              # ", Test Hits@k= " + "{:.6f}".format(np.mean(test_score))
              )
        train_loss = []
    step += 1

print "Finished!\n----------------------------------------------------------------"
print "Time:", time.time() - start
