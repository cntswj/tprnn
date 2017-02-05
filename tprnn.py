# Notes on dims:
#   length = sequences.shape[0]
#   n_samples = sequences.shape[1]
#   seqs: dims = (time, example), shape = length * n_samples
#   a-masks: dims = (time, example, mask), shape = length * n_samples * length
#   q-masks: dims = (example, mask), shape = n_samples * length
#   labels: dims = (example, label), shape = n_samples * 0

import read_data
import numpy as np
import theano
from theano import tensor
from theano import config
from collections import OrderedDict
import time
import six.moves.cPickle as pickle
import tprnn_model
from optimizers import adadelta


def numpy_floatX(data):
    return np.asarray(data, dtype=config.floatX)


def ortho_weight(ndim):
    W = np.random.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return u.astype(config.floatX)


def init_params(options):
    """
    Initializes values of shared variables.
    """
    params = OrderedDict()

    # word embedding, shape = #words * dim_proj
    randn = np.random.rand(options['n_words'],
                           options['dim_proj'])
    params['Wemb'] = (0.1 * randn).astype(config.floatX)

    # shape = dim_proj * (4*dim_proj)
    lstm_W = np.concatenate([ortho_weight(options['dim_proj']),
                             ortho_weight(options['dim_proj']),
                             ortho_weight(options['dim_proj']),
                             ortho_weight(options['dim_proj'])], axis=1)
    params['lstm_W'] = lstm_W

    # shape = dim_proj * (4*dim_proj)
    lstm_U = np.concatenate([ortho_weight(options['dim_proj']),
                             ortho_weight(options['dim_proj']),
                             ortho_weight(options['dim_proj']),
                             ortho_weight(options['dim_proj'])], axis=1)
    params['lstm_U'] = lstm_U

    lstm_b = np.zeros((4 * options['dim_proj'],))
    params['lstm_b'] = lstm_b.astype(config.floatX)

    # shape = dim_proj * 0
    theta = 0.1 * np.random.randn(options['dim_proj'])
    params['theta'] = theta.astype(config.floatX)

    return params


def init_tparams(params):
    '''
    Set up Theano shared variables.
    '''
    tparams = OrderedDict()
    for kk, pp in params.items():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams


def unzip(zipped):
    """
    When we pickle the model. Needed for the GPU stuff.
    """
    new_params = OrderedDict()
    for kk, vv in zipped.items():
        new_params[kk] = vv.get_value()
    return new_params


def load_params(path, params):
    pp = np.load(path)
    for kk, vv in params.items():
        if kk not in pp:
            raise Warning('%s is not in the archive' % kk)
        params[kk] = pp[kk]

    return params


data_dir = 'data/toy/'


def tprnn_training(dim_proj=64,
                   n_words=10000,
                   maxlen=100,
                   batch_size=200,
                   is_shuffle_for_batch=False,
                   lrate=0.01,
                   max_epochs=10000,
                   disp_freq=100,
                   save_freq=1000,
                   saveto=data_dir + 'saved/params.npz',
                   reload_model=False,
                   decay_lstm_W=0.01,
                   decay_lstm_U=0.01,
                   decay_lstm_b=0.01,
                   decay_theta=0.01):
    """
    Topo-LSTM model training.
    """
    options = locals().copy()

    # loads all training data.
    print 'Loading data...'
    training_examples, _ = read_data.load_cascade_examples(data_dir, dataset='train')

    # makes batches of data indices
    batches = read_data.get_minibatches_idx(len(training_examples), batch_size, is_shuffle_for_batch)

    # creates and initializes shared variables.
    print 'Initializing variables...'
    params = init_params(options)
    if reload_model:
        load_params('lstm_model.npz', params)

    tparams = init_tparams(params)

    # builds tprnn model
    print 'Building model...'
    input_list, labels, cost, f_prob, f_pred = tprnn_model.build_model(tparams, options)

    # generates gradients and optimizers.
    print 'Generating gradients and optimizers...'
    grads = tensor.grad(cost, wrt=list(tparams.values()))
    lr = tensor.scalar(name='lr')
    f_grad_shared, f_update = adadelta(lr, tparams, grads, input_list, labels, cost)

    # training loop.
    print 'Training starts.'
    start_time = time.time()
    print 'start time: ', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))
    global_steps = 0
    for i_epoch in range(max_epochs):
        for batch in batches:
            global_steps += 1

            # prepares batch of training data.
            batch_training_examples = [training_examples[i] for i in batch]
            batch_data = read_data.prepare_batch_data(batch_training_examples)

            # training steps.
            cost = f_grad_shared(*batch_data)
            f_update(lrate)

            # checks for bad cost.
            if np.isnan(cost) or np.isinf(cost):
                print('bad cost detected: ', cost)
                return

            # shows cost.
            if np.mod(global_steps, disp_freq) == 0:
                print 'epoch=%d, global_step=%d, cost=%f' % (i_epoch, global_steps, cost)

            # saves model.
            if saveto and np.mod(global_steps, save_freq) == 0:
                params = unzip(tparams)
                np.savez(saveto, **params)
                pickle.dump(options, open('%s.pkl' % saveto, 'wb'), -1)

    end_time = time.time()
    print 'Training finished! Time elapsed ', end_time - start_time, 's'


if __name__ == '__main__':
    tprnn_training()
