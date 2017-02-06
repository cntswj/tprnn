# Notes on dims:
#   length = sequences.shape[0]
#   n_samples = sequences.shape[1]
#   seqs: dims = (time, example), shape = length * n_samples
#   a-masks: dims = (time, example, mask), shape = length * n_samples * length
#   q-masks: dims = (example, mask), shape = n_samples * length
#   labels: dims = (example, label), shape = n_samples * 0

import numpy as np
import theano
# from theano import tensor
from theano import config
from collections import OrderedDict
import timeit
import six.moves.cPickle as pickle
import downhill

import read_data
import tprnn_model


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


def train(dim_proj=64,
          n_words=10000,
          maxlen=100,
          batch_size=64,
          shuffle_for_batch=False,
          learning_rate=0.01,
          max_epochs=100,
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

    # creates and initializes shared variables.
    print 'Initializing variables...'
    params = init_params(options)
    if reload_model:
        load_params('lstm_model.npz', params)

    tparams = init_tparams(params)

    # builds tprnn model
    print 'Building model...'
    input_list, labels, cost, f_prob, f_pred = tprnn_model.build_model(tparams, options)

    # prepares training data.
    print 'Loading data...'
    training_examples, _ = read_data.load_cascade_examples(data_dir, dataset='train')
    batch_loader = read_data.Loader(training_examples,
                                    batch_size=batch_size,
                                    shuffle=shuffle_for_batch)

    # training loop.
    start_time = timeit.default_timer()

    downhill.minimize(
        loss=cost,
        algo='adam',
        train=batch_loader,
        params=tparams.values(),
        inputs=input_list + [labels],
        # patience=0,
        max_gradient_clip=1,
        # max_gradient_norm=1,
        learning_rate=learning_rate,
        monitor_gradients=False)

    end_time = timeit.default_timer()
    print 'time used: %d seconds.' % (end_time - start_time)

    # dump model parameters.
    params = unzip(tparams)
    np.savez(saveto, **params)
    pickle.dump(options, open('%s.pkl' % saveto, 'wb'), -1)


if __name__ == '__main__':
    train()
