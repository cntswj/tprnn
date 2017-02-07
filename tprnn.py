import numpy as np
import theano
# from theano import tensor
from theano import config
from collections import OrderedDict
import timeit
import six.moves.cPickle as pickle
import downhill
# import pdb
# import pprint

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

    # word embedding, shape = (n_words, dim_proj)
    randn = np.random.randn(options['n_words'],
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

    # decoding matrix
    randn = np.random.randn(options['dim_proj'],
                            options['n_words'])
    params['Wout'] = (0.1 * randn).astype(config.floatX)

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


def train(data_dir='data/dblp/',
          dim_proj=128,
          n_words=1000000,
          maxlen=50,
          batch_size=128,
          shuffle_for_batch=True,
          learning_rate=0.001,
          max_epochs=100,
          disp_freq=100,
          save_freq=100,
          saveto_file='params.npz',
          reload_model=False,
          decay_lstm_W=0.01,
          decay_lstm_U=0.01,
          decay_lstm_b=0.01,
          decay_Wout=0.01):
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
    print 'Loaded %d training examples.' % len(training_examples)
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
    saveto = data_dir + saveto_file
    np.savez(saveto, **params)
    pickle.dump(options, open('%s.pkl' % saveto, 'wb'), -1)


if __name__ == '__main__':
    train()
