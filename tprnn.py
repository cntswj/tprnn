# TODO: reload model.

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
import six.moves.cPickle as pickle  # @UnresolvedImport
import tprnn_model

import pdb


def numpy_floatX(data):
    return np.asarray(data, dtype=config.floatX)


def adadelta(lr, tparams, grads, input_list, labels, cost):
    """
    An adaptive learning rate optimizer
    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [ADADELTA]_.

    .. [ADADELTA] Matthew D. Zeiler, *ADADELTA: An Adaptive Learning
       Rate Method*, arXiv:1212.5701.
    """
    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.items()]
    running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.items()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.items()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function(input_list + [labels], cost, updates=zgup + rg2up,
                                    on_unused_input='ignore',
                                    name='adadelta_f_grad_shared')

    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]

    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]

    f_update = theano.function([lr], [], updates=ru2up + param_up,
                               on_unused_input='ignore',
                               name='adadelta_f_update')

    return f_grad_shared, f_update


def ortho_weight(ndim):
    W = np.random.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return u.astype(config.floatX)


def init_params(options):
    """
    Initializes values of shared variables.
    """
    print 'init shared variables......'
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
                   n_words=100000,
                   maxlen=100,
                   batch_size=200,
                   is_shuffle_for_batch=False,
                   lrate=0.001,
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
    training_examples, _ = read_data.load_cascade_examples(data_dir, dataset='train')

    # makes batches: tuples of (index, batch)
    batches = read_data.get_minibatches_idx(len(training_examples), batch_size, is_shuffle_for_batch)

    # creates and initializes shared variables.
    params = init_params(options)
    if reload_model:
        load_params('lstm_model.npz', params)

    tparams = init_tparams(params)

    # builds tprnn model
    print 'Generate models...'
    input_list, labels, cost, f_prob, f_pred = tprnn_model.build_model(tparams, options)

    # generates gradients and optimizers.
    grads = tensor.grad(cost, wrt=list(tparams.values()))
    lr = tensor.scalar(name='lr')
    f_grad_shared, f_update = adadelta(lr, tparams, grads, input_list, labels, cost)

    # training loop.
    print 'Start training models...'
    start_time = time.time()
    print 'start time: ', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))
    global_steps = 0
    for i_epoch in range(max_epochs):
        for _, batch in batches:
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
    print 'Start running tprnn_training...'
    tprnn_training()
