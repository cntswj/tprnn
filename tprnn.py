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

import data_utils
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
    params['dec_W'] = (0.1 * randn).astype(config.floatX)

    dec_b = np.zeros(options['n_words'])
    params['dec_b'] = dec_b.astype(config.floatX)

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


def top_k_accuracy(y_prob, y, k=10):
    acc = []
    for p_, y_ in zip(y_prob, y):
        top_k = p_.argsort()[-k:][::-1]
        acc += [1. if y_ in top_k else 0.]
    return acc


def evaluate(f_prob, test_examples, batch_size=64):
    '''
    Evaluates trained model.
    '''
    n_test_examples = len(test_examples)
    loader = data_utils.Loader(test_examples, batch_size=batch_size)
    acc = []
    for _ in range(n_test_examples // batch_size + 1):
        batch_data = loader()
        labels = batch_data[-1]
        prob = f_prob(*batch_data[:-1])
        acc += top_k_accuracy(prob, labels)

    return sum(acc) / len(acc)


def simulate(f_pred, f_prob, seeds, n_timesteps=20, G=None):
    '''
    Simulates a cascade given seeding nodes.
    '''
    sequence = seeds
    # probs = []
    for _ in range(n_timesteps):
        # constructs input for current sequence.
        example = data_utils.convert_cascade_to_examples(sequence, G=G,
                                                         inference=True)
        data_batch = data_utils.prepare_minibatch([example], inference=True)
        prob = f_prob(*data_batch[:-1])[0]
        prob[sequence] = 0.
        prob /= prob.sum()
        pred = np.random.choice(range(len(prob)), p=prob)
        sequence += [pred]
        # probs += [prob]

    # print probs
    return sequence


def train(data_dir='data/digg/',
          dim_proj=512,
          n_words=20000,
          maxlen=20,
          batch_size=256,
          shuffle_for_batch=True,
          learning_rate=0.0001,
          global_steps=50000,
          disp_freq=100,
          save_freq=1000,
          test_freq=1000,
          saveto_file='params.npz',
          weight_decay=0.0005,
          reload_model=True,
          train=True):
    """
    Topo-LSTM model training.
    """
    options = locals().copy()
    saveto = data_dir + saveto_file

    # creates and initializes shared variables.
    print 'Initializing variables...'
    params = init_params(options)
    if reload_model:
        print 'reusing saved model.'
        load_params(saveto, params)
    tparams = init_tparams(params)

    # builds Topo-LSTM model
    print 'Building model...'
    model = tprnn_model.build_model(tparams, options)

    print 'Loading test data...'
    G, node_index = data_utils.load_graph(data_dir)
    test_examples = data_utils.load_examples(data_dir,
                                             dataset='test',
                                             node_index=node_index,
                                             maxlen=maxlen,
                                             G=G)
    print 'Loaded %d test examples' % len(test_examples)

    if train:
        # prepares training data.
        print 'Loading train data...'
        train_examples = data_utils.load_examples(data_dir,
                                                  dataset='train',
                                                  node_index=node_index,
                                                  maxlen=maxlen,
                                                  G=G)
        print 'Loaded %d training examples.' % len(train_examples)

        batch_loader = data_utils.Loader(train_examples,
                                         batch_size=batch_size,
                                         shuffle=shuffle_for_batch)

        # compiles updates.
        optimizer = downhill.build(algo='adam',
                                   loss=model['cost'],
                                   params=tparams.values(),
                                   inputs=model['data'])

        updates = optimizer.get_updates(max_gradient_elem=5.,
                                        learning_rate=learning_rate)

        f_update = theano.function(model['data'],
                                   model['cost'],
                                   updates=list(updates))

        # training loop.
        start_time = timeit.default_timer()

        # downhill.minimize(
        #     loss=cost,
        #     algo='adam',
        #     train=batch_loader,
        #     # inputs=input_list + [labels],
        #     # params=tparams.values(),
        #     # patience=0,
        #     max_gradient_clip=1,
        #     # max_gradient_norm=1,
        #     learning_rate=learning_rate,
        #     monitors=[('cost', cost)],
        #     monitor_gradients=False)

        n_examples = len(train_examples)
        batches_per_epoch = n_examples // batch_size + 1
        n_epochs = global_steps // batches_per_epoch + 1

        global_step = 0
        for _ in range(n_epochs):
            for _ in range(batches_per_epoch):
                cost = f_update(*batch_loader())

                if global_step % disp_freq == 0:
                    print 'global step %d, cost: %f' % (global_step, cost)

                # dump model parameters.
                if global_step % save_freq == 0:
                    params = unzip(tparams)
                    np.savez(saveto, **params)
                    pickle.dump(options, open('%s.pkl' % saveto, 'wb'), -1)

                # evaluate on test data.
                if global_step % test_freq == 0:
                    score = evaluate(model['f_prob'], test_examples)
                    print 'eval score: %f' % score

                global_step += 1

                # for debugging use.
                # if global_step > 1000:
                #     break

        end_time = timeit.default_timer()
        print 'time used: %d seconds.' % (end_time - start_time)

    err = evaluate(model['f_prob'], test_examples)
    print 'test error: %f' % err

    # runs some simulations for debugging.
    test_example = test_examples[1000]
    sequence = test_example['sequence']
    print 'true cascade: ', sequence

    seeds = sequence[:3]
    preds = simulate(model['f_pred'], model['f_prob'], seeds, G=G)
    print 'simulated: ', preds


if __name__ == '__main__':
    train()
