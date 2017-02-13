import os
import numpy as np
# from sklearn.linear_model import LogisticRegression

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.models import load_model
from keras.utils import np_utils
import metrics

data_dir = 'data/memes'
emb_size = 128
save_path = os.path.join(data_dir, 'emb.h5')

print data_dir


def load_embedding():
    embeddings = {}

    nodes_file = os.path.join(data_dir, 'seen_nodes.txt')
    with open(nodes_file, 'rb') as f:
        for line in f:
            embeddings[line.strip()] = np.zeros(emb_size)

    filepath = os.path.join(data_dir, 'deepwalk.txt')
    with open(filepath, 'rb') as f:
        for line in f:
            v, emb = line.strip().split(' ', 1)
            emb = map(float, emb.split())
            embeddings[v] = emb

    return embeddings


embeddings = load_embedding()
node_map = {v: i for i, v in enumerate(embeddings.keys())}


def convert_cascade_to_examples(sequence):
    X = []
    y = []
    for i, node in enumerate(sequence[:-1]):
        prefix = sequence[:i + 1]
        label = sequence[i + 1]
        if label not in node_map:
            continue
        embs = [embeddings[v] for v in prefix if v in embeddings]
        if len(embs) == 0:
            continue
        emb_mean = np.array(embs).mean(axis=0)
        X += [emb_mean]
        y += [label]

    y = [node_map[v] for v in y]
    return X, y


def load_dataset(dataset=None, maxlen=30):
    X = []
    y = []
    filepath = os.path.join(data_dir, dataset + '.txt')
    with open(filepath, 'rb') as f:
        for line in f:
            action, cascade = line.strip().split(' ', 1)
            sequence = cascade.split()[::2][:maxlen]
            X_sub, y_sub = convert_cascade_to_examples(sequence)
            X.extend(X_sub)
            y.extend(y_sub)

    X = np.array(X)
    y = np.array(y, dtype=np.int)
    return X, y


print 'training...'
if os.path.isfile(save_path):
    model = load_model(save_path)
else:
    # model = LogisticRegression()
    # model.fit(X, y)

    n_words = len(node_map)
    model = Sequential()
    model.add(Dense(n_words, input_shape=(emb_size,)))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')

X, y = load_dataset(dataset='train')
y_cat = np_utils.to_categorical(y)
model.fit(X, y_cat, nb_epoch=30, verbose=1)
model.save(save_path)

print 'testing...'
# prob_test = model.predict_proba(X_test)
X_test, y_test = load_dataset(dataset='test')
y_prob = model.predict_proba(X_test)

# n_classes = len(model.classes_)
# class_map = {c: i for i, c in enumerate(model.classes_)}
# y_test = [class_map[c] if c in class_map else -1 for c in y_test]

acc = metrics.top_k_accuracy(y_prob, y_test, k=10)
print np.array(acc).mean()

acc = metrics.top_k_accuracy(y_prob, y_test, k=50)
print np.array(acc).mean()

acc = metrics.top_k_accuracy(y_prob, y_test, k=100)
print np.array(acc).mean()
