import os
# import re
# import sys
import codecs
import networkx as nx
import numpy as np
import pickle
# import pprint
from theano import config


def load_graph(data_dir):
    # loads nodes observed in any cascade.
    node_file = os.path.join(data_dir, 'seen_nodes.txt')
    with open(node_file, 'rb') as f:
        seen_nodes = [x.strip() for x in f]

    # builds node index
    node_index = {v: i for i, v in enumerate(seen_nodes)}

    # loads graph
    graph_file = os.path.join(data_dir, 'graph.txt')
    pkl_file = os.path.join(data_dir, 'graph.pkl')

    if os.path.isfile(pkl_file):
        G = pickle.load(open(pkl_file, 'rb'))
    else:
        G = nx.Graph()
        G.name = data_dir
        n_nodes = len(node_index)
        G.add_nodes_from(range(n_nodes))
        with open(graph_file, 'rb') as f:
            f.next()
            for line in f:
                u, v = line.strip().split()
                if (u in node_index) and (v in node_index):
                    u = node_index[u]
                    v = node_index[v]
                    G.add_edge(u, v)
        pickle.dump(G, open(pkl_file, 'wb'))

    return G, node_index


def convert_cascade_to_examples(sequence,
                                G=None,
                                inference=False):
    length = len(sequence)

    # grows the series of dags incrementally.
    examples = []
    dag = nx.DiGraph()
    for i, node in enumerate(sequence):
        # grows the DAG.
        prefix = sequence[: i + 1]
        dag.add_node(node)
        predecessors = set(G[node]) & set(prefix)
        dag.add_edges_from(
            [(v, node) for v in predecessors])

        # (optional) adds chronological edges
        if i > 0:
            dag.add_edge(sequence[i - 1], node)

        if i == length - 1 and not inference:
            return examples

        if i < length - 1 and inference:
            continue

        # compiles example from DAG.
        node_pos = {v: i for i, v in enumerate(prefix)}
        prefix_len = len(prefix)
        topo_mask = np.zeros((prefix_len, prefix_len), dtype=np.int)
        for i_v, v in enumerate(prefix):
            i_p = [node_pos[x] for x in dag.predecessors(v)]
            topo_mask[i_v, i_p] = 1

        neighborhoods = [G[v] for v in prefix]
        all_neighbors = [w for neighbors in neighborhoods for w in neighbors]
        frontier = list(set(all_neighbors) - set(prefix))

        if not inference:
            label = sequence[i + 1]
        else:
            label = None

        example = {'sequence': prefix,
                   'topo_mask': topo_mask,
                   'nbr_mask': frontier,
                   'label': label}

        if not inference:
            examples.append(example)
        else:
            return example


def load_examples(data_dir, dataset=None, G=None, node_index=None, maxlen=None):
    """
    Load the train/dev/test data
    Return: list of example tuples
    """

    pkl_path = os.path.join(data_dir, dataset + '.pkl')
    if os.path.isfile(pkl_path):
        print 'pickle exists.'
        examples = pickle.load(open(pkl_path, 'rb'))
    else:
        # loads cascades
        filename = os.path.join(data_dir, dataset + '.txt')
        examples = []
        with codecs.open(filename, 'r', encoding='utf-8') as input_file:
            for line_index, line in enumerate(input_file):
                # parses the input line.
                query, cascade = line.strip().split(' ', 1)
                sequence = [query] + cascade.split(' ')[::2]
                if maxlen is not None:
                    sequence = sequence[:maxlen]
                sequence = [node_index[x] for x in sequence]

                sub_examples = convert_cascade_to_examples(sequence, G=G)
                examples.extend(sub_examples)

        pickle.dump(examples, open(pkl_path, 'wb'))

    return examples


def prepare_minibatch(tuples, inference=False, n_words=None):
    '''
    produces a mini-batch of data in format required by model.
    '''
    seqs = [t['sequence'] for t in tuples]
    lengths = map(len, seqs)
    n_timesteps = max(lengths)
    n_samples = len(tuples)

    # prepare sequences data
    seqs_matrix = np.zeros((n_timesteps, n_samples)).astype('int32')
    for i, seq in enumerate(seqs):
        seqs_matrix[: lengths[i], i] = seq

    # prepare topo-masks data
    topo_masks = [t['topo_mask'] for t in tuples]
    topo_masks_tensor = np.zeros((n_timesteps, n_samples, n_timesteps)).astype(config.floatX)
    for i, topo_mask in enumerate(topo_masks):
        topo_masks_tensor[: lengths[i], i, : lengths[i]] = topo_mask

    # prepare sequence masks
    seq_masks_matrix = np.zeros((n_timesteps, n_samples)).astype(config.floatX)
    for i, length in enumerate(lengths):
        seq_masks_matrix[: length, i] = 1.

    # prepare neighborhood masks
    nbr_masks = [t['nbr_mask'] for t in tuples]
    nbr_masks_matrix = np.zeros((n_samples, n_words)).astype(config.floatX)
    for i, nbr_mask in enumerate(nbr_masks):
        nbr_masks_matrix[i, nbr_mask] = 1.

    # prepare labels data
    if not inference:
        labels = [t['label'] for t in tuples]
        labels_vector = np.array(labels).astype('int32')
    else:
        labels_vector = None

    return (seqs_matrix,
            seq_masks_matrix,
            topo_masks_tensor,
            nbr_masks_matrix,
            labels_vector)


class Loader:
    def __init__(self, data, batch_size=64, shuffle=False, n_words=None):
        self.batch_size = batch_size
        self.idx = 0
        self.data = data
        self.shuffle = shuffle
        self.n = len(data)
        self.n_words = n_words
        self.indices = np.arange(self.n, dtype="int32")

    def __len__(self):
        return len(self.data) // self.batch_size + 1

    def __call__(self):
        if self.shuffle and self.idx == 0:
            np.random.shuffle(self.indices)

        batch_indices = self.indices[self.idx: self.idx + self.batch_size]
        batch_examples = [self.data[i] for i in batch_indices]

        self.idx += self.batch_size
        if self.idx >= self.n:
            self.idx = 0

        return prepare_minibatch(batch_examples, n_words=self.n_words)
