import os
# import re
# import sys
import codecs
import networkx as nx
import numpy as np
# import pprint
import theano


def load_graph(data_dir):
    filename = os.path.join(data_dir, 'graph.txt')
    G = nx.Graph()
    with open(filename, 'r') as f:
        f.next()
        for line in f:
            u, v = line.strip().split()
            G.add_edge(u, v)
    return G


def convert_cascade_to_examples(line, G=None, node_map=None, max_length=50):
    # parses the input line.
    action, cascade = line.strip().split(' ', 1)
    sequence = cascade.split(' ')[::2]
    sequence = sequence[:max_length]

    # grows the series of dags incrementally.
    examples = []
    sub_dag = nx.DiGraph()
    for i, node in enumerate(sequence[:-1]):
        # grows the current dag.
        seen_nodes = sequence[: i + 1]
        sub_dag.add_edges_from(
            [(node, v) for v in G[node] if v not in seen_nodes])

        # will compute hidden states for nodes following their topological ordering.
        ordered_nodes = nx.topological_sort(sub_dag)
        node_index = {v: i for i, v in enumerate(ordered_nodes)}
        length = len(ordered_nodes)

        # targets (candidates) masks
        targets = list(set(sub_dag.nodes()) - set(seen_nodes))
        i_targets = [node_index[v] for v in targets]
        target_mask = np.zeros(length, dtype=np.int)
        target_mask[i_targets] = 1

        # structure masks
        topo_mask = np.zeros((length, length), dtype=np.int)
        for i_v, v in enumerate(ordered_nodes):
            i_p = [node_index[x] for x in sub_dag.predecessors(v)]
            topo_mask[i_v, i_p] = 1

        # next node as label (catch: timestep id instead of node index)
        label = i + 1

        example = {'sequence': [node_map[v] for v in ordered_nodes],
                   'topo_mask': topo_mask,
                   'target_mask': target_mask,
                   'label': label}
        examples.append(example)

    return examples


def load_cascade_examples(data_dir, dataset="train"):
    """
    Load the train/dev/test data
    Return: list of example tuples
    """
    # loads graph
    G = load_graph(data_dir)
    node_map = {node: i for i, node in enumerate(G.nodes())}

    # loads cascades
    filename = os.path.join(data_dir, dataset + '.txt')
    example_tuples = []
    with codecs.open(filename, 'r', encoding='utf-8') as input_file:
        for line_index, line in enumerate(input_file):
            examples = convert_cascade_to_examples(line, G=G, node_map=node_map)
            example_tuples.extend(examples)

    return example_tuples, node_map


class Loader:
    def __init__(self, data, batch_size=64, shuffle=False):
        self.batch_size = batch_size
        self.idx = 0
        self.data = data
        self.shuffle = shuffle
        self.n = len(data)
        self.indices = np.arange(self.n, dtype="int32")
        self.n_batches = self.n // batch_size

    def __prepare_minibatch(self, tuples):
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
        topo_masks_tensor = np.zeros((n_timesteps, n_samples, n_timesteps)).astype(theano.config.floatX)
        for i, topo_mask in enumerate(topo_masks):
            topo_masks_tensor[: lengths[i], i, : lengths[i]] = topo_mask

        # prepare target-masks data
        target_masks = [t['target_mask'] for t in tuples]
        target_masks_matrix = np.zeros((n_samples, n_timesteps)).astype('int32')
        for i, target_mask in enumerate(target_masks):
            target_masks_matrix[i, : lengths[i]] = target_mask

        # prepare labels data
        labels = [t['label'] for t in tuples]
        labels_vector = np.array(labels).astype('int32')

        # prepare sequence masks
        seq_masks_matrix = np.zeros((n_timesteps, n_samples)).astype(theano.config.floatX)
        for i, length in enumerate(lengths):
            seq_masks_matrix[: length, i] = 1.

        return (seqs_matrix, seq_masks_matrix, topo_masks_tensor, target_masks_matrix, labels_vector)

    def __call__(self):
        if self.shuffle and self.idx == 0:
            np.random.shuffle(self.indices)
        try:
            batch_indices = self.indices[self.idx: self.idx + self.batch_size]
            batch_examples = [self.data[i] for i in batch_indices]
            return self.__prepare_minibatch(batch_examples)
        finally:
            self.idx += self.batch_size
            if self.idx >= self.n:
                self.idx = 0
