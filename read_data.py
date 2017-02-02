import os
# import re
# import sys
import codecs
import networkx as nx
import numpy as np
import pprint


def load_graph(data_dir):
    filename = os.path.join(data_dir, 'graph.txt')
    G = nx.Graph()
    with open(filename, 'r') as f:
        f.next()
        for line in f:
            u, v = line.strip().split()
            G.add_edge(u, v)
    return G


def convert_cascade_to_examples(line, G=None, max_length=50):
    # parses the input line.
    action, cascade = line.strip().split(';')
    sequence = cascade.split(',')[::2]
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

        # next node as label
        label = sequence[i + 1]

        example = (ordered_nodes, topo_mask, target_mask, label)
        examples.append(example)

    return examples


def load_cascade_examples(data_dir, dataset="train", G=None):
    """
    Load the train/dev/test data
    Return: dict of examples
    """
    filename = os.path.join(data_dir, dataset + '.txt')

    example_dict = {}
    vocab = set()

    # read in all the strings, convert them to trees, and store them in a dict
    with codecs.open(filename, 'r', encoding='utf-8') as input_file:
        for line_index, line in enumerate(input_file):
            examples = convert_cascade_to_examples(line, G)
            for example_index, example in enumerate(examples):
                key = str(line_index) + '.' + str(example_index)
                sequence, topo_mask, target_mask, label = example
                example_dict[key] = {'sequence': sequence,
                                     'topo_mask': topo_mask,
                                     'target_mask': target_mask,
                                     'label': label}
                vocab.update(set(sequence))

    return example_dict, vocab


G = load_graph('data/toy')
examples, _ = load_cascade_examples('data/toy', dataset='train', G=G)

pp = pprint.PrettyPrinter(indent=4)
pp.pprint(examples)
