import os

maxlen = 50
data_dir = 'data/twitter'


def process_dataset(data_dir, dataset):
    node_set = set()
    filename = os.path.join(data_dir, dataset + '.txt')
    with open(filename, 'rb') as f:
        for line in f:
            action, cascade = line.strip().split(' ', 1)
            sequence = cascade.split(' ')[::2]
            if maxlen is not None:
                sequence = sequence[:maxlen]
            node_set.update(sequence)
    return node_set


train_nodes = process_dataset(data_dir, 'train')
test_nodes = process_dataset(data_dir, 'test')
seen_nodes = train_nodes | test_nodes

print '%d seen nodes.' % len(seen_nodes)

filename = os.path.join(data_dir, 'seen_nodes.txt')
with open(filename, 'wb') as f:
    for v in seen_nodes:
        f.write('%s\n' % v)

graph_file = os.path.join(data_dir, 'graph.txt')
output_file = os.path.join(data_dir, 'subgraph.txt')
edge_set = set()
with open(graph_file, 'rb') as f, open(output_file, 'wb') as fo:
    f.next()
    for line in f:
        u, v = line.strip().split()
        if u in seen_nodes and v in seen_nodes:
            if (u, v) not in edge_set:
                fo.write('%s %s\n' % (u, v))
                edge_set.add((u, v))
