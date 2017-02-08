import os

maxlen = 50
data_dir = 'data/digg'


def process_dataset(data_dir, dataset):
    node_set = set()
    filename = os.path.join(data_dir, dataset + '.txt')
    with open(filename, 'rb') as f:
        for line in f:
            action, cascade = line.strip().split(' ', 1)
            sequence = map(int, cascade.split(' ')[::2])
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
