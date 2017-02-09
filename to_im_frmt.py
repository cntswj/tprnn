import os

# Convert cascades to graph file.

data_dir = 'data/dblp'
maxlen = 30

input_actions = os.path.join(data_dir, 'train.txt')
output_actions = os.path.join(data_dir, 'actionslog.txt')

actions = set()
with open(input_actions, 'r') as f, open(output_actions, 'wb') as output:
    for line in f:
        action_id, timeseries = line.strip().split(' ', 1)
        actions.add(action_id)
        timeseries = timeseries.split(' ')
        cascade = timeseries[::2][:maxlen]
        timestamps = timeseries[1::2][:maxlen]
        timestamps = [int(float(x)) for x in timestamps]

        for node, timestamp in zip(cascade, timestamps):
            output.write('%s %s %d\n' % (node, action_id, timestamp))

# Convert cascades to actions file.
input_graph = os.path.join(data_dir, 'subgraph.txt')
output_graph = os.path.join(data_dir, 'im_graph.txt')
with open(input_graph, 'rb') as f, open(output_graph, 'wb') as fo:
    for line in f:
        u, v = line.strip().split()
        fo.write('%s %s 0\n' % (u, v))

output_train_actions = os.path.join(data_dir, 'train_action_ids.txt')
with open(output_train_actions, 'wb') as f:
    for i in actions:
        f.write('%s\n' % i)
