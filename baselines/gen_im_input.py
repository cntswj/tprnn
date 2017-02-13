import os

# Convert cascades to graph file.

data_dir = 'data/twitter'
maxlen = 30
infer_graph_from_cascades = True

input_actions = os.path.join(data_dir, 'train.txt')
output_actions = os.path.join(data_dir, 'actionslog.txt')

actions = set()
edge_set = set()
with open(input_actions, 'r') as f, open(output_actions, 'wb') as output:
    for line in f:
        query, timeseries = line.strip().split(' ', 1)
        actions.add(query)
        timeseries = timeseries.split(' ')
        cascade = [query] + timeseries[::2][:maxlen]
        timestamps = ['0'] + timeseries[1::2][:maxlen]
        timestamps = [int(float(x)) for x in timestamps]

        for node, timestamp in zip(cascade, timestamps):
            output.write('%s %s %d\n' % (node, query, timestamp))

        if infer_graph_from_cascades:
            length = len(cascade)
            for i in range(length):
                for j in range(i + 1, length):
                    u, v = cascade[i], cascade[j]
                    edge_set.add((u, v))

# outputs ids of training actions.
output_train_actions = os.path.join(data_dir, 'train_action_ids.txt')
with open(output_train_actions, 'wb') as f:
    for i in actions:
        f.write('%s\n' % i)

# Convert cascades to actions file.
output_graph = os.path.join(data_dir, 'im_graph.txt')
with open(output_graph, 'wb') as fo:
    if infer_graph_from_cascades:
        for u, v in edge_set:
            fo.write('%s %s 0\n' % (u, v))
    else:
        input_graph = os.path.join(data_dir, 'subgraph.txt')
        with open(input_graph, 'rb') as f:
            for line in f:
                u, v = line.strip().split()
                fo.write('%s %s 0\n' % (u, v))
