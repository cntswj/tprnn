"""Convert IM (Goyal et al., PVLDB'12) outputs to digraph with probs."""

import argparse
from igraph import *

# Parses input arguments.
parser = argparse.ArgumentParser(description='Convert IM outputs to digraphs with probs.')
addarg = parser.add_argument

addarg('--input_graph_file', type=str, default='../data/digg/im_graph.txt')
addarg('--input_user_counts_file', type=str, default='../data/digg/im/usersCounts.txt')
addarg('--input_edge_counts_file', type=str, default='../data/digg/im/edgesCounts.txt')

addarg('--output_file', type=str, default='../data/digg/prob_graph.txt')

args = parser.parse_args()

# Read the directed graph (output of preprocess.py)
g = Graph.Read_Ncol(args.input_graph_file, directed=True)
g.es['weight'] = 0.0
print('Number of nodes: ', g.vcount())
print('Number of directed edges: ', g.ecount())

# Read actions performed by each user
print('Loading usersCounts...')
f_nodes = open(args.input_user_counts_file, 'r')
user_action = {}
for line in f_nodes:
    v, a_v = [int(x) for x in line.split()]
    user_action[v] = a_v
f_nodes.close()

# Read edges, and replace the weights in g by influence probabilities
print('Loading edgesCounts...')
f_edges = open(args.input_edge_counts_file, 'r')
for i, line in enumerate(f_edges):
    u, v, a_u2v, a_v2u = [int(x) for x in line.split()[:4]]
    if a_u2v == 0 and a_v2u == 0:
        continue
    if u in user_action and v in user_action:
        uid = g.vs.find(str(u))
        vid = g.vs.find(str(v))
        # u -> v
        a_u = user_action[u]
        if a_u2v > 0 and a_u > 0:
            eid_uv = g.get_eid(uid, vid, error=False)
            if eid_uv >= 0:
                g.es[eid_uv]['weight'] = round(float(a_u2v) / a_u, 6)
        # v -> u
        a_v = user_action[v]
        if a_v2u > 0 and a_v > 0:
            eid_vu = g.get_eid(vid, uid, error=False)
            if eid_vu >= 0:
                g.es[eid_vu]['weight'] = round(float(a_v2u) / a_v, 6)
f_edges.close()

# Delete edges with < 0 influence probabilities
g.delete_edges(g.es.select(weight_le=0))
print('Number of edges after deletion: ', g.ecount())

g.write_ncol(args.output_file)
