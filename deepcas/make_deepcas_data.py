import os
import networkx as nx


data_dir = 'data/memes'
out_dir = os.path.join(data_dir, 'deepcas/')

maxlen = 30

# creates global graph
graph_file = os.path.join(data_dir, 'graph.txt')
out_graph_file = os.path.join(out_dir, 'global_graph.txt')

G = nx.Graph()
with open(graph_file, 'rb') as fin:
    fin.next()
    for line in fin:
        u, v = line.strip().split()
        G.add_edge(u, v)

with open(out_graph_file, 'wb') as fout:
    for u in G.nodes():
        nbr_str = '\t'.join(['%s:1' % v_ for v_ in G[u]])
        if nbr_str == '':
            nbr_str = 'null'
        fout.write(u + '\t\t' + nbr_str + '\n')


def create_cacades(dataset):
    # creates cascades
    cas_file = os.path.join(data_dir, dataset + '.txt')
    out_cas_file = os.path.join(out_dir, 'cascade_' + dataset + '.txt')
    n_samples = 0
    with open(cas_file, 'rb') as fin, open(out_cas_file, 'wb') as fout:
        for line in fin:
            query, cas = line.strip().split(' ', 1)
            cas = [query] + cas.split()[::2][:maxlen]
            edge_set = set()
            for i in range(len(cas) - 1):
                u = cas[i]
                pf = cas[: i + 1]
                y = cas[i + 1]
                sz = i + 1
                # v_set = set(pf) & set(G[u])
                # edge_set.update([(u, v_) for v_ in v_set])
                # edge_set.update([(v_, u) for v_ in v_set])
                edge_set.update([(v_, u) for v_ in pf[:-1]])

                # edge_strs = ' '.join(['%s:%s:1' % (u_, v_) for u_, v_ in edge_set])
                edge_strs = ' '.join(['%s:%s:1' % (u_, v_) for u_, v_ in edge_set])

                starters = ' '.join(pf)
                example = str(n_samples) + '\t' + starters + '\t2017\t' + str(sz) + '\t' + edge_strs + '\t' + y
                fout.write(example + '\n')

                n_samples += 1


create_cacades('train')
create_cacades('test')
