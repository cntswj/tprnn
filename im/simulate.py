"""Simulate information cascade based the independent cascade model."""

import argparse
import networkx as nx
import random
import numpy as np
from collections import deque, Counter
import sys
import pdb
import math


# Parses input arguments.
parser = argparse.ArgumentParser(description='IC Simulator')
addarg = parser.add_argument

addarg('--model', type=str, default='lt',
       choices=['ic', 'lt', 'netrate'], help='Cascade model')
addarg('--input_graph_file', type=str,
       default='../data/memes/digraph_with_probs.txt')
addarg('--input_seeds_file', type=str, default='../data/memes/test_ids.txt')
addarg('--output_file', type=str, default='../data/memes/lt_results.txt')
# addarg('--output_file', type=str, default='../data/memes/ic_results.txt')

addarg('--max_steps', type=int, default=20,
       help='Max length of simulated cascade.')
addarg('--num_rounds', type=int, default=100,
       help='Number of simulations per test case.')

args = parser.parse_args()

# For debugging convenience.
random.seed(0)
np.random.seed(0)


def simulate_once_ic(seed, g):
    """
    Simulates IC once.
    Arguments:
        seed: The initial infected node.
        g: The input graph with diffusion probabilities.
    Returns:
        The set of infected nodes.
    """
    q = deque()
    q.appendleft(seed)
    active_nodes = set([seed])

    while len(q) > 0 and len(active_nodes) < args.max_steps:
        src = q.pop()
        for target, atrbs in g[src].iteritems():
            prob = atrbs['weight']
            success = random.random() < prob
            if success and (target not in active_nodes):
                q.appendleft(target)
                active_nodes.add(target)

    return active_nodes


def simulate_once_lt(seed, g):
    """
    Simulates LT once.
    Arguments:
        seed: The initial infected node.
        g: The input graph with diffusion probabilities.
    Returns:
        The set of infected nodes.
    """
    q = deque()
    q.appendleft(seed)
    active_nodes = set([seed])
    node_to_status = {}

    tol = 0.0001

    while len(q) > 0 and len(active_nodes) < args.max_steps:
        src = q.pop()
        for target, atrbs in g[src].iteritems():
            prob = atrbs['weight']

            # Updates in-weights of neighbors of newly infected node.
            if target not in node_to_status:
                node_to_status[target] = {'threshold': random.random(),
                                          'inweight': 0}
            status = node_to_status[target]
            status['inweight'] += prob

            if target not in active_nodes and status['inweight'] > tol + status['threshold']:
                active_nodes.add(target)
                q.appendleft(target)

    return active_nodes


def log_survival(a, t1, t2):
    return -a * (t1 - t2)


def transmission_rate(a, t1, t2):
    return a * math.exp(-a * (t1 - t2))


def simulate_once_netrate(seed, g):
    """
    Simulates NetRate once.
    Arguments:
        seed: The initial infected node.
        g: The input graph with diffusion probabilities.
    Returns:
        The set of infected nodes.
    """
    cascade = [seed]

    # Set of nodes that can be infected at next timestamp. Considers only
    # nodes that have non-zero transmission rates from any node in the current
    # cascade.
    candidates = set()

    for _ in xrange(args.max_steps - 1):
        candidates.update(g.neighbors(cascade[-1]))
        candidates -= set(cascade)
        candidate_to_prob = {}
        for i in candidates:
            candidate_to_prob[i] = sum(g[j][i]['weight']
                                       for j in cascade if g.has_edge(j, i))

            # prob = 0
            # for j in cascade:
            #     if not g.has_edge(j, i):
            #         continue
            #     # sum_log_survival = 0.
            #     # for k in cascade:
            #     #     if k == j or not g.has_edge(k, i):
            #     #         continue
            #     #     sum_log_survival += log_survival(g[k][i]['weight'], 0, 0)
            #     # prob += (math.exp(sum_log_survival) *
            #     #          transmission_rate(g[j][i]['weight'], 0, 0))
            #     prob += g[j][i]['weight']
            # # candidate_to_prob[i] = prob

        sum_prob = sum(candidate_to_prob.values())
        next = np.random.choice(candidate_to_prob.keys(),
                                p=[x / sum_prob for x in candidate_to_prob.values()])
        cascade += [next]

    return set(cascade)


def simulate_and_estimate(seed, g):
    """
    Simulates IC several times and estimate infection probabilities.
    Returns:
        A dictionary mapping nodes to infection probabilities.
    """
    if args.model == 'ic':
        simulate_once = simulate_once_ic
    elif args.model == 'lt':
        simulate_once = simulate_once_lt
    elif args.model == 'netrate':
        simulate_once = simulate_once_netrate

    rounds = args.num_rounds
    counter = Counter()
    for _ in xrange(rounds):
        active_nodes = simulate_once(seed, g)
        counter.update(active_nodes)

    counter = dict(counter).iteritems()
    probs = {k: float(v) / rounds for k, v in counter}
    return probs


# Constructs direced graph with probabilities.
g = nx.DiGraph()

with open(args.input_graph_file, 'r') as f:
    for line in f:
        u, v, p = line.strip().split()
        p = float(p)
        g.add_edge(u, v, weight=p)

print nx.info(g)

# Simulates for each test case.
with open(args.input_seeds_file, 'r') as f, open(args.output_file, 'w') as output:
    num_lines = sum(1 for _ in f)
    f.seek(0)

    for i, line in enumerate(f):
        seed, action_id = line.strip().split()

        # This happens when seed is an isolated node.
        if seed not in g:
            probs = {}
        else:
            probs = simulate_and_estimate(seed=seed, g=g)

        # Formats results for output.
        sorted_probs = sorted(probs.items(), key=lambda x: -x[1])
        output.write('%s,%s;' % (seed, action_id))
        output.write(','.join(
            ['%s:%f' % (k, prob) for k, prob in sorted_probs]))
        output.write('\n')

        # Shows progress.
        print '\r%.3f%% of %d cases processed.' % (float(i) / num_lines * 100, num_lines),
        # print '\r%d cases processed.' % (i + 1),
        sys.stdout.flush()

print
