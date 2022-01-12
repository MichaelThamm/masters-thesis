#!/usr/bin/env python
"""
Main program for demonstrating the random-walk pagerank implementation.

python main.py [<n_communities>]

Arguments:
    n_communities : int; defaults to 100
        The number of communities in the randomly generated graph.
"""

import sys
from timeit import default_timer as timer
import operator
import math
import random
import contextlib

import networkx as nx
import numpy as np

import Numba
import Numba.cpu_randomwalk
import Numba.cuda_randomwalk


def make_random_graph(n_communities):
    seed = 123
    random.seed(seed)
    # Generate a very sparse community graph.
    communities = [random.randrange(5, 100) for _ in range(n_communities)]
    prob_inner_connect = 0.002
    prob_outer_connect = 0.0001
    G = nx.random_partition_graph(communities, prob_inner_connect,
                                  prob_outer_connect,
                                  directed=True, seed=seed)
    print('# of nodes: {}'.format(G.number_of_nodes()))
    print('# of edges: {}'.format(G.number_of_edges()))
    # To draw the graph with `dot`,
    # use the following to write out the graph:
    #   nx.nx_pydot.write_dot(G, 'my.dot')
    return G


def numpy_reference_implementation(G):
    pr = nx.pagerank_numpy(G)
    toplist = list(sorted(pr.items(), key=operator.itemgetter(1), reverse=True))
    print('top5', [i[0] for i in toplist[:5]])


def random_walk_implementation(G, gpu=False):
    spmat = nx.to_scipy_sparse_matrix(G)
    colidx = spmat.indptr
    edges = spmat.indices

    nodes = list(G.nodes())
    compute(colidx, edges, nodes, gpu=gpu)


def compute(colidx, edges, nodes, gpu=False):
    coupons = np.zeros(len(nodes), dtype="uint32")

    resetprob = 0.01

    assert 0 < resetprob < 1

    K = int(2 * math.log(len(nodes)))
    assert K >= 1

    coupons.fill(K)

    colidx = colidx.astype(np.uint32)
    edges = edges.astype(np.uint32)

    if gpu:
        module = Numba.cuda_randomwalk
    else:
        module = Numba.cpu_randomwalk
    ranks = module.random_walk(nodes, coupons, colidx, edges, resetprob)
    print('top5', [v for k, v in sorted(zip(ranks, nodes), reverse=True)][:5])


@contextlib.contextmanager
def timing():
    s = timer()
    yield
    e = timer()
    print('execution time: {:.2f}s'.format(e - s))


if __name__ == '__main__':
    n_communities = 100
    try:
        n_communities = int(sys.argv[1])
    except IndexError:
        pass
    print('# of communities: {}'.format(n_communities))

    G = make_random_graph(n_communities)
    print('multiCPU random walk')
    with timing():
        random_walk_implementation(G, gpu=False)
    print('single-GPU random walk')
    with timing():
        random_walk_implementation(G, gpu=True)
    print('numpy reference')
    if G.number_of_edges() > 100000:
        # the numpy based implementation will consume too much memory and time
        # for large graphs.
        print('disabled for large graph')
    else:
        with timing():
            numpy_reference_implementation(G)
