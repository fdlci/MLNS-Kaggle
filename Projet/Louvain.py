import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
from copy import deepcopy
import networkx.algorithms.community.quality as nxq
import math
import matplotlib.cm as cm
from time import time
from utils import *

def phase1(graph):
    """Gets the best partition by maximizing greedily the modularity function"""

    # initialize the communities: each node in a different community
    partition = singleton_partition(graph)

    # initialize the best modularity to spot convergence
    best_mod = nxq.modularity(graph, partition, weight='weight')
    best_partition = partition

    nodes = list(graph.nodes())
    random.shuffle(nodes)

    while 1:
        for node in nodes:
            part = deepcopy(best_partition)
            best_mod, best_partition = get_best_partition(graph, best_partition, best_mod, node, part)

        if part == best_partition:
            break

    return best_partition, best_mod


def phase2(partition, graph):
    """Given a partition, computes the new graph with each community being a node"""

    # Get information from current graph
    nodes = graph.nodes()
    edges = graph.edges()

    # Initialize new graph
    new_nodes = [i for i in range(len(partition))]
    new_edges = []

    new_graph = nx.Graph()
    new_graph.add_nodes_from([i for i in range(len(partition))])


    for i, com1 in enumerate(partition):
        for j, com2 in enumerate(partition):
            if i >= j:
                if i == j:
                    # self_loops
                    nodes_in_com = [node for node in com1]
                    subGraph = graph.subgraph(nodes_in_com)
                    new_edges.append((i,i, {'weight':subGraph.size(weight='weight')}))
                else:
                    # weights between communities
                    link = size_link_between_com(com1, com2, graph)
                    if link > 0:
                        new_edges.append((i,j, {'weight':link}))

    new_graph.add_edges_from(new_edges)

    return new_graph


def louvain_algorithm(G):
    """Computes the best partition using the Louvain algorithm"""

    prev_best_partition, prev_best_mod = None, None

    partition = singleton_partition(G)

    First = True

    while 1:

        # phase 1
        best_partition, best_mod = phase1(G)

        # phase 2
        G = phase2(best_partition, G)

        if best_mod == prev_best_mod:
            break
        else:
            prev_best_partition, prev_best_mod = best_partition, best_mod

        if First:
            First = False
            partition = best_partition
        else:
            partition = merge_communities(partition, best_partition)

    return partition, best_mod

def deg_best_result_of_N_Louvain(graph_file, N):
    """Applies Louvain N times to increase the chances of getting the maximum modularity"""
    best_mod = -math.inf
    best_partition = None
    G = load_graph(graph_file)
    for i in range(N):
        partition, q = louvain_algorithm(G)
        if q > best_mod:
            best_mod = q
            best_partition = partition
    return best_partition, best_mod


if __name__ == '__main__':

    t0 = time()

    G = load_graph('Projet/karate.txt')
    partition, q = louvain_algorithm(G)

    print(f'Time: {time() - t0}s')
    print(partition)

    partition = community_to_dict(partition)
    print(f'Best modularity found: {q}')

    drawing_partition(partition, G)