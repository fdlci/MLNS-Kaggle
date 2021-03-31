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


def move_nodes_fast(graph, partition):
    """Move nodes to its neighbors communities to maximize the modularity"""

    # randomize the order of the visited nodes
    Q = list(graph.nodes())
    random.shuffle(Q)

    # initialize best modularity and partition
    best_modularity = nxq.modularity(graph, partition, weight='weight')
    best_partition = partition

    # visit all nodes at least once
    while len(Q) > 0:

        new_partition = deepcopy(best_partition)
        # get next node and neighbors
        next_node = Q.pop(0)
        neigh_node = graph.neighbors(next_node)
        ind_node = find_community_i(new_partition, next_node)

        # visit all neighbors
        for neigh in neigh_node:

            partition_copy = deepcopy(new_partition)
            ind_neigh = find_community_i(partition_copy, neigh)
            partition_copy = delete_from_com(partition_copy, ind_node, next_node)
            partition_copy = add_to_community(partition_copy, ind_neigh, next_node)
            partition_copy = [s for s in partition_copy if s != set()] 
            mod = nxq.modularity(graph, partition_copy, weight='weight')

            if mod > best_modularity:
                best_modularity = mod
                best_partition = partition_copy
                new_ind_node = find_community_i(partition_copy, next_node)
                neigh_left = get_neighbors_not_in_com(graph, new_ind_node, partition_copy, next_node)
                neigh_not_in_Q = [neigh for neigh in neigh_left if neigh not in Q]
                # add those neighbors to Q again
                Q += neigh_not_in_Q

    return best_partition, best_modularity


def move_node_to_other_com(graph, v, partition, initial_partition, best_modularity, theta, T):
    """Computes the probabilities used for the function merge_nodes_subset and finds a new partition according to those probabilities"""

    prob = []
    new_partition = []
    ind_node = find_community_i(partition, v)

    for C, ind_com in T:
        partition_copy = deepcopy(partition)
        partition_copy = delete_from_com(partition_copy, ind_node, v)
        partition_copy = add_to_community(partition_copy, ind_com, v)
        new_com = partition_copy[ind_com]
        partition_copy = [s for s in partition_copy if s != set()] 
        mod = nxq.modularity(graph, partition_copy, weight='weight')

        if mod > best_modularity and is_in_initial_partition(new_com, initial_partition) == True:
            prob.append(np.exp((mod - best_modularity)/theta))
            best_modularity = mod
            best_partition = partition_copy
        else:
            prob.append(0)

        new_partition.append(partition_copy)

    return prob, new_partition


def merge_nodes_subset(graph, partition, initial_partition, subset, theta):
    """From the initial refined partition, merges subsets only if those subsets are a subset of the communities from the initial partition"""

    R = get_connected_nodes(graph, subset)

    best_modularity = nxq.modularity(graph, partition, weight='weight')

    for v in R:

        ind_community = find_community_i(partition, v)
        if len(partition[ind_community]) == 1:
            T = get_connected_communities(graph, subset, partition)
            prob, new_partition = move_node_to_other_com(graph, v, partition, initial_partition, best_modularity, theta, T)
            if prob.count(0) == len(prob):
                pass
            else:
                partition = random.choices(new_partition, weights = prob)[0]
                
    return partition


def refine_partition(graph, partition):
    """Computes the refined partition according to the partition obtained in the first phase"""

    part_refined = singleton_partition(graph)

    for community in partition:
        part_refined = merge_nodes_subset(graph, part_refined, partition, community, theta=0.001)
    return part_refined


def aggregate_graph(graph, partition):
    """Given a refined partition, computes the new graph with each community being a node"""

    nodes = graph.nodes()
    edges = graph.edges()

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
                    link = size_link_between_com(com1, com2, graph)
                    if link > 0:
                        new_edges.append((i,j, {'weight':link}))

    new_graph.add_edges_from(new_edges)

    return new_graph


def leiden_algorithm(graph):

    prev_best_community, prev_best_mod = None, None

    First = True
    partition = singleton_partition(graph)

    while 1:

        # Phase 1
        init_partition = singleton_partition(graph)
        best_partition, best_mod = move_nodes_fast(graph, init_partition)

        # Phase 2
        part_refined = refine_partition(graph, best_partition)

        # Phase 3
        graph = aggregate_graph(graph, part_refined)

        if is_single_node_partition(best_partition):
            break

        if First:
            First = False
            partition = part_refined
        else:
            partition = merge_communities(partition, part_refined)

    return partition, best_mod

def deg_best_result_of_N_Leiden(graph_file, N):
    """Applies Leiden N times to increase the chances of getting the maximum modularity"""
    best_mod = -math.inf
    best_partition = None
    G = load_graph(graph_file)
    for i in range(N):
        partition, q = leiden_algorithm(G)
        if q > best_mod:
            best_mod = q
            best_partition = partition
    return best_partition, best_mod


if __name__ == '__main__':

    t0 = time()

    G = load_graph('Projet/karate.txt')
    partition, q = leiden_algorithm(G)

    print(f'Time: {time() - t0}s')
    print(f'Partition {partition}')

    partition = community_to_dict(partition)
    print(f'Best modularity found: {q}')

    drawing_partition(partition, G)