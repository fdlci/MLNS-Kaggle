import networkx as nx
from networkx import drawing
import numpy as np
from copy import deepcopy
import networkx.algorithms.community.quality as nxq
from numpy.linalg import inv, matrix_power
from time import time
from utils import *

def P_t_C_func(graph, community, P_t, nodes2id):
    """Computes the Ptc defined in [6]"""

    n = P_t.shape[0]
    P_t_C = np.zeros(n)

    for j in range(n):
        summing = 0
        for node in community:
            ind = nodes2id[node]
            summing += P_t[ind,j]
        P_t_C[j] = summing/len(community)

    return P_t_C

def distance_com1_com2(graph, C1, C2, P_t, nodes2id):
    """Computes the distance between two communities"""

    summing = 0

    for i, k in enumerate(graph.nodes()):

        square = (P_t_C_func(graph, C1, P_t, nodes2id)[i] - P_t_C_func(graph, C2, P_t, nodes2id)[i])**2
        degree_k = graph.degree(k)

        summing += square/degree_k

    return summing

def delta_sigma(graph, C1, C2, P_t, nodes2id):
    """Computes the delta sigma defined in [6]"""

    n = graph.number_of_nodes()

    r_c1c2 = distance_com1_com2(graph, C1, C2, P_t, nodes2id)

    return (1/n) * len(C1) * len(C2) * r_c1c2**2 /(len(C1) + len(C2))

def adjacent_communities_dist(graph, partition, P_t, nodes2id):
    """Computes all distances between adjacent communities"""

    adjacent_coms_dist = {}

    num_com = len(partition)
    for ind1 in range(num_com-1):
        for ind2 in range(ind1+1,num_com):
            C1, C2 = partition[ind1], partition[ind2]
            link = size_link_between_com(C1, C2, graph)
            if link > 0:
                adjacent_coms_dist[(ind1, ind2)] = delta_sigma(graph, C1, C2, P_t, nodes2id)

    return adjacent_coms_dist


def walktrap_algorithm(graph, t=5):

    # check that graph is connected and undirected
    assert nx.is_connected(graph) == True, "The graph must be connected"
    assert nx.is_directed(graph) == False, "The graph must be undirected"

    partitions = []
    modularities = []

    #initialization
    partition = singleton_partition(graph)
    modularity = nxq.modularity(graph, partition, weight='weight') 
    partitions.append(deepcopy(partition))
    modularities.append(modularity)

    # number of iterations
    num_nodes = graph.number_of_nodes()

    # adjacency matrix
    A = nx.to_numpy_matrix(graph, dtype=int)
    A += np.diag([1 for i in range(len(A))])

    # diagonal matrix
    D = nx.laplacian_matrix(graph) + A
    Ddiag = np.diagonal(D)
    Dd = np.diag(np.power(Ddiag, (-0.5)))
    # Transition prob matrix P
    P = inv(D) @ A
    P_t = matrix_power(P, t)

    nodes2id = nodes_to_ind(graph)

    for iteration in tqdm(range(num_nodes - 1)):

        # index_to_id
        id2p = index_to_partition(partition)

        # computing distances
        dist = adjacent_communities_dist(graph, partition, P_t, nodes2id)
        (ind1, ind2) = min(dist, key=dist.get)
        C1 = id2p[ind1]
        C2 = id2p[ind2]

        # union of communities
        C3 = C1.union(C2)

        # redefine the partition
        partition.remove(C1)
        partition.remove(C2)
        partition.append(C3)

        partitions.append(deepcopy(partition))
        modularities.append(nxq.modularity(graph, partition, weight='weight'))

    return list(reversed(partitions)), list(reversed(modularities))

if __name__ == '__main__':

    t0 = time()

    G = load_graph('Projet/karate.txt')
    partitions, modularities = walktrap_algorithm(G)

    ind = modularities.index(max(modularities))
    q = max(modularities)
    partition = partitions[ind]

    print(f'Time: {time() - t0}s')
    print(f'Partition {partition}')
    print(f'Best modularity found: {q}')

    partition = community_to_dict(partition)

    drawing_partition(partition, G)
