import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm.notebook import tqdm
from copy import deepcopy
import networkx.algorithms.community.quality as nxq
import math
import matplotlib.cm as cm
from time import time
from scipy.cluster import hierarchy


def load_graph(file_path):
    """Loads graph from file txt"""
    
    assert ('txt' in file_path), 'Please choose a graph file of type txt'

    G = nx.read_edgelist(file_path,create_using=nx.Graph(), nodetype = int)
    return G

def singleton_partition(graph):
    """Initializes the partition to all nodes form a community"""
    partition = []
    for node in graph.nodes():
        partition.append({node})
    return partition

def community_to_dict(partition):
    """Converts a partition of the following form:
    partition = [{nodes of com 1}, {nodes of com 2}, ..., ] into a dict with the nodes as keys and the community as value"""
    part = {}
    for i, com in enumerate(partition):
        for node in com:
            part[node] = i
    return part

def from_dict_to_list(partition):
    """Perfomrs the inverse operation of the community_to_dict function"""
    num_part = max(partition.values())
    part = [[] for i in range(num_part+1)]
    for node in partition:
        part[partition[node]].append(node)
    return part

def find_community_i(partition, node):
    """Finds the index of the community of node"""
    for ind, com in enumerate(partition):
        if node in com:
            return ind

def delete_from_com(partition, ind, node):
    """Deletes node from community with index ind"""
    part = deepcopy(partition)
    part[ind].remove(node)
    return part

def add_to_community(partition, ind, node):
    """Adds node to the community of index ind"""
    partition = deepcopy(partition)
    partition[ind].add(node)
    return partition

def get_neighbors_not_in_com(graph, ind_node, partition, node):
    """Given a node, returns the neighbors of that node that are not in the community"""
    neighbors = graph.neighbors(node)
    return [neigh for neigh in list(neighbors) if neigh not in partition[ind_node]]


def weight_subset(graph, nodes):
    """For a given degree of nodes, gives the sum of the weights of the edges"""
    total = 0
    degree = graph.degree(nodes, weight='weight')
    for edge in degree:
        data = graph.get_edge_data(edge[0], edge[1])
        if data == None:
            pass
        elif data != {}:
            total += data['weight']
        else:
            total += 1
    return total

def get_connected_nodes(graph, subset, gamma=1):
    """Gets the nodes that are well connected within subset S"""
    nodes = []
    for v in subset:
        deg_v = weight_subset(graph, [v])
        deg_sub = weight_subset(graph, subset)
        subset_wo_v = [s for s in subset if s != v]
        E = weight_subset(graph, subset_wo_v)
        if E >= deg_v * (deg_sub - deg_v):
            nodes.append(v)
    return nodes

def get_connected_communities(graph, subset, partition, gamma=1):
    """Gets the communities that are well-connected"""

    communities = []
    
    for i, C in enumerate(partition):
        deg_C = weight_subset(graph, C)
        deg_sub = weight_subset(graph, subset)
        subset_wo_C = [commu for commu in subset if commu != C]
        E = weight_subset(graph, subset_wo_C)
        if E >= deg_C * (deg_sub - deg_C):
            communities.append((C, i))
    return communities

def is_in_initial_partition(new_com, initial_partition):
    """Returns True if the considered new communityis a subset of a community from the initial partition"""
    for com in initial_partition:
        if new_com.issubset(com):
            return True
    return False

def size_link_between_com(com1, com2, graph):
    """Gets the weight of the edge between two communities"""

    link = 0
    for node1 in com1:
        for node2 in com2:
            data = graph.get_edge_data(node1, node2)
            if data != None:
                if data != {}:
                    link += data['weight']
                else:
                    link += 1
    return link

def get_best_partition(graph, best_partition, best_mod, node, part):
    """Gets best partition by removing node to its neighbors' communities"""

    ind_node = find_community_i(part, node)
    neigh_node = graph.neighbors(node)

    # visit all nieghbors of the node
    for neigh in neigh_node:

        # make copy of part to not change the initial part
        part_bis = deepcopy(part)
        ind_neigh = find_community_i(part_bis, neigh)
        part_bis = delete_from_com(part_bis, ind_node, node)
        part_bis = add_to_community(part_bis, ind_neigh, node)
        part_bis = [s for s in part_bis if s != set()]

        # compute modularity of new partition
        mod = nxq.modularity(graph, part_bis, weight='weight') 

        # update modularity
        if mod > best_mod:
            best_mod = mod
            best_partition = part_bis

    return best_mod, best_partition 

def merge_communities(communities, best_communities):
    """Merges communities during the iterations of the Louvain algorithm (it > 1)"""

    new_communities = []

    for com in best_communities:
        uni = set()
        for element in com:
            uni = uni.union(communities[element])
        new_communities.append(uni)
    return new_communities 

def is_single_node_partition(partition):
    """Returns True if the partition is only composed of single nodes"""
    for com in partition:
        if len(com) > 1:
            return False
    return True

def nodes_to_ind(graph):
    """Returns a dictionnary with all nodes associated to their id"""

    nodes = {}
    for i, node in enumerate(graph.nodes()):
        nodes[node] = i

    return nodes

def index_to_partition(partition):
    """Returns dictionnary with ids as keys and communities as values"""

    id2p = {}

    for ind, com in enumerate(partition):
        id2p[ind] = com

    return id2p

def drawing_partition(partition, graph):
    """Draws the graph according to the communities found"""

    # draw the graph
    pos = nx.spring_layout(graph)

    # color the nodes according to their partition
    cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
    nx.draw_networkx_nodes(graph, pos, partition.keys(), node_size=40,
                        cmap=cmap, node_color=list(partition.values()))
    nx.draw_networkx_edges(graph, pos, alpha=0.5)
    plt.show()

def format_dendrogram(partitions):
    """Converting the partitions into the right format for the dendrogram function"""
    part = []
    for parti in partitions:
        community = []
        for com in parti:
            new_com = set()
            for el in com:
                new_com.add(str(el))

            community.append(new_com)
        part.append(community)
    return part

def plot_dendrogram(G, partitions):
    """Plots the dendrogram. Function taken from lab 5"""

    num_of_nodes = G.number_of_nodes()
    dist = np.ones( shape=(num_of_nodes, num_of_nodes), dtype=np.float )*num_of_nodes
    d = num_of_nodes-1
    for partition in partitions:
        for subset in partition:
            for i in range(len(subset)):
                for j in range(i+1, len(subset)):
                    subsetl = list(subset)

                    dist[int(subsetl[i]), int(subsetl[j])] = d
                    dist[int(subsetl[j]), int(subsetl[i])] = d
        d -= 1



    dist_list = [dist[i,j] for i in range(num_of_nodes) for j in range(i+1, num_of_nodes)]


    Z = hierarchy.linkage(dist_list, 'complete')
    plt.figure()
    dn = hierarchy.dendrogram(Z)