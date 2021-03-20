import os
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, auc
import pandas as pd
import csv
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn import preprocessing
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import random
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

nltk.download('punkt') # for tokenization
nltk.download('stopwords')
stpwds = set(nltk.corpus.stopwords.words("english"))
stemmer = nltk.stem.PorterStemmer()

def compute_network_characteristics(graph):
    prop = {}
    prop['N'] =  graph.number_of_nodes() # number of nodes
    prop['M'] = graph.number_of_edges() # number of edges
    degrees = [degree for node, degree in graph.degree()] # degree list
    prop['min_degree'] =  np.min(degrees) # minimum degree
    prop['max_degree'] =  np.max(degrees) # maximum degree
    prop['mean_degree'] = np.mean(degrees) # mean of node degrees
    prop['median_degree'] = np.median(degrees) # median of node degrees
    prop['density'] =  nx.density(graph) # density of the graph
    return prop

def generate_samples(graph, train_set_ratio):
    """
    Graph pre-processing step required to perform supervised link prediction
    Create training and test sets
    """
        
    # --- Step 0: The graph must be connected ---
    if nx.is_connected(graph) is not True:
        raise ValueError("The graph contains more than one connected component!")
       
    
    # --- Step 1: Generate positive edge samples for testing set ---
    residual_g = graph.copy()
    test_pos_samples = []
      
    # Store the shuffled list of current edges of the graph
    edges = list(residual_g.edges())
    np.random.shuffle(edges)
    
    # Define number of positive test samples desired
    test_set_size = int((1.0 - train_set_ratio) * graph.number_of_edges())
    train_set_size = graph.number_of_edges() - test_set_size
    num_of_pos_test_samples = 0
    
    # Remove random edges from the graph, leaving it connected
    # Fill in the blanks
    for edge in tqdm(edges):
        
        # Remove the edge
        residual_g.remove_edge(edge[0], edge[1])
        
        # Add the removed edge to the positive sample list if the network is still connected
        if nx.is_connected(residual_g):
            num_of_pos_test_samples += 1
            test_pos_samples.append(edge)
        # Otherwise, re-add the edge to the network
        else: 
            residual_g.add_edge(edge[0], edge[1])
        
        # If we have collected enough number of edges for testing set, we can terminate the loop
        if num_of_pos_test_samples == test_set_size:
            break
    
    # Check if we have the desired number of positive samples for testing set 
    if num_of_pos_test_samples != test_set_size:
        raise ValueError("Enough positive edge samples could not be found!")

        
    # --- Step 2: Generate positive edge samples for training set ---
    # The remaining edges are simply considered for positive samples of the training set
    train_pos_samples = list(residual_g.edges())
        
        
    # --- Step 3: Generate the negative samples for testing and training sets ---
    # Fill in the blanks
    non_edges = list(nx.non_edges(graph))
    random.seed(10)
    np.random.shuffle(non_edges)
    
    train_neg_samples = non_edges[:train_set_size] 
    test_neg_samples = non_edges[train_set_size:train_set_size + test_set_size]

    
    # --- Step 4: Combine sample lists and create corresponding labels ---
    # For training set
    train_samples = train_pos_samples + train_neg_samples
    train_labels = [1 for _ in train_pos_samples] + [0 for _ in train_neg_samples]
    # For testing set
    test_samples = test_pos_samples + test_neg_samples
    test_labels = [1 for _ in test_pos_samples] + [0 for _ in test_neg_samples]
    
    return residual_g, train_samples, train_labels, test_samples, test_labels

def feature_extractor(graph, samples, deg_centrality):
    """
    Creates a feature vector for each edge of the graph contained in samples 
    """
    feature_vector = []
    number_nodes_out = 0

    for edge in tqdm(samples):
        source_node, target_node = edge[0], edge[1]

        # Degree Centrality
        if (source_node not in list(deg_centrality.keys())) or (target_node not in list(deg_centrality.keys())):
            feature_vector.append(np.array([0, 0, 0, 0, 0, 0]))
            number_nodes_out += 1

        else:

            source_degree_centrality = deg_centrality[source_node]
            target_degree_centrality = deg_centrality[target_node]
            
            # # Betweeness centrality measure 
            # diff_bt = betweeness_centrality[target_node] - betweeness_centrality[source_node]

            # Preferential Attachement 
            pref_attach = list(nx.preferential_attachment(graph, [(source_node, target_node)]))[0][2]

            # AdamicAdar
            aai = list(nx.adamic_adar_index(graph, [(source_node, target_node)]))[0][2]

            # Jaccard
            jacard_coeff = list(nx.jaccard_coefficient(graph, [(source_node, target_node)]))[0][2]
            # Ressource allocation index
            res_all = list(nx.resource_allocation_index(graph, [(source_node, target_node)]))[0][2]
            
            # Create edge feature vector with all metric computed above
            feature_vector.append(np.array([source_degree_centrality, target_degree_centrality, pref_attach, aai, jacard_coeff, res_all])) 
    print(f"Number nodes out: {number_nodes_out}")
        
    return np.array(feature_vector)

def get_true_test_labels(test_samples, t_set_red):
    real_test_labels = []

    for edge in test_samples:
        if (edge[0], edge[1]) in list(t_set_red.keys()):
            real_test_labels.append(int(t_set_red[(edge[0], edge[1])]))
        else:
            real_test_labels.append(0)
    return real_test_labels

def prediction(graph, train_features, test_features, train_labels, test_labels):
    """
    Downstream ML task using edge embeddings to classify them 
    """
    
    # --- Build the model and train it ---
    # Fill in the blanks
    clf = LogisticRegression()
    clf.fit(train_features, train_labels)

    train_preds = clf.predict_proba(train_features)[:, 1]
    test_preds = clf.predict_proba(test_features)[:, 1]
    labels_pred = clf.predict(test_features)

    print(f'Accuracy: {accuracy_score(test_labels, labels_pred)}')
    print(f'F1 score: {f1_score(test_labels, labels_pred)}')

    # --- Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from predictions ---
    # Fill in the blanks
    fpr, tpr, _ = roc_curve(test_labels, test_preds)
    roc_auc = auc(fpr, tpr)
    
    # plt.figure(figsize=(6,6))
    # plt.plot(fpr, tpr, color='darkred', label='ROC curve (area = %0.3f)' % roc_auc)
    # plt.plot([0, 1], [0, 1], color='lightgray', linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic Curve')
    # plt.legend(loc="lower right")
    # plt.show()
    
    return roc_auc

def preprocessing_info(sample, abstract_vectorized, IDs, node_info):
    # number of overlapping words in title
    overlap_title = []

    # temporal distance between the papers
    temp_diff = []

    # number of common authors
    comm_auth = []

    # Cosine sim between abstracts
    cosine_sim = []

    dense_matrix = abstract_vectorized.todense()

    counter = 0
    for i in range(len(sample)):
        source = sample[i][0]
        target = sample[i][1]
        
        index_source = IDs.index(source)
        index_target = IDs.index(target)
        
        source_info = [element for element in node_info if element[0]==source][0]
        target_info = [element for element in node_info if element[0]==target][0]
        
        # convert to lowercase and tokenize
        source_title = source_info[2].lower().split(" ")
        # remove stopwords
        source_title = [token for token in source_title if token not in stpwds]
        source_title = [stemmer.stem(token) for token in source_title]
        
        target_title = target_info[2].lower().split(" ")
        target_title = [token for token in target_title if token not in stpwds]
        target_title = [stemmer.stem(token) for token in target_title]
        
        source_auth = source_info[3].split(",")
        target_auth = target_info[3].split(",")
        
        overlap_title.append(len(set(source_title).intersection(set(target_title))))
        temp_diff.append(int(source_info[1]) - int(target_info[1]))
        comm_auth.append(len(set(source_auth).intersection(set(target_auth))))

        v1 = dense_matrix[index_source,:]
        v2 = dense_matrix[index_target,:]

        sim = cosine_similarity(v1, v2)
        cosine_sim.append(sim[0][0])
    
        counter += 1
        if counter % 1000 == True:
            print(counter, "training examples processsed")

    return overlap_title, temp_diff, comm_auth, cosine_sim

def get_training_features(overlap_title, temp_diff, comm_auth, sim):
    training_features = np.array([overlap_title, temp_diff, comm_auth, sim]).T
    training_features = preprocessing.scale(training_features)
    df = pd.DataFrame(training_features, columns=['overl_title', 'temp_diff', 'comm_author', 'sim'])
    return training_features, df

