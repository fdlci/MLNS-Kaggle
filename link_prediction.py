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
from functions import *

random.seed(10)

print("------------- Step 1 --------------")
with open("training_set.txt", "r") as f:
    reader = csv.reader(f)
    training_set  = list(reader)

training_set = [element[0].split(" ") for element in training_set]

to_keep = random.sample(range(len(training_set)), k=int(round(len(training_set)*0.05)))
training_set_reduced = [training_set[i] for i in to_keep]
print(f'We are only working on {len(training_set_reduced)} nodes of the dataset')

# Pandas dataframe
df = pd.DataFrame(training_set_reduced, columns=['source', 'target', 'connected'])
edges = df.loc[df['connected'] == '1']

# Number of edges
print(f'There are {len(edges)} edges and {len(df) - len(edges)} non edges')

# networkx graph !!!!!!! The graph is directed
Graphtype = nx.DiGraph()
G = nx.from_pandas_edgelist(edges, create_using=Graphtype)

# Take the largest weakly conected component
nodes = max(nx.weakly_connected_components(G), key=len) 
G0 = G.subgraph(nodes)

# Make that graph undirected
G0 = G0.to_undirected()
nx.is_connected(G0)

print(compute_network_characteristics(G0))

print("------------- Step 2 --------------")
residual_g, train_samples, train_labels, test_samples, test_labels = generate_samples(G0, train_set_ratio=0.8)

print("degree centrality")
deg_centrality = nx.degree_centrality(G0)
print("done!")

train_features = feature_extractor(G0, train_samples, deg_centrality)
test_features = feature_extractor(G0, test_samples, deg_centrality)

feat_train = pd.DataFrame(train_features, columns=['source_degree_centrality', 'target_degree_centrality', 'pref_attach', 'aai', 'jacard_coeff', 'res_all'])
feat_test = pd.DataFrame(test_features, columns=['source_degree_centrality', 'target_degree_centrality', 'pref_attach', 'aai', 'jacard_coeff', 'res_all'])

t_set_red = {}
for source, target, lab in training_set_reduced:
    t_set_red[(source, target)] = lab

real_test_labels = get_true_test_labels(test_samples, t_set_red)

column_names = ['id', 'year', 'title', 'authors', 'journal', 'abstract']
info = pd.read_csv('node_information.csv', sep=',', names=column_names)

print("------------- Step 3 --------------")
with open("node_information.csv", "r") as f:
    reader = csv.reader(f)
    node_info  = list(reader)

IDs = [element[0] for element in node_info]

vect = TfidfVectorizer(stop_words="english")
abstract_vectorized = vect.fit_transform(info['abstract'])

overlap_title_train, temp_diff_train, comm_auth_train, cosine_sim_train = preprocessing_info(train_samples, abstract_vectorized, IDs, node_info)
overlap_title_test, temp_diff_test, comm_auth_test, cosine_sim_test = preprocessing_info(test_samples, abstract_vectorized, IDs, node_info)

training_add_feat = get_training_features(overlap_title_train, temp_diff_train, comm_auth_train, cosine_sim_train)[1]
testing_add_feat = get_training_features(overlap_title_test, temp_diff_test, comm_auth_test, cosine_sim_test)[1]

all_train_feat = pd.concat([feat_train, training_add_feat], axis=1)
all_test_feat = pd.concat([feat_test, testing_add_feat], axis=1)

print(prediction(G0, all_train_feat.to_numpy(), all_test_feat.to_numpy(), train_labels, real_test_labels))

print("------------- Step 4 --------------")
with open("testing_set.txt", "r") as f:
    reader = csv.reader(f)
    testing_set  = list(reader)

testing_set = [element[0].split(" ") for element in testing_set]

overlap_title_test = []
temp_diff_test = []
comm_auth_test = []
cosine_sim_test = []
dense_matrix = abstract_vectorized.todense()
   
counter = 0
for i in range(len(testing_set)):

    source = testing_set[i][0]
    target = testing_set[i][1]
    
    index_source = IDs.index(source)
    index_target = IDs.index(target)
    
    source_info = [element for element in node_info if element[0]==source][0]
    target_info = [element for element in node_info if element[0]==target][0]
    
    source_title = source_info[2].lower().split(" ")
    source_title = [token for token in source_title if token not in stpwds]
    source_title = [stemmer.stem(token) for token in source_title]
    
    target_title = target_info[2].lower().split(" ")
    target_title = [token for token in target_title if token not in stpwds]
    target_title = [stemmer.stem(token) for token in target_title]
    
    source_auth = source_info[3].split(",")
    target_auth = target_info[3].split(",")
    
    overlap_title_test.append(len(set(source_title).intersection(set(target_title))))
    temp_diff_test.append(int(source_info[1]) - int(target_info[1]))
    comm_auth_test.append(len(set(source_auth).intersection(set(target_auth))))

    v1 = dense_matrix[index_source,:]
    v2 = dense_matrix[index_target,:]

    sim = cosine_similarity(v1, v2)
    cosine_sim_test.append(sim[0][0])
   
    counter += 1
    if counter % 1000 == True:
        print(counter, "testing examples processsed")

testing_features = np.array([overlap_title_test,temp_diff_test,comm_auth_test, cosine_sim_test]).T

testing_features = preprocessing.scale(testing_features)

test_feat = pd.DataFrame(testing_features, columns=['overl_title', 'temp_diff', 'comm_author', 'sim'])

test_graph_feat = feature_extractor(G0, testing_set, deg_centrality)
print(test_graph_feat.shape)

test_graph_feat = pd.DataFrame(test_graph_feat, columns=['source_degree_centrality', 'target_degree_centrality', 'pref_attach', 'aai', 'jacard_coeff', 'res_all'])

total_test_feat = pd.concat([test_graph_feat, test_feat], axis=1)

# initialize basic SVM
classifier = svm.LinearSVC(max_iter=10000)

# train
classifier.fit(all_train_feat.to_numpy(), train_labels)

predictions_SVM = list(classifier.predict(total_test_feat.to_numpy()))

predictions_SVM = zip(range(len(testing_set)), predictions_SVM)

with open("test.csv","w", newline='') as pred1:
    csv_out = csv.writer(pred1)
    for row in predictions_SVM:
        csv_out.writerow(row)