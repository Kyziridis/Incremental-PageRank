#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 17:03:12 2018

@author: s2077981
"""

import numpy as np
import networkx as nx
from time import time 
import random
from sklearn.metrics import f1_score, precision_recall_curve, precision_recall_fscore_support
import pandas as pd
import operator
import os
import sys
import matplotlib.pyplot as plt
###################################################
os.chdir("/home/dead/Documents/SNACS/Snacs_Final")
#os.chdir("/home/melidell024/Desktop/snacs/project/Incremental-PageRank/")
####################################################
from incremental import IncrementalPersonalizedPageRank2 as inc
"""
Gnutella Fasoula
Nodes 	6301
Edges 	20777

RoadsPA
Nodes: 1088092
Edges: 1541898
"""

# Fixing the dataset format
def preprocess(raw_data,head=0):
    # Input: path of a raw dataset
    # Output: data_path of the corrected data
    # Export: Correct Dataset
    
    os.chdir("datasets/")
    if not os.path.isfile('corrected'+raw_data):
        start = time()
        pan = pd.read_csv(raw_data, sep="\t", header=head)
        #pan = pan.drop(0, axis=0)
        #pan.columns= ["Source", "Target"]
        data_path = 'corrected'+raw_data
        pan.to_csv(data_path, header=False, sep='\t', index = False )
        os.chdir("../")
        print("Time taken for Preprocess: " + str(time()-start))
        print("Dataset preprocessed and exported >_")
        return data_path    
    else: 
        data_path = 'corrected'+raw_data
        print("Dataset already preprocessed   >_\n")
        os.chdir("../")
        return data_path

#data_path = preprocess('p2p-Gnutella08.txt')
####################################################

def Import(datapath, discriptives=False, directed=True):
    # Function for importing the dataset into networkx
    # Print some statistics
    if directed:
        graph_type = nx.DiGraph()
    else:
        graph_type = nx.Graph()
    start = time()
    with open('datasets/'+datapath, 'rb') as inf:
        data = nx.read_edgelist(inf,create_using=graph_type,\
                                delimiter='\t', encoding="utf-8")
    print("Time taken for edgelist loading: %.2fsec" % (time()-start))

    # Discriptives
    if discriptives:
        print('\n-------------------------')
        print('DESCRIPTIVE_STATISTICS')
        print('-------------------------')
        print('Is Graph directed?: ' + str(nx.is_directed(data)))
        print('Number of nodes: %i' % data.number_of_nodes())
        print('Number of edges: %i' % data.number_of_edges())
        print('Density: %.5f' % nx.density(data))
        print('-------------------------\n')    
    return data

# Pagerank
def PPR (data, node, maxiter=500, alpha=0.7):
    start = time()    
    truePPR = nx.pagerank(data, alpha=alpha, personalization={node: 1}, max_iter=maxiter)
    print("Time taken for PageRank computation: %.2fsec" % (time()-start))
    return truePPR

def Approximate(data, node, n_walks=1000):
    start = time()
    increment = inc(graph=data, node=node, number_of_random_walks=n_walks)
    increment.initial_random_walks()
    hat_PPR = increment.compute_personalized_page_ranks()
    print('Time taken for Approximation: %.2fsec' % (time()-start))    
    return hat_PPR

def Evaluate_values(true,pred):
    true = sorted(true.items(), key=operator.itemgetter(0), reverse=True)
    pred = sorted(pred.items(), key=operator.itemgetter(0), reverse=True)
    true_values = [v for _,v in true]
    pred_values = [v for _,v in pred]
    
    true = np.array(true_values)
    pred = np.array(pred_values)
    
    MSE = np.mean((true-pred)**2)
    RMSE = np.sqrt(MSE)
    MAE = np.mean(np.abs((true-pred)))
    eucl = np.linalg.norm(true-pred)
    eucl_norm = np.linalg.norm(true-pred)/np.linalg.norm(pred)
    lala = np.max(true-pred)/np.max(true)
    cor = np.corrcoef(true,pred)
    print("\n-------------")
    print("ERROR_METRICS")
    print("-------------------")  
    print("| MAE   : %.5f |" % MAE)    
    print("| RMSE  : %.5f |" % RMSE)
    print("| Eucl  : %.5f |" % eucl)    
    print("| Eucl_n: %.5f |" % eucl_norm)
    print("| Suprem: %.5f |" % lala)
    print("| Correl: %.5f |" % cor[0,1])
    print("-------------------")
    return MAE, RMSE, eucl, eucl_norm, lala, cor[0,1]
    

def Evaluate_retrieval(true,pred, k=10):
    true = sorted(true.items(), key=operator.itemgetter(1), reverse=True)
    pred = sorted(pred.items(), key=operator.itemgetter(1), reverse=True)
    true_nodes = [n for n,_ in true[0:k]]
    pred_nodes = [n for n,_ in pred[0:k]]
        
    ff1 = f1_score(true_nodes, pred_nodes, average='micro')
    true = set(true_nodes)
    pred = set(pred_nodes)
    
    Acc = len(true.intersection(pred))/(len(true))
    jac = len(true.intersection(pred))/len(true.union(pred))
    print("\n-----------------")
    print("RETRIEVAL_METRICS")
    print("--------------------")
    print("| f1_score: %.4f |" % ff1)
    print("| Accuracy: %.4f |" % Acc)
    print("| Jaccard : %.4f |" % jac)
    print("--------------------")
    return ff1, Acc, jac # kai oti allo valoume edw:P


if __name__=='__main__':
    data = ['p2p-Gnutella08.txt', 'roadNet-PA-sample.txt']
    for dataset in data:
        print('')
        print("Network-Dataset: || %s ||" % dataset)
        print("-----------------------------------------------")
        os.mkdir(dataset)
        results_r = []
        results_v = []
        data_path = preprocess(dataset, head=3)
        data = Import(data_path, discriptives=True, directed=False)
        for _ in range(10):
            node = random.choice(list(data.nodes()))
            true = PPR(data, node=node )
            hat = Approximate(data,node=node)
            results_v.append( Evaluate_values(true,hat))
            results_r.append(Evaluate_retrieval(true,hat))
        
        np.save(dataset+'/retrieval', np.array(results_r))
        np.save(dataset+'/statistics', np.array(results_v))
        print("\n-------------------------")
        print(">_ SUPPORT GNU/Linux  >_")
        print("-------------------------\n")
        print('')
    





























    