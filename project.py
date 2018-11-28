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
        pan.columns= ["Source", "Target"]
        data_path = 'corrected'+raw_data
        pan.to_csv(data_path, header=False, sep='\t', index = False )
        os.chdir("../")
        print("Time taken for Preprocess: " + str(time()-start))
        print("Exported Dataset ready >_")
        return data_path    
    else: 
        data_path = 'corrected'+raw_data
        print("Existing Dataset is Ready >_\n")
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
    print("Time taken for loading final.csv in secs: " + str(time()-start))

    # Discriptives
    if discriptives:
        print('\n-------------------------')
        print('DISCRIPTIVE_STATISTICS')
        print('-------------------------')
        print('Is Graph directed?: ' + str(nx.is_directed(data)))
        print('Number of nodes: %i' % data.number_of_nodes())
        print('Number of edges: %i' % data.number_of_edges())
        print('Density: %.5f' % nx.density(data))
        print('-------------------------')
    
    return data

#data = Import(data_path, discriptives=True)

# Pagerank
def PPR (data, node, sort = False,  maxiter=500, alpha=0.7, num=10):
    start = time()    
    truePPR = nx.pagerank(data, alpha=alpha, personalization={node: 1}, max_iter=maxiter)
    print("Time taken for PageRank computation: " + str(time()-start))
    if sort:
        true_spr = sorted(truePPR.items(),key=operator.itemgetter(1),reverse=True)
        nodes = [n for n,_ in true_spr]
        values = [v for _,v in true_spr]
        return nodes[0:num], values[0:num]
    else:
        nodes,values = truePPR.keys(), truePPR.values()
        nodes, values = list(nodes), list(values)
        return nodes, values

def Approximate(data, node, sort=False, random=False, num=10):
    start = time()
    if random: random_node = random.choice(list(data.nodes()))
    else: random_node = node 
    #node = '4'
    # Init the Class
    increment = inc(graph=data, node=random_node)
    # initial random walks
    increment.initial_random_walks()
    # Hat sorted_Approximation PPR
    hat_PPR = increment.compute_personalized_page_ranks()
    if sort:
        sort_hat_PPR = sorted(hat_PPR.items(), key=operator.itemgetter(1), reverse=True)
        nodes = [n for n,_ in sort_hat_PPR]
        values = [v for _,v in sort_hat_PPR]
        nodes, values = nodes[0:num], values[0:num]
    else:
        nodes, values = hat_PPR.keys(),hat_PPR.values()
        nodes, values = list(nodes), list(values)
    print('Time taken for approximation: ' + str(time()-start))    
    return nodes, values

def Evaluate_values(true,pred):
    true = np.array(true)
    pred = np.array(pred)
    MSE = np.mean((true-pred)**2)
    RMSE = np.sqrt(MSE)
    MAE = np.mean(np.abs((true-pred)))
    eucl = np.linalg.norm(true-pred)
    print("\n-----------------")
    print("ERROR_METRICS")
    print("-----------------")
    print("| MSE : %.5f |" % MSE)    
    print("| MAE : %.5f |" % MAE)    
    print("| RMSE: %.5f |" % RMSE)    
    print("| Eucl: %.5f |" % eucl)
    print("-----------------")
    return MSE, RMSE, MAE, eucl    

def Evaluate_retrieval(true,pred,plot=False):
    ff1 = f1_score(true, pred, average='micro')
    true = set(true)
    pred = set(pred)
    Acc = len(true.intersection(pred))/(len(true) +len(pred))
    jac = len(true.intersection(pred))/len(true.union(pred))
    print("\n--------------------")
    print("RETRIEVAL_METRICS")
    print("--------------------")
    print("| f1_score: %.4f" % ff1)
    print("| Accuracy: %.4f" % Acc)
    print("| Jaccard : %.4f" % jac)
    print("--------------------")
#recall = np.linspace(0.0, 1.0, num=11)
#precision = np.random.rand(42)*(1.-recall)

# take a running maximum over the reversed vector of precision values, reverse the
# result to match the order of the recall vector
#decreasing_max_precision = np.maximum.accumulate(precision[::-1])[::-1]    
    

if __name__=='__main__':
    results=[]
    data_path = preprocess('p2p-Gnutella08.txt',head = 4)
    data = Import(data_path, discriptives=True, directed=False)
    t_nodes,t_values = PPR(data, node='4', sort=True, maxiter=500, alpha=0.7, num=10 )
    hat_nodes, hat_values = Approximate(data,node='4', sort=True, random=False, num=10 )
    results_v = Evaluate_values(t_values,hat_values)
    results_r = Evaluate_retrieval(t_nodes,hat_nodes)
    
    
    





























    