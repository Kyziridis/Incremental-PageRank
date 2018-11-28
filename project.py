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
import pandas as pd
import operator
import os
import sys
import matplotlib.pyplot as plt
from incremental import IncrementalPersonalizedPageRank2 as inc
"""
Gnutella Fasoula
Nodes 	6301
Edges 	20777
"""
###################################################
os.chdir("/home/dead/Documents/SNACS/Snacs_Final")
####################################################
# Fixing the dataset format
def preprocess(raw_data):
    # Input: path of a raw dataset
    # Output: data_path of the corrected data
    # Export: Correct Dataset
    if not os.path.isfile('./'+'corrected'+raw_data):
        start = time()
        pan = pd.read_csv(raw_data, sep="\t", header=None)
        pan = pan.drop(0, axis=0)
        pan.columns= ["Source", "Target"]
        data_path = 'corrected'+raw_data
        pan.to_csv(data_path, header=False, sep='\t', index = False )
        print("Time taken for Preprocess: " + str(time()-start))
        print("Exported Dataset ready >_")
        return data_path    
    else: 
        data_path = 'corrected'+raw_data
        print("Existing Dataset is Ready >_")
        return data_path

#data_path = preprocess('p2p-Gnutella08.txt')
####################################################

def Import(datapath, discriptives=False):
    # Function for importing the dataset into networkx
    # Print some statistics
    start = time()
    with open(datapath, 'rb') as inf:
        data = nx.read_edgelist(inf,create_using=nx.DiGraph(),\
                                delimiter='\t', encoding="utf-8")
    print("Time taken for loading final.csv in secs: " + str(time()-start))

    # Discriptives
    if discriptives:
        print('Is Graph directed?: ' + str(nx.is_directed(data)))
        print('Number of nodes: ' + str(data.number_of_nodes()))
        print('Number of edges: ' + str(data.number_of_edges()))
        print('Density: ' + str(nx.density(data)))
    
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
    return nodes, values

def Evaluate_values(true,pred):
    true = np.array(true)
    pred = np.array(pred)
    MSE = np.mean((true-pred)**2)
    RMSE = np.sqrt(MSE)
    MAE = np.mean(np.abs((true-pred)))
    eucl = np.linalg.norm(true-pred)
    print("-------------------")
    print("| MSE : %.5f |" % MSE)    
    print("| MAE : %.5f |" % MAE)    
    print("| RMSE: %.5f |" % RMSE)    
    print("| Eucl: %.5f |" % eucl)
    print("-------------------")
    return MSE, RMSE, MAE, eucl    

def Evaluate_retrieval(true,pred,plot=False):
    true = set(true)
    pred = set(pred)
    Acc = len(true.intersection(pred))/(len(true) +len(pred))
    jac = len(true.intersection(pred))/len(true.union(pred))
    
recall = np.linspace(0.0, 1.0, num=42)
precision = np.random.rand(42)*(1.-recall)

# take a running maximum over the reversed vector of precision values, reverse the
# result to match the order of the recall vector
decreasing_max_precision = np.maximum.accumulate(precision[::-1])[::-1]    
    

if __name__=='__main__':
    results=[]
    data_path = preprocess('p2p-Gnutella08.txt')
    data = Import(data_path, discriptives=True)
    t_nodes,t_values = PPR(data, node='4', sort=True, maxiter=500, alpha=0.7, num=20 )
    hat_nodes, hat_values = Approximate(data,node='4', sort=True, random=False, num=20 )
    results = Evaluate_values(t_values,hat_values)
    
    
    





























    