#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 17:03:12 2018

@author: s2077981
"""

import numpy as np
import networkx as nx
from time import time as tm
import random
import pandas as pd
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
    pan = pd.read_csv(raw_data, sep="\t", header=None)
    pan = pan.drop(0, axis=0)
    pan.columns= ["Source", "Target"]
    data_path = 'corrected'+raw_data
    pan.to_csv(data_path, header=False, sep='\t', index = False )
    return data_path    

data_path = preprocess('p2p-Gnutella08.txt')
####################################################

def Import(datapath, discriptives=False):
    # Function for importing the dataset into networkx
    # Print some statistics
    start = tm()
    with open(datapath, 'rb') as inf:
        data = nx.read_edgelist(inf,create_using=nx.DiGraph(),\
                                delimiter='\t', encoding="utf-8")
    print("Time taken for loading final.csv in secs: " + str(tm()-start))

    # Discriptives
    if discriptives:
        print('Is Graph directed?: ' + str(nx.is_directed(data)))
        print('Number of nodes: ' + str(data.number_of_nodes()))
        print('Number of edges: ' + str(data.number_of_edges()))
        print('Density: ' + str(nx.density(data)))
    
    return data

data = Import(data_path, discriptives=True)

# Pagerank
def PR(data, sort = False):
    start = tm()    
    pr = nx.pagerank_numpy(data, alpha=0.75)
    print("Time taken for PageRank computation: " + str(tm()-start))
    if sort:
        spr = sorted(pr.keys(),key=pr.get,reverse=True)
        return pr, spr
    else:
        return pr
#pagerank, spr = PR(data, sort=True)    
    

random_node = random.choice(list(data.nodes()))
lala = '4'
# Init the Class
increment = inc(graph=data, node=lala)
# initial random walks
increment.initial_random_walks()
# Hat sorted_Approximation PPR
PPR = increment.compute_personalized_page_ranks()
sort_hatPPR = sorted(PPR.keys(), key=PPR.get, reverse=True)


# True networkX PPR
truePPR = nx.pagerank(data, alpha=0.7, personalization={lala: 1}, max_iter=500)
sort_truePPR= sorted(truePPR.keys(), key=truePPR.get, reverse=True)































    