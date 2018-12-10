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
from multiprocessing import Pool
import operator
import os
import sys
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from tqdm import tqdm
#import numba as nb
###################################################
os.chdir("/home/dead/Documents/SNACS/Snacs_Final")
#os.chdir("/home/melidell024/Desktop/snacs/project/Incremental-PageRank/")
####################################################
from incremental import IncrementalPersonalizedPageRank2 as inc
from retrieval_metrics import mean_average_precision,plot_precision
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
#@nb.jit
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
#@nb.jit(parallel=True)
def PPR (data, node, maxiter=500, alpha=0.7):
    #start = time()    
    truePPR = nx.pagerank(data, alpha=alpha, personalization={node: 1}, max_iter=maxiter)
    #print("\nTime taken for PageRank computation: %.2fsec" % (time()-start))
    return truePPR

#@nb.jit(parallel=True)
def Approximate(data, node, n_walks=1000):
    #start = time()
    increment = inc(graph=data, node=node, number_of_random_walks=n_walks)
    increment.initial_random_walks()
    hat_PPR = increment.compute_personalized_page_ranks()
    #print('\nTime taken for Approximation: %.2fsec' % (time()-start))    
    return hat_PPR

#@nb.jit
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
    return MAE, RMSE, eucl, eucl_norm, lala, cor[0,1]
    
#@nb.jit
def Evaluate_retrieval(true,pred, k):
    true = sorted(true.items(), key=operator.itemgetter(1), reverse=True)
    pred = sorted(pred.items(), key=operator.itemgetter(1), reverse=True)
    true_nodes = [n for n,_ in true[0:k]]
    pred_nodes = [n for n,_ in pred[0:k*10]]
        
    #ff1 = f1_score(true_nodes, pred_nodes, average='micro')
    true = set(true_nodes)
    pred = set(pred_nodes[0:k])
    
    #ranked retrieval
    true_r = np.array(pred_nodes)
    pred_r = np.array(true_nodes)
    
    retrieval_array = np.isin(true_r,pred_r)
    #precision = r_precision(retrieval_array)
    #avg_precision = average_precision(retrieval_array)
    #m_avg_precision = mean_average_precision(list(retrieval_array))
    
    #print("precision:",precision)
   # print("average precision:",avg_precision)
    #print("out :",len(retrieval_array))
    #print("mean average precision:",m_avg_precision)
    
    #Acc = len(true.intersection(pred))/(len(true))
    jac = len(true.intersection(pred))/len(true.union(pred))
    
    return  retrieval_array, jac # kai oti allo valoume edw:P

#@nb.jit(parallel=True)
def mean_statistics(results,k):
    v =[]
    r = []
    r_jacc=[]
    for i in range(20): v.append(results[i][0][0]); r.append(results[i][1][0]), r_jacc.append(results[i][2][0])
    res = np.array(v)
    #retrieval = np.array(r)
    avg_v = np.mean(res, axis=0)
    Map = mean_average_precision(r)#[0]
    #poutses = mean_average_precision(r)[1]
    poutses,outs = plot_precision(r,k)
    avg_jacc = np.mean(r_jacc)
    return avg_v, Map, poutses, avg_jacc,outs
  
def plotting(x):
    plt.figure(figsize=(8,6))
    plt.tick_params(size=5, labelsize=15)
    plt.plot(x)
    plt.xlabel("Recall", fontsize=15)
    plt.ylabel("Precission", fontsize=15)
    plt.title("Mean Interpolated Average Precission", fontsize=15)
    plt.savefig(dataset+'/plot.png')
    plt.show()


def lolen(node):
    true = PPR(data, node=node)
    hat = Approximate(data,node=node)
    results_v.append(Evaluate_values(true,hat))
    results_retrieval.append(Evaluate_retrieval(true,hat,k)[0])
    results_jaccard.append(Evaluate_retrieval(true,hat,k)[1])        
    return results_v, results_retrieval, results_jaccard

if __name__=='__main__':
    data = ['p2p-Gnutella08.txt']
    k=100
    runs = 20
    for dataset in data:
        print('')
        print("Network-Dataset: || %s ||" % dataset)
        print("-----------------------------------------------")
        if not os.path.exists(dataset):
            os.mkdir(dataset)
        results_retrieval = []
        results_jaccard = []
        results_v = []
        data_path = preprocess(dataset, head=None)
        if dataset == 'cit-HepPh.txt' or dataset=='sx-superuser.txt': direct = True
        else: direct = False
        data = Import(data_path, discriptives=True, directed=direct)
        node = random.sample(list(data.nodes()),20)
        results = Parallel(n_jobs=4)(delayed(lolen)(i) for i in node)
        avg_v, Map, p, avg_jacc,outs= mean_statistics(results,k)
        plotting(p)
        np.save(dataset+'/Map', Map)
        np.save(dataset+'/avg_v', avg_v)
        np.save(dataset+'/retrieval', np.array(results_retrieval))
        np.save(dataset+'/statistics', np.array(results_v))
        print("\n-------------------------")
        print(">_ SUPPORT GNU/Linux  >_")
        print("-------------------------\n")
        print('')
    





























    