#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 17:21:19 2018

@author: s2077981
"""
import random
import numpy as np
from numpy import cumsum, array
import networkx as nx

class IncrementalPersonalizedPageRank2:
   
    def __init__(self, graph, node, number_of_random_walks=1000, reset_probability=0.2):
        """
        Initializes the incremental personalized page rank class by determining the graph, the seed node, the number
        of random walks, the reset probability and the length of each random walk.

        :type node: The seed node at which all random walks begin
        :param graph: The graph for which the incremental page rank is computed
        :param number_of_random_walks: The number of random walks starting at the seed node
        :param reset_probability: The probability with which a random walk jumps back to the seed node
        """
        self.graph = graph
        self.node = node
        self.number_of_random_walks = number_of_random_walks
        self.reset_probability = reset_probability

        self.random_walks = list()
        self.added_edges = list()
        self.removed_edges = list()

    def initial_random_walks(self):
        """
        Initiates the random_walk_from_node function starting from the seed node, number_of_random_walks times
        """
        while len(self.random_walks) < self.number_of_random_walks:
            self.regular_random_walk(self.node)
        return

    def regular_random_walk(self, node):
        """
        Computes a random walk starting from node and appending all nodes it passes though to the list random_walk
        :param node: The node at which the random walk begins
        """
        random_walk = [node]
        c = random.uniform(0, 1)
        while c > self.reset_probability:
            if len(list(self.graph.neighbors(random_walk[-1]))) > 0:
                current_node = random_walk[-1]
                current_neighbors = list(self.graph.neighbors(current_node))    
                next_node = random.choice(current_neighbors)
                random_walk.append(next_node)
                c = random.uniform(0, 1)
            else:
                break
        self.random_walks.append(random_walk)
        return

    def add_random_walk(self, previous_random_walk):
        """
        Takes a given random walk segment and computes random walk of length random_walk_length starting at the final
        node in the previous random walk. The idea is that once the graph is modified some random walks will be
        recomputed starting at a given node in the graph.
        :param previous_random_walk: A random walk segment which is not as long as random_walk_length
        """
        random_walk = previous_random_walk
        c = random.uniform(0, 1)
        while c > self.reset_probability:
            if len(list(self.graph.neighbors(random_walk[-1]))) > 0:
                current_node = random_walk[-1]
                current_neighbors = list(self.graph.neighbors(current_node))
                current_edge_weights = array(
                    [self.graph[current_node][neighbor]['weight'] for neighbor in current_neighbors])
                cumulated_current_edge_weights = cumsum(current_edge_weights)
                if cumulated_current_edge_weights[-1] == 0:
                    break
                random_id = list(
                    cumulated_current_edge_weights < (random.uniform(0, 1) * cumulated_current_edge_weights[-1])).index(
                    False)
                next_node = current_neighbors[random_id]
                random_walk.append(next_node)
                c = random.uniform(0, 1)
            else:
                break
        self.random_walks.append(random_walk)
        return

    def compute_personalized_page_ranks(self):
        """
        Determines the personalized page ranks based the random walks in the list random_walks
        :return: A dictionary of nodes and corresponding page ranks
        """
        zeros = [0 for _ in range(len(self.graph.nodes()))]
        #zeros = np.zeros_like(len(self.graph.nodes()))
        page_ranks = dict(zip(self.graph.nodes(), zeros))
        visit_times = dict(zip(self.graph.nodes(), zeros))
        nodes_in_random_walks = []
        for random_walk in self.random_walks:
            nodes_in_random_walks.extend(random_walk)
        for node in self.graph.nodes():
            visit_times[node] = nodes_in_random_walks.count(node)
        for node in self.graph.nodes():
            try:
                page_ranks[node] = float(visit_times[node]) / sum(visit_times.values())
            except ZeroDivisionError:
                print ("List of visit times is empty...")
        return page_ranks



#arxidies