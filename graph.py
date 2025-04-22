import argparse
from collections import deque
import logging
import os
import pickle
import sys
import time

import numpy as np


class Graph:
    def __init__(self, directed=False, file_path=any, cached=bool):
        
        start_time = time.perf_counter()
        
        self.file_path = file_path
        self.cached = cached
        
        self.pickle_path = os.path.splitext(file_path)[0] + ".pkl"
        
        if self.cached and os.path.exists(self.pickle_path):
            self.deserialize_graph()
        else:
            status = f"Reading graph from {self.file_path}"
            print(status)
            logging.info(status)
            self.build_graph()
            self.serialize_graph()
            
        output = f"Parsing Time: {(time.perf_counter() - start_time)*1000:.1f}ms"    
        print(output)
        logging.info(output)
        
    def build_graph(self):
        start_time = time.perf_counter()
        status = f"Reading graph from {self.file_path}"
        print(status)
        logging.info(status)
        
        with open(self.file_path, 'r') as f:
            data = f.read().split()
    
        self.node_count = int(data[0])                # First token is node count
        edges = data[1:]                     # Remaining tokens are edges
        self.adjacency = [[] for _ in range(self.node_count)]   # Adjacency List: Preallocate an empty list for each node

        # Process edges in pairs (i, j)
        for k in range(0, len(edges), 2):
            i = int(edges[k])
            j = int(edges[k+1])
            self.adjacency[i].append(j)
            # also vice versa since undirected
            self.adjacency[j].append(i)
            
            output = f"Read Graph Time: {(time.perf_counter() - start_time)*1000:.1f}ms"    
            print(output)
            logging.info(output)
    
    def serialize_graph(self):
        with open(self.pickle_path, 'wb') as f:
            pickle.dump({
                'node_count': self.node_count,
                'adjacency_list': self.adjacency
            }, f)
    
    def deserialize_graph(self):
        # Load adjacency list from a cached binary file
        #pkl_file = os.path.splitext(file_path)[0] + ".pkl"
        if os.path.exists(self.pickle_path):
            with open(self.pickle_path, 'rb') as f:
                data = pickle.load(f)
        self.node_count = data['node_count']
        self.adjacency = data['adjacency_list']
    

def bfs(adjacency, visited, start):
    """BFS traversal implementation"""
    queue = deque([start])
    visited[start] = True
    while queue:
        node = queue.popleft()
        for neighbor in adjacency[node]:
            if not visited[neighbor]:
                visited[neighbor] = True
                queue.append(neighbor)
   
def dfs(adjacency, visited, start):
    """DFS traversal implementation"""
    stack = [start]
    visited[start] = True
    while stack:
        node = stack.pop()
        for neighbor in adjacency[node]:
            if not visited[neighbor]:
                visited[neighbor] = True
                stack.append(neighbor)
                
def count_components(graph, traversal):
    start_time = time.perf_counter()
    
    # Component counter using specified traversal algorithm
    n = len(graph.adjacency)
    visited = np.zeros(n, dtype=bool)
    components = 0
    
    for u in range(n):
        if not visited[u]:
            components += 1
            traversal(graph.adjacency, visited, u)
    
    # output = f"Traversal completed using {traversal.__name__.upper()}"
    output = f"Analysis Time: {(time.perf_counter() - start_time)*1000:.1f}ms"                    
    print(output)
    logging.info(output)
    return components             

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='P1: Connected Components')
    parser.add_argument('input_file', help='Path to graph file')
    parser.add_argument('-a', '--algorithm', 
                        choices=['bfs', 'dfs'],
                        default='bfs',
                        help='Traversal algorithm (bfs|dfs)')
    parser.add_argument('--no-cache', 
                        action='store_false',
                        dest='use_cache',
                        help='Disable graph caching')
    
    args = parser.parse_args()
    
    # Map algorithm names to strategy functions
    algorithms = {
        'bfs': bfs,
        'dfs': dfs
    }
    
    g = Graph(file_path=args.input_file, cached=args.use_cache)
    print(g.adjacency)
    
    
    components = count_components(g, algorithms[args.algorithm])
    
    print(f"Component count: {components}")
    logging.info(f"Component count: {components}")