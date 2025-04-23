import argparse
from collections import deque
import logging
import os
import pickle
import sys
import time
from line_profiler import profile

import numpy as np


class Graph:
    def __init__(self, directed=False, file_path=any, cached=bool):
        
        start_time = time.perf_counter()
        
        self.directed = directed
        self.node_count = int()                
        self.adjacency = []   # Adjacency List: Preallocate an empty list for each node
        
        self.file_path = file_path
        self.cached = cached
        
        self.pickle_path = os.path.splitext(file_path)[0] + ".pkl"
        
        if self.cached and os.path.exists(self.pickle_path):
            self.deserialize_graph()
        else:
            
            status = f"Reading graph from {self.file_path}"
            print(status)
            logging.info(status)
            
            self.read_graph()
            self.build_csr()
            self.serialize_graph()
            
        output = f"Parsing Time: {(time.perf_counter() - start_time)*1000:.1f}ms"    
        print(output)
        logging.info(output)
      
    # @profile   
    def read_graph(self):
        start_time = time.perf_counter()
        status = f"Reading graph from {self.file_path}"
        print(status)
        logging.info(status)
        
        with open(self.file_path, 'r') as f:
            data = f.read().split()

        self.node_count = int(data[0])
        edges = data[1:]  
        edge_count = len(edges)     # Count elements to determine if weights are present
        
        self.adjacency = [[] for _ in range(self.node_count)]   # Adjacency List: Preallocate an empty list for each node
        self.edge_list = []

        # Detect whether weighted or not: If divisible by 3 -> weighted
        #                                 if divisible by 2 -> unweighted
        if edge_count % 3 == 0:
            step = 3
        elif edge_count % 2 == 0:
            step = 2
        else:
            raise ValueError("Invalid edge count")

        # Process edges in pairs (i, j)
        for k in range(0, len(edges), step):
            i = int(edges[k])
            j = int(edges[k+1])
            
            weight = float(edges[k+2]) if step == 3 else None
            self.add_edge(i, j , weight)
            # self.add_edge(j, i , weight)
            self.edge_list.append((i, j, weight))
            
            # self.adjacency[i].append((j, weight))
            # also vice versa since undirected
            # self.adjacency[j].append((i, weight))
             
        output = f"Read Graph Time: {(time.perf_counter() - start_time)*1000:.1f}ms"    
        print(output)
        logging.info(output)
        
        
    def build_csr(self):
        self.indptr = np.zeros(self.node_count + 1, dtype=np.int32)
        self.indices = []
        self.data = []
        current = 0
        for node in range(self.node_count):
            self.indptr[node] = current
            for neighbor, weight in self.adjacency[node]:
                self.indices.append(neighbor)
                self.data.append(weight)
                current += 1
        self.indptr[self.node_count] = current
        self.indices = np.array(self.indices, dtype=np.int32)
        self.data = np.array(self.data, dtype=np.float32)
    
    # def serialize_graph(self):
    #     with open(self.pickle_path, 'wb') as f:
    #         pickle.dump({
    #             'node_count': self.node_count,
    #             'adjacency_list': self.adjacency
    #         }, f)
    
    def serialize_graph(self):
        with open(self.pickle_path, 'wb') as f:
            pickle.dump({
                'node_count': self.node_count,
                'adjacency_list': self.adjacency,
                'indptr': self.indptr,
                'indices': self.indices,
                'data': self.data,
                'edge_list': self.edge_list
            }, f)
    
    # def deserialize_graph(self):
    #     # Load adjacency list from a cached binary file
    #     #pkl_file = os.path.splitext(file_path)[0] + ".pkl"
    #     if os.path.exists(self.pickle_path):
    #         with open(self.pickle_path, 'rb') as f:
    #             data = pickle.load(f)
    #     self.node_count = data['node_count']
    #     self.adjacency = data['adjacency_list']
    
    def deserialize_graph(self):
        if os.path.exists(self.pickle_path):
            with open(self.pickle_path, 'rb') as f:
                data = pickle.load(f)
                
        self.node_count = data['node_count']
        self.adjacency = data['adjacency_list']
        
        self.indptr = data['indptr']
        self.indices = data['indices']
        self.data = data['data']
        self.edge_list = data['edge_list']
    
    def __repr__(self):
        output = ""
        for node, neighbors in enumerate(self.adjacency):
            output += f"{node} -> {neighbors}\n"
        return output
    
    def add_edge(self, from_node, to_node, weight):
        self.adjacency[from_node].append((to_node, weight))
        # add edge both ways if graph is not directed
        if not self.directed:
            self.adjacency[to_node].append((from_node, weight))
            
    # def remove_edge(self, from_node, to_node):
    
    def get_neighbors(self, node):
        # this gets neighboring nodes from the CSR data structure
        start = self.indptr[node]
        end = self.indptr[node + 1]
        return self.indices[start:end], self.data[start:end]
    
    #@profile
    def bfs(self, adjacency, visited, start):
        """BFS traversal USING CSR"""
        queue = deque([start])
        visited[start] = True
        while queue:
            node = queue.popleft()
            
            # Get neighbors from CSR
            # start = self.indptr[node]
            # end = self.indptr[node + 1]
            # neighbors = self.indices[start:end]
            
            neighbors = adjacency[node]
            # neighbors = self.get_neighbors(node)[0]            
            
            for neighbor in neighbors:
                neighbor = neighbor[0]  # Extract node from (node, weight)
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
                    
    def count_components(self, traversal):
        start_time = time.perf_counter()

        # Component counter using specified traversal algorithm
        n = len(self.adjacency)
        visited = np.zeros(self.node_count, dtype=bool)
        components = 0

        for u in range(self.node_count):
            if not visited[u]:
                components += 1
                traversal(self.adjacency, visited, u)

        # output = f"Traversal completed using {traversal.__name__.upper()}"
        output = f"Analysis Time: {(time.perf_counter() - start_time)*1000:.1f}ms"                    
        print(output)
        logging.info(output)
        return components  
        

def bfs(adjacency, visited, start):
    """BFS traversal implementation"""
    queue = deque([start])
    visited[start] = True
    while queue:
        node = queue.popleft()
        neighbors = adjacency[node]
        for neighbor in neighbors:
            neighbor = neighbor[0]  # Extract node from (node, weight)
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
            neighbor = neighbor[0]  # Extract node from (node, weight)
            if not visited[neighbor]:
                visited[neighbor] = True
                stack.append(neighbor)
                
def count_components(graph, traversal):
    start_time = time.perf_counter()
    
    # Component counter using specified traversal algorithm
    n = len(graph.adjacency)
    visited = np.zeros(graph.node_count, dtype=bool)
    components = 0
    
    for u in range(graph.node_count):
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
    
    g = Graph(file_path=args.input_file, cached=args.use_cache)
    
    # Map algorithm names to strategy functions
    algorithms = {
        'bfs': g.bfs,
        'dfs': dfs
    }
    
    #g = Graph(file_path=args.input_file, cached=args.use_cache)
    
    
    components = count_components(g, algorithms[args.algorithm])
    
    print(f"Component count: {components}")
    logging.info(f"Component count: {components}")
    
    # print(g)