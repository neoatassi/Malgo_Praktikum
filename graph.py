import argparse
from collections import deque
import logging
import os
import pickle
import sys
import time
from line_profiler import profile
import heapq

import numpy as np

#from algorithms.connected_components import count_components
from algorithms.mst import kruskal_mst, prim_mst
from logger import log, setup_logger

class Graph:
    def __init__(self, directed=False, file_path='data/Graph1.txt', cached=True):
        
        start_time = time.perf_counter()
        
        self.directed = directed
        self.node_count = int()                
        self.adjacency = []   # Adjacency List: Preallocate an empty list for each node
        
        self.file_path = file_path
        self.cached = cached
        
        self.pickle_path = os.path.splitext(self.file_path)[0] + ".pkl"
        
        if self.cached and os.path.exists(self.pickle_path):
            self.deserialize_graph()
        else:    
            self.build_graph()
            self.build_csr()
            
            self.serialize_graph()
            
        output = f"Loading Time: {(time.perf_counter() - start_time)*1000:.1f}ms"    
        print(output)
        logging.info(output)
      
    # @profile   
    def build_graph(self):
        start_time = time.perf_counter()
        log(f"Building graph from {self.file_path}")
        
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
            
        # Sort edge list for Kruskal's
        # self.edge_list.sort(key=lambda x: x[2])
             
        log(f"Build Time: {(time.perf_counter() - start_time)*1000:.1f}ms")
        
        
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
        log(f"Saving graph to {self.pickle_path}")
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
            log(f"Loading graph from {self.pickle_path}")
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
    
    def get_neighbors_csr(self, node):
        # this gets neighboring nodes from the CSR data structure
        start = self.indptr[node]
        end = self.indptr[node + 1]
        return self.indices[start:end], self.data[start:end]
    
    def get_edges(self):
        return self.edge_list
    
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
    
    def dfs(adjacency, visited, start):
        """DFS traversal implementation"""
        stack = [start]
        visited[start] = True
        while stack:
            node = stack.pop()
            
            # Get neighbors from CSR
            # start = self.indptr[node]
            # end = self.indptr[node + 1]
            # neighbors = self.indices[start:end]
            
            for neighbor in adjacency[node]:
                neighbor = neighbor[0]  # Extract node from (node, weight)
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)
                    
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
    
    def prim_mst(self):
        start_time = time.perf_counter()
        visited = np.zeros(self.node_count, dtype=bool)
        heap = []
        mst_edges = []
        mst_weight = np.int32()

        # Start from node 0
        heapq.heappush(heap, (0.0, 0, -1))  # (weight, node, parent)

        while heap:
            weight, node, parent = heapq.heappop(heap)
            if not visited[node]:
                visited[node] = True
                if parent != -1:
                    mst_edges.append((parent, node, weight))
                    mst_weight += weight
                
                '''For the CSR data structure'''
                # neighbors, weights = self.get_neighbors(node)
                # for neighbor, w in zip(neighbors, weights):
                #     if not visited[neighbor]:
                #         heapq.heappush(heap, (w, neighbor, node))
                
                neighbors = self.adjacency[node]
                for neighbor, weight in neighbors:
                    if not visited[neighbor]:
                        heapq.heappush(heap, (weight, neighbor, node))
                        
        run_time = f"Analysis Time: {(time.perf_counter() - start_time)*1000:.1f}ms"
        return {"mst_edges": mst_edges, 
                "weight": mst_weight.round(5),
                "time": run_time }
    
    def kruskal_mst(self):
        start_time = time.perf_counter()
        
        edge_list = self.get_edges()  # Pre-sorted edge list
        parent = list(range(self.node_count))
        mst_edges = []
        mst_weight = np.int32()

        def find(u):
            """Find the root of node u with path compression."""
            if parent[u] != u:
                parent[u] = find(parent[u])
            return parent[u]

        def union(u, v):
            """Merge the sets of u and v."""
            root_u = find(u)
            root_v = find(v)
            if root_u != root_v:
                parent[root_v] = root_u

        for u, v, w in edge_list:
            if find(u) != find(v):
                mst_edges.append((u, v, w))
                mst_weight += w
                union(u, v)

        run_time = f"Analysis Time: {(time.perf_counter() - start_time)*1000:.1f}ms"
        return {"mst_edges": mst_edges, 
                "weight": mst_weight.round(5),
                "time": run_time }
        

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
                
# def count_components(graph, traversal):
#     start_time = time.perf_counter()
#     
#     # Component counter using specified traversal algorithm
#     n = len(graph.adjacency)
#     visited = np.zeros(graph.node_count, dtype=bool)
#     components = 0
#     
#     for u in range(graph.node_count):
#         if not visited[u]:
#             components += 1
#             traversal(graph.adjacency, visited, u)
#     
#     # output = f"Traversal completed using {traversal.__name__.upper()}"
#     output = f"Analysis Time: {(time.perf_counter() - start_time)*1000:.1f}ms"                    
#     print(output)
#     logging.info(output)
#     return components             


