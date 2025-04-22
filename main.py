from collections import deque
import sys
import os
import logging
import time
from line_profiler import profile
import numpy as np
import pickle
from typing import Callable
import argparse

# Set up logging
logging.basicConfig(
    filename='connected_components.log',
    # INFO as minimum log level
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='a'
)

def serialize_graph(adjacency, file_path):
    # Save adjacency list to binary file for persistence
    pkl_file = os.path.splitext(file_path)[0] + ".pkl"
    with open(pkl_file, 'wb') as f:
        pickle.dump(adjacency, f)
    logging.info(f"Serialized graph to {pkl_file}")
    
def deserialize_graph(file_path):
    # Load adjacency list from a cached binary file
    pkl_file = os.path.splitext(file_path)[0] + ".pkl"
    if os.path.exists(pkl_file):
        with open(pkl_file, 'rb') as f:
            return pickle.load(f)
    return None

def read_graph(file_path, use_cache=True):
    start_time = time.perf_counter()
    
    if use_cache:
        # Try to load cached version first
        adjacency = deserialize_graph(file_path)
        if adjacency is not None:
            elapsed = (time.perf_counter() - start_time) * 1000
            # output = f"Loaded graph from cache {os.path.splitext(file_path)[0]}"
            output = f"Loaded graph from cache"
            print(output)
            logging.info(output)
            return adjacency
    
    
    # Read entire file and split into tokens
    status = f"Reading graph from {file_path}"
    print(status)
    logging.info(status)
    
    with open(file_path, 'r') as f:
        data = f.read().split()
    
    n = int(data[0])                     # First token is node count
    edges = data[1:]                     # Remaining tokens are edges
    adjacency = [[] for _ in range(n)]   # Adjacency List: Preallocate an empty list for each node
    
    # Process edges in pairs (i, j)
    for k in range(0, len(edges), 2):
        i = int(edges[k])
        j = int(edges[k+1])
        adjacency[i].append(j)
        # also vice versa since undirected
        adjacency[j].append(i)
        
    if use_cache:
        serialize_graph(adjacency, file_path)
    
    output = f"Read Graph Time: {(time.perf_counter() - start_time)*1000:.1f}ms"    
    print(output)
    logging.info(output)
    return adjacency

#@profile
def bfs_count(adjacency):
    start_time = time.perf_counter()
    
    n = len(adjacency)
    visited = np.zeros(n, dtype=bool)
    # visited = [False] * n
    components = 0
    
    for u in range(n):
        if not visited[u]:
            components += 1
            # deques are implemented as double-linked lists and are quite efficient for appending and popping
            queue = deque([u])
            visited[u] = True
            while queue:
                node = queue.popleft()
                #if not visited[node]:
                #   visited[node] = True
                #   queue.append(adjacency[node]) 
                for neighbor in adjacency[node]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)
    
    output = f"BFS Component Count Time: {(time.perf_counter() - start_time)*1000:.1f}ms"                    
    print(output)
    logging.info(output)
    return components

def dfs_count(adjacency):
    start_time = time.perf_counter()
    
    n = len(adjacency)
    visited = np.zeros(n, dtype=bool)
    marked = np.zeros(n, dtype=bool)
    components = 0

    for u in range(n):
        if not visited[u]:
            components += 1
            stack = [u]
            visited[u] = True
            while stack:
                node = stack.pop()
                for neighbor in adjacency[node]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        stack.append(neighbor)
    
    output = f"DFS Component Count Time: {(time.perf_counter() - start_time)*1000:.1f}ms"                    
    print(output)
    logging.info(output)
    return components

#@profile
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

# @profile
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

def count_components(adjacency, traversal):
    start_time = time.perf_counter()
    
    # Component counter using specified traversal algorithm
    n = len(adjacency)
    visited = np.zeros(n, dtype=bool)
    components = 0
    
    for u in range(n):
        if not visited[u]:
            components += 1
            traversal(adjacency, visited, u)
    
    # output = f"Traversal completed using {traversal.__name__.upper()}"
    output = f"Analysis Time: {(time.perf_counter() - start_time)*1000:.1f}ms"                    
    print(output)
    logging.info(output)
    return components



def run_tests(algorithm):
    """Embedded test cases with assertions."""
    test_cases = [
        # Format: (graph_content, expected_components)
        ("0", 0),  # Empty graph
        ("1", 1),  # Single node
        ("2", 2),  # Two disconnected nodes
        ("2\n0 1", 1),  # Two connected nodes
        ("8\n0 1\n0 2\n0 3\n1 2\n3 4\n4 5\n4 6\n5 6\n6 7", 1),  # Your example
        ("8\n0 1\n2 3\n4 5\n6 7", 4),  # 4 components (similar to graph2.txt)
    ]
    
    dir = ".\\data"
    test_cases = [
        #(os.path.join(dir, "Graph1.txt"), 1),
        (os.path.join(dir, "Graph2.txt"), 4),
        (os.path.join(dir, "Graph3.txt"), 4),
        (os.path.join(dir, "Graph_gross.txt"), 222),
        (os.path.join(dir, "Graph_ganzgross.txt"), 9560),
        (os.path.join(dir, "Graph_ganzganzgross.txt"), 306)
    ]
    
    for (graph, expected) in test_cases:
        try:
            adjacency = read_graph(graph)
            result = count_components(adjacency, algorithm)
            assert result == expected, \
                f"Test {graph} failed: Expected {expected}, got {result}"
            print(f"Test {graph} passed")
        except:
            print("Error")
            return
        
    print("All tests passed!")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <graph_file>")
        sys.exit(1)
    
    # print(f"Running Command: {''.join(sys.argv)}")
    # logging.info(f"Running Command: {''.join(sys.argv)}")
    
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
    parser.add_argument('--test',
                        action='store_false',
                        dest='test',
                        help='Run testing assertions for all graphs')
    
    args = parser.parse_args()
    
    # Map algorithm names to strategy functions
    algorithms = {
        'bfs': bfs,
        'dfs': dfs
    }
    
    if args.test:
        print('Running tests...')
        run_tests(algorithms[args.algorithm])
        sys.exit(0)
    
    print(f"Counting connected components in {os.path.basename(args.input_file)} using {args.algorithm.upper()}")
    
    start_time = time.perf_counter()
    #adjacency = read_graph(sys.argv[1], use_cache=True)
    
    adjacency = read_graph(args.input_file, use_cache=args.use_cache)
    
    output = f"Loading Time: {(time.perf_counter() - start_time)*1000:.1f}ms"    
    print(output)
    logging.info(output)
    
    # search_method = 
    
    # components = dfs_count(adjacency)
    # components = bfs_count(adjacency)
    
    components = count_components(adjacency, algorithms[args.algorithm])
    
    print(f"Component count: {components}")
    logging.info(f"Component count: {components}")
    
    elapsed = (time.perf_counter() - start_time)*1000
    print(f"Total Script Time: {elapsed:.1f}ms")
    logging.info(f"Total Script Time: {elapsed:.1f}ms")
    logging.info("="*50)
    # print(f"Process Time: {time.process_time()}")