from collections import deque
import sys
import logging
import time
from line_profiler import profile
import numpy as np

# Set up logging
logging.basicConfig(
    filename='connected_components.log',
    # INFO as minimum log level
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='a'
)

def read_graph(file_path):
    start_time = time.perf_counter()
    # Read entire file and split into tokens
    with open(file_path, 'r') as f:
        data = f.read().split()
    
    n = int(data[0])                     # First token is node count
    edges = data[1:]                     # Remaining tokens are edges
    adjacency = [[] for _ in range(n)]   # Adjacency Matrix: Preallocate an empty list for each node
    
    # Process edges in pairs (i, j)
    for k in range(0, len(edges), 2):
        i = int(edges[k])
        j = int(edges[k+1])
        adjacency[i].append(j)
        # also vice versa since undirected
        adjacency[j].append(i)
    
    output = f"Read Graph Time: {(time.perf_counter() - start_time)*1000:.1f}ms"    
    print(output)
    logging.info(output)
    return adjacency

@profile
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

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <graph_file>")
        sys.exit(1)
    
    print(f"Running Command: {''.join(sys.argv)}")
    logging.info(f"Running Command: {''.join(sys.argv)}")
    
    start_time = time.perf_counter()
    adjacency = read_graph(sys.argv[1])
    
    # search_method = 
    
    # components = dfs_count(adjacency)
    components = bfs_count(adjacency)
    
    print(f"Components: {components}")
    logging.info(f"Components: {components}")
    
    elapsed = (time.perf_counter() - start_time)*1000
    print(f"Total Time: {elapsed:.1f}ms")
    logging.info(f"Total Time: {elapsed:.1f}ms")
    logging.info("="*50)
    # print(f"Process Time: {time.process_time()}")
'''

import numpy as np
from collections import defaultdict
import sys
import time

def read_graph_csr(file_path):
    """Build a CSR (Compressed Sparse Row) adjacency list."""
    with open(file_path, 'r') as f:
        data = f.read().split()
    n = int(data[0])
    edges = list(map(int, data[1:]))
    
    # Count neighbors for each node
    adj = defaultdict(list)
    for i, j in zip(edges[::2], edges[1::2]):
        adj[i].append(j)
        adj[j].append(i)  # Undirected
    
    # Build CSR arrays
    indices = np.zeros(n + 1, dtype=np.int32)
    edges_csr = []
    current = 0
    for u in range(n):
        neighbors = adj.get(u, [])
        edges_csr.extend(neighbors)
        current += len(neighbors)
        indices[u + 1] = current
    edges_csr = np.array(edges_csr, dtype=np.int32)
    return n, edges_csr, indices

def count_components_csr(n, edges_csr, indices):
    """BFS using CSR and list-based queue."""
    start_time = time.perf_counter()
    visited = np.zeros(n, dtype=bool)
    components = 0
    
    for u in range(n):
        if not visited[u]:
            components += 1
            queue = [u]
            visited[u] = True
            front = 0  # Track the front of the queue with an index
            
            while front < len(queue):
                node = queue[front]
                front += 1
                # Get neighbors from CSR
                start = indices[node]
                end = indices[node + 1]
                neighbors = edges_csr[start:end]
                
                for neighbor in neighbors:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)
    output = f"Component Count Time: {(time.perf_counter() - start_time)*1000:.1f}ms"                    
    print(output)
    
    return components

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <graph_file>")
        sys.exit(1)
    
    start_time = time.time()
    n, edges_csr, indices = read_graph_csr(sys.argv[1])
    components = count_components_csr(n, edges_csr, indices)
    print(f"Components: {components}")
    print(f"Time: {time.time() - start_time:.2f}s")
    
'''