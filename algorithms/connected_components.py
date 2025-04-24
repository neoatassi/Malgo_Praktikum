from collections import deque
import time
import numpy as np

from logger import log

def bfs(adjacency, visited, start):
    """Breadth-First Search implementation"""
    queue = deque([start])
    visited[start] = True
    
    while queue:
        node = queue.popleft()
        #neighbors = [neighbor[0] for neighbor in adjacency[node]] # discard weights
        neighbors = adjacency[node]
        for neighbor, _ in neighbors:
            if not visited[neighbor]:
                visited[neighbor] = True
                queue.append(neighbor)
    return visited

def dfs(adjacency, visited, start):
    """Depth-First Search implementation"""
    stack = [start]
    visited[start] = True
    
    while stack:
        node = stack.pop()
        neighbors = adjacency[node]
        for neighbor, _ in neighbors:
            if not visited[neighbor]:
                visited[neighbor] = True
                stack.append(neighbor)
    return visited

def count_components(graph, algorithm) -> int:
    start_time = time.perf_counter()
    """Count connected components using specified algorithm"""
    visited = np.zeros(graph.node_count, dtype=bool)
    components = 0
    adjacency_list = graph.adjacency
    
    for node in range(graph.node_count):
        if not visited[node]:
            components += 1
            algorithm(adjacency_list, visited, node)
    
    run_time = f"Analysis Time: {(time.perf_counter() - start_time)*1000:.1f}ms"
    log(run_time)
    return components