import heapq
import time
from typing import List, Tuple
import numpy as np
from logger import log

def prim_mst(graph):
        start_time = time.perf_counter()
        visited = np.zeros(graph.node_count, dtype=bool)
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
                # neighbors, weights = graph.get_neighbors(node)
                # for neighbor, w in zip(neighbors, weights):
                #     if not visited[neighbor]:
                #         heapq.heappush(heap, (w, neighbor, node))
                
                neighbors = graph.adjacency[node]
                for neighbor, weight in neighbors:
                    if not visited[neighbor]:
                        heapq.heappush(heap, (weight, neighbor, node))
                        
        run_time = f"Analysis Time: {(time.perf_counter() - start_time)*1000:.1f}ms"
        return {"mst_edges": mst_edges, 
                "weight": mst_weight.round(5),
                "time": run_time }
    
def kruskal_mst(graph):
    start_time = time.perf_counter()
    
    edge_list = graph.get_edges()
    
    # Sort edge list after the third element (weight)
    edge_list.sort(key=lambda x: x[2])
    
    parent = list(range(graph.node_count))
    mst_edges = []
    mst_weight = np.int32()
    def find(u):
        """Find the root of node u with path compression."""
        if parent[u] != u:
            parent[u] = find(parent[u])
        return parent[u]
    for u, v, w in edge_list:
        root_u = find(u)
        root_v = find(v)
        if root_u != root_v:
            mst_edges.append((u, v, w))
            mst_weight += w
            parent[root_v] = root_u 
            
    run_time = f"Analysis Time: {(time.perf_counter() - start_time)*1000:.1f}ms"
    return {"mst_edges": mst_edges, 
            "weight": mst_weight.round(5),
            "time": run_time }