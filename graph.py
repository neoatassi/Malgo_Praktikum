import argparse
import os
import pickle
import sys


class Graph:
    def __init__(self, directed=False, file_path=any, cached=bool):
        
        self.file_path = file_path
        self.cached = cached
        
        self.pickle_path = os.path.splitext(file_path)[0] + ".pkl"
        
        if self.cached and os.path.exists(self.pickle_path):
            self.adjacency = self.deserialize_graph()
        else:
            self.build_graph()
            self.serialize_graph()
        
    def build_graph(self):
        status = f"Reading graph from {self.file_path}"
        
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
    
    
    g = Graph(file_path=sys.argv[1], cached=args.use_cache)