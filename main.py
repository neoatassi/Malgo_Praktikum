import argparse
import logging
from graph import Graph
from algorithms.connected_components import bfs, count_components, dfs
from algorithms.mst import prim_mst, kruskal_mst
from logger import log, setup_logger

def main():
    # Env Variables
    file_path = "data/G_1_2.txt"
    
    parser = argparse.ArgumentParser(description='Math. Algo. Graph Tool')
    parser.add_argument('input_file', 
                        default=file_path,
                        help='Path to graph file'
                        )
    parser.add_argument('-a', '--algorithm', 
                        #choices=['bfs', 'dfs'],
                        default='prim',
                        help='Traversal algorithm (bfs|dfs)')
    parser.add_argument('--no-cache', 
                        action='store_false',
                        dest='use_cache',
                        #default='--no-cache',
                        help='Disable graph caching')
    
    args = parser.parse_args()
    
    g = Graph(file_path=args.input_file, cached=args.use_cache)
    
    # Map algorithm names to strategy functions
    algorithms = {
        'bfs': bfs,
        'dfs': dfs,
        'prim': prim_mst,
        'kruskal': kruskal_mst
    }
    
    #g = Graph(file_path=args.input_file, cached=args.use_cache)

    if args.algorithm in ('bfs', 'dfs'):
        log(f"Running COMPONENT COUNT using {args.algorithm.upper()} ...")
        components = count_components(g, algorithms[args.algorithm])
        log(f"Component count: {components}")
    elif args.algorithm == 'prim':
        mst = prim_mst(g)
        # print(mst["mst_edges"])
        log(f"Total Weight: {mst['weight']}")
    elif args.algorithm == 'kruskal':
        mst = kruskal_mst(g)
        # print(mst["mst_edges"])
        log(f"Total Weight: {mst['weight']}")
    
    
if __name__ == '__main__':
    setup_logger('info.log')
    main()