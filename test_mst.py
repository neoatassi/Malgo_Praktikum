import os
import unittest
import numpy as np
from graph import Graph
from algorithms.mst import kruskal_mst, prim_mst

class TestMST(unittest.TestCase):
    TEST_CASES = [
        ("G_1_2.txt", 287.32286),
        ("G_1_20.txt", 36.86275),
        ("G_1_200.txt", 12.68182),
        ("G_10_20.txt", 2785.62417),
        ("G_10_200.txt", 372.14417)
    ]
    data_dir = os.path.join(os.path.abspath(__file__), '..' , 'data')
    
    def _test_mst(self, file_name, expected_weight):
        print("="*100)
        # for file_name, expected_weight in self.TEST_CASES:
        graph = Graph(file_path=os.path.join(self.data_dir, file_name), cached=True)
    
        prim_result = prim_mst(graph)
        self.assertAlmostEqual(
            prim_result['weight'],
            expected_weight,
            places=5,
            msg=f"\n\nPrim's MST weight mismatch for {os.path.basename(file_name)}\n"
            f"Expected: {expected_weight}\n"
            f"Actual: {prim_result['weight']}"
        )
        print(f"✓ Prim's passed for {os.path.basename(file_name)}")
        
        kruskal_result = kruskal_mst(graph)
        self.assertAlmostEqual(
            kruskal_result['weight'],
            expected_weight,
            places=5,
             msg=f"\n\nKruskal's MST weight mismatch for {os.path.basename(file_name)}\n"
            f"Expected: {expected_weight}\n"
            f"Actual: {kruskal_result['weight']}"
        )
        print(f"✓ Kruskal's passed for {os.path.basename(file_name)}")
    
    
    # Test edge count (nodes-1 for tree)
    # self.assertEqual(len(result['mst_edges']), graph.node_count - 1)
        
def test_generator(filename, expected_weight):
    """Dynamically generates individual test methods"""
    def test(self):
        self._test_mst(filename, expected_weight)
    return test

for filename, weight in TestMST.TEST_CASES:
    test_name = f'test_{filename.replace(".", "_")}'
    
    setattr(TestMST, test_name, test_generator(filename, weight))

if __name__ == '__main__':
    unittest.main(verbosity=2)