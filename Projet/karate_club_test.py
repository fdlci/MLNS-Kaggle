import unittest
import math

from Louvain import deg_best_result_of_N_Louvain
from Leiden import deg_best_result_of_N_Leiden

class PylouvainTest(unittest.TestCase):

    def test_karate_club_louvain(self):
        graph_file = 'Projet/karate.txt'
        partition, q = deg_best_result_of_N_Louvain(graph_file, N=5)
        print(f'Best modularity found: {q}')
        print(f'Partition {partition}')
        q_ = q * 10000
        self.assertEqual(4, len(partition))
        self.assertEqual(4298, math.floor(q_))
        self.assertEqual(4299, math.ceil(q_))
        print('Test Louvain passed!')

    def test_karate_club_leiden(self):
        graph_file = 'Projet/karate.txt'
        partition, q = deg_best_result_of_N_Leiden(graph_file, N=5)
        print(f'Best modularity found: {q}')
        print(f'Partition {partition}')
        q_ = q * 10000
        self.assertEqual(4, len(partition))
        self.assertEqual(4298, math.floor(q_))
        self.assertEqual(4299, math.ceil(q_))
        print('Test Leiden passed!')


if __name__ == '__main__':
    
    PylouvainTest().test_karate_club_louvain()
    PylouvainTest().test_karate_club_leiden()
