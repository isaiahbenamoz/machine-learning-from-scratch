import unittest
from decomposition import lower_upper


class LowerUpperTester(unittest.TestCase):
    def test_matrix_multiply(self):

        A = [[1, 0], [0, 1]]
        self.assertEqual(lower_upper.matrix_multiply(A, A), A)

        A = [[2, 0], [2, 2]]
        B = [[1, 0], [0, 1]]

        self.assertEqual(lower_upper.matrix_multiply(A, B), A)
        self.assertEqual(lower_upper.matrix_multiply(A, A), [[4, 0], [8, 4]])

    def test_matrix_pivot(self):

        A = [[1, 0], [2, 1]]
        P = [[0, 1], [1, 0]]

        self.assertEqual(lower_upper.matrix_pivot(A), P)

        A = [[1, 2, 0], [2, 2, 1], [3, 4, 5]]
        P = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]

        self.assertEqual(lower_upper.matrix_pivot(A), P)

    def test_lower_upper_decomposition(self):
        A = [[7, 3, -1, 2], [3, 8, 1, -4], [-1, 1, 4, -1], [2, -4, -1, 6]]
        P, L, U = lower_upper.lower_upper_decomposition(A)

        P_t = [[1.0, 0.0, 0.0, 0.0],
               [0.0, 1.0, 0.0, 0.0],
               [0.0, 0.0, 1.0, 0.0],
               [0.0, 0.0, 0.0, 1.0]]

        self.assertEqual(P, P_t)

        L_t = [[1.0, 0.0, 0.0, 0.0],
               [0.42857142857142855, 1.0, 0.0, 0.0],
               [-0.14285714285714285, 0.2127659574468085, 1.0, 0.0],
               [0.2857142857142857, -0.7234042553191489, 0.0898203592814371, 1.0]]

        self.assertEqual(L, L_t)

        U_t = [[7.0, 3.0, -1.0, 2.0],
               [0.0, 6.714285714285714, 1.4285714285714286, -4.857142857142857],
               [0.0, 0.0, 3.5531914893617023, 0.31914893617021267],
               [0.0, 0.0, 0.0, 1.88622754491018]]

        self.assertEqual(U, U_t)
