
def matrix_multiply(A, B):
    """ Multiply two matrices A and B.

    :param A: the right matrix
    :param B: the left matrix
    :return: A * B
    """
    # define m and n for the matrix as well as l, the connecting dimension between A and B
    m, l, n = len(A), len(A[0]), len(B[0])

    # initialize an all zeros matrix
    C = [[0.0 for _ in range(len(B[0]))] for _ in range(len(A))]

    # iterative over the rows of C
    for i in range(m):
        # iterative over the columns of C
        for j in range(n):
            # set C[i][j] to the dot product of ith row of A and the jth column of B
            C[i][j] = sum(A[i][k] * B[k][j] for k in range(l))

    # return the matrix C = A @ B
    return C


def matrix_pivot(A):
    """ Find the optimal pivoting matrix for A.

    :return: the permutation matrix, P
    """
    # define m and n for matrix A
    m, n = len(A), len(A[0])

    # create an identity permutation matrix
    P = [[float(i == j) for i in range(m)] for j in range(n)]

    # iterate over the rows of A
    for i in range(m):
        # find the row with the max value of in the it column of A
        row = max(range(i, m), key=lambda j: abs(A[j][i]))

        # swap P[row] with the ith row of P
        P[i], P[row] = P[row], P[i]

    # return the permuted matrix P
    return P


def lower_upper_decomposition(A):
    """ Factorize A in upper and lower matrices, L and U.

    :return: P, L and U
    """
    n = len(A)

    # get the pivot matrix P
    P = matrix_pivot(A)

    # permute the rows of A with P
    PA = matrix_multiply(P, A)

    # initialize the L and U matrices
    L = [[0.0 for _ in range(n)] for _ in range(n)]
    U = [[0.0 for _ in range(n)] for _ in range(n)]

    for j in range(n):
        # set the diagonal values of L to 1.0
        L[j][j] = 1.0

        for i in range(j + 1):
            U[i][j] = PA[i][j] - sum(U[k][j] * L[i][k] for k in range(i))

        for i in range(j, n):
            L[i][j] = (PA[i][j] - sum(U[k][j] * L[i][k] for k in range(j))) / U[j][j]

    return P, L, U









