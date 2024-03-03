import numpy as np


def qr_alg(matrix, tol=1e-12, max_iter=1000):
    n = matrix.shape[0]
    eigenvalues = np.zeros(n, dtype=complex)
    for i in range(max_iter):
        shift = matrix[-1, -1]
        Q, R = custom_qr(matrix - shift * np.eye(n))
        matrix = R @ Q + shift * np.eye(n)
        if frobenius_norm_upper_triangle(matrix) < tol:
            break
        eigenvalues = np.diag(matrix)
    return eigenvalues


def custom_qr(matrix):
    n = matrix.shape[0]
    Q = np.eye(n)
    R = matrix.copy()
    for i in range(n - 1):
        x = R[i:, i].copy()
        norm_x = np.linalg.norm(x)
        v = x - norm_x * np.eye(len(x))[:, 0]
        v = v / np.linalg.norm(v)
        H = np.eye(len(x)) - 2 * np.outer(v, v)
        R[i:, i:] = H @ R[i:, i:]
        Q[:, i:] = Q[:, i:] @ H.T
    return Q, R


def frobenius_norm_upper_triangle(matrix):
    return np.linalg.norm(np.triu(matrix, k=1), ord='fro')

def arnoldi_method(matrix, k):
    n = len(matrix)
    q = np.random.rand(n)
    q = q / np.linalg.norm(q)
    Q = np.zeros((n, k + 1))
    H = np.zeros((k + 1, k))
    Q[:, 0] = q

    for j in range(k):
        v = np.dot(matrix, Q[:, j])
        for i in range(j + 1):
            H[i, j] = np.dot(Q[:, i], v)
            v = v - H[i, j] * Q[:, i]
        H[j + 1, j] = np.linalg.norm(v)
        Q[:, j + 1] = v / H[j + 1, j]
    eigenvalues_H = qr_alg(H[:k, :k])
    eigenvectors_A = np.dot(Q[:, :k], np.linalg.inv(Q[:, :k].T @ Q[:, :k]) @ Q[:, :k].T)

    return eigenvalues_H, eigenvectors_A


n = int(input("Введите размер матрицы: "))
matrix = np.zeros((n, n))
print("Введите элементы матрицы построчно:")

for i in range(n):
    matrix[i, :] = list(map(float, input().split()))


k = n
eigenvalues, eigenvectors = arnoldi_method(matrix, k)

print("\nМатрица:")
print(matrix)
print("\nВсе собственные значения:")
print(eigenvalues)
print("\nПриближенные собственные векторы:")
print(eigenvectors)