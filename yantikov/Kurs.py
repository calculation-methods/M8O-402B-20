import numpy as np
import matplotlib.pyplot as plt

def matrixCreate():
    A = np.zeros((25, 25))
    n = np.random.randint(20, size=40)
    for k in range(len(n)):
            i = np.random.randint(25)
            j = np.random.randint(25)
            A[i][j] = n[k]
            A[j][i] = n[k]
    return A

def maxElem(A):
    i_max, j_max, max_elem = 0, 0, 0
    for i in range(A[0].size):
        for j in range(i+1, A[0].size):
            if (abs(A[i][j])>max_elem):
                max_elem = abs(A[i][j])
                i_max = i
                j_max = j
    return i_max, j_max, max_elem

def rotation(A, eps):
    Ak = np.copy(A)
    eigen_vectors =  np.eye(A[0].size)
    i_max, j_max, max_elem = maxElem(Ak)
    count = 0
    while (max_elem>eps):
        phi = 0.5*np.arctan(2*Ak[i_max][j_max]/(Ak[i_max][i_max]-Ak[j_max][j_max]))
        U = np.eye(Ak.shape[0])
        U[i_max][j_max] = -np.sin(phi)
        U[j_max][i_max] = np.sin(phi)
        U[i_max][i_max] = np.cos(phi)
        U[j_max][j_max] = np.cos(phi)
        Ak = U.T @ Ak @ U
        eigen_vectors = eigen_vectors @ U
        count += 1
        i_max, j_max, max_elem = maxElem(Ak)
    eigen_values = np.array([Ak[i][i] for i in range(A[0].size)])
    return eigen_vectors, eigen_values, count

def lanchosh(A, b, iters, EPSILON):
    Q = np.zeros((A.shape[0], iters + 1))
    alpha = np.zeros((iters))
    beta = np.zeros((iters))
    Q[:,0] = b/np.linalg.norm(b)
    for m in range(iters):
        if np.linalg.norm(b) <= EPSILON:
            break
        v = np.dot(A, Q[:,m])
        alpha[m] = np.dot(Q[:,m],v)
        if m == 0:
            v = v - alpha[m]*Q[:,m]
        else:
            v = v - alpha[m]*Q[:,m] - beta[m-1]*Q[:,m-1]
        beta[m] = np.linalg.norm(v)
        Q[:,m+1] = v/beta[m]

    T = np.dot(np.dot(Q.T,A), Q)
    Vec_T, Val, _ = rotation(T, 1e-16)
    Vec = Q@Vec_T
    return Vec, Val

A = matrixCreate()
np.set_printoptions(suppress=True)
EPSILON = 0.00000000000000001
b = np.random.rand(A.shape[0])

iters = 25
Vec, Val = lanchosh(A, b, iters, EPSILON)
linalg_eigenvalues = np.sort(np.real(np.linalg.eigvals(A)))
kp_eigenvalues = np.sort(Val)
print("np.linalg:", linalg_eigenvalues)
print("lanchosh:", kp_eigenvalues)

plt.plot(linalg_eigenvalues, 'o-', label='np.linalg',color='green')
plt.plot(kp_eigenvalues, 'o-', label='lanchosh',color='red')
plt.title("Сравнение соб. значений")
plt.xlabel("Индекс")
plt.ylabel("Значение")
plt.legend()
plt.show()