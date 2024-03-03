import numpy as np
import matplotlib.pyplot as plt


def Fredgolm_II(K, g, x, h, lamb):
    n = x.size
    A = np.zeros((n, n))
    for i in range(n):
        if i == 0:
            A[i, 0] = 1 - 0.5 * lamb * h * K(x[i], x[0])
        else:
            A[i, 0] = - 0.5 * lamb * h * K(x[i], x[0])
    for i in range(n):
        for j in range(n):
            if j == i:
                A[i, j] = 1 - lamb * h * K(x[i], x[j])
            else:
                A[i, j] = - lamb * h * K(x[i], x[j])

    for i in range(n):
        if i == n - 1:
            A[i, n - 1] = 1 - lamb * 0.5 * h * K(x[i], x[n - 1])
        else:
            A[i, n - 1] = - lamb * 0.5 * h * K(x[i], x[n - 1])
    return solve(A, g(x))

def error(x, y, f):
    max = 0
    for i in range(x.size):
        e = np.abs(f(x[i]) - y[i])
        if (e > max):
            max = e
    return max

def solve(A, b):
    n = len(A)
    for k in range(n - 1):
        for i in range(k + 1, n):
            factor = A[i, k] / A[k, k]
            for j in range(k + 1, n):
                A[i, j] -= factor * A[k, j]
            b[i] -= factor * b[k]
        x = [0] * n
        x[n - 1] = b[n - 1] / A[n - 1, n - 1]
        for i in range(n - 2, -1, -1):
            sum_ax = sum(A[i, j] * x[j] for j in range(i + 1, n))
            x[i] = (b[i] - sum_ax) / A[i, i]
        return x

def K(x, t):
    #return 1 / (0.64 * numpy.cos((x + t) / 2) ** 2 - 1)
    return np.exp(x-t)
def g(x):
    # return 25 - 16 * numpy.sin(x) ** 2 return numpy.exp(x)
    return np.exp(x)
def f(x):
    # return 17 / 2 + (128 / 17) * numpy.cos(2 * x) return 2*numpy.exp(x)
    return 2*np.exp(x)

a = 0
b = 1
lamb = 1 / 2
#a = -numpy.pi # b = numpy.pi
# lamb = 3 / (10 * numpy.pi)

h_exact = 0.01
x_exact = np.arange(a, b, h_exact)
y_exact = f(x_exact)
plt.plot(x_exact, y_exact, 'C9', label='точное', linewidth=3, linestyle='--')

h_approx = 0.03
x_approx = np.arange(a, b, h_approx)
y_approx = Fredgolm_II(K, g, x_approx, h_approx, lamb)
y_approx[0] = f(x_approx[0])
y_approx[len(y_approx) - 1] = f(x_approx[x_approx.size - 1])
plt.plot(x_approx, y_approx, 'C3', label='численное')
plt.title('h = ' + str(h_approx) + ' Погрешность = ' + str(error(x_approx, y_approx, f)))
plt.grid(True)
plt.legend()
plt.show()
