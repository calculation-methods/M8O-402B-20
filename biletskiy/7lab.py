import numpy as np


def analyt_func(x, y):
    return np.exp(-x - y) * np.cos(x) * np.cos(y)

def border1(x, y):
    return np.exp(-y) * np.cos(y)

def border2(x, y):
    return 0

def border3(x, y):
    return np.exp(-x) * np.cos(x)

def func_border4(x, y):
    return 0

def norm(cur_u, prev_u):
    max = 0
    for i in range(cur_u.shape[0]):
        for j in range(cur_u.shape[1]):
            if abs(cur_u[i, j] - prev_u[i, j]) > max:
                max = abs(cur_u[i, j] - prev_u[i, j])

    return max


def liebman( x, y, h, eps):
    N = len(x)
    count = 0
    prev_u = np.zeros((N, N))
    cur_u = np.zeros((N, N))

    for j in range(len(y)):
        coeff = (border2(x[j], y[j]) - border1(x[j], y[j])) / (len(x) - 1)
        addition = border1(x[j], y[j])
        for i in range(len(x)):
            cur_u[i][j] = coeff * i + addition
    for i in range(1, N - 1):
        cur_u[i, 0] = border3(x[i], y[i])
        cur_u[i, -1] = func_border4(x[i], y[i])

    while norm(cur_u, prev_u) > eps:
        count += 1
        prev_u = np.copy(cur_u)
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                cur_u[i, j] = ((1 + h) * (prev_u[i + 1, j] + prev_u[i, j + 1]) + (1 - h) * (prev_u[i - 1, j] + prev_u[i, j - 1]) \
                               + 4 * h**2 * prev_u[i, j]) / 4
    U = np.copy(cur_u)

    return U, count


def relaxation(x, y, h, eps, tau):
    N = len(x)
    count = 0
    prev_u = np.zeros((N, N))
    cur_u = np.zeros((N, N))
    for j in range(len(y)):
        coeff = (border2(x[j], y[j]) - border1(x[j], y[j])) / (len(x) - 1)
        addition = border1(x[j], y[j])
        for i in range(len(x)):
            cur_u[i][j] = coeff * i + addition
    for i in range(1, N - 1):
        cur_u[i, 0] = border3(x[i], y[i])
        cur_u[i, -1] = func_border4(x[i], y[i])

    while norm(cur_u, prev_u) > eps:
        count += 1
        prev_u = np.copy(cur_u)
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                cur_u[i, j] = (1 - tau) * prev_u[i, j] + tau * (((1 + h) * (prev_u[i + 1, j] + prev_u[i, j + 1]) + (1 - h) \
                                                                 * (prev_u[i - 1, j] + prev_u[i, j - 1]) + 4 * h**2 * prev_u[i, j]) / 4)
    U = np.copy(cur_u)

    return U, count


def Zeidel(x, y, h, eps, tau):
    N = len(x)
    count = 0
    prev_u = np.zeros((N, N))
    cur_u = np.zeros((N, N))
    for j in range(len(y)):
        coeff = (border2(x[j], y[j]) - border1(x[j], y[j])) / (len(x) - 1)
        addition = border1(x[j], y[j])
        for i in range(len(x)):
            cur_u[i][j] = coeff * i + addition
    for i in range(1, N - 1):
        cur_u[i, 0] = border3(x[i], y[i])
        cur_u[i, -1] = func_border4(x[i], y[i])

    while norm(cur_u, prev_u) > eps:
        count += 1
        prev_u = np.copy(cur_u)

        for i in range(1, N - 1):
            for j in range(1, N - 1):
                cur_u[i, j] = (1 - tau) * prev_u[i, j] + tau * (((1 + h) * (prev_u[i + 1, j] + prev_u[i, j + 1]) + (1 - h) \
                                                                 * (cur_u[i - 1, j] + cur_u[i, j - 1]) + 4 * h**2 * prev_u[i, j]) / 4)
    U = np.copy(cur_u)

    return U, count


def main():

    eps = 0.0001
    N = 50
    h = (np.pi / 2 - 0) / N
    x = np.arange(0, np.pi / 2 + h / 2, h)
    y = np.arange(0, np.pi / 2 + h / 2, h)
    # tau = float(input("Введите параметр tau от 0 до 1:"))
    tau=0.5

    # Ul, count_l = liebman(x, y, h, eps)
    # Uz, count_z = Zeidel(x, y, h, eps, tau)
    # Ur, count_r = relaxation(x, y, h, eps, tau)
    return 0



main()