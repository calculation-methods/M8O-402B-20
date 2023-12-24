import numpy as np

def analyt_func(x, t):
    return np.exp(-t) * np.sin(x)

def func(x, t):
    return np.cos(x) * np.exp(-t)

def border1(t):
    return np.exp(-t)

def border2(t):
    return -np.exp(-t)

def run_through(a, b, c, d, s):
    P = np.zeros(s + 1)
    Q = np.zeros(s + 1)

    P[0] = -c[0] / b[0]
    Q[0] = d[0] / b[0]

    k = s - 1
    for i in range(1, s):
        P[i] = -c[i] / (b[i] + a[i] * P[i - 1])
        Q[i] = (d[i] - a[i] * Q[i - 1]) / (b[i] + a[i] * P[i - 1])
    P[k] = 0
    Q[k] = (d[k] - a[k] * Q[k - 1]) / (b[k] + a[k] * P[k - 1])

    x = np.zeros(s)
    x[k] = Q[k]

    for i in range(s - 2, -1, -1):
        x[i] = P[i] * x[i + 1] + Q[i]

    return x


def explicit(K, t, tau, h, x, approx_st, approx_bo):
    N = len(x)
    U = np.zeros((K, N))
    t += tau
    for j in range(N):
        U[0, j] = np.sin(x[j])
        if approx_st == 1:
            U[1][j] = np.sin(x[j]) - np.sin(x[j]) * tau
        if approx_st == 2:
            U[1][j] = np.sin(x[j]) - np.sin(x[j]) * tau + (-2 * np.sin(x[j]) + np.cos(x[j]) + func(x[j], t)) * tau**2 / 2

    for k in range(1, K - 1):
        t += tau
        for j in range(1, N - 1):
            U[k + 1, j] = ((2 + 3 * tau) * U[k, j] - U[k - 1, j] + \
                           tau**2 * ((U[k, j + 1] - 2 * U[k, j] + U[k, j - 1]) / h**2 + (U[k, j + 1] - U[k, j - 1]) / (2 * h) - U[k, j] - func(x[j], t))) \
                          / (1 + 3 * tau)
        if approx_bo == 1:
            U[k + 1, 0] = -h * border1(t) + U[k + 1, 1]
            U[k + 1, N - 1] = h * border2(t) + U[k + 1, N - 2]
        elif approx_bo == 2:
            U[k + 1, 0] = (2 * h * border1(t) - 4 * U[k + 1, 1] + U[k + 1, 2]) / -3
            U[k + 1, N - 1] = (2 * h * border2(t) + 4 * U[k + 1, N - 2] - U[k + 1, N - 3]) / 3
        elif approx_bo == 3:
            U[k + 1, 0] = (2 * tau ** 2 * (h - h**2 / 2) * border1(t) - 2 * tau ** 2 * U[k + 1, 1] + tau ** 2 * h ** 2 * func(x[0], t) \
                           - h ** 2 * (2 + 3 * tau) * U[k, 0] + h ** 2 * U[k - 1, 0]) / (-2 * tau**2 - tau**2 * h**2 - h**2 - 3 * h**2 * tau)
            U[k + 1, N - 1] = (2 * tau ** 2 * (h + h**2 / 2) * border2(t) + 2 * tau ** 2 * U[k + 1, N - 2] - tau ** 2 * h ** 2 * func(x[N - 1], t)
                               + h ** 2 * (2 + 3 * tau) * U[k, N - 1] - h ** 2 * U[k - 1, N - 1]) / (2 * tau**2 + tau**2 * h**2 + h**2 + 3 * h**2 * tau)

    return U


def implicit(K, t, tau, h, x, approx_st, approx_bo):
    N = len(x)
    U = np.zeros((K, N))
    t += tau
    for j in range(N):
        U[0, j] = np.sin(x[j])
        if approx_st == 1:
            U[1][j] = np.sin(x[j]) - np.sin(x[j]) * tau
        if approx_st == 2:
            U[1][j] = np.sin(x[j]) - np.sin(x[j]) * tau + (-2 * np.sin(x[j]) + np.cos(x[j]) + func(x[j], t)) * tau**2 / 2

    for k in range(1, K - 1):
        a = np.zeros(N)
        b = np.zeros(N)
        c = np.zeros(N)
        d = np.zeros(N)
        t += tau

        for j in range(1, N - 1):
            a[j] = 1 / h**2 - 1 / (2 * h)
            b[j] = -2 / h**2 - 1 - 1 / tau**2 - 3 / tau
            c[j] = 1 / h**2 + 1 / (2 * h)
            d[j] = func(x[j], t) - (2 * U[k, j]) / tau**2 - (3 * U[k, j]) / tau + U[k - 1, j] / tau**2
        if approx_bo == 1:
            b[0] = -1 / h
            c[0] = 1 / h
            d[0] = border1(t)

            a[N - 1] = -1 / h
            b[N - 1] = 1 / h
            d[N - 1] = border2(t)
        elif approx_bo == 2:
            k0 = 1 / (2 * h) / c[1]
            b[0] = (-3 / (2 * h)) + a[1] * k0
            c[0] = 2 / h + b[1] * k0
            d[0] = border1(t) + d[1] * k0

            k1 = -(1 / (h * 2)) / a[N - 2]
            a[N - 1] = (-2 / h) + b[N - 2] * k1
            b[N - 1] = (3 / (h * 2)) + c[N - 2] * k1
            d[N - 1] = border2(t) + d[N - 2] * k1
        elif approx_bo == 3:
            b[0] = -1 - h**2 / 2 - h**2 / (2 * tau**2) - (3 * h**2) / (2 * tau)
            c[0] = 1
            d[0] = border1(t) * (h - h ** 2 / 2) + (func(x[0], t) * h ** 2) / 2 - (U[k, 0] * h ** 2) / tau ** 2 \
                   + (U[k - 1, 0] * h**2) / (2 * tau**2) - (U[k , 0] * 3 * h**2) / (2 * tau)

            a[N - 1] = -1
            b[N - 1] = 1 + h**2 / 2 + h**2 / (2 * tau**2) + (3 * h**2) / (2 * tau)
            d[N - 1] = border2(t) * (h + h ** 2 / 2) - (func(x[N - 1], t) * h ** 2) / 2 + (U[k, N - 1] * h ** 2) / tau ** 2 \
                       - (U[k - 1, N - 1] * h**2) / (2 * tau**2) + (U[k , N - 1] * 3 * h**2) / (2 * tau)

        u_new = run_through(a, b, c, d, N)
        for i in range(N):
            U[k + 1, i] = u_new[i]

    return U


def main():

    N = 50
    K = 5000
    time = 3
    h = (np.pi - 0) / N
    tau = time / K
    x = np.arange(0, np.pi + h / 2, h)
    T = np.arange(0, time, tau)
    t = 0

    # схема крест
    # апроксимации начальных условий первого порядка
    Ue11 = explicit(K, t, tau, h, x, 1, 1)
    Ue12 = explicit(K, t, tau, h, x, 1, 2)
    Ue13 = explicit(K, t, tau, h, x, 1, 3)
    # апроксимации начальных условий второго порядка
    Ue21 = explicit(K, t, tau, h, x, 2, 1)
    Ue22 = explicit(K, t, tau, h, x, 2, 2)
    Ue23 = explicit(K, t, tau, h, x, 2, 3)

    # неявная схема
    # апроксимации начальных условий первого порядка
    Ue11 = implicit(K, t, tau, h, x, 1, 1)
    Ue12 = implicit(K, t, tau, h, x, 1, 2)
    Ue13 = implicit(K, t, tau, h, x, 1, 3)
    # апроксимации начальных условий второго порядка
    Ui21 = implicit(K, t, tau, h, x, 2, 1)
    Ui22 = implicit(K, t, tau, h, x, 2, 2)
    Ui23 = implicit(K, t, tau, h, x, 2, 3)
    return 0

main()