import numpy as np

def analyt_func(mu, x, y, t):
    return np.sin(x) * np.sin(y) * np.sin(mu * t)

def func(a, b, mu, x, y, t):
    return np.sin(x) * np.sin(y) * (mu * np.cos(mu * t) + (a + b) * np.sin(mu * t))

def border1(mu, y, t):
    return 0

def border2(mu, y, t):
    return -np.sin(y) * np.sin(mu * t)

def border3(mu, x, t):
    return 0

def border4(mu, x, t):
    return -np.sin(x) * np.sin(mu * t)

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


def variable_directions(a1, b1, mu, x, y, hx, hy, K, tau, t):
    U = np.zeros((K, len(x), len(y)))
    sigma_a = (a1 * tau) / (2 * hx**2)
    sigma_b = (b1 * tau) / (2 * hy**2)

    for k in range(K):
        for i in range(len(x)):
            U[k, i, 0] = border3(mu, x[i], t)
            U[k, i, -1] = border4(mu, x[i], t) * hy + U[k, i, -2]
    for k in range(K):
        for j in range(len(y)):
            U[k, 0, j] = border1(mu, y[j], t)
            U[k, -1, j] = border2(mu, y[j], t) * hx + U[k, -2, j]

    for k in range(1, K):
        U_temp = np.zeros((len(x), len(y)))
        a = np.zeros(len(y))
        b = np.zeros(len(y))
        c = np.zeros(len(y))
        d = np.zeros(len(y))
        t += tau / 2
        for i in range(1, len(x) - 1):
            for j in range(1, len(y) - 1):
                a[j] = -sigma_a
                b[j] = 1 + 2 * sigma_a
                c[j] = -sigma_a
                d[j] = (func(a1, b1, mu, x[i], y[j], t) * tau) / 2 + sigma_b * (U[k - 1, i, j + 1] - 2 * U[k - 1, i, j] + U[k - 1, i, j - 1]) \
                       + U[k - 1, i, j]
            b[0] = 1
            c[0] = 0
            d[0] = border3(mu, x[i], t)
            a[-1] = -1
            b[-1] = 1
            d[-1] = border4(mu, x[i], t) * hy
            u_new = run_through(a, b, c, d, len(d))
            U_temp[i] = u_new
            for j in range(len(y)):
                U_temp[0, j] = border1(mu, y[j], t)
                U_temp[-1, j] = border2(mu, y[j], t) * hx + U_temp[-2, j]

        a = np.zeros(len(x))
        b = np.zeros(len(x))
        c = np.zeros(len(x))
        d = np.zeros(len(x))
        t += tau / 2
        for j in range(1, len(y) - 1):
            for i in range(1, len(x) - 1):
                a[i] = -sigma_b
                b[i] = 1 + 2 * sigma_b
                c[i] = -sigma_b
                d[i] = (func(a1, b1, mu, x[i], y[j], t) * tau) / 2 + sigma_a * (U_temp[i + 1, j] - 2 * U_temp[i, j] + U_temp[i - 1, j]) \
                       + U_temp[i, j]
            b[0] = 1
            c[0] = 0
            d[0] = border1(mu, y[j], t)
            a[-1] = -1
            b[-1] = 1
            d[-1] = border2(mu, y[j], t) * hx
            u_new = run_through(a, b, c, d, len(d))
            for i in range(len(u_new)):
                U[k, i, j] = u_new[i]
            for i in range(len(x)):
                U[k, i, 0] = border3(mu, x[i], t)
                U[k, i, -1] = border4(mu, x[i], t) * hy + U[k, i, -2]

    return U.transpose()


def fractional_step(a1, b1, mu, x, y, hx, hy, K, tau, t):
    U = np.zeros((K, len(x), len(y)))
    sigma_a = (a1 * tau) / hx**2
    sigma_b = (b1 * tau) / hy**2

    for k in range(K):
        for i in range(len(x)):
            U[k, i, 0] = border3(mu, x[i], t)
            U[k, i, -1] = border4(mu, x[i], t) * hy + U[k, i, -2]
    for k in range(K):
        for j in range(len(y)):
            U[k, 0, j] = border1(mu, y[j], t)
            U[k, -1, j] = border2(mu, y[j], t) * hx + U[k, -2, j]

    for k in range(1, K):
        U_temp = np.zeros((len(x), len(y)))
        a = np.zeros(len(y))
        b = np.zeros(len(y))
        c = np.zeros(len(y))
        d = np.zeros(len(y))
        t += tau / 2
        for i in range(1, len(x) - 1):
            for j in range(1, len(y) - 1):
                a[j] = -sigma_a
                b[j] = 1 + 2 * sigma_a
                c[j] = -sigma_a
                d[j] = (func(a1, b1, mu, x[i], y[j], t) * tau) / 2 + U[k - 1, i, j]
            b[0] = 1
            c[0] = 0
            d[0] = border3(mu, x[i], t)
            a[-1] = -1
            b[-1] = 1
            d[-1] = border4(mu, x[i], t) * hy
            u_new = run_through(a, b, c, d, len(d))
            U_temp[i] = u_new
            for j in range(len(y)):
                U_temp[0, j] = border1(mu, y[j], t)
                U_temp[-1, j] = border2(mu, y[j], t) * hx + U_temp[-2, j]

        a = np.zeros(len(x))
        b = np.zeros(len(x))
        c = np.zeros(len(x))
        d = np.zeros(len(x))
        t += tau / 2
        for j in range(1, len(y) - 1):
            for i in range(1, len(x) - 1):
                a[i] = -sigma_b
                b[i] = 1 + 2 * sigma_b
                c[i] = -sigma_b
                d[i] = (func(a1, b1, mu, x[i], y[j], t) * tau) / 2 + U_temp[i, j]
            b[0] = 1
            c[0] = 0
            d[0] = border1(mu, y[j], t)
            a[-1] = -1
            b[-1] = 1
            d[-1] = border2(mu, y[j], t) * hx
            u_new = run_through(a, b, c, d, len(d))
            for i in range(len(u_new)):
                U[k, i, j] = u_new[i]
            for i in range(len(x)):
                U[k, i, 0] = border3(mu, x[i], t)
                U[k, i, -1] = border4(mu, x[i], t) * hy + U[k, i, -2]

    return U.transpose()

def main():

    N_x = 30
    N_y = 30
    K = 2000
    time = 3

    h_x = (np.pi - 0) / N_x
    h_y = (np.pi - 0) / N_y
    x = np.arange(0, np.pi + h_x / 2, h_x)
    y = np.arange(0, np.pi + h_y / 2, h_y)
    tau = time / K
    t = 0

    # # метод переменных направлений
    # # a,b,mu,x,y...
    # Uv1 = variable_directions(1, 1, 1, x, y, h_x, h_y, K, tau, t)
    # Uv2 = variable_directions(2, 1, 1, x, y, h_x, h_y, K, tau, t)
    # Uv3 = variable_directions(1, 2, 1, x, y, h_x, h_y, K, tau, t)
    # Uv4 = variable_directions(1, 1, 2, x, y, h_x, h_y, K, tau, t)
    #
    # # метод дробных шагов
    # Uf1 = fractional_step(1, 1, 1, x, y, h_x, h_y, K, tau, t)
    # Uf2 = fractional_step(2, 1, 1, x, y, h_x, h_y, K, tau, t)
    # Uf3 = fractional_step(1, 2, 1, x, y, h_x, h_y, K, tau, t)
    # Uf4 = fractional_step(1, 1, 2, x, y, h_x, h_y, K, tau, t)

    return 0

main()