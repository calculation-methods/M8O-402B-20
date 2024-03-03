import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['figure.figsize'] = (15, 12)

def analitic(x, t):
    return np.exp(-x) * np.cos(x) * np.cos(2 * t)

def u_x_0(x):
    return np.exp(-x) * np.cos(x)

def u_x_0t(x):
    return 0

def u_r(t):
    return 0

def u_l(t):
    return np.cos(2 * t)

def compute_analytical_on_grid(time_p, l, h_steps):
    node = int(np.pi / (2 * h_steps)) + 1
    grid = np.zeros((l, node))
    for i in range(l):
        for j in range(node):
            x_val = j * h_steps
            t_val = i * time_p
            grid[i, j] = analitic(x_val, t_val)
    return grid

def explicit(time_p, l, h_steps, start_aprox):
    node = int(np.pi / (2 * h_steps)) + 1
    grid = np.zeros((l, node))
    for i in range(l):
        grid_ = np.zeros(node)
        if i == 0:
            for j in range(node):
                grid_[j] = u_x_0(j * h_steps)
        elif i == 1:
            for j in range(node):
                if start_aprox == 1:
                    grid_[j] = grid[i - 1][j] + u_x_0t(j * h_steps) * time_p
                elif start_aprox == 2:
                    term = ((-u_x_0(j * h_steps)) * (time_p ** 2)) / 2
                    grid_[j] = grid[i - 1][j] + u_x_0t(j * h_steps) * time_p + term
        else:
            grid_[0] = u_l(i * time_p)
            for j in range(1, node - 1):
                term = (time_p ** 2) / (h_steps ** 2) * (grid[i - 1][j + 1] - 2 * grid[i - 1][j] + grid[i - 1][j - 1])
                grid_[j] = term + 2 * grid[i - 1][j] - grid[i - 2][j]
            grid_[-1] = u_r(i * time_p)
        grid[i] = grid_
    return grid

def implicit(time_p, l, h_steps, start_aprox):
    node = int(np.pi / (2 * h_steps)) + 1
    grid = np.zeros((l, node))
    alpha = np.zeros(node - 1)
    beta = np.zeros(node - 1)
    a = -(time_p ** 2) / (h_steps ** 2)
    b = 1 + (2 * (time_p ** 2) / (h_steps ** 2))
    c = a
    for i in range(l):
        grid_ = np.zeros(node)
        if i == 0:
            for j in range(node):
                grid_[j] = u_x_0(j * h_steps)
        elif i == 1:
            for j in range(node):
                if start_aprox == 1:
                    grid_[j] = grid[i - 1][j] + u_x_0t(j * h_steps) * time_p
                elif start_aprox == 2:
                    term = ((-u_x_0(j * h_steps)) * (time_p ** 2)) / 2
                    grid_[j] = grid[i - 1][j] + u_x_0t(j * h_steps) * time_p + term
        else:
            beta[0] = u_l(i * time_p)
            for j in range(1, node - 1):
                alpha[j] = -a / (b + c * alpha[j - 1])
                beta[j] = (2 * grid[i - 1][j] - grid[i - 2][j] - c * beta[j - 1]) / (b + c * alpha[j - 1])
            grid_[node - 1] = u_r(i * time_p)
            for j in range(node - 2, -1, -1):
                grid_[j] = grid_[j + 1] * alpha[j] + beta[j]
        grid[i] = grid_
    return grid

def error(res, analitic):
    er = max(abs(res - analitic) ** 2)
    return er

def update_plot(analitic, time_p, l, start_aprox, h_steps, cur_display_mode):
    if cur_display_mode == 1:
        name = 'Явная схема.'
        res = explicit(time_p, l, h_steps, start_aprox)
    elif cur_display_mode == 2:
        name = 'He явная схема.'
        res = implicit(time_p, l, h_steps, start_aprox)

    plot_function(name, analitic, res, time_p, l, h_steps)
    plt.draw()

def plot_function(name, analitic, res, time_p, l, h_steps, num_last_layers=4):
    node = int(np.pi / (2 * h_steps)) + 1
    X = np.array([i * h_steps for i in range(node)])

    for layer in analitic[-num_last_layers:]:
        fst.plot(X, layer)

    fst.legend(['t = {}'.format(time_p - i * time_p) for i in range(num_last_layers)], fontsize=7, loc='upper right')

    scd.set_title(name + 'Результаты сетке', fontsize=10)
    scd.set_xlabel('x', fontsize=9)
    scd.set_ylabel('U(x, t)', fontsize=9)

    for layer in res[-num_last_layers:]:
        scd.plot(X, layer)
    scd.legend(['t = {}'.format(time_p - i * time_p) for i in range(num_last_layers)], fontsize=5, loc='upper right')

    T = np.array([i * time_p for i in range(l)])
    MSE_error = (np.array([error(i, j) for i, j in zip(res, analitic)]))

    thd.set_title('1e-5            MSE: h = {}, tau = {}'.format(h_steps, time_p), fontsize=9, pad=5, loc='left')
    thd.set_xlabel('t', fontsize=9)
    thd.set_ylabel('mse_error', fontsize=9)
    thd.plot(T, MSE_error)
    thd.scatter(T, MSE_error, marker='o', c='r', s=50)
    thd.legend(['Значение ошибки в разные моменты времени'], fontsize=7, loc='upper right')


time_p = 0.001
h_steps = 0.001
l = 3000

cur_display_mode = int(input("Выберите схему (1 - Явная схема, 2 - He явная схема): "))
start_aprox = int(input("Введите порядок аппроксимации для начальных условий (1, 2): "))

analitic = compute_analytical_on_grid(time_p, l, h_steps)
fig = plt.figure(figsize=(15, 12))
fst = fig.add_subplot(2, 3, 1)
scd = fig.add_subplot(2, 3, 2)
thd = fig.add_subplot(2, 3, 3)

fst.set_title('Аналитическое решение. Результаты сетке ', fontsize=10)
fst.set_xlabel('x', fontsize=9)
fst.set_ylabel('U(x, t)', fontsize=9)

display_modes = [1, 2]
display_index = 0

update_plot(analitic, time_p, l, start_aprox, h_steps, cur_display_mode)
plt.show()
