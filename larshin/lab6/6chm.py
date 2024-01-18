import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['figure.figsize'] = (15, 12)

def AnSol(x, t):
    return np.exp(-x) * np.cos(x) * np.cos(2 * t)

def Stx(x):
    return np.exp(-x) * np.cos(x)

def Stxt(x):
    return 0

def St(t):
    return 0

def Kt(t):
    return np.cos(2 * t)

def AnGrid(time, l, h):
    # Количество узлов
    node = int(np.pi / (2 * h)) + 1
    grid = np.zeros((l, node))
    # Итерации по времени
    for i in range(l):
        #По пространству
        for j in range(node):
            x = j * h
            t = i * time
            grid[i, j] = AnSol(x, t)
    return grid


def explicit(time, l, h, approx):
    node = int(np.pi / (2 * h)) + 1
    grid = np.zeros((l, node))

    for i in range(l):
        grid_ = np.zeros(node)
        if i == 0:
            for j in range(node):
                grid_[j] = Stx(j * h)
        elif i == 1:
            for j in range(node):
                if approx == 1:
                    grid_[j] = grid[i - 1][j] + Stxt(j * h) * time
                elif approx == 2:
                    term = ((-Stx(j * h)) * (time ** 2)) / 2
                    grid_[j] = grid[i - 1][j] + Stxt(j * h) * time + term
        else:
            grid_[0] = Kt(i * time)
            for j in range(1, node - 1):
                term = (time ** 2) / (h ** 2) * (grid[i - 1][j + 1] - 2 * grid[i - 1][j] + grid[i - 1][j - 1])
                grid_[j] = term + 2 * grid[i - 1][j] - grid[i - 2][j]
            grid_[-1] = St(i * time)
        grid[i] = grid_
    return grid

def implicit(time, l, h, approx):
    node = int(np.pi / (2 * h)) + 1
    grid = np.zeros((l, node))
    alpha = np.zeros(node - 1)
    beta = np.zeros(node - 1)

    a = -(time ** 2) / (h ** 2)
    b = 1 + (2 * (time ** 2) / (h ** 2))
    c = a

    for i in range(l):
        grid_ = np.zeros(node)

        if i == 0:
            for j in range(node):
                grid_[j] = Stx(j * h)
        elif i == 1:
            for j in range(node):
                if approx == 1:
                    grid_[j] = grid[i - 1][j] + Stxt(j * h) * time
                elif approx == 2:
                    term = ((-Stx(j * h)) * (time ** 2)) / 2
                    grid_[j] = grid[i - 1][j] + Stxt(j * h) * time + term
        else:
            beta[0] = Kt(i * time)
            for j in range(1, node - 1):
                alpha[j] = -a / (b + c * alpha[j - 1])
                beta[j] = (2 * grid[i - 1][j] - grid[i - 2][j] - c * beta[j - 1]) / (b + c * alpha[j - 1])
            grid_[node - 1] = St(i * time)

            for j in range(node - 2, -1, -1):
                grid_[j] = grid_[j + 1] * alpha[j] + beta[j]

        grid[i] = grid_
    return grid



def error(res, AnSol):
    er = max(abs(res - AnSol) ** 2)
    return er





def Draw(name, AnSol, res, time, l, h, num_last_layers=4):
    node = int(np.pi / (2 * h)) + 1
    X = np.array([i * h for i in range(node)])

    for layer in AnSol[-num_last_layers:]:
        fig2.plot(X, layer)

    fig2.legend(['t = {}'.format(time - i * time) for i in range(num_last_layers)], loc='upper right')
    fig4.set_title(name + ' Результаты работы по сетке')
    fig4.set_xlabel('x')
    fig4.set_ylabel('U(x, t)')

    for layer in res[-num_last_layers:]:
        fig4.plot(X, layer)
    fig4.legend(['t = {}'.format(time - i * time) for i in range(num_last_layers)], loc='upper right')

    T = np.array([i * time for i in range(l)])
    MSE_error = (np.array([error(i, j) for i, j in zip(res, AnSol)]))

    fig3.set_xlabel('t')
    fig3.set_ylabel('mse_error')
    fig3.plot(T, MSE_error)
    fig3.scatter(T, MSE_error, marker='o', c='b', s=50)
    fig3.legend(['Ощибка для разных временных точек'], loc='upper right')

def RefreshPlot(AnSol, time, l, approx, h, cur_display_mode):
    if cur_display_mode == 1:
        name = 'Явная схема.'
        res = explicit(time, l, h, approx)
    elif cur_display_mode == 2:
        name = 'Heявная схема.'
        res = implicit(time, l, h, approx)

    Draw(name, AnSol, res, time, l, h)
    plt.draw()

time = 0.001
h = 0.001
l = 3000

strstart = int(input("1 - Явная схема\n 2 - He явная схема: "))
approx = int(input("Порядок аппроксимации(1 или 2): "))

AnSol = AnGrid(time, l, h)
fig = plt.figure(figsize=(15, 12))
fig2 = fig.add_subplot(2, 3, 1)
fig4 = fig.add_subplot(2, 3, 2)
fig3 = fig.add_subplot(2, 3, 3)

fig2.set_title('Аналитическое решение. Результаты работы по сетке ')
fig2.set_xlabel('x')
fig2.set_ylabel('U(x, t)')

display_modes = [1, 2]
display_index = 0

RefreshPlot(AnSol, time, l, approx, h, strstart)
plt.show()
