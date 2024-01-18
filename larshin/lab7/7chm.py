import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def Sx(x):
    return np.sin(x)

def Kx(x):
    return np.sin(x) * np.e

def Sy(y):
    return np.exp(y)

def Ky(y):
    return -np.exp(y)

def AnS(x, y):
    return np.exp(y)*np.sin(x)


def Func(lx=[0, np.pi], ly=[0, 1], Nx=10, Ny=10, eps=0.001, metod='zeidel', w=1, print_itters=False):
    lx = np.array(lx)
    ly = np.array(ly)

    hx = (lx[1] - lx[0]) / (Nx - 1)
    hy = (ly[1] - ly[0]) / (Ny - 1)

    def zeidel(X, Y, M, w):
        return relaxation(X, Y, M, 1)

    def relaxation(X, Y, M, w):

        norm = 0.0
        hx2 = hx * hx
        hy2 = hy * hy

        for i in range(1, Ny - 1):
            diff = w * ((-2 * hx * Sy(Y[i][0]) + 4 * M[i][1] - M[i][2]) / 3 - M[i][0])
            M[i][0] += diff
            diff = abs(diff)
            norm = diff if diff > norm else norm
            for j in range(1, Nx - 1):
                diff = hy2 * (M[i][j - 1] + M[i][j + 1])
                diff += hx2 * (M[i - 1][j] + M[i + 1][j])
                diff /= 2 * (hy2 + hx2)
                diff -= M[i][j]
                diff *= w
                M[i][j] += diff
                diff = abs(diff)
                norm = diff if diff > norm else norm
            diff = w * ((2 * hx * Ky(Y[i][-1]) + 4 * M[i][-2] - M[i][-3]) / 3 - M[i][-1])
            M[i][-1] += diff
            diff = abs(diff)
            norm = diff if diff > norm else norm

        return norm

    def Simple(X, Y, M):
        temp = [[0.0 for _ in range(Nx)] for _ in range(Ny)]
        norm = 0.0
        hx2 = hx * hx
        hy2 = hy * hy

        for i in range(1, Ny - 1):
            temp[i][0] = (-2 * hx * Sy(Y[i][0]) + 4 * M[i][1] - M[i][2]) / 3
            diff = abs(temp[i][0] - M[i][0])
            norm = diff if diff > norm else norm
            for j in range(1, Nx - 1):
                temp[i][j] = hy2 * (M[i][j - 1] + M[i][j + 1])
                temp[i][j] += hx2 * (M[i - 1][j] + M[i + 1][j])
                temp[i][j] /= 2 * (hy2 + hx2)
                diff = abs(temp[i][j] - M[i][j])
                norm = diff if diff > norm else norm
            temp[i][-1] = (2 * hx * Ky(Y[i][-1]) + 4 * M[i][-2] - M[i][-3]) / 3
            diff = abs(temp[i][0] - M[i][0])
            norm = diff if diff > norm else norm

        for i in range(1, Ny - 1):
            M[i] = temp[i]

        return norm

    if metod == 'zeidel':
        method = zeidel
    elif metod == 'limban':
        method = Simple
    else:
        method = relaxation

    x = list(np.arange(lx[0], lx[1] + 1 / (10 * Nx), hx))
    y = list(np.arange(ly[0], ly[1] + 1 / (10 * Nx), hy))

    X = [x for _ in range(Ny)]
    Y = [[y[i] for _ in x] for i in range(Ny)]

    ans = [[0 for _ in range(Nx)] for _ in range(Ny)]
    for j in range(Nx):
        coeff = (Kx(X[-1][j]) - Sx(X[0][j])) / (ly[1] - ly[0])
        addition = Sx(X[0][j])
        for i in range(Ny):
            ans[i][j] = coeff * (Y[i][j] - ly[0]) + addition

    itters = 0

    while (method(X, Y, ans, w) >= eps):
        itters += 1

    if print_itters:
        print("Кол-во итераций: ", {itters})

    return np.array(X), np.array(Y), np.array(ans)


def RealZord(lx0, lx1, ly0, ly1, f):
    x = np.arange(lx0, lx1 + 0.005, 0.005)
    y = np.arange(ly0, ly1 + 0.005, 0.005)
    X = np.ones((y.shape[0], x.shape[0]))
    Y = np.ones((x.shape[0], y.shape[0]))
    Z = np.ones((y.shape[0], x.shape[0]))
    for i in range(Y.shape[0]):
        Y[i] = y
    Y = Y.T
    for i in range(X.shape[0]):
        X[i] = x
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            Z[i, j] = f(X[i, j], Y[i, j])
    return X, Y, Z




def epsilon(x, y, z, f):
    ans = 0.0
    for i in range(len(z)):
        for j in range(len(z[i])):
            ans += (z[i][j] - f(x[i][j], y[i][j]))**2
    return (ans/(len(z[0])*len(z)))**0.5

def Geth(solver, real_f):
    h = []
    e = []
    for N in range(4, 50):
        x, y, z = solver(Nx=N, metod='zeidel')
        h.append(np.pi/N)
        e.append(epsilon(x, y, z, real_f))
    return h, e


def epsilon1(x, y, z, f):
    ans = 0.0
    for i in range(len(z)):
        for j in range(len(z[i])):
            ans += (z[i][j] - f(x[i][j], y[i][j]))**2
    return (ans/(len(z[0])*len(z)))**0.5

def Geth1(solver, real_f):
    h = []
    e = []
    for N in range(4, 50):
        x, y, z = solver(Ny=N, metod='zeidel')
        h.append(1/N)
        e.append(epsilon1(x, y, z, real_f))
    return h, e



def epsilon2(x, y, z, f):
    ans = 0.0
    for i in range(len(z)):
        for j in range(len(z[i])):
            ans += (z[i][j] - f(x[i][j], y[i][j]))**2
    return (ans/(len(z[0])*len(z)))**0.5

def Geth2(solver, real_f):
    h = []
    e = []
    for N in range(4, 50):
        x, y, z = solver(Nx=N, metod='relax', w=0.5)
        h.append(np.pi/N)
        e.append(epsilon2(x, y, z, real_f))
    return h, e


def epsilon3(x, y, z, f):
    ans = 0.0
    for i in range(len(z)):
        for j in range(len(z[i])):
            ans += (z[i][j] - f(x[i][j], y[i][j]))**2
    return (ans/(len(z[0])*len(z)))**0.5

def Geth3(solver, real_f):
    h = []
    e = []
    for N in range(4, 50):
        x, y, z = solver(Ny=N, metod='relax', w=0.5)
        h.append(1/N)
        e.append(epsilon3(x, y, z, real_f))
    return h, e



def epsilon4(x, y, z, f):
    ans = 0.0
    for i in range(len(z)):
        for j in range(len(z[i])):
            ans += (z[i][j] - f(x[i][j], y[i][j]))**2
    return (ans/(len(z[0])*len(z)))**0.5

def Geth4(solver, real_f):
    h = []
    e = []
    for N in range(4, 50):
        x, y, z = solver(Nx=N, metod='limban')
        h.append(np.pi/N)
        e.append(epsilon4(x, y, z, real_f))
    return h, e


def epsilon5(x, y, z, f):
    ans = 0.0
    for i in range(len(z)):
        for j in range(len(z[i])):
            ans += (z[i][j] - f(x[i][j], y[i][j]))**2
    return (ans/(len(z[0])*len(z)))**0.5

def Geth5(solver, real_f):
    h = []
    e = []
    for N in range(4, 50):
        x, y, z = solver(Ny=N, metod='limban')
        h.append(1/N)
        e.append(epsilon5(x, y, z, real_f))
    return h, e
lx=[0, np.pi]
ly=[0, 1]
Nx=10
Ny=10
w = 0.5

X, Y, Z = Func(lx, ly, Nx, Ny, metod='zeidel', print_itters=True)


fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(projection='3d')
ax.plot_wireframe(*RealZord(lx[0], lx[1], ly[0], ly[1], AnS), color="green")
ax.plot_surface(X, Y, Z)
ax.set(xlabel='x', ylabel='t', zlabel='z', title='График приближения и реальной функции методом Зейделя')
plt.show()


plt.figure(figsize = (10, 5))
plt.title("Зависимость погрешности от длины шага")
h, e = Geth(Func, AnS)

plt.plot(h, e, color = "green")
plt.xlabel("$h_x$")
plt.ylabel("e")
plt.legend()
plt.show()


plt.figure(figsize = (10, 5))
plt.title("Зависимость погрешности от длины шага")
h, e = Geth1(Func, AnS)

plt.plot(h, e, color = "green")
plt.xlabel("$h_y$")
plt.ylabel("e")
plt.legend()
plt.show()


X, Y, Z = Func(lx, ly, Nx, Ny, metod='relax', w=w, print_itters=True)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(projection='3d')
ax.plot_wireframe(*RealZord(lx[0], lx[1], ly[0], ly[1], AnS), color="green")
ax.plot_surface(X, Y, Z)
ax.set(xlabel='x', ylabel='t', zlabel='z', title='График приближения и реальной функции явным методом')
plt.show()


plt.figure(figsize = (10, 5))
plt.title("Зависимость погрешности от длины шага")
h, e = Geth2(Func, AnS)

plt.plot(h, e, color = "green")
plt.xlabel("$h_x$")
plt.ylabel("e")
plt.show()


plt.figure(figsize = (10, 5))
plt.title("Зависимость погрешности от длины шага")
h, e = Geth3(Func, AnS)

plt.plot(h, e, color = "red")
plt.xlabel("$h_y$")
plt.ylabel("e")
plt.show()


X, Y, Z = Func(lx, ly, Nx, Ny, metod='limban', print_itters=True)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(projection='3d')
ax.plot_wireframe(*RealZord(lx[0], lx[1], ly[0], ly[1], AnS), color="red")
ax.plot_surface(X, Y, Z)
ax.set(xlabel='x', ylabel='t', zlabel='z', title='График приближения и реальной функции явным методом')
plt.show()


plt.figure(figsize = (10, 5))
plt.title("Зависимость погрешности от длины шага")
h, e = Geth4(Func, AnS)

plt.plot(h, e, color = "green")
plt.xlabel("$h_x$")
plt.ylabel("e")
plt.show()


plt.figure(figsize = (10, 5))
plt.title("Зависимость погрешности от длины шага")
h, e = Geth5(Func, AnS)

plt.plot(h, e, color = "green")
plt.xlabel(" h_y ")
plt.ylabel("e")
plt.show()