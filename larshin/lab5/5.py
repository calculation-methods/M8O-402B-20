import math
import matplotlib.pyplot as plt
import numpy as np
import ipywidgets as widgets
from IPython.display import display


def Sx(x):
    return 0
def f(x, t):
    return np.cos(x)*(np.cos(t)+np.sin(t))
def St(t):
    return np.sin(t)
def Kt(t):
    return -np.sin(t)
def Sol(x,t):
    return np.sin(t)*np.cos(x)

N = 10
K = 500
T = 5
l = (np.pi) / 2
h = l/N
tau = T/K
sigma = tau/(h ** 2)
x = np.linspace(0,l,N)
t = np.linspace(0,T,K)
Xp, Tp = np.meshgrid(x, t)

if sigma <= 0.5:
    print("Условие Куррата выполнено")
else:
    print("Условие Куррата не выполнено : ", sigma)

def RefreshPlot(index, u_array, t_array):
    plt.figure(figsize=(10, 6))
    plt.plot(x, u_array[index])
    plt.xlabel('x')
    plt.ylabel('u')
    plt.title(f'u от x при t = {t_array[index]:.2f}')
    plt.grid()
    plt.xlim(0, (np.pi) / 2)
    plt.show()
def AnM():
    u = [0]*K
    for i in range(K):
        u[i] = [0]*N
    for i in range(K):
        for j in range(N):
            u[i][j] = Sol(x[j], t[i])
    return u



def explicit(N, K,app,T):
    u = [0]*K
    for i in range(K):
        u[i] = [0]*N
    for j in range(N):
        u[0][j] = Sx(j * h)
    for k in range(K):
        u[k][0] = St(tau * k)

    for k in range(1, K):
        for j in range(1, N-1):
            u[k][j] = u[k-1][j]+tau*(u[k-1][j-1]-2*u[k-1][j]+u[k-1][j+1])/h**2+tau*f(j*h, tau*k)
    if app == 1:
        for k in range(K):
            u[k][-1] = Kt(tau * k)*h + u[k][-2]
    if app == 2:
        for k in range(K):
            u[k][-1] = (Kt(k * tau) * 2 * h - u[k][-3] + 4 * u[k][-2]) / 3
    if app == 3:
        for k in range(K):
            u[k][-1] = (Kt(k * tau) + u[k][-2] / h + 2 * tau * u[k - 1][-1] / h) / (1 / h + 2 * tau / h)

    return u



def pr(a, b, c, d):
    n = len(a)
    p, q = [], []
    p.append(-c[0] / b[0])
    q.append(d[0] / b[0])
    for i in range(1, n):
        p.append(-c[i] / (b[i] + a[i] * p[i - 1]))
        q.append((d[i] - a[i] * q[i - 1]) / (b[i] + a[i] * p[i - 1]))
    x = [0 for _ in range(n)]
    x[n - 1] = q[n - 1]
    for i in range(n-2, -1, -1):
        x[i] = p[i] * x[i+1] + q[i]
    return x

def impicit(N, K, app,T):
    a = np.zeros(N)
    b = np.zeros(N)
    c = np.zeros(N)
    d = np.zeros(N)
    u = [0]*K
    for i in range(K):
        u[i] = [0]*N
    for j in range(N):
        u[0][j] = Sx(j * h)
    for k in range(K):
        u[k][0] = St(tau*k)
    for k in range(1, K):
        a[0] = 0
        b[0] = -(1 + 2 * sigma)
        c[0] = sigma
        d[0] = -u[k-1][0]-sigma*St((k)*tau)
        for j in range(1, N):
            a[j] = sigma
            b[j] = -(1 + 2 * sigma)
            c[j] = sigma
            d[j] = -u[k-1][j] - tau * f(j * h, (k-1) * tau)
        if app == 1:
            a[-1] = sigma
            b[-1] = -(1 + 2 * sigma)
            c[-1] = 0
            d[-1] = -(u[k][-1] - 2 * tau * Kt(k * tau) / h)
        elif app == 2:
            a[-1] = sigma
            b[-1] = -(1 + 2 * sigma)
            c[-1] = 0
            d[-1] = -h*Kt(tau*(k))*h-u[k][-1]-tau*f(N*h,tau*(k+1))
        elif app == 3:
            a[-1] = sigma
            b[-1] = -(1 + 2 * sigma)
            c[-1] = 0
            d[-1] = -((1 - sigma) * u[k - 1][-1] + sigma / 2 * u[k - 1][-2]) - sigma * Kt(k * tau)
        u[k] = pr(a, b, c, d)
    return u



def Crank(N, K, app,T):
    omega = 0.5
    a = np.zeros(N)
    b = np.zeros(N)
    c = np.zeros(N)
    d = np.zeros(N)
    u = [0]*K
    for i in range(K):
        u[i] = [0]*N
    for j in range(N):
        u[0][j] = Sx(j * h)
    for k in range(K):
        u[k][0] = St(tau*k)
    for k in range(1, K):
        a[0] = 0
        b[0] = -(1 + 2 * sigma)
        c[0] = sigma
        d[0] = -u[k-1][0]-sigma*St((k)*tau)
        for j in range(1, N):
            a[j] = sigma
            b[j] = -(1 + 2 * sigma)
            c[j] = sigma
            d[j] = -u[k-1][j] - tau * f(j * h, (k-1) * tau)
        if app == 1:
            a[-1] = sigma
            b[-1] = -(1 + 2 * sigma)
            c[-1] = 0
            d[-1] = -(u[k][-1] - 2 * tau * Kt(k * tau) / h)
        elif app == 2:
            a[-1] = sigma
            b[-1] = -(1 + 2 * sigma)
            c[-1] = 0
            d[-1] = -h*Kt(tau*(k))*h-u[k][-1]-tau*f(N*h,tau*(k+1))
        elif app == 3:
            a[-1] = sigma
            b[-1] = -(1 + 2 * sigma)
            c[-1] = 0
            d[-1] = -((1 - sigma) * u[k - 1][-1] + sigma / 2 * u[k - 1][-2]) - sigma * Kt(k * tau)
        u[k] = pr(a, b, c, d)
    return u


u = AnM()
index_slider = widgets.IntSlider(value=95, min=0, max=len(u) - 1, description='Index u:')
widgets.interactive(RefreshPlot, index=index_slider, u_array=widgets.fixed(u), t_array=widgets.fixed(t))

exp1 = explicit(l, N, K, T, 1)
index_slider = widgets.IntSlider(value=95, min=0, max=len(exp1) - 1, description='Index u:')
widgets.interactive(RefreshPlot, index=index_slider, u_array=widgets.fixed(exp1), t_array=widgets.fixed(t))

exp2 = explicit(l, N, K, T, 2)
index_slider = widgets.IntSlider(value=95, min=0, max=len(exp2) - 1, description='Index u:')
widgets.interactive(RefreshPlot, index=index_slider, u_array=widgets.fixed(exp2), t_array=widgets.fixed(t))

exp3 = explicit(l, N, K, T, 3)
index_slider = widgets.IntSlider(value=95, min=0, max=len(exp3) - 1, description='Index u:')
widgets.interactive(RefreshPlot, index=index_slider, u_array=widgets.fixed(exp3), t_array=widgets.fixed(t))

imp = impicit(l, N, K, T, 1)
index_slider = widgets.IntSlider(value=95, min=0, max=len(imp) - 1, description='Index u:')
widgets.interactive(RefreshPlot, index=index_slider, u_array=widgets.fixed(imp), t_array=widgets.fixed(t))

imp = impicit(l, N, K, T, 2)
index_slider = widgets.IntSlider(value=95, min=0, max=len(imp) - 1, description='Index u:')
widgets.interactive(RefreshPlot, index=index_slider, u_array=widgets.fixed(imp), t_array=widgets.fixed(t))

imp = impicit(l, N, K, T, 3)
index_slider = widgets.IntSlider(value=95, min=0, max=len(imp) - 1, description='Index u:')
widgets.interactive(RefreshPlot, index=index_slider, u_array=widgets.fixed(imp), t_array=widgets.fixed(t))

kn = Crank(l, N, K, T, 3)

index_slider = widgets.IntSlider(value=95, min=0, max=len(kn) - 1, description='Index u:')
widgets.interactive(RefreshPlot, index=index_slider, u_array=widgets.fixed(kn), t_array=widgets.fixed(t))

def Error(analytic, num):
    errors = []
    for i in range(len(analytic)):
        max_error = np.max(np.abs(np.array(analytic[i]) - np.array(num[i])))
        errors.append(max_error)
    return errors

analytic_result = AnM()
explicit_result = explicit(l, N, K, T, 3)
implicit_result = impicit(l, N, K, T, 3)
kn_res = kn

# Вычисление ошибки
errors1 = Error(analytic_result, explicit_result)
errors2 = Error(analytic_result, implicit_result)
errors3 = Error(analytic_result, kn_res)

# Построение графика ошибки от времени
time_steps = np.linspace(0, T, K)
plt.figure(figsize=(10, 6))
plt.plot(time_steps, errors1, label='Явный метод')
plt.plot(time_steps, errors2, label='Неявный метод')
plt.plot(time_steps, errors3, label='Метод Кранка-Николсона')
plt.xlabel('Время')
plt.ylabel('Максимальная ошибка')
plt.title('Зависимость ошибки от времени')
plt.grid()
plt.legend()
plt.show()