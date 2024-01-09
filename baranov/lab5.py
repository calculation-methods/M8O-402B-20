import numpy as np
import matplotlib.pyplot as plt
def analyt_func(x, a, c, t):
 return x + np.exp(-np.pi * np.pi * a * t) * np.sin(np.pi * x)
def func_border1(a, c, t):
 return 0
def func_border2(a, c, t):
 return 1
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
def explicit(K, t, tau, h, a, c, x, approx): # явный метод
 N = len(x)
 U = np.zeros((K, N))
 for j in range(N):
 U[0, j] = x[j] + np.sin(np.pi * x[j])
 for k in range(K - 1):
 t += tau
 for j in range(1, N - 1):
 U[k + 1, j] = tau * (a * (U[k, j - 1] - 2 * U[k, j] + U[k, j + 1]) / h**2 + \
 + c * U[k, j]) + U[k, j]
 if approx == 1:
 U[k + 1, 0] = func_border1(a, c, t)
 U[k + 1, N - 1] = func_border2(a, c, t)
 elif approx == 2:
 U[k + 1, 0] = func_border1(a, c, t)
 U[k + 1, N - 1] = func_border2(a, c, t)
 elif approx == 3:
 U[k + 1, 0] = func_border1(a, c, t)
 U[k + 1, N - 1] = func_border2(a, c, t)
 return U
def implicit(K, t, tau, h, a1, c1, x, approx): # неявный метод
 N = len(x)
 U = np.zeros((K, N))
 for j in range(N):
 U[0, j] = x[j] + np.sin(np.pi * x[j])
 for k in range(0, K - 1):
 a = np.zeros(N)
 b = np.zeros(N)
 c = np.zeros(N)
 d = np.zeros(N)
 t += tau
 for j in range(1, N - 1):
 a[j] = tau * (a1 / h**2 - 0 / (2 * h))
 b[j] = tau * ((-2 * a1) / h**2 + c1) - 1
 c[j] = tau * (a1 / h**2 + 0 / (2 * h))
 d[j] = -U[k][j]
 b[0] = 1
 c[0] = 0
 d[0] = 0
 a[N - 1] = 0
 b[N - 1] = 1
 d[N - 1] = 1
 u_new = run_through(a, b, c, d, N)
 for i in range(N):
 U[k + 1, i] = u_new[i]
 return U
def Krank_Nikolson(K, t, tau, h, a1, c1, x, approx, theta):
 N = len(x)
 if theta == 0:
 U = explicit(K, t, tau, h, a1, c1, x, approx)
 elif theta == 1:
 U = implicit(K, t, tau, h, a1, c1, x, approx)
 else:
 U_ex = explicit(K, t, tau, h, a1, c1, x, approx)
 U_im = implicit(K, t, tau, h, a1, c1, x, approx)
 U = np.zeros((K, N))
 for i in range(K):
 for j in range(N):
 U[i, j] = theta * U_im[i][j] + (1 - theta) * U_ex[i][j]
 return U
N = 50
K = 30000
time = 3
h = (2 - 0) / N
tau = time / K
x = np.arange(0, 1 + h / 2 - 1e-4, h)
T = np.arange(0, time, tau)
a = 1
c = 0
t = 0
dt = float(input("Введите момент времени от 0 до 3: "))
dt = int(dt*2333)
while (1):
 print("Выберите метод:\n"
 "|0| – ВЫХОД                  |\n"
 "|1| - ЯВНАЯ СХЕМА            |\n"
 "|2| - НЕЯВНАЯ СХЕМА          |\n"
 "|3| - СХЕМА КРАНКА-НИКОЛСОНА |")
 method = int(input())
 if method == 0:
 break
 else:
 print("АПРОКСИМАЦИЯ:\n"
 "1 - двухточечная аппроксимация с 1-ым порядком\n"
 "2 - трехточечная аппроксимация со 2-ым порядком\n"
 "3 - двухточечная аппроксимация со 2-ым порядком")
 approx = int(input())
 if method == 1:
 if a * tau / h**2 <= 0.5:
 print("Условие Куррента выполнено:", a * tau / h**2, "<= 0.5\n")
 U = explicit(K, t, tau, h, a, c, x, approx)
 else:
 print("Условие Куррента не выполнено:", a * tau / h**2, "> 0.5")
 break
 elif method == 2:
 U = implicit(K, t, tau, h, a, c, x, approx)
 elif method == 3:
 theta = 0.5;
 U = Krank_Nikolson(K, t, tau, h, a, c, x, approx, theta)
 U_analytic = analyt_func(x, a, c, T[dt])
 error = abs(U_analytic - U[dt, :])
 plt.title("График точного и численного решения задачи")
 plt.xlabel("x")
 plt.ylabel("U")
 plt.text(0.2, -0.8, "Максимальная ошибка метода: " + str(max(error)))
 plt.axis([0, 3, -1.2, 2])
 plt.plot(x, U_analytic, label = "Точное решение", color = "red")
 plt.scatter(x, U[dt, :], label = "Численное решение")
 plt.grid()
 plt.legend()
 plt.show()
########################################################################################
 plt.title("График ошибки по шагам")
 error_time = np.zeros(len(T))
 for i in range(len(T)):
 error_time[i] = max(abs(analyt_func(x, a, c, T[i]) - U[i, :]))
 plt.plot(T, error_time, label = "По времени")
 plt.plot(x, error, label = "По пространству")
 plt.legend()
 plt.grid()
 plt.show() 
