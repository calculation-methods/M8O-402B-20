import numpy as np, matplotlib.pyplot as plt
def analyt_func(x, t):
 return np.sin(x-t)+np.cos(x+t)
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
def explicit(K, t, tau, h, x, approx_st): # явный метод
 N = len(x)
 U = np.zeros((K, N))
 t += tau
 for j in range(N):
 U[0, j] = (np.sin(x[j])+np.cos(x[j]))
 if approx_st == 1:
 U[1][j] = (np.sin(x[j]) + np.cos(x[j])) - tau * (np.cos(x[j]) - np.sin(x[j]))
 if approx_st == 2:
 U[1][j] = (np.sin(x[j]) + np.cos(x[j])) - tau * (np.cos(x[j]) + np.sin(x[j])) - (tau ** 2) / 2 * (
 np.cos(x) - np.sin(x)) + (-np.sin(x[j]) - np.cos(x[j])) * (tau ** 2) / 2
 for k in range(1, K - 1):
 t += tau
 for j in range(1, N - 1):
 U[k + 1, j] = ((tau ** 2) * (U[k, j + 1] - 2 * U[k, j] + U[k, j - 1]) / (h ** 2)) + 2 * U[k, j] - U[k -
1, j]
 U[k, 0] = (h * (np.sin(tau * k)+np.cos(tau * k)) - U[k, 1]) / -1
 U[k, N - 1] = (-h * (np.sin(tau * k)+np.cos(tau * k)) - U[k, N - 2]) / -1
 return U
def implicit(K, t, tau, h, x, approx_st): # неявный метод
 N = len(x)
 U = np.zeros((K, N))
 t += tau
 for j in range(N):
 U[0, j] = (np.sin(x[j])+np.cos(x[j]))
 if approx_st == 1:
 U[1][j] = (np.sin(x[j])+np.cos(x[j])) - tau * (np.cos(x[j])-np.sin(x[j]))
 if approx_st == 2:
 U[1][j] = (np.sin(x[j])+np.cos(x[j])) - tau * (np.cos(x[j])+np.sin(x[j]))- (tau ** 2) / 2 * 
(np.cos(x)-np.sin(x)) + (-np.sin(x[j])-np.cos(x[j])) * (tau ** 2) / 2
 for k in range(K - 1):
 U[k + 1, 0] = U[k + 1, 1]
 U[k + 1, N - 1] = U[k + 1, N - 2]
 for k in range(1, K - 1):
 a = np.zeros(N)
 b = np.zeros(N)
 c = np.zeros(N)
 d = np.zeros(N)
 t += tau
 for j in range(1, N - 1):
 a[j] = (tau ** 2) * (1 / h ** 2 - 1 / (2 * h))
 b[j] = -2 * tau ** 2 / (h ** 2) - tau ** 2 - 1 - 3 * tau / 2
 c[j] = (tau ** 2) * (1 / h ** 2 + 1 / (2 * h))
 d[j] = -2 * U[k, j]+ U[k - 1, j]
 b[0] = 1
 c[0] = 0
 d[0] = 0
 a[N - 1] = 0
 b[N - 1] = 1
 d[N - 1] = 0
 u_new = run_through(a, b, c, d, N)
 for i in range(N):
 U[k + 1, i] = u_new[i]
 return U
N = 100
K = 7000
time = 3
h = (np.pi - 0) / N
tau = time / K
x = np.arange(0, np.pi + h / 2 - 1e-4, h)
T = np.arange(0, time, tau)
t = 0
dt = float(input("Введите момент времени от 0 до 3: "))
dt = int(dt*2333)
while (1):
 print("МЕТОД:\n"
 "0 - ВЫХОД\n"
 "1 - ЯВНАЯ СХЕМА\n"
 "2 - НЕЯВНАЯ СХЕМА")
 method = int(input())
 if method == 0:
 break
 else:
 print("АПРОКСИМАЦИЯ НАЧАЛЬНЫХ УСЛОВИЙ:\n"
 "1 - 1-ОГО ПОРЯДКА\n"
 "2 - 2-ОГО ПОРЯДКА")
 approx_st = int(input())
 if method == 1:
 if tau / h**2 <= 1:
 print("Условие Куррента выполнено:", tau / h**2, "<= 1\n")
 U = explicit(K, t, tau, h, x, approx_st)
 else:
 print("Условие Куррента не выполнено:", tau / h**2, "> 1")
 break
 elif method == 2:
 U = implicit(K, t, tau, h, x, approx_st)
 U_analytic = analyt_func(x, T[dt])
 error = abs(U_analytic - U[dt, :])
 plt.title("График точного и численного решения задачи")
 plt.xlabel("x")
 plt.ylabel("y")
 plt.text(0.2, -0.8, "Максимальная ошибка метода: " + str(max(error)))
 plt.axis([0, 3, -2.2, 2.2])
 plt.scatter(x, U_analytic, label = "Точное решение", color = "green")
 plt.plot(x, U[dt, :], label = "Численное решение", color = "red")
 plt.grid()
 plt.legend()
 plt.show()
 plt.title("График зависимости ошибок по времени и пространству")
 error_time = np.zeros(len(T))
 for i in range(len(T)):
 error_time[i] = max(abs(analyt_func(x, T[i]) - U[i, :]))
 plt.plot(T, error_time, label = "По времени")
 plt.plot(x, error, label = "По пространству")
 plt.legend()
 plt.grid()
 plt.show()
