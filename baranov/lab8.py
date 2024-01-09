import random
import matplotlib.pyplot as plt
import math
import numpy as np
def psi_0(x, t, a=1):
 return math.cos(x)*math.exp(-2*a*t)
def psi_1(x, t, a=1):
 return 0
def phi_0(y, t, a=1):
 return math.cos(y)*math.exp(-2*a*t)
def phi_1(y, t, a=1):
 return 0
def u0(x, y):
 return math.cos(x)*math.cos(y)
def u(x, y, t, a=1):
 return math.cos(x)*math.cos(y)*math.exp(-2*a*t)
class Schema:
 def __init__(self, rho=u0, psi0=psi_0, psi1=psi_1, phi0=phi_0, phi1=phi_1, lx0=0, lx1=math.pi/2, 
ly0=0, ly1=math.pi/2, T=3, order2nd=True):
 self.psi0 = psi0
 self.psi1 = psi1
 self.phi0 = phi0
 self.phi1 = phi1
 self.rho0 = rho
 self.T = T
 self.lx0 = lx0
 self.lx1 = lx1
 self.ly0 = ly0
 self.ly1 = ly1
 self.tau = None
 self.hx = None
 self.hy = None
 self.order = order2nd
 self.Nx = None
 self.Ny = None
 self.K = None
 self.cx = None
 self.bx = None
 self.cy = None
 self.by = None
 self.hx2 = None
 self.hy2 = None
 def set_l0_l1(self, lx0, lx1, ly0, ly1):
 self.lx0 = lx0
 self.lx1 = lx1
 self.ly0 = ly0
 self.ly1 = ly1
 def set_T(self, T):
 self.T = T
 def compute_h(self):
 self.hx = (self.lx1 - self.lx0) / (self.Nx - 1)
 self.hy = (self.ly1 - self.ly0) / (self.Ny - 1)
 self.hx2 = self.hx * self.hx
 self.hy2 = self.hy * self.hy
 def compute_tau(self):
 self.tau = self.T / (self.K - 1)
 @staticmethod
 def progonka(A, b):
 P = [-item[2] for item in A]
 Q = [item for item in b]
 P[0] /= A[0][1]
 Q[0] /= A[0][1]
 for i in range(1, len(b)):
 z = (A[i][1] + A[i][0] * P[i - 1])
 P[i] /= z
 Q[i] -= A[i][0] * Q[i - 1]
 Q[i] /= z
 for i in range(len(Q) - 2, -1, -1):
 Q[i] += P[i] * Q[i + 1]
 return Q
 @staticmethod
 def nparange(start, end, step=1):
 now = start
 e = 0.00000000001
 while now - e <= end:
 yield now
 now += step
 def compute_left_edge(self, X, Y, t, square):
 for i in range(self.Ny):
 square[i][0] = self.phi0(Y[i][0], t)
 def compute_right_edge(self, X, Y, t, square):
 for i in range(self.Ny):
 square[i][-1] = self.phi1(Y[i][-1], t)
 def compute_bottom_edge(self, X, Y, t, square):
 for j in range(1, self.Nx - 1):
 square[0][j] = self.psi0(X[0][j], t)
 def compute_top_edge(self, X, Y, t, square):
 for j in range(1, self.Nx - 1):
 square[-1][j] = self.psi1(X[-1][j], t)
 def compute_line_first_step(self, i, X, Y, t, last_square, now_square): #
 hy2 = self.hy2
 hx2 = self.hx2
 b = self.bx
 c = self.cx
 A = [(0, b, c)]
 w = [
 -self.cy * self.order * last_square[i - 1][1] -
 ((self.order + 1) * hx2 * hy2 - 2 * self.cy * self.order) * last_square[i][1] -
 self.cy * self.order * last_square[i + 1][1] -
 c * now_square[i][0]
 ]
 A.extend([(c, b, c) for _ in range(2, self.Nx - 2)])
 w.extend([
 -self.cy * self.order * last_square[i - 1][j] -
 ((self.order + 1) * hx2 * hy2 - 2 * self.cy * self.order) * last_square[i][j] -
 self.cy * self.order * last_square[i + 1][j]
 for j in range(2, self.Nx - 2)
 ])
 A.append((c, b, 0))
 w.append(
 -self.cy * self.order * last_square[i - 1][-2] -
 ((self.order + 1) * hx2 * hy2 - 2 * self.cy * self.order) * last_square[i][-2] -
 self.cy * self.order * last_square[i + 1][-2] -
 c * now_square[i][-1]
 )
 line = self.progonka(A, w)
 for j in range(1, self.Nx - 1):
 now_square[i][j] = line[j - 1]
 def compute_line_second_step(self, j, X, Y, t, last_square, now_square):
 hx2 = self.hx2
 hy2 = self.hy2
 c = self.cy
 b = self.by
 A = [(0, b, c)]
 w = [
 -self.cx * self.order * last_square[1][j - 1] -
 ((self.order + 1) * hx2 * hy2 - 2 * self.cx * self.order) * last_square[1][j] -
 self.cx * self.order * last_square[1][j + 1] -
 c * now_square[0][j]
 ]
 A.extend([(c, b, c) for _ in range(2, self.Ny - 2)])
 w.extend([
 -self.cx * self.order * last_square[i][j - 1] -
 ((self.order + 1) * hx2 * hy2 - 2 * self.cx * self.order) * last_square[i][j] -
 self.cx * self.order * last_square[i][j + 1]
 for i in range(2, self.Ny - 2)
 ])
 A.append((c, b, 0))
 w.append(
 -self.cx * self.order * last_square[-2][j - 1] -
 ((self.order + 1) * hx2 * hy2 - 2 * self.cx * self.order) * last_square[-2][j] -
 self.cx * self.order * last_square[-2][j + 1] -
 c * now_square[-1][j]
 )
 line = self.progonka(A, w)
 for i in range(1, self.Ny - 1):
 now_square[i][j] = line[i - 1]
 def compute_square(self, X, Y, t, last_square):
 square = [[0.0 for _ in range(self.Nx)] for _ in range(self.Ny)]
 self.compute_left_edge(X, Y, t - 0.5 * self.tau, square)
 self.compute_right_edge(X, Y, t - 0.5 * self.tau, square)
 self.compute_bottom_edge(X, Y, t - 0.5 * self.tau, square)
 self.compute_top_edge(X, Y, t - 0.5 * self.tau, square)
 for i in range(1, self.Ny - 1):
 self.compute_line_first_step(i, X, Y, t - 0.5 * self.tau, last_square, square)
 last_square = square
 square = [[0.0 for _ in range(self.Nx)] for _ in range(self.Ny)]
 self.compute_left_edge(X, Y, t, square)
 self.compute_right_edge(X, Y, t, square)
 self.compute_bottom_edge(X, Y, t, square)
 self.compute_top_edge(X, Y, t, square)
 for j in range(1, self.Nx - 1):
 self.compute_line_second_step(j, X, Y, t, last_square, square)
 return square
 def init_t0(self, X, Y):
 first = [[0.0 for _ in range(self.Nx)] for _ in range(self.Ny)]
 for i in range(self.Ny):
 for j in range(self.Nx):
 first[i][j] = self.rho0(X[i][j], Y[i][j])
 return first
 def __call__(self, Nx=20, Ny=20, K=20):
 self.Nx, self.Ny, self.K = Nx, Ny, K
 self.compute_tau()
 self.compute_h()
 self.bx = -2 * self.tau * self.hy2
 self.bx -= (1 + self.order) * self.hx2 * self.hy2
 self.cx = self.tau * self.hy2
 self.cy = self.tau * self.hx2
 self.by = -2 * self.tau * self.hx2
 self.by -= (1 + self.order) * self.hx2 * self.hy2
 x = list(self.nparange(self.lx0, self.lx1, self.hx))
 y = list(self.nparange(self.ly0, self.ly1, self.hy))
 X = [x for _ in range(self.Ny)]
 Y = [[y[i] for _ in x] for i in range(self.Ny)]
 taus = [0.0]
 ans = [self.init_t0(X, Y)]
 for t in self.nparange(self.tau, self.T, self.tau):
 ans.append(self.compute_square(X, Y, t, ans[-1]))
 taus.append(t)
 return X, Y, taus, ans
def real_z_by_time(lx0, lx1, ly0, ly1, t, f):
 x = np.arange(lx0, lx1 + 0.002, 0.002)
 y = np.arange(ly0, ly1 + 0.002, 0.002)
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
 Z[i, j] = f(X[i, j], Y[i, j], t)
 return X, Y, Z
def error(X, Y, t, z, ut = u):
 ans = 0.0
 for i in range(len(z)):
 for j in range(len(z[i])):
 ans = max(abs(ut(X[i][j], Y[i][j], t) - z[i][j]), ans)
 return (ans / (len(z) * len(z[0])))
def plot_by_time(X, Y, T, Z, j, extrems):
 t = T[j]
 z = Z[j]
 fig = plt.figure(num=1, figsize=(10, 7), clear=True)
 ax = fig.add_subplot(1, 1, 1, projection='3d')
 ax.plot_surface(np.array(X), np.array(Y), np.array(z))
 ax.plot_wireframe(*real_z_by_time(0, math.pi/2, 0, math.pi/2, t, u), color="green")
 ax.set_xlabel('x')
 ax.set_ylabel('y')
 ax.set_zlabel('z')
 ax.set_title(
 't = ' + str(round(t, 8)) + " error = " + str(round(error(X, Y, t, z), 11)),
 loc="center", fontsize=12)
 ax.set_zlim(extrems[0], extrems[1])
 fig.tight_layout()
 plt.show()
 return fig
def square_minmax(z):
 minimum, maximum = z[0][0], z[0][0]
 for i in range(len(z)):
 for j in range(len(z[i])):
 minimum = z[i][j] if z[i][j] < minimum else minimum
 maximum = z[i][j] if z[i][j] > maximum else maximum
 return minimum, maximum
def search_minmax(zz):
 minimum, maximum = 0.0, 0.0
 for z in zz:
 minmax = square_minmax(z)
 minimum = minmax[0] if minmax[0] < minimum else minimum
 maximum = minmax[1] if minmax[1] > maximum else maximum
 return minimum, maximum
def plot(nx, ny, k, t, order):
 schema = Schema(T=t, order2nd=order)
 xx, yy, tt, zz = schema(Nx=nx, Ny=ny, K=k)
 extrems = search_minmax(zz)
 plots = []
 for j in range(len(tt)):
 plots.append(plot_by_time(xx, yy, tt, zz, j, extrems))
def get_graphic_h(solver, time = 0, tsteps = 40):
 h, e = [], []
 for N in range(10, 100, 10):
 x, y, t, z = solver(Nx = N, Ny = N, K = tsteps)
 h.append(solver.hx)
 e.append(error(x, y, t[time], z[time]))
 return h, e
def get_graphic_tau(solver):
 tau = []
 e = []
 for K in range(4, 101, 2):
 x, y, t, z = solver(Nx = 10, Ny = 10, K = K)
 tau.append(solver.tau)
 time = K // 2
 e.append(error(x, y, t[time], z[time]))
 return tau, e
def plot(nx, ny, k, t, order):
 schema = Schema(T=t, order2nd=order)
 xx, yy, tt, zz = schema(Nx=nx, Ny=ny, K=k)
 extrems = search_minmax(zz)
 plots = []
 for j in range(len(tt)):
 plots.append(plot_by_time(xx, yy, tt, zz, j, extrems))
def get_graphic_h(solver, time = 0, tsteps = 40):
 h, e = [], []
 for N in range(10, 100, 10):
 x, y, t, z = solver(Nx = N, Ny = N, K = tsteps)
 h.append(solver.hx)
 e.append(error(x, y, t[time], z[time]))
 return h, e
def get_graphic_tau(solver):
 tau = []
 e = []
 for K in range(4, 101, 2):
 x, y, t, z = solver(Nx = 10, Ny = 10, K = K)
 tau.append(solver.tau)
 time = K // 2
 e.append(error(x, y, t[time], z[time]))
 return tau, e
def error_tau():
 plt.figure(figsize = (10, 7))
 plt.title("Зависимость погрешности от длины шага по времени")
 first = Schema(T=1, order2nd=False) # метод дробных шагов
 second = Schema(T=1, order2nd=True) # метод переменных направлений
 tau1, e1 = get_graphic_tau(first)
 tau2, e2 = get_graphic_tau(second)
 plt.plot(tau1, e1, label="Метод дробных шагов")
 plt.plot(tau2, e2, label="Метод переменных направлений")
 plt.xlabel("$tau$")
 plt.ylabel("$\epsilon$")
 plt.legend()
 plt.grid()
 plt.show()
def error_h():
 TSTEPS = 100
 time = 50
 plt.figure(figsize = (10, 7))
 plt.title("Зависимость погрешности от длины шага при t = " + str(time / TSTEPS))
 first = Schema(T=1, order2nd=False) # метод дробных шагов
 second = Schema(T=1, order2nd=True) # метод переменных направлений
 h1, e1 = get_graphic_h(first, time, TSTEPS)
 h2, e2 = get_graphic_h(second, time, TSTEPS)
 plt.plot(h1, e1, label="Метод дробных шагов")
 plt.plot(h2, e2, label="Метод переменных направлений")
 plt.xlabel("$h_x$")
 plt.ylabel("$\epsilon$")
 plt.legend()
 plt.grid()
 plt.show()
plot(nx=40, ny=40, k=3, t=1, order=False)
error_h()
error_tau()
