import matplotlib.pyplot as plt
import numpy as np
def psi_0(y):
 return 0
def psi_1(y):
 return -(1 - y ** 2)
def phi_0(x):
 return 0
def phi_1(x):
 return -(x ** 2 - 1)
def u(x, y):
 return x ** 2 - y ** 2
class Schema:
 def __init__(self, psi0=psi_0, psi1=psi_1, phi0=phi_0, phi1=phi_1, lx0=0,
 lx1=1, ly0=0, ly1=1, solver="zeidel", relax=1.5, epsilon=0.01):
 self.psi1 = psi1
 self.psi0 = psi0
 self.phi0 = phi0
 self.phi1 = phi1
 self.lx0 = lx0
 self.ly0 = ly0
 self.lx1 = lx1
 self.ly1 = ly1
 self.eps = epsilon
 self.method = None
 if solver == "zeidel":
 self.method = self.ZeidelStep
 elif solver == "simple":
 self.method = self.SimpleEulerStep
 elif solver == "relaxation":
 self.method = lambda x, y, m: self.RelaxationStep(x, y, m, relax)
 else:
 raise ValueError("Wrong solver name")
 def ZeidelStep(self, X, Y, M):
 return self.RelaxationStep(X, Y, M, w=1)
 def RelaxationStep(self, X, Y, M, w):
 norm = 0.0
 hx2 = self.hx * self.hx
 hy2 = self.hy * self.hy
 for i in range(1, self.Ny - 1):
 for j in range(1, self.Nx - 1):
 diff = hy2 * (M[i][j - 1] + M[i][j + 1])
 diff += hx2 * (M[i - 1][j] + M[i + 1][j])
 diff /= 2 * (hy2 + hx2 - hx2 * hy2)
 diff -= M[i][j]
 diff *= w
 M[i][j] += diff
 diff = abs(diff)
 norm = diff if diff > norm else norm
 return norm
 def SimpleEulerStep(self, X, Y, M):
 tmp = [[0.0 for _ in range(self.Nx)] for _ in range(self.Ny)]
 norm = 0.0
 hx2 = self.hx * self.hx
 hy2 = self.hy * self.hy
 for i in range(1, self.Ny - 1):
 tmp[i][0] = M[i][0]
 for j in range(1, self.Nx - 1):
 tmp[i][j] = hy2 * (M[i][j - 1] + M[i][j + 1])
 tmp[i][j] += hx2 * (M[i - 1][j] + M[i + 1][j])
 tmp[i][j] /= 2 * (hy2 + hx2 - hx2 * hy2)
 diff = abs(tmp[i][j] - M[i][j])
 norm = diff if diff > norm else norm
 tmp[i][-1] = M[i][-1]
 for i in range(1, self.Ny - 1):
 M[i] = tmp[i]
 return norm
 def set_l0_l1(self, lx0, lx1, ly0, ly1):
 self.lx0 = lx0
 self.lx1 = lx1
 self.ly0 = ly0
 self.ly1 = ly1
 def _compute_h(self):
 self.hx = (self.lx1 - self.lx0) / (self.Nx - 1)
 self.hy = (self.ly1 - self.ly0) / (self.Ny - 1)
 @staticmethod
 def nparange(start, end, step=1):
 now = start
 e = 0.00000000001
 while now - e <= end:
 yield now
 now += step
 def init_values(self, X, Y):
 ans = [[0.0 for _ in range(self.Nx)] for _ in range(self.Ny)]
 for j in range(1, self.Nx - 1):
 coeff = (self.psi1(X[-1][j]) - self.psi0(Y[0][j])) / (self.ly1 - self.ly0)
 addition = self.psi0(X[0][j])
 for i in range(self.Ny):
 ans[i][j] = coeff * (Y[i][j] - self.ly0) + addition
 for i in range(self.Ny):
 ans[i][0] = self.phi0(Y[i][0])
 ans[i][-1] = self.phi1(Y[i][-1])
 return ans
 def __call__(self, Nx=10, Ny=10):
 self.Nx, self.Ny = Nx, Ny
 self._compute_h()
 x = list(self.nparange(self.lx0, self.lx1, self.hx))
 y = list(self.nparange(self.ly0, self.ly1, self.hy))
 X = [x for _ in range(self.Ny)]
 Y = [[y[i] for _ in x] for i in range(self.Ny)]
 ans = self.init_values(X, Y)
 self.itters = 0
 while (self.method(X, Y, ans) >= self.eps):
 self.itters += 1
 ans=np.array(ans)
 return X, Y, ans
def real_z(lx0, lx1, ly0, ly1, f):
 x = np.arange(lx0, lx1 + 0.001, 0.001)
 y = np.arange(ly0, ly1 + 0.001, 0.001)
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
def plot(Nx=5, Ny=5, eps=1, solv="simple"):
 schema = Schema(epsilon=eps, solver=solv, relax=1.6)
 x, y, z = schema(Nx, Ny)
 fig = plt.figure(num=1, figsize=(10, 7), clear=True)
 ax = fig.add_subplot(1, 1, 1, projection='3d')
 #ax.plot_wireframe(*real_z(0, 1, 0, 1, u), color="green")
 ax.plot_surface(np.array(x), np.array(y), np.array(z))
 ax.set(xlabel='x', ylabel='y', zlabel='u', title=f"График приближения функции конечно разностным методом")
 fig.tight_layout()
 plt.show()
def error(x, y, z, f):
 ans = 0.0
 for i in range(len(z)):
 for j in range(len(z[i])):
 ans += (f(x[i][j], y[i][j]) - z[i][j])**2
 return ((ans)**0.5)/(len(z)*len(z[0]))
def get_step_error(solver, real_f): # вычисление погрешности
 h = []
 e = []
 N=10
 x, y, z = solver(N, N)
 h.append(solver.hx)
 e.append(error(x, y, z, real_f))
 N=20
 x, y, z = solver(N, N)
 h.append(solver.hx)
 e.append(error(x, y, z, real_f))
 N=40
 x, y, z = solver(N, N)
 h.append(solver.hx)
 e.append(error(x, y, z, real_f))
 return h, e
def error_step(eps, r): # построение графика погрешности
 simp = Schema(epsilon=eps,solver="simple")
 zeid = Schema(epsilon=eps,solver="zeidel")
 relax = Schema(epsilon=eps,solver="relaxation", relax=r)
 plt.figure(figsize = (10, 7))
 plt.title(f"Зависимость погрешности от длины шага eps={eps}, w={r}, (N=10,20,40)")
 h_s, e_s = get_step_error(simp, u)
 h_z, e_z = get_step_error(zeid, u)
 h_r, e_r = get_step_error(relax, u)
 plt.plot(h_s, e_s, label="Метод Простых Итераций", color = "red")
 plt.plot(h_z, e_z, label="Метод Зейделя", color="green")
 plt.plot(h_r, e_r, label="Метод Простых Итераций С Верхней Релаксацией", color="blue")
 plt.xlabel("h")
 plt.ylabel("error")
 plt.legend()
 plt.grid()
 plt.show()
eps_0, eps_1, eps_2, eps_3, eps_4, eps_5, eps_6, eps_7 = 1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 
0.000001, 0.0000001
Nx = 10
Ny = 10
r = 1.6
eps = eps_2
plot(Nx, Ny, eps, solv="simple")
plot(Nx, Ny, eps, solv="zeidel")
plot(Nx, Ny, eps, solv="relaxation")
error_step(eps, r=r)
schema = Schema(epsilon=eps, solver="simple")
schema(Nx, Ny)
print("Количество итераций метода простых итераций:", schema.itters)
schema = Schema(epsilon=eps)
schema(Nx, Ny)
print("Количество итераций метода Зейделя:", schema.itters)
schema = Schema(epsilon=eps, solver="relaxation", relax=r)
schema(Nx, Ny)
print("Количество итераций метода верхних релаксаций:", schema.itters)
