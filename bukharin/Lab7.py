import matplotlib.pyplot as plt
import math


def psi_0(x):
    return math.cos(x)

def psi_1(x):
    return 0.0

def phi_0(y):
    return math.cos(y)

def phi_1(y):
    return 0.0

def u(x, y):
    return math.cos(y) * math.cos(x)


class Schema:
    def __init__(self, psi0=psi_0, psi1=psi_1, phi0=phi_0, phi1=phi_1,
                 lx0=0, lx1=math.pi / 2, ly0=0, ly1=math.pi / 2,
                 solver="zeidel", relax=0.1, epsilon=0.01):
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

    def ZeidelStep(self, X, Y, M):
        return self.RelaxationStep(X, Y, M, w=1)

    def Set_l0_l1(self, lx0, lx1, ly0, ly1):
        self.lx0 = lx0
        self.lx1 = lx1
        self.ly0 = ly0
        self.ly1 = ly1

    def CalculateH(self):
        self.hx = (self.lx1 - self.lx0) / (self.Nx - 1)
        self.hy = (self.ly1 - self.ly0) / (self.Ny - 1)

    @staticmethod
    def nparange(start, end, step=1):
        now = start
        e = 0.00000000001
        while now - e <= end:
            yield now
            now += step

    def InitializeValues(self, X, Y):
        ans = [[0.0 for _ in range(self.Nx)] for _ in range(self.Ny)]
        for j in range(1, self.Nx - 1):
            coeff = (self.psi1(X[-1][j]) - self.psi0(X[0][j])) / (self.ly1 - self.ly0)
            addition = self.psi0(X[0][j])
            for i in range(self.Ny):
                ans[i][j] = coeff * (Y[i][j] - self.ly0) + addition
        for i in range(self.Ny):
            ans[i][0] = self.phi0(Y[i][0])
            ans[i][-1] = self.phi1(Y[i][-1])
        return ans

    def __call__(self, Nx=10, Ny=10):
        self.Nx, self.Ny = Nx, Ny
        self.CalculateH()
        x = list(self.nparange(self.lx0, self.lx1, self.hx))
        y = list(self.nparange(self.ly0, self.ly1, self.hy))
        X = [x for _ in range(self.Ny)]
        Y = [[y[i] for _ in x] for i in range(self.Ny)]
        ans = self.InitializeValues(X, Y)
        self.itters = 0
        while (self.method(X, Y, ans) >= self.eps):
            self.itters += 1
        return X, Y, ans

def Error(x, y, z, f):
    ans = 0.0
    for i in range(len(z)):
        for j in range(len(z[i])):
            ans += (z[i][j] - f(x[i][j], y[i][j]))**2
    return (ans / (len(z) * len(z[0])))**0.5

def GetStepHandError(solver, real_f):
    h , e = [], []
    for N in range(5, 50, 4):
        x, y, z = solver(N, N)
        h.append(solver.hx)
        e.append(Error(x, y, z, real_f))
    return h, e


schema = Schema(epsilon=0.001, solver="simple")
schema(10, 10)
print("Количество итераций метода простых итераций = {}".format(schema.itters))

schema = Schema(epsilon=0.001)
schema(10, 10)
print("Количество итераций метода Зейделя = {}".format(schema.itters))

schema = Schema(epsilon=0.001, solver="relaxation", relax=1.5)
schema(10, 10)
print("Количество итераций метода релаксаций = {}".format(schema.itters))

explict1 = Schema(epsilon=0.00001, solver="simple", relax=1.55)
h1, e1 = GetStepHandError(explict1, u)
explict2 = Schema(epsilon=0.000001, solver="simple", relax=1.55)
h2, e2 = GetStepHandError(explict2, u)
explict3 = Schema(epsilon=0.0000001, solver="simple", relax=1.55)
h3, e3 = GetStepHandError(explict3, u)
explict4 = Schema(epsilon=0.00000001, solver="simple", relax=1.55)
h4, e4 = GetStepHandError(explict4, u)
explict5 = Schema(epsilon=0.000000001, solver="simple", relax=1.55)
h5, e5 = GetStepHandError(explict5, u)

plt.plot(h1,e1)
plt.plot(h2,e2)
plt.plot(h3,e3)
plt.plot(h4,e4)
plt.plot(h5,e5)
plt.xlabel("Step h")
plt.ylabel("e")
fig = plt.gcf()
fig.canvas.manager.set_window_title('Зависимость ошибки от размера шага')
plt.show()


