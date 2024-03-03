import matplotlib.pyplot as plt
import math
import numpy as np


def phi_0(t):
    return math.exp(-0.5*t)

def phi_l(t):
    return -math.exp(-0.5*t)

def u_0(x):
    return math.sin(x)

# Вместо cos нужно sin, так как ошибка в условии
def g(x, t):
    return 0.5 * math.exp(-0.5*t) * math.sin(x)

def u(x, t):
    return math.exp(-0.5*t)*math.sin(x)


class Schema:
    def __init__(self, f0=phi_0, fl=phi_l,
                 u0=u_0, g=g,
                 O=0.5, l0=0, l1=math.pi,
                 T=5, aprx_cls=None):
        self.fl = fl
        self.f0 = f0
        self.u0 = u0
        self.g = g
        self.T = T
        self.l0 = l0
        self.l1 = l1
        self.tau = None
        self.h = None
        self.O = O
        self.approx = None
        if aprx_cls is not None:
            self._init_approx(aprx_cls)
        self.sigma = None

    def _init_approx(self, a_cls):
        self.approx = a_cls(self.f0, self.fl)

    def SetApprox(self, aprx_cls):
        self._init_approx(self, aprx_cls)

    def Set_l0_l1(self, l0, l1):
        self.l0 = l0
        self.l1 = l1

    def SetT(self, T):
        self.T = T

    def CalculateH(self, N):
        self.h = (self.l1 - self.l0) / N

    def CalculateTau(self, K):
        self.tau = self.T / K

    def CalculateSigma(self):
        self.sigma = self.tau / (self.h * self.h)

    @staticmethod
    def nparange(start, end, step=1):
        now = start
        e = 0.00000000001
        while now - e <= end:
            yield now
            now += step

    def CalculateLine(self, t, x, lastLine):
        pass

    def __call__(self, N=50, K=70):
        N, K = N - 1, K - 1
        self.CalculateTau(K)
        self.CalculateH(N)
        self.CalculateSigma()
        ans = []
        x = list(np.arange(self.l0, self.l1 + 0.5 * self.h, self.h))
        lastLine = list(map(self.u0, x))
        ans.append(list(lastLine))
        X, Y = [], []
        X.append(x)
        Y.append([0.0 for _ in x])
        for t in np.arange(self.tau, self.T + 0.5 * self.tau, self.tau):
            ans.append(self.CalculateLine(t, x, lastLine))
            X.append(x)
            Y.append([t for _ in x])
            lastLine = ans[-1]
        return X, Y, ans


class ExplictSchema(Schema):
    def CalculateSigma(self):
        self.sigma = self.tau / (self.h * self.h)
        if self.sigma > 0.5:
            print("Sigma > 0.5")

    def CalculateLine(self, t, x, lastLine):
        line = [None for _ in lastLine]
        for i in range(1, len(x) - 1):
            line[i] = self.sigma * lastLine[i - 1]
            line[i] += (1 - 2 * self.sigma) * lastLine[i]
            line[i] += self.sigma * lastLine[i + 1]
            line[i] += self.tau * self.g(x[i], t - self.tau)
        line[0] = self.approx.explict_0(t, self.h, self.sigma,
                                        self.g, self.l0, lastLine,
                                        line, t - self.tau)
        line[-1] = self.approx.explict_l(t, self.h, self.sigma,
                                         self.g, self.l0, lastLine,
                                         line, t - self.tau)
        return line


class ImplictExplict(Schema):
    def SetO(self, O):
        self.O = O

    @staticmethod
    def SweepMethod(A, b):
        P = [-item[2] for item in A]
        Q = [item for item in b]
        P[0] /= A[0][1]
        Q[0] /= A[0][1]
        for i in range(1, len(b)):
            z = (A[i][1] + A[i][0] * P[i - 1])
            P[i] /= z
            Q[i] -= A[i][0] * Q[i - 1]
            Q[i] /= z
        x = [item for item in Q]
        for i in range(len(x) - 2, -1, -1):
            x[i] += P[i] * x[i + 1]
        return x

    def CalculateLine(self, t, x, lastLine):
        a = self.sigma * self.O
        b = -1 - 2 * self.sigma * self.O
        A = [(a, b, a) for _ in range(1, len(x) - 1)]
        w = [-(lastLine[i] + self.O * self.tau * self.g(x[i], t) + (1 - self.O) * self.sigma * (
                    lastLine[i - 1] - 2 * lastLine[i] + lastLine[i + 1] + self.h * self.h * self.g(x[i], t - self.tau)))
             for i in range(1, len(x) - 1)]
        koeffs = self.approx.nikolson_0(t, self.h, self.sigma,
                                        self.g, self.l0, lastLine,
                                        self.O, t - self.tau)
        A.insert(0, koeffs[:-1])
        w.insert(0, koeffs[-1])
        koeffs = self.approx.nikolson_l(t, self.h, self.sigma,
                                        self.g, self.l1, lastLine,
                                        self.O, t - self.tau)
        A.append(koeffs[:-1])
        w.append(koeffs[-1])

        return self.SweepMethod(A, w)


class Approx:
    def __init__(self, f0, fl):
        self.f0 = f0
        self.fl = fl

    def explict_0(self, t, h, sigma, g, x0, l0, l1, t0):
        pass

    def explict_l(self, t, h, sigma, g, xN, l0, l1, t0):
        pass

    def nikolson_0(self, t, h, sigma, g, x0, l0, O, t0):
        pass

    def nikolson_l(self, t, h, sigma, g, xN, l0, O, t0):
        pass


class Approx2pointFirstOrder(Approx):
    def explict_0(self, t, h, sigma, g, x0, l0, l1, t0):
        return -h * self.f0(t) + l1[1]

    def explict_l(self, t, h, sigma, g, xN, l0, l1, t0):
        return h * self.fl(t) + l1[-2]

    def nikolson_0(self, t, h, sigma, g, x0, l0, O, t0):
        return 0, -1, 1, h * self.f0(t)

    def nikolson_l(self, t, h, sigma, g, xN, l0, O, t0):
        return -1, 1, 0, h * self.fl(t)


class Approx3pointSecondOrder(Approx):
    def explict_0(self, t, h, sigma, l0, l1, t0):
        return (-2 * h * self.f0(t) + 4 * l1[1] - l1[2]) / 3

    def explict_l(self, t, h, sigma, l0, l1, t0):
        return (2 * h * self.fl(t) + 4 * l1[-2] - l1[-3]) / 3

    def nikolson_0(self, t, h, sigma, g, x0, l0, O, t0):
        d = 2 * sigma * O * h * self.f0(t)
        d -= l0[1] + O * (t - t0) * g(x0 + h, t)
        d -= (1 - O) * sigma * (l0[0] - 2 * l0[1] + l0[2] + h * h * g(x0 + h, t0))
        return 0, -2 * sigma * O, 2 * sigma * O - 1, d

    def nikolson_l(self, t, h, sigma, g, xN, l0, O, t0):
        d = 2 * sigma * O * h * self.fl(t)
        d += l0[-2] + O * (t - t0) * g(xN - h, t)
        d += (1 - O) * sigma * (l0[-3] - 2 * l0[-2] + l0[-1] + h * h * g(xN - h, t0))
        return 1 - 2 * sigma * O, 2 * sigma * O, 0, d


class Approx2pointSecondOrder(Approx):
    def explict_0(self, t, h, sigma, g, x0, l0, l1, t0):
        return -2 * sigma * h * self.f0(t0) + 2 * sigma * l0[1] + \
               (1 - 2 * sigma) * l0[0] + (t - t0) * g(x0, t0)

    def explict_l(self, t, h, sigma, g, xN, l0, l1, t0):
        return 2 * sigma * h * self.fl(t0) + 2 * sigma * l0[-2] + \
               (1 - 2 * sigma) * l0[-1] + (t - t0) * g(xN, t0)

    def nikolson_0(self, t, h, sigma, g, x0, l0, O, t0):
        d = 2 * sigma * O * h * self.f0(t) - l0[0] - O * (t - t0) * g(x0, t)
        d -= 2 * (1 - O) * sigma * (l0[1] - l0[0] - h * self.f0(t0) + 0.5 * h * h * g(x0, t0))
        return 0, -(2 * sigma * O + 1), 2 * sigma * O, d

    def nikolson_l(self, t, h, sigma, g, xN, l0, O, t0):
        d = -2 * sigma * O * h * self.fl(t) - l0[-1] - O * (t - t0) * g(xN, t)
        d -= 2 * (1 - O) * sigma * (l0[-2] - l0[-1] + h * self.fl(t0) + 0.5 * h * h * g(xN, t0))
        return 2 * sigma * O, -(2 * sigma * O + 1), 0, d

def Error(x, y, z, f):
    ans = 0.0
    for i in range(len(z)):
        for j in range(len(z[i])):
            ans += (z[i][j] - f(x[i][j], y[i][j]))**2
    return ans **0.5

def GetStepHandError(solver, real_f):
    h = []
    e = []
    for N in range(3, 20):
        x, y, z = solver(N, 70)
        h.append(solver.h)
        e.append(Error(x, y, z, real_f))
    return h, e

explict = ExplictSchema(T = 1, aprx_cls=Approx2pointSecondOrder)
h, e = GetStepHandError(explict, u)
plt.subplot(1,3,1)
plt.plot(h,e)
plt.xlabel("Step h")
plt.ylabel("e")
# ImplictExplict с параметром O=1 это неявная схема
implict = ImplictExplict(T=1, aprx_cls=Approx2pointFirstOrder, O=1)
h, e = GetStepHandError(implict, u)
plt.plot(h,e)

# ImplictExplict с параметром O=0.5 это схема Кранка-Николсона
krank = ImplictExplict(T=1, aprx_cls=Approx2pointSecondOrder, O=0.5)
h, e = GetStepHandError(krank, u)
plt.plot(h,e)



def GetTandError(solver, real_f):
    tau, e = [], []
    for K in range(3, 70):
        x, y, z = solver(20, K)
        tau.append(solver.tau)
        e.append(Error(x, y, z, real_f))
    return tau, e

explict = ExplictSchema(T=5, aprx_cls=Approx2pointSecondOrder)
tau, e = GetTandError(explict, u)
plt.subplot(1,3,2)

plt.xlabel("Step tau")
plt.ylabel("e")
print(tau, e)
plt.plot(tau,e)

implict = ImplictExplict(T=5, aprx_cls=Approx2pointSecondOrder, O=1)
tau, e = GetTandError(implict, u)
plt.subplot(1,3,3)
plt.plot(tau,e)


krank = ImplictExplict(T=5, aprx_cls=Approx2pointSecondOrder)
tau, e = GetTandError(krank, u)
plt.plot(tau,e)
plt.xlabel("Step tau")
plt.ylabel("e")
fig = plt.gcf()
fig.canvas.manager.set_window_title('Зависимость ошибки от размера шага')
plt.show()
