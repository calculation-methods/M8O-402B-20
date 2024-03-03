import matplotlib.pyplot as plt
import math
import sys
import warnings
import numpy as np



def psi_1(x):
    return math.exp(-x) * math.cos(x)

def psi_2(x):
    return -math.exp(-x) * math.cos(x)

def phi_0(t):
    return math.exp(-t) * math.cos(2 * t)

def phi_l(t):
    return 0

def dpsi2_dx2(x):
    return 2 * math.sin(x) * math.exp(-x)

def u(x, t):
    return math.exp(-x - t) * math.cos(2 * t) * math.cos(x)


class Schema:
    def __init__(self, psi1=psi_1, psi2=psi_2,
                 diffpsi2=dpsi2_dx2, f0=phi_0, fl=phi_l,
                 l0=0, l1=math.pi / 2, T=5, order2nd=True):
        self.psi1 = psi1
        self.diffpsi = diffpsi2
        self.psi2 = psi2
        self.T = T
        self.l0 = l0
        self.l1 = l1
        self.tau = None
        self.h = None
        self.order = order2nd
        self.sigma = None
        self.f0 = f0
        self.fl = fl

    def CalculateH(self, N):
        self.h = (self.l1 - self.l0) / N

    def CalculateTau(self, K):
        self.tau = self.T / K

    def CalculateSigma(self):
        self.sigma = self.tau * self.tau / (self.h * self.h)

    def Set_l0_l1(self, l0, l1):
        self.l0 = l0
        self.l1 = l1

    def SetT(self, T):
        self.T = T

    @staticmethod
    def nparange(start, end, step=1):
        curr = start
        e = 0.00000000001
        while curr - e <= end:
            yield curr
            curr += step

    def CalculateLine(self, t, x, lastLine1, lastLine2):
        pass

    def __call__(self, N=30, K=70):
        N, K = N - 1, K - 1
        self.CalculateTau(K)
        self.CalculateH(N)
        self.CalculateSigma()
        ans = []
        x = list(self.nparange(self.l0, self.l1, self.h))
        lastLine = list(map(self.psi1, x))
        ans.append(list(lastLine))
        if self.order:
            lastLine = list(
                map(lambda a: self.psi1(a) + self.tau * self.psi2(a) + self.tau * self.tau * self.diffpsi(a) / 2, x))
        else:
            lastLine = list(map(lambda a: self.psi1(a) + self.tau * self.psi2(a), x))
        ans.append(list(lastLine))
        X = [x, x]
        Y = [[0.0 for _ in x]]
        Y.append([self.tau for _ in x])
        for t in self.nparange(self.tau + self.tau, self.T, self.tau):
            ans.append(self.CalculateLine(t, x, ans[-1], ans[-2]))
            X.append(x)
            Y.append([t for _ in x])
        return X, Y, ans


class ExplictSchema(Schema):
    def CalculateSigma(self):
        self.sigma = self.tau * self.tau / (self.h * self.h)
        if self.sigma > 1:
            warnings.warn("Sigma > 1")

    def CalculateLine(self, t, x, lastLine1, lastLine2):
        line = [None for _ in lastLine1]
        for i in range(1, len(x) - 1):
            line[i] = self.sigma * (lastLine1[i - 1] - 2 * lastLine1[i] + lastLine1[i + 1])
            line[i] -= 3 * self.tau * self.tau * lastLine1[i]
            line[i] += 2 * lastLine1[i]
            line[i] += (self.tau - 1) * lastLine2[i]
            line[i] += self.tau * self.tau * (lastLine1[i + 1] - lastLine1[i - 1]) / self.h
            line[i] /= (1 + self.tau)
        line[0] = self.f0(t)
        line[-1] = self.fl(t)
        return line


class ImplictSchema(Schema):
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

    def CalculateLine(self, t, x, lastLine1, lastLine2):
        a = 1 - self.h
        c = 1 + self.h
        b = -2 - 3 * self.h * self.h - (1 + self.tau) / self.sigma
        A = [(a, b, c) for _ in range(2, len(x) - 2)]
        w = [((1 - self.tau) * lastLine2[i] - 2 * lastLine1[i]) / self.sigma for i in range(2, len(x) - 2)]
        koeffs = (0, b, c, (((1 - self.tau) * lastLine2[1] - 2 * lastLine1[1]) / self.sigma) - a * self.f0(t))
        A.insert(0, koeffs[:-1])
        w.insert(0, koeffs[-1])
        koeffs = (a, b, 0, (((1 - self.tau) * lastLine2[-2] - 2 * lastLine1[-2]) / self.sigma) - c * self.fl(t))
        A.append(koeffs[:-1])
        w.append(koeffs[-1])
        ans = self.SweepMethod(A, w)
        ans.insert(0, self.f0(t))
        ans.append(self.fl(t))
        return ans


def Error(x, y, z, f):
    ans = 0.0
    for i in range(len(z)):
        for j in range(len(z[i])):
            tmp = abs(z[i][j] - f(x[i][j], y[i][j]))
            ans = tmp if tmp > ans else ans
    return ans

def GetStepHandEror(solver, real_f):
    h = []
    e = []
    for N in range(5, 30, 2):
        x, y, z = solver(N, 100)
        h.append(solver.h)
        e.append(Error(x, y, z, real_f))
    return h, e

explict = ExplictSchema(T=1)
h, e = GetStepHandEror(explict, u)
plt.subplot(1,3,1)
plt.plot(h,e)
plt.xlabel("Step h")
plt.ylabel("e")

implict = ImplictSchema(T=1)
h, e = GetStepHandEror(implict, u)
plt.plot(h,e)


def GetTandError(methodToSolve, realF):
    tau, e = [], []
    for K in range(3, 90):
        x, y, z = methodToSolve(K=K)
        tau.append(methodToSolve.tau)
        e.append(Error(x, y, z, realF))
    return tau, e

explict = ExplictSchema(T=1)
tau, e = GetTandError(explict, u)
plt.subplot(1,3,2)
plt.plot(tau,e)
plt.xlabel("Step tau")
plt.ylabel("e")
implict = ImplictSchema(T=1)
tau, e = GetTandError(implict, u)
plt.subplot(1,3,3)
plt.plot(tau,e)
plt.xlabel("Step tau")
plt.ylabel("e")
fig = plt.gcf()
fig.canvas.manager.set_window_title('Зависимость ошибки от размера шага')
plt.show()