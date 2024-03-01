import numpy as np
from matplotlib import pyplot as plt


# Разреженная матрица
class SparseMatrix:
    def __init__(self, object={}, shape=None, dtype=np.float32):
        self.__data = object
        if len(object) > 0 and shape is None:
            self.__shape = (max(object) + 1, max(max(object.values(), key=lambda x: max(x))) + 1)
        elif shape is not None:
            self.__shape = shape
        else:
            self.__shape = (0, 0)

    # Доступ к элементу
    def __getitem__(self, key):
        if self.shape[0] == 1:  # вектор
            key = (0, key)
        if self.shape[1] == 1:  # вектор
            key = (key, 0)
        if isinstance(key, int):  # (int)
            if key in self.__data:
                return SparseMatrix({0: self.__data[key]}, shape=(1, self.shape[1]))
            else:
                return SparseMatrix(dict(), shape=(1, self.shape[1]))
        elif isinstance(key[0], int) and isinstance(key[1], int):  # (int, int)
            if key[0] in self.__data and key[1] in self.__data[key[0]]:
                return self.__data[key[0]][key[1]]
            elif key[0] < self.__shape[0] and key[1] < self.__shape[1]:
                return 0
            else:
                raise IndexError
        elif isinstance(key[0], int) and key[1].start is None and key[1].stop is None and key[
            1].step is None:  # (int, :)
            if key[0] in self.__data:
                return SparseMatrix({0: self.__data[key[0]]}, shape=(1, self.shape[1]))
            else:
                return SparseMatrix(dict(), shape=(1, self.shape[1]))
        elif key[0].start is None and key[0].stop is None and key[0].step is None and isinstance(key[1],
                                                                                                 int):  # (:, int)
            d = dict()
            for i in self.__data:  # цикл по строкам
                if key[1] in self.__data[i]:
                    d[i] = dict()
                    d[i][0] = self.__data[i][key[1]]
            return SparseMatrix(d, shape=(self.shape[0], 1))

    # Назначение по ключу
    def __setitem__(self, key, value):
        if value == 0:
            if key[0] in self.__data and key[1] in self.__data[key[0]]:
                del self.__data[key[0]][key[1]]
                if len(self.__data[key[0]]) == 0:
                    del self.__data[key[0]]
        else:
            if not key[0] in self.__data:
                self.__data[key[0]] = dict()
            self.__data[key[0]][key[1]] = value

    def __iter__(self):  # итератор на начало
        self.__ij = [0, 0]
        return self

    def __str__(self):
        return str(self.to_dense()) + "\nIn memory: " + str(self.__data)

    def __next__(self):
        m, n = self.shape
        i, j = self.__ij  # текущее значение
        if i < m and j < n:
            self.__ij[1] += 1
            return i, j
        elif i == m - 1 and j == n:
            raise StopIteration
        else:
            self.__ij = [i + 1, 0]
            return i + 1, 0

    @property
    def shape(self):
        return self.__shape

    # Вернуть массив значений
    @property
    def values(self):
        res = []
        for i in sorted(self.__data):
            for j in sorted(self.__data[i]):
                res += [self.__data[i][j]]
        return res

    # Вернуть индексы ненулевых значений
    @property
    def indices(self):
        x = []
        y = []
        for i in sorted(self.__data):
            for j in sorted(self.__data[i]):
                x += [j]
                y += [i]
        return x, y

    def transpose(A):
        d = {}
        for i in sorted(A.__data):
            for j in sorted(A.__data[i]):
                if not j in d:
                    d[j] = {}
                d[j][i] = A.__data[i][j]
        d_sort = {}
        return SparseMatrix(d_sort, shape=(A.shape[1], A.shape[0]))
    
    # Преобразовать в плотную матрицу
    def to_dense(self):
        res = np.zeros(self.shape)
        for i in sorted(self.__data):  # цикл по строкам
            for j in sorted(self.__data[i]):  # цикл по столбцам
                res[i, j] = self.__data[i][j]
        return res

    # Преобразовать в разреженную матрицу
    def to_sparse(A):
        d = {}
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if A[i, j] != 0:
                    if not i in d:
                        d[i] = {}
                    d[i][j] = A[i, j]
        return SparseMatrix(d, shape=A.shape)

    # Умножение матриц
    def dot(A, B, sparse=True):
        if isinstance(A, SparseMatrix) and isinstance(B, SparseMatrix):  # разреженная на разреженную
            res = dict()
            b = dict()  # оставляем только столбцы
            for i in range(B.shape[1]):
                tmp = B[:, i]
                if len(tmp.__data) != 0:  # отбрасываем нулевые столбцы
                    b[i] = tmp
            for i in sorted(A.__data):  # по ненулевым строкам 1-й матрицы
                for j in b:  # по ненулевым столбцам 2-й матрицы
                    inter = set(A.__data[i]) & set(b[j].__data)  # пересечение
                    if len(inter) > 0:  # имеют общие элементы
                        s = 0
                        for k in inter:  # вычисление суммы
                            s += A.__data[i][k] * b[j][k]
                        if s != 0:
                            if not i in res:
                                res[i] = dict()
                            res[i][j] = s
            return SparseMatrix(res, shape=(A.shape[0], B.shape[1])) if sparse else SparseMatrix(res, shape=(
            A.shape[0], B.shape[1])).to_dense()
        elif isinstance(A, SparseMatrix) and isinstance(B, np.ndarray):  # разреженная на плотную
            res = dict()
            for i in sorted(A.__data):  # по ненулевым строкам 1-й матрицы
                for j in range(B.shape[1]):  # по столбцам 2-й матрицы
                    s = 0
                    for k in sorted(A.__data[i]):  # по ненулевым столбцам 1-й матрицы
                        s += A.__data[i][k] * B[k][j]
                    if s != 0:
                        if not i in res:
                            res[i] = dict()
                        res[i][j] = s
            return SparseMatrix(res, shape=(A.shape[0], B.shape[1])) if sparse else SparseMatrix(res, shape=(
            A.shape[0], B.shape[1])).to_dense()


# Равномерно-распределённая матрица
def uniform(shape=(10, 10), alpha=None, a=0, b=1):
    if alpha is None:
        alpha = 1 / shape[0]

    data = {}
    l = np.random.randint(0, shape[0], np.ceil(shape[0] * shape[1] * alpha).astype(int))
    m = np.random.randint(0, shape[1], np.ceil(shape[0] * shape[1] * alpha).astype(int))
    for i, j in zip(l, m):
        if not i in data:
            data[int(i)] = {}
        data[int(i)][int(j)] = np.random.uniform(a, b)
    return SparseMatrix(data, shape=shape)


# Генерирование положительно-определеннной и симметричной матрицы
def SPD(n=10, alpha=None, a=1, b=None):
    if alpha is None:
        alpha = 1 / n
    if b is None:
        b = n

    l = np.random.randint(0, n, np.ceil(n * n * alpha).astype(int))
    m = np.random.randint(0, n, np.ceil(n * n * alpha).astype(int))

    A = np.zeros(shape=(n, n))

    for i, j in zip(l, m):
        A[i, j] = np.random.uniform(0, 1)

    A = 0.5 * (A + np.transpose(A))
    A = A + np.diag(np.random.uniform(a, b, size=n))

    return SparseMatrix.to_sparse(A)


# Метод сопряжённых градиентов
def conjugate_gradient_method(A, b, x0 = None, norm = lambda x: np.linalg.norm(x, ord = np.inf), eps = 1e-5):
    b = np.reshape(b, (-1, 1))
    x = b if x0 is None else np.reshape(x0, (-1, 1))
    r = b - SparseMatrix.dot(A, x, sparse = False)
    z = r
    b_norm = norm(b)
    r_dot_new = np.dot(np.reshape(r, -1), np.reshape(r, -1))
    i = 0
    while norm(r) / b_norm >= eps:
        r_dot_old = r_dot_new
        alpha = r_dot_old / np.dot(np.reshape(SparseMatrix.dot(A, z, sparse = False), -1), np.reshape(z, -1))
        x = x + alpha * z
        r = r - alpha * SparseMatrix.dot(A, z, sparse = False)
        r_dot_new = np.dot(np.reshape(r, -1), np.reshape(r, -1))
        beta = r_dot_new / r_dot_old
        z = r + beta * z
        i += 1
    print("Число итераций =", i)
    return np.reshape(x, -1)


# Размер матрицы
print("Метод сопряженных градиентов для разряжнных матриц большой размерности")
n = 100
A = SPD(n)
print("n =", n)
x_true = np.random.uniform(- n / 2, n / 2, size = (A.shape[1], 1)).astype(np.float32)
b = np.reshape(SparseMatrix.dot(A, x_true, sparse = False), -1)
x, y = A.indices
d = A.values
print("Число ненулевых элементов =", len(A.values))

plt.scatter(x, -np.array(y), s=1)
plt.xlabel("Столбец")
plt.ylabel('Строка')
plt.show()

eps = 1e-8
print("eps =", eps)
x = conjugate_gradient_method(A, b, eps = eps)
print("Проверим решение:")
A_dense = A.to_dense()
print("норма вектора Ax - b =", np.linalg.norm(np.dot(A_dense, x) - b, ord = np.inf))
print("Полученное значение\t Точное значение\n", np.concatenate((np.reshape(x, (-1, 1)), np.reshape(x_true, (-1, 1))), axis = 1))