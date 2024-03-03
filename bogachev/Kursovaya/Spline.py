import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as spi


def print_entered_value():
    global value
    global value1
    value = entry.get().split()
    value = np.array(value, dtype=float)
    value1 = entry1.get().split()
    value1 = np.array(value1, dtype=float)

    window.quit()


def ff(x):
    return 16 * x * x * x/ (x * x + 1) ** 3 - 12 * x / (x ** 2 + 1) ** 2


def F(A, B, C, x):
    return C + B * np.exp(A * x)


def check(x, y):
    swapped = True
    while swapped:
        swapped = False
        for i in range(len(x) - 1):
            if x[i] > x[i + 1]:
                x[i], x[i + 1] = x[i + 1], x[i]
                y[i], y[i + 1] = y[i + 1], y[i]
                swapped = True
    for i in range((len(x) - 1)):
        if (y[i + 1] < y[i]):
            return 1, x, y

    return 0, x, y


def exp_interpolation(x, y):
    A = np.zeros(len(x) - 2)
    B = np.zeros(len(x) - 2)
    C = np.zeros(len(x) - 2)
    res = np.zeros(len(x) - 2)

    d = 0.00001
    for i in range(len(A)):
        dif = (y[i + 2] - y[i + 1]) * (x[i + 1] - x[i]) / ((y[i + 1] - y[i]) * (x[i + 2] - x[i + 1]))

        if check(x, y)[0] == 1:
            A[i], B[i], C[i], res = 0, 0, 0, 0
            break
        else:
            x, y = check(x, y)[1:]

        if abs(dif - 1) < d:
            A[i] = (y[i + 2] - y[i]) / (x[i + 2] - x[i])
            B[i] = y[i] - A[i] * x[i]
            print('При данном наборе точек применима только линейная интерполяция')
            C[i] = 0
            res[i] = 1

        if x[i + 2] + x[i] == x[i + 1] + x[i + 1]:
            z = (2 * y[i + 1] - y[i] - y[i + 2])
            r = (y[i + 2] - y[i + 1]) / (y[i + 1] - y[i])
            A[i] = np.log(r) / (x[i + 2] - x[i + 1])
            C[i] = (y[i + 1] * y[i + 1] - y[i] * y[i + 2]) / z
            B[i] = (y[i] - C[i]) * np.exp(-A[i] * x[i])
            res[i] = 2

        else:

            Amin = np.log(dif) / (x[i + 2] - x[i])
            A0 = 2 * Amin
            while (1):
                u = np.exp(A0 * (x[i + 2] - x[i + 1]))
                v = np.exp(-A0 * (x[i + 1] - x[i]))
                F = (y[i + 1] - y[i]) * (u - 1) + (y[i + 2] - y[i + 1]) * (v - 1)
                FF = (y[i + 1] - y[i]) * (x[i + 2] - x[i + 1]) * u - (y[i + 2] - y[i + 1]) * (x[i + 1] - x[i]) * v
                dA = -F / FF
                A0 += dA
                if (abs(dA / Amin) < 0.000000001):
                    break

            A[i] = A0
            B[i] = (y[i] - y[i + 1]) / (np.exp(A0 * x[i]) - np.exp(A0 * x[i + 1]))
            C[i] = y[i] - B[i] * np.exp(A0 * x[i])
            res[i] = 2

    return A, B, C, res


def plot_graphics(A, B, C, x, y):
    max = 0
    n = len(A)

    x1 = np.arange(x[0], x[1] + 0.01, 0.01)
    y1 = C[0] + B[0] * np.exp(A[0] * x1)
    for i in range (len(x1)):
        if abs(y1[i]-ff(x1)[i])> max:
            max = abs(y1[i]-ff(x1)[i])
    yy = ff(x1)
    plt.title("Интерполяция экспонентой с использованием G(x)")
    plt.plot(x1, y1, color="red")
    plt.plot(x1, yy, color="black")
    plt.scatter(x, y, c='b')

    if (n > 1):
        for i in range(1, n):
            x1 = np.arange(x[i], x[i + 1] + 0.01, 0.01)
            y1 = ((x[i + 1] - x1) * (C[i - 1] + B[i - 1] * np.exp(A[i - 1] * x1)) + (x1 - x[i]) * (
                    C[i] + B[i] * np.exp(A[i] * x1))) / (x[i + 1] - x[i])
            yy = ff(x1)
            for i in range(len(x1)):
                if abs(y1[i] - ff(x1)[i]) > max:
                    max = abs(y1[i] - ff(x1)[i])
            plt.plot(x1, y1, color="red")
            plt.plot(x1, yy, color="black")
    x1 = np.arange(x[-2], x[-1] + 0.01, 0.01)
    y1 = C[n - 1] + B[n - 1] * np.exp(A[n - 1] * x1)
    yy = ff(x1)
    plt.plot(x1, y1, color="red",label="Интерполяционная функция")
    plt.plot(x1, yy, color="black", label="Искомая функция  ")
    plt.legend()
    plt.grid()
    plt.show()
    plt.title("Интерполяция экспонентой с использованием H(x)")
    x1 = np.arange(x[0], x[1] + 0.01, 0.01)
    y1 = C[0] + B[0] * np.exp(A[0] * x1)
    yy = ff(x1)
    for i in range (len(x1)):
        if abs(y1[i]-ff(x1)[i])> max:
            max = abs(y1[i]-ff(x1)[i])

    plt.plot(x1, y1, color="green")
    plt.plot(x1, yy, color="black")
    plt.scatter(x, y, c='b')

    if (n > 1):
        for i in range(1, n):
            x1 = np.arange(x[i], x[i + 1] + 0.01, 0.01)
            y2 = (C[i] + B[i] * np.exp(A[i] * x1) + C[i - 1] + B[i - 1] * np.exp(A[i - 1] * x1)) / 2
            yy = ff(x1)
            plt.plot(x1, y2, color="green")
            plt.plot(x1, yy, color="black")

    x1 = np.arange(x[-2], x[-1] + 0.01, 0.01)
    y1 = C[n - 1] + B[n - 1] * np.exp(A[n - 1] * x1)
    yy = ff(x1)
    plt.plot(x1, y1, color="green", label="Интерполяционная функция")
    plt.plot(x1, yy, color="black", label="Искомая функция  ")
    # ynew = [lagranz(x, y, i) for i in xx]
    # plt.plot(xx, ynew)
    plt.grid()
    plt.show()
    return max


def lagranz(x, y, t):
    z = 0
    for j in range(len(y)):
        p1 = 1;
        p2 = 1
        for i in range(len(x)):
            if i == j:
                p1 = p1 * 1;
                p2 = p2 * 1
            else:
                p1 = p1 * (t - x[i])
                p2 = p2 * (x[j] - x[i])
        z = z + y[j] * p1 / p2
    return z

def _poly_newton_coefficient(x, y):
    """
    x: list or np array contanining x data points
    y: list or np array contanining y data points
    """

    m = len(x)

    x = np.copy(x)
    a = np.copy(y)
    for k in range(1, m):
        a[k:m] = (a[k:m] - a[k - 1])/(x[k:m] - x[k - 1])

    return a

def newton_polynomial(x_data, y_data, x):
    """
    x_data: data points at x
    y_data: data points at y
    x: evaluation point(s)
    """
    a = _poly_newton_coefficient(x_data, y_data)
    n = len(x_data) - 1  # Degree of polynomial
    p = a[n]

    for k in range(1, n + 1):
        p = a[n - k] + (x - x_data[n - k])*p

    return p




window = tk.Tk()
window.geometry('500x300+200+100')
window.title("Ввод данных")
label = tk.Label(window, text="Введите координаты точек по х и у без запятых")
label.pack()

entry = tk.Entry(window)
entry.pack()
entry1 = tk.Entry(window)
entry1.pack()

button = tk.Button(window, text="Построить график", command=print_entered_value)
button.pack()

window.mainloop()

x = value
y = value1
#x = [0.45,  0.52,  0.54,  0.6,  0.72,  0.79,  0.9,  1.06,  1.1,  1.23,  1.3,  1.4,  1.5,  1.6,  1.7]
#y = [-2.896,  -2.769,  -2.715,  -2.519,  -2.042,  -1.753,  -1.33,  -0.831,  -0.73,  -0.461,  -0.35,  -0.225,  -0.131,  -0.062,  -0.012]
res = check(x, y)[0]
x, y = check(x, y)[1:]
res
if res == 1:

    window1 = tk.Tk()
    window1.geometry('300x200+200+100')
    window1.title("Ошибка")
    label1 = tk.Label(window1, text="Нарушен порядок возрастания")
    label1.pack(expand=True, ipadx=10, ipady=10)
    window1.mainloop()
else:
    A, B, C, res = exp_interpolation(x, y)
    max = plot_graphics(A, B, C, x, y)
    max_l = 0
    max_n = 0
    max_lin = 0
    max_q = 0
    max_c = 0
    xx = np.arange(x[0],x[-1]+0.01,0.01)
    ynew = [lagranz(x, y, i) for i in xx]
    y_n = newton_polynomial(x,y,xx)
    yy = ff(xx)
    f1 = spi.interp1d(x, y, kind='linear',bounds_error = False,fill_value= y[-1])
    f2 = spi.interp1d(x, y, kind='quadratic', bounds_error=False, fill_value=y[-1])
    f3 = spi.interp1d(x, y, kind='cubic', bounds_error=False, fill_value=y[-1])
    for i in range(len(xx)):
        if abs(ynew[i] - yy[i]) > max_l:
            max_l = abs(ynew[i] - yy[i])
        if abs(y_n[i] - yy[i]) > max_n:
            max_n = abs(y_n[i] - yy[i])
        if abs(f1(xx)[i] - yy[i]) > max_lin:
            max_lin = abs(f1(xx)[i] - yy[i])
        if abs(f2(xx)[i] - yy[i]) > max_q:
            max_q = abs(f2(xx)[i] - yy[i])
        if abs(f3(xx)[i] - yy[i]) > max_c:
            max_c = abs(f3(xx)[i] - yy[i])
    print('Ошибки интерполяции:')
    print('Экспоненциальная функция:', max)
    print('Полином Лагранжа: ',max_l)
    print('Полином Ньютона: ',max_n)
    print('Линейная интерполяция: ',max_lin)
    print('Квадратичная интерполяция: ', max_q)
    print('Кубическая интерполяция: ', max_c)
    plt.title("Интерполяционный многочлен Лагранжа")
    plt.plot(xx, ynew,label="многочлен Лагранжа  ")
    plt.plot(xx, yy, color="black", label="Искомая функция  ")
    plt.scatter(x, y, c='b')
    plt.legend()
    plt.grid()
    plt.show()

    plt.title("Интерполяционный многочлен Ньютона")
    plt.plot(xx, y_n,label="многочлен Ньютона  ")
    plt.plot(xx, yy, color="black", label="Искомая функция  ")
    plt.scatter(x, y, c='b')
    plt.legend()
    plt.grid()
    plt.show()
    plt.title("Линейная интерполяция")
    plt.plot(xx, f1(xx),label ='Линейная интерполяция')
    plt.plot(xx, yy, color="black", label="Искомая функция  ")
    plt.scatter(x, y, c='b')
    plt.legend()
    plt.grid()
    plt.show()
    plt.title("Квадратичная интерполяция")
    plt.plot(xx, f2(xx),label ='Квадратичная интерполяция')
    plt.plot(xx, yy, color="black", label="Искомая функция  ")
    plt.scatter(x, y, c='b')
    plt.legend()
    plt.grid()
    plt.show()
    plt.title("Кубическая интерполяция")
    plt.plot(xx, f3(xx),label ='Кубическая интерполяция')
    plt.plot(xx, yy, color="black", label="Искомая функция  ")
    plt.scatter(x, y, c='b')
    plt.legend()
    plt.grid()
    plt.show()