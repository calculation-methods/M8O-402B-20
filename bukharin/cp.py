
INF = 1e9


def f(x):
    """
    Подинтегральная функция
    """
    return 1 / (1 + x**2)


def integrate_rectangle_method(f, l, r, h):
    """
    Расчет интеграла f(x)dx на интервале [l; r] используя метод прямоугольников с шагом h
    """
    result = 0
    cur_x = l
    while cur_x < r:
        result += h * f((cur_x + cur_x + h) * 0.5)
        cur_x += h
    return result


def integrate_with_definite_integral(f, l, r, h=0.01, eps=1e-6):
    """
    Расчет несобственного интеграла первого типа методом перехода к определенному интегралу
    """

    def f_new(t):
        return (1. / t ** 2) * f(1. / t)

    result = 0
    if r == INF:
        new_r = max(eps, l)
        result += integrate_rectangle_method(f_new, eps, 1. / new_r - eps, h)
    else:
        new_r = r
    if l == -INF:
        new_l = min(-eps, r)
        result += integrate_rectangle_method(f_new, 1. / new_l + eps, -eps, h)
    else:
        new_l = l
    if new_l < new_r:
        result += integrate_rectangle_method(f, new_l, new_r, h)
    return result


def integrate_lim(f, l, r, h=0.1, eps=1e-6):
    """
    Расчет несобственного интеграла первого типа методом перехода к пределу
    """
    result = 0
    iters = 0
    if r == INF:
        finish = False
        cur_x = max(l, 0)
        while not finish:
            iters += 1
            new_result = result + h * f((cur_x + cur_x + h) * 0.5)
            cur_x += h
            if abs(new_result - result) < eps:
                finish = True
            result = new_result
    else:
        result += integrate_rectangle_method(f, 0, r, h)
    if l == -INF:
        finish = False
        cur_x = min(0, r)
        while not finish:
            iters += 1
            new_result = result + h * f((cur_x - h + cur_x) * 0.5)
            cur_x -= h
            if abs(new_result - result) < eps:
                finish = True
            result = new_result
    else:
        result += integrate_rectangle_method(f, l, 0, h)
    return result, iters


if __name__ == '__main__':
    a = -INF
    b = -0.1
    h = 0.1
    eps = 1e-9
    print('Переход к определенному интегралу')
    res_definite = integrate_with_definite_integral(f, a, b, h, eps)
    print('Интеграл =', res_definite)
    print()

    print('Предельный метод')
    res_limit, iters_limit = integrate_lim(f, a, b, h, eps)
    print('Интеграл =', res_limit)
    print('Итерации:', iters_limit)
    print()
