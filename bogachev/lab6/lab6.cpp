#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

double phi0(double t) {
    return sin(t * 2.);
}

double phi1(double t) {
    return -sin(t * 2.);
}

double psi0(double x) {
    return 0.;
}

double psi1(double x) {
    return cos(x) * 2.;
}

double solution(double x, double t) {
    return cos(x) * sin(t * 2);
}

std::vector<std::vector<double>> analytical_solution(double x_begin, double x_end, double t_begin,
    double t_end, double h, double sigma) {
    double tau = sigma * h * h;
    int length_x = static_cast<int>((x_end - x_begin) / h) + 1;
    int length_t = static_cast<int>((t_end - t_begin) / tau) + 1;
    std::vector<double> x(length_x), t(length_t);
    x[0] = x_begin;
    for (int i = 1; i < length_x; ++i) {
        x[i] = x[i - 1] + h;
    }
    t[0] = t_begin;
    for (int i = 1; i < length_x; ++i) {
        t[i] = t[i - 1] + tau;
    }
    std::vector<std::vector<double>> result(length_t, std::vector<double>(length_x));
    for (int i = 0; i < length_x; ++i) {
        for (int j = 0; j < length_t; ++j) {
            result[j][i] = solution(x[i], t[j]);
        }
    }
    return result;
}

std::vector<std::vector<double>> explicit_finite_difference_method(double x_begin, double x_end,
    double t_begin, double t_end, double h, double sigma) {
    double tau = sigma * h * h;
    int length_x = static_cast<int>((x_end - x_begin) / h) + 1;
    int length_t = static_cast<int>((t_end - t_begin) / tau) + 1;
    std::vector<double> x(length_x), t(length_t);
    x[0] = x_begin;
    for (int i = 1; i < length_x; ++i) {
        x[i] = x[i - 1] + h;
    }
    t[0] = t_begin;
    for (int i = 1; i < length_x; ++i) {
        t[i] = t[i - 1] + tau;
    }
    std::vector<std::vector<double>> result(length_t, std::vector<double>(length_x));
    for (int i = 0; i < length_x; ++i) {
        result[0][i] = psi0(x[i]);
        result[1][i] = psi0(x[i]) + tau * psi1(x[i]);
    }
    for (int i = 2; i < length_t; ++i) {
        result[i][0] = phi0(t[i]);
        for (int j = 1; j < length_x - 1; ++j) {
            result[i][j] = sigma * (result[i - 1][j + 1] - result[i - 1][j] * 2. + result[i - 1][j - 1]) +
                (2. - tau * tau * 3.) * result[i - 1][j] - result[i - 2][j];
        }
        result[i][length_x - 1] = phi1(t[i]);
    }
    return result;
}

std::vector<double> tridiagonal_solve(std::vector<std::vector<double>>& A, std::vector<double>& b) {
    int n = A.size();
    std::vector<double> v(n), u(n), x(n);
    v[0] = A[0][1] / -A[0][0];
    u[0] = b[0] / A[0][0];
    for (int i = 1; i < n - 1; ++i) {
        v[i] = A[i][i + 1] / (-A[i][i] - A[i][i - 1] * v[i - 1]);
        u[i] = (A[i][i - 1] * u[i - 1] - b[i]) / (-A[i][i] - A[i][i - 1] * v[i - 1]);
    }
    u[n - 1] = (A[n - 1][n - 2] * u[n - 2] - b[n - 1]) / (-A[n - 1][n - 1] - A[n - 1][n - 2] * v[n - 2]);
    x[n - 1] = u[n - 1];
    for (int i = n - 1; i > 0; --i) {
        x[i - 1] = v[i - 1] * x[i] + u[i - 1];
    }
    return x;
}

std::vector<std::vector<double>> implicit_finite_difference_method(double x_begin, double x_end,
    double t_begin, double t_end, double h, double sigma) {
    double tau = sigma * h * h;
    int length_x = static_cast<int>((x_end - x_begin) / h) + 1;
    int length_t = static_cast<int>((t_end - t_begin) / tau) + 1;
    std::vector<double> x(length_x), t(length_t);
    x[0] = x_begin;
    for (int i = 1; i < length_x; ++i) {
        x[i] = x[i - 1] + h;
    }
    t[0] = t_begin;
    for (int i = 1; i < length_x; ++i) {
        t[i] = t[i - 1] + tau;
    }
    std::vector<std::vector<double>> result(length_t, std::vector<double>(length_x));
    for (int i = 0; i < length_x; ++i) {
        result[0][i] = psi0(x[i]);
        result[1][i] = psi0(x[i]) + tau * psi1(x[i]);
    }
    for (int i = 2; i < length_t; ++i) {
        std::vector<std::vector<double>> A(length_x - 2, std::vector<double>(length_x - 2));
        A[0][0] = -(tau * tau * 3 + sigma * 2. + 1.);
        A[0][1] = sigma;
        for (int j = 1; j < A.size() - 1; ++j) {
            A[j][j - 1] = sigma;
            A[j][j] = -(tau * tau * 3 + sigma * 2. + 1.);
            A[j][j + 1] = sigma;
        }
        A[length_x - 3][length_x - 4] = sigma;
        A[length_x - 3][length_x - 3] = -(tau * tau * 3 + sigma * 2. + 1.);
        std::vector<double> b(result[i - 1].size() - 2);
        for (int j = 0; j < b.size(); ++j) {
            b[j] = result[i - 2][j + 1] - result[i - 1][j + 1] * 2.;
        }
        b[0] -= sigma * phi0(t[i]);
        b[b.size() - 1] -= sigma * phi1(t[i]);
        result[i][0] = phi0(t[i]);
        std::vector<double> interior = tridiagonal_solve(A, b);
        std::move(interior.begin(), interior.end(), result[i].begin() + 1);
        result[i][result[i].size() - 1] = phi1(t[i]);
    }
    return result;
}

double max_abs_error(std::vector<std::vector<double>>& A, std::vector<std::vector<double>>& B) {
    int n = A.size(), m = A[0].size();
    double max = 0.;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            max = std::max(max, std::abs(A[i][j] - B[i][j]));
        }
    }
    return max;
}

double mean_abs_error(std::vector<std::vector<double>>& A, std::vector<std::vector<double>>& B) {
    int n = A.size(), m = A[0].size();
    double mean = 0., prod = static_cast<double>(n * m);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            mean += std::abs(A[i][j] - B[i][j]) / prod;
        }
    }
    return mean;
}

void output_to_file(std::string filepath, const std::vector<std::vector<double>>& arr) {
    std::ofstream fout(filepath);
    int n = arr.size(), m = arr[0].size();
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            fout << arr[i][j] << " ";
        }
        fout << "\n";
    }
    fout.close();
}

int main()
{
    double x_begin = 0., x_end = acos(-1), t_begin = 0., t_end = 5., h = 0.01, sigma = 1;
    std::vector<std::vector<double>> as = analytical_solution(x_begin, x_end, t_begin, t_end, h, sigma);
    std::vector<std::vector<double>> efdm = explicit_finite_difference_method(x_begin, x_end, t_begin, t_end, h, sigma);
    std::vector<std::vector<double>> ifdm = implicit_finite_difference_method(x_begin, x_end, t_begin, t_end, h, sigma);
    std::cout << "Max abs error between analytical solution and explicit finite difference method solution: " << max_abs_error(as, efdm) << "\n";
    std::cout << "Mean abs error between analytical solution and explicit finite difference method solution: " << mean_abs_error(as, efdm) << "\n";
    std::cout << "Max abs error between analytical solution and implicit finite difference method solution: " << max_abs_error(as, ifdm) << "\n";
    std::cout << "Mean abs error between analytical solution and implicit finite difference method solution: " << mean_abs_error(as, ifdm) << "\n";
    output_to_file("analytical_solution.txt", as);
    output_to_file("explicit_finite_difference_method.txt", efdm);
    output_to_file("implicit_finite_difference_method.txt", ifdm);
    return 0;
}