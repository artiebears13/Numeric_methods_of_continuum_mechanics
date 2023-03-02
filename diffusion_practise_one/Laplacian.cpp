//
// Created by artem on 3/1/23.
//
#include "../progonka/thomas.h"

/*
 EQUATION:
-d2u/dx2 = f(x)
u(0) = g(0)
u(1) = g(1)
g(x) = sin(pi*x)
------------------------
scheme:
    -(u[i+1] - 2*u[i] + u[i+1])/(h*h) = f(x[i])
    u[0] = g(x[0])
    u[N] = g(x[N])

matrix:
           |h**2    0                                 |   g(x[0])
           | -1     2     -1                          |    f(x[1])
           |       -1       2       -1                |       .
1/(h**2) * |        . . .. .........                  | =     .
           |                                          |       .
           |                          -1    2     -1  |     f(x[N-1])
           |                                0    h**2 |     g(x[N])

 */

double f(double x) {
    return M_PI * M_PI * sin(M_PI * x);
}

double g(double x) {
    return sin(M_PI * x);
}

std::vector<double> solve(int N) {

    double h = 1. / (N - 1);
    ThreeDiagMatrix A(N);
    std::vector<double> b(N, 0.0);
    A.set_diag_element(0, 1);
    A.set_diag_element(N - 1, 1);

    b[0] = g(0.);
    b[N - 1] = g(1.);

    for (int i = 1; i < N - 1; ++i) {
        A.set_diag_element(i, 2. / (h * h));
        A.set_upper_diag_element(i, -1. / (h * h));
        A.set_lower_diag_element(i, -1. / (h * h));
        b[i] = f(i * h);

    }
    return A.thomas_solver(b);

}

void check_for_N(int N) {
    double h = 1./(N-1);
    std::cout<<"h = "<<h<<std::endl;
    std::vector<double> numeric_solution;
    numeric_solution = solve(N);
    double norm = 0;
    for (int i = 0; i < N; ++i) {
        norm += (numeric_solution[i] - g(i * (1. / (N - 1)))) * (numeric_solution[i] - g(i * (1. / (N - 1))));

    }
//    norm = sqrt(norm)*sqrt(h);
    std::cout<<"Equation solver:"<<std::endl;
    std::cout << "N = " << N << " L2 norm: " << sqrt(norm)*sqrt(h) << std::endl;
}