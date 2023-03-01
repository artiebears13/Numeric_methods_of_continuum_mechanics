#include "thomas.h"


std::vector<double> ThreeDiagMatrix::thomas_solver(std::vector<double> &b) {
    if (d[0] == 0) {
        std::cerr << "first diagonal element shouldn't be zero" << std::endl;
        exit(-1);
    }
    int N = b.size();
    double w;
    std::vector<double> q(N, 0.0);
    std::vector<double> g(N, 0.0);
    std::vector<double> u(N, 0.0);
    auto start = std::chrono::high_resolution_clock::now();
//      forward
    q[0] = du[0] / d[0];
    g[0] = b[0] / d[0];

    for (int i = 1; i < N; ++i) {
        w = d[i] - dl[i] * q[i - 1];

        if (w == 0) {
            std::cerr << "condition w[i]==0 not met" << std::endl;
            exit(-1);
        }

        if (i != N - 1) {
            q[i] = du[i] / w;
        }
        g[i] = (b[i] - dl[i] * g[i - 1]) / w;
    }
//        backward

    u[N - 1] = g[N - 1];
    for (int i = N - 2; i >= 0; i -= 1) {
        u[i] = g[i] - q[i] * u[i + 1];
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "time of execution of thomas method: " <<
              std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " microseconds"
              << std::endl;
    std::vector<double> tmp = mul_mat_vec(u);
    std::cout << "L2 norm: " << norm_l2(tmp, b) << std::endl;

    return u;

}

std::vector<double> ThreeDiagMatrix::mul_mat_vec(std::vector<double> b) {

    if (d.size() != b.size()) {
        std::cerr << "can't multiply: wrong sizes" << std::endl;
        exit(-1);
    }
    std::vector<double> res(d.size(), 0.0);
    for (int i = 0; i < d.size(); ++i) {
        if (i > 0) {
            res[i] += dl[i] * b[i - 1];
        }
        res[i] += d[i] * b[i];
        if (i < d.size() - 1) {
            res[i] += du[i] * b[i + 1];
        }
    }
    return res;
}

double ThreeDiagMatrix::norm_l2(std::vector<double> res, std::vector<double> b) {
    double norm = 0;

    for (int i = 0; i < b.size(); ++i) {
        norm += ((res[i] - b[i]) * (res[i] - b[i]));
    }
    return sqrt(norm);
}

void ThreeDiagMatrix::print() {
    std::cout << "upper diag: ";
    for (int i = 0; i < d.size(); ++i) {
        std::cout << du[i] << " ";
    }
    std::cout << std::endl;

    std::cout << " main diag: ";
    for (int i = 0; i < d.size(); ++i) {
        std::cout << d[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "lower diag: ";
    for (int i = 0; i < d.size(); ++i) {
        std::cout << dl[i] << " ";
    }
    std::cout << std::endl;
}

