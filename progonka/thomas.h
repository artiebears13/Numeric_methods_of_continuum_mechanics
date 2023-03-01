#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>

class ThreeDiagMatrix {
private:
    std::vector<double> d = {};
    std::vector<double> du = {};
    std::vector<double> dl = {};
public:
    ThreeDiagMatrix(std::vector<double> const &diag,
                    std::vector<double> const &upper_diag,
                    std::vector<double> const &lower_diag) {
        d = diag;
        du = upper_diag;
        dl = lower_diag;
    };
    ThreeDiagMatrix(int N) {
        for (int i = 0; i < N; ++i) {
            d.push_back(0.0);
            du.push_back(0.0);
            dl.push_back(0.0);
        }
    };

    std::vector<double> thomas_solver(std::vector<double> &b);

    std::vector<double> mul_mat_vec(std::vector<double> b);

    double norm_l2(std::vector<double> res, std::vector<double> b);

    void print();
};