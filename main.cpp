#include "progonka/thomas.h"


int main() {
    int N = 100;
    std::vector<double> du(N, -1.);
    std::vector<double> d(N, 2.);
    std::vector<double> dl(N, -1.);
    std::vector<double> b(N, 1.);

    ThreeDiagMatrix A(d, du, dl);
//    A.print();
    A.thomas_solver(b);


}