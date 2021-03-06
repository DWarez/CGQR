#include <iostream>
#include <armadillo>
#include <sys/stat.h>
#include <chrono>
#include "../include/utils.hpp"
#include "../include/conjugate_gradient.hpp"
#include "../include/qr_factorization.hpp"

void cg_experiment() {
    std::cout << "\n\nStart of the CG experiment\n==========================" << std::endl;

    mkdir("../results", S_IRWXU);

    arma::mat X, n_X;
    arma::vec b, n_b;

    std::tie(X, b) = grab_mlcup_dataset();
    add_columns(X);
    std::tie(n_X, n_b) = to_normal_equations(X, b);

    // solution vector
    arma::vec solution = arma::solve(X, b);
    arma::vec w = arma::vec(X.n_cols, arma::fill::zeros);

    w = conjugate_gradient(n_X, n_b, 30);

    std::cout << "Norm of CG: " << arma::norm(X*w - b)/arma::norm(b) << std::endl;
    std::cout << "Distance from optimal solution: " << arma::norm(w - solution) << std::endl;
    std::cout << "Gradient: " << arma::norm(X.t() * X * w - X.t() * b) << "\n";
    std::cout << "==========================\nEnd of the CG experiment" << std::endl;
}

void qr_experiment() {
    std::cout << "\n\nStart of the QR experiment\n==========================" << std::endl;
    arma::mat X, Q, R;
    arma::vec b;

    std::tie(X, b) = grab_mlcup_dataset();
    add_columns(X);
    arma::vec solution = arma::solve(X, b);

    arma::vec w(X.n_cols, arma::fill::zeros);

    auto start = std::chrono::steady_clock::now();
    std::tie(Q, R) = thin_qr(X, b);

    w = solve_thin_qr(Q, R);

    auto end = std::chrono::steady_clock::now();

    std::cout << "Execution time: " <<
                    std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()<< " us \n";
    std::cout << "Norm of QR: " << arma::norm(X*w - b)/arma::norm(b)  << "\n";
    std::cout << "Distance from optimal solution: " << arma::norm(w - solution) << "\n";
    std::cout << "Gradient: " << arma::norm(X.t() * X * w - X.t() * b) << "\n";
    std::cout << "==========================\nEnd of the QR experiment" << std::endl;
}

int main(int argc, char** argv) {
    problem_properties();
    cg_experiment();
    qr_experiment();
    return 0;
}