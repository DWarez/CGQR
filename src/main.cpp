#include <iostream>
#include <armadillo>
#include <sys/stat.h>
#include "../include/utils.hpp"
#include "../include/conjugate_gradient.hpp"
#include "../include/qr_factorization.hpp"

void cg_experiment() {

    mkdir("../results", S_IRWXU);

    arma::mat X, n_X;
    arma::vec b, n_b;

    std::tie(X, b) = grab_mlcup_dataset();

    std::tie(n_X, n_b) = to_normal_equations(X, b);

    // solution vector
    arma::vec w = arma::vec(X.n_cols, arma::fill::zeros);

    w = conjugate_gradient(n_X, n_b, X.n_cols);

    std::cout << "Norm: " << arma::norm(X*w - b)/arma::norm(b) << std::endl;
}

void qr_experiment() {
    arma::mat X, Q, R;
    arma::vec b;

    std::tie(X, b) = grab_mlcup_dataset();
    arma::vec w(X.n_cols, arma::fill::zeros);
    std::tie(Q, R) = thin_qr(X);

    w = solve_thin_qr(Q, R, b);

    std::cout << "Norm: " << arma::norm(X*w - b)/arma::norm(b) << std::endl;
}

int main(int argc, char** argv) {
    cg_experiment();
    qr_experiment();
    return 0;
}