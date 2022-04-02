#include <iostream>
#include <armadillo>
#include <sys/stat.h>
#include "../include/utils.hpp"
#include "../include/conjugate_gradient.hpp"

void cg_experiment() {

    mkdir("../results", S_IRWXU);

    arma::mat X;
    arma::vec b;
    std::vector<double> residuals;

    std::tie(X, b) = grab_mlcup_dataset();

    to_normal_equations(X, b);

    // solution vector
    arma::vec w = arma::vec(X.n_cols, arma::fill::randn);

    w = conjugate_gradient(X, b, 10);
}

int main(int argc, char** argv) {
    cg_experiment();
    return 0;
}