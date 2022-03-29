#include <iostream>
#include <armadillo>
#include "../include/conjugate_gradient.hpp"

bool cg_random_test() {
    arma::arma_rng::set_seed_random();
    arma::mat X(5, 5, arma::fill::randn);
    arma::vec b(5, arma::fill::randn);
    arma::vec w = arma::zeros(5);

    X = X.t() * X;

    std::cout << "Norm before: " << arma::norm(X*w - b, 2) << std::endl;
    w = conjugate_gradient(X, b, 5);

    std::cout << "Norm after: " << arma::norm(X*w - b, 2) << std::endl;
    std::cout << "Result\n w = " << w << std::endl;

    return true;
}

int main() {
    cg_random_test();
    return 0;
}