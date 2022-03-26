//
// Created by dwarez on 24/03/22.
//

#include <iostream>
#include <armadillo>
#include "../include/conjugate_gradient.hpp"

int main(int argc, char** argv) {
    arma::arma_rng::set_seed_random();
    arma::mat X(5, 5, arma::fill::randn);
    arma::vec b(5, arma::fill::randn);
    arma::vec w = arma::zeros(5);

    X = X.t() * X;

    std::cout << "Norm before: " << arma::norm(X*w - b, 2) << std::endl;
    conjugate_gradient(X, b, w, 5);

    std::cout << "Norm after: " << arma::norm(X*w - b, 2) << std::endl;
    std::cout << "Result\n w = " << w << std::endl;

    return 0;
}