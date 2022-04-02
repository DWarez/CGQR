#include <iostream>
#include <armadillo>
#include "../include/conjugate_gradient.hpp"
#include "../include/utils.hpp"

bool cg_random_test() {
    arma::arma_rng::set_seed_random();
    arma::mat X(5, 5, arma::fill::randn);
    arma::vec b(5, arma::fill::randn);
    arma::vec w = arma::zeros(5);

    to_normal_equations(X, b);

    std::cout << "Norm before: " << arma::norm(X*w - b)/arma::norm(b) << std::endl;
    w = conjugate_gradient(X, b, 5);

    std::cout << "Norm after: " << arma::norm(X*w - b)/arma::norm(b) << std::endl;
    std::cout << "Result\n w = " << w << std::endl;

    return true;
}

int main() {
    cg_random_test();
    return 0;
}