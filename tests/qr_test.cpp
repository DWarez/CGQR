#include <iostream>
#include <armadillo>
#include "../include/qr_factorization.hpp"

bool hh_random_test() {
    arma::arma_rng::set_seed_random();
    arma::mat X(5, 5, arma::fill::randn);
    arma::vec v = X.col(0);

    std::pair<arma::vec, double> householder;

    std::cout << X << std::endl;
    std::cout << v << std::endl;
    std::cout << "Computing householder vector from v" << std::endl;

    householder = compute_householder(v);

    std::cout << "HH Vector: " << householder.first << "\nNorm: " << householder.second << std::endl;

    arma::mat H = arma::eye(householder.first.n_elem, householder.first.n_elem) - 2 * householder.first * householder.first.t();

    std::cout << "Modified matrix X: " << std::endl;
    std::cout << H * X << std::endl;

    return true;
}

int main() {
    hh_random_test();
    return 0;
}

