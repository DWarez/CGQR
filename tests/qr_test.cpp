#include <iostream>
#include <armadillo>
#include "../include/qr_factorization.hpp"

bool hh_random_test() {
    arma::arma_rng::set_seed_random();
    arma::mat X(5, 5, arma::fill::randn);
    arma::vec v = X.col(0);

    std::cout << X << std::endl;
    std::cout << v << std::endl;
    std::cout << "Computing householder vector from v" << std::endl;

    arma::vec householder;
    householder = compute_householder(v);

    std::cout << "HH Vector: " << householder << std::endl;
    arma::mat H = arma::eye(householder.n_elem, householder.n_elem) - 2 * householder * householder.t();
    std::cout << "Modified matrix X: " << std::endl;
    std::cout << H * X << std::endl;

    return true;
}

bool hh_set_random_test() {
    arma::arma_rng::set_seed_random();
    arma::mat X(5, 5, arma::fill::randn);

    std::vector<arma::vec> hh_set;
    hh_set = householder_set(X);

    for(const auto& hh_vector:hh_set) {
        std::cout << hh_vector << std::endl;
    }

    return true;
}

int main() {
    hh_random_test();
    hh_set_random_test();
    return 0;
}

