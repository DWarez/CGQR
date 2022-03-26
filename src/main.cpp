#include <iostream>
#include <armadillo>
#include "../include/conjugate_gradient.hpp"
#include "../include/qr_factorization.hpp"

int main(int argc, char** argv) {
    arma::arma_rng::set_seed_random();
    arma::vec v(5, arma::fill::randn);
    std::pair<arma::vec, double> householder;

    std::cout << v << std::endl;
    std::cout << "Computing householder vector from v" << std::endl;

    householder = compute_householder(v);

    std::cout << "HH Vector: " << householder.first << "\nNorm: " << householder.second << std::endl;

    return 0;
}