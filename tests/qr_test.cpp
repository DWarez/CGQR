#include <iostream>
#include <armadillo>
#include "../include/qr_factorization.hpp"
#include "../include/utils.hpp"

bool hh_random_test() {
    arma::arma_rng::set_seed_random();
    arma::mat X(5, 5, arma::fill::randn);
    arma::vec v = X.col(0);

    std::cout << X << std::endl;
    std::cout << v << std::endl;
    std::cout << "Computing householder vector from v" << std::endl;

    arma::vec householder;
    int s;
    std::tie(householder, s) = compute_householder(v);

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

bool matrix_expansion_random_test() {
    arma::arma_rng::set_seed_random();
    arma::mat X(5, 5, arma::fill::randn);

    std::cout << "X: \n" << X << std::endl;
    auto new_X = expand_matrix(X, 5);
    std::cout << "Expanded X: \n " << new_X << std::endl;
    return true;
}

bool thin_qr_random_test() {
    arma::arma_rng::set_seed_random();
    arma::mat X(5, 4, arma::fill::randn);
    std::cout << "X: \n" << X << std::endl;

    arma::mat Q(X.n_rows, X.n_cols);
    arma::mat R(X.n_rows, X.n_cols);
    std::tie(Q, R) = thin_qr(X);

    std::cout << "Q: \n" << Q << std::endl;
    std::cout << "R: \n" << R << std::endl;
    return true;
}


int main() {
    // hh_random_test();
    // hh_set_random_test();
    matrix_expansion_random_test();
    // thin_qr_random_test();
    return 0;
}

