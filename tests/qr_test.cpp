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
    arma::mat X(15, 4, arma::fill::randn);
    std::cout << "X: \n" << X << std::endl;

    arma::mat Q(X.n_rows, X.n_cols);
    arma::mat R(X.n_rows, X.n_cols);
    std::tie(Q, R) = thin_qr(X);

    std::cout << "Q: \n" << Q << std::endl;
    std::cout << "R: \n" << R << std::endl;
    return true;
}

bool thin_qr_specific_test() {
    arma::mat X(4,4);
    X(0,0) = 7;
    X(0,1) = 9;
    X(0,2) = 1;
    X(0,3) = 6;
    X(1,0) = 5;
    X(1,1) = 8;
    X(1,2) = 9;
    X(1,3) = 4;
    X(2,0) = 9;
    X(2,1) = 4;
    X(2,2) = 10;
    X(2,3) = 5;
    X(3,0) = 4;
    X(3,1) = 6;
    X(3,2) = 8;

    std::pair result = thin_qr(X);
    arma::mat Q = std::get<0>(result);
    arma::mat R = std::get<1>(result);

    std::cout << "X: \n" << X << std::endl;
    std::cout << "Q1 \n" << Q << std::endl;
    std::cout << "R \n" << R << std::endl;

    return true;
}

bool solve_thin_qr_random_test() {
    arma::arma_rng::set_seed_random();
    arma::mat X(15, 4, arma::fill::randn);

    arma::mat Q(X.n_rows, X.n_cols);
    arma::mat R(X.n_rows, X.n_cols);
    std::tie(Q, R) = thin_qr(X);

    arma::vec b(X.n_rows, arma::fill::randn);
    arma::vec w(X.n_cols, arma::fill::randn);

    std::cout << "Norm before: " << arma::norm(X*w - b)/arma::norm(b) << std::endl;

    w = solve_thin_qr(Q, R, b);

    std::cout << "Norm after: " << arma::norm(X*w - b)/arma::norm(b) << std::endl;

    return true;
}

int main() {
    // hh_random_test();
    // hh_set_random_test();
    //matrix_expansion_random_test();
    // thin_qr_random_test();
    // thin_qr_specific_test();
    solve_thin_qr_random_test();
    return 0;
}

