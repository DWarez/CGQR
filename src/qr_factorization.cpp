#include <armadillo>
#include "../include/qr_factorization.hpp"
#include "../include/utils.hpp"

arma::vec compute_householder(const arma::vec &x) {
    arma::vec householder = arma::vec(x);
    double s = arma::norm(x, 2);

    if(x(0) >= 0)
        s = -s;

    householder(0) = householder(0) - s;
    double norm = arma::norm(householder, 2);
    householder = householder/norm;
    return householder;
}

std::vector <arma::vec> householder_set(const arma::mat &X) {
    std::vector<arma::vec> hh_set;
    arma::vec current_column;

    for(size_t i = 0; i < X.n_cols; ++i) {
        current_column = arma::conv_to<arma::vec>::from(X.col(i));
        hh_set.push_back(compute_householder(trim_head_vector(current_column, i)));
    }

    return hh_set;
}