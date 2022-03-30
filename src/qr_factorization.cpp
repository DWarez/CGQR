#include <armadillo>
#include "../include/qr_factorization.hpp"
#include "../include/utils.hpp"

std::tuple<arma::vec, int> compute_householder(const arma::vec &x) {
    arma::vec householder = arma::vec(x);
    double s = arma::norm(x, 2);

    if(x(0) >= 0)
        s = -s;

    householder(0) = householder(0) - s;
    double norm = arma::norm(householder, 2);
    householder = householder/norm;
    return std::make_tuple(householder, s);
}

std::vector <arma::vec> householder_set(const arma::mat &X) {
    std::vector<arma::vec> hh_set;
    arma::vec current_column;
    arma::vec current_hh;
    int s;

    for(size_t i = 0; i < X.n_cols; ++i) {
        current_column = arma::conv_to<arma::vec>::from(X.col(i));
        std::tie(current_hh, s) = compute_householder(trim_head_vector(current_column, i));
        hh_set.push_back(current_hh);
    }

    return hh_set;
}


std::pair<arma::mat, arma::mat> thin_qr(const arma::mat &X) {
    std::vector<arma::vec> hhs = householder_set(X);
    arma::mat Q(X.n_rows, X.n_cols);
    arma::mat R(X.n_rows, X.n_cols);
    arma::mat current_hh_mat;
    int m = X.n_rows;
    int n = X.n_cols;

    // first iteration
    current_hh_mat = arma::eye(hhs[hhs.size() - 1].n_elem, hhs[hhs.size() - 1].n_elem) - 2 * hhs[hhs.size() - 1] * hhs[hhs.size() - 1].t();
    arma::mat expanded = expand_matrix(current_hh_mat, m);
    Q = expanded * arma::eye(m, n);
    R = expanded;

    for(int i = hhs.size() - 2; i >= 0; --i) {
        current_hh_mat = arma::eye(hhs[i].n_elem, hhs[i].n_elem) - 2 * hhs[i] * hhs[i].t();
        expanded = expand_matrix(current_hh_mat, m);
        Q = expanded * Q;
        R = R * expanded;
    }
    R = R * X;

    return {Q, R};
}