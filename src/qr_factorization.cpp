#include <armadillo>
#include <ranges>
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


std::pair<arma::mat, arma::mat> thin_qr(const arma::mat &X, const arma::vec &b) {
    int s = 0;
    uint m = X.n_rows;
    uint n = X.n_cols;

    std::vector<arma::mat> hhs;
    arma::mat Q1b = arma::eye(m, 1);
    arma::mat R(X);
    arma::mat current_hh_mat;
    arma::vec hh;
    arma::mat expanded;

    Q1b = b;

    for(int i = 0; i < n; ++i){
        //Compute householder
        std::tie(hh, s)= compute_householder(R.col(i).tail(R.col(i).n_elem - i));
        //Compute H
        current_hh_mat = arma::eye(hh.n_elem, hh.n_elem) - 2*hh*hh.t();
        expanded = expand_matrix(current_hh_mat, m);
        R = expanded * R;
        Q1b = expanded.t() * Q1b;
    }

    return {Q1b, R};
}

arma::vec solve_thin_qr(const arma::mat &Q1b, const arma::mat &R) {
    arma::mat R1;
    R1 = R.head_rows(R.n_cols);
    arma::vec w(Q1b.n_rows, arma::fill::zeros);

    w = R1.i() * (arma::eye(R1.n_cols, Q1b.n_rows) * Q1b);

    return w;
}