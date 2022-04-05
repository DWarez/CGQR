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


std::pair<arma::mat, arma::mat> thin_qr(const arma::mat &X) {
    int s = 0;
    uint m = X.n_rows;
    uint n = X.n_cols;

    std::vector<arma::mat> hhs;
    arma::mat Q = arma::eye(m, n);
    arma::mat R(X);
    arma::mat current_hh_mat;
    arma::vec hh;
    arma::mat expanded;

    for(int i = 0; i < n; i++){
        //Compute householder
        std::tie(hh, s)= compute_householder(R.col(i).tail(R.col(i).n_elem - i));
        //Compute H
        current_hh_mat = arma::eye(hh.n_elem, hh.n_elem) - 2*hh*hh.t();
        expanded = expand_matrix(current_hh_mat, m);
        R = expanded*R;
        hhs.push_back(expanded);
    }
    for (auto &x: std::ranges::reverse_view(hhs))
        Q = x * Q;

    return {Q, R};
}

arma::vec solve_thin_qr(const arma::mat &Q, const arma::mat &R, const arma::vec &b) {
    uint m = Q.n_rows;
    uint n = Q.n_cols;
    arma::mat R1;
    R1 = R.head_rows(R.n_cols);
    arma::vec w(b.n_elem, arma::fill::zeros);

    w = R1.i() * (Q.t() * b);

    return w;
}