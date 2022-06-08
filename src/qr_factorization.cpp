#include <armadillo>
//#include <ranges>
#include "../include/qr_factorization.hpp"

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
    arma::mat Q1b(b);
    arma::mat R(X);
    arma::vec hh;

    for(int i = 0; i < n; ++i){
        std::tie(hh, s)= compute_householder(R.col(i).tail(R.col(i).n_elem - i));
        R.submat(i, i, m-1, n-1) = R.submat(i, i, m-1, n-1) -
                2*hh*(hh.t() * R.submat(i, i, m-1, n-1));
        Q1b.submat(i, 0, m-1, 0) = Q1b.submat(i, 0, m-1, 0) -
              2*hh*(hh.t() * Q1b.submat(i, 0, m-1, 0));
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
