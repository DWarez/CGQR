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


std::pair<arma::mat, arma::mat> thin_qr(const arma::mat &X) {
    int s = 0;
    uint m = X.n_rows;
    uint n = X.n_cols;

    std::vector<arma::vec> hhs;
    arma::mat Q = arma::eye(m, n);
    arma::mat R(X);
    arma::mat current_hh_mat;
    arma::vec hh;
    arma::mat expanded;

    for(int i = 0; i < n; i++){
        //Compute householder
        std::tie(hh, s)= compute_householder(trim_head_vector(R.col(i),i));
        //Compute H
        current_hh_mat = arma::eye(hh.n_elem, hh.n_elem) - 2*hh*hh.t();
        expanded = expand_matrix(current_hh_mat, m);
        R = expanded*R;
        hhs.push_back(hh);
    }
    for (int i = hhs.size() - 1; i >= 0; --i){
        expanded = expand_matrix(arma::eye(hhs[i].size(), hhs[i].size()) - 2*hhs[i]*hhs[i].t(),m);
        Q = expanded * Q;
    }

    return {Q, R};
}