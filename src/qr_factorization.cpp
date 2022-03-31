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
        current_column = X.col(i);
        std::tie(current_hh, s) = compute_householder(trim_head_vector(current_column, i));
        hh_set.push_back(current_hh);
    }


    return hh_set;
}


std::pair<arma::mat, arma::mat> thin_qr(const arma::mat &X) {

    int s = 0;
    int m = X.n_rows;
    int n = X.n_cols;

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
    for (int i = hhs.size() - 1; i >= 0; i--){
        expanded = expand_matrix(arma::eye(hhs[i].size(), hhs[i].size()) - 2*hhs[i]*hhs[i].t(),m);
        Q = expanded * Q;
    }

    /*
     * Old code, here the whole householder set is calculated before the iterations, wrong.
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
    */

    return {Q, R};
}