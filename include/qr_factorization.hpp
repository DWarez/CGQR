#ifndef CGQR_QR_FACTORIZATION_HPP
#define CGQR_QR_FACTORIZATION_HPP

#include <iostream>
#include <armadillo>

typedef std::pair<arma::vec, double> hh_vec;

hh_vec compute_householder(const arma::vec &x) {
    arma::vec v(x);
    double s = arma::norm(x, 2);

    if(x(0) >= 0)
        s = -s;

    v(0) = v(0) - s;
    double norm = arma::norm(v, 2);
    v = v/norm;
    return {v, s};
}

#endif //CGQR_QR_FACTORIZATION_HPP
