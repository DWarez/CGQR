#ifndef CGQR_UTILS_HPP
#define CGQR_UTILS_HPP

#include <armadillo>
#include <cassert>

/**
 * Transforms the original system into the system of normal equations
 * @param X matrix of the system
 * @param b vector of the system
 */
void to_normal_equations(arma::mat &X, arma::vec &b) {
    X = X * X.t();
    b = X * b;
}

/**
 * Produces a copy of a vector in which the first n elements are removed, counting from the head
 * @param x constant reference to the vector
 * @param n number of elements to trim
 * @return copy of x with n elements removed, from the head
 */
arma::vec trim_head_vector(const arma::vec &x, int n) {
    assert(n >= 0 && "n must be >= 0");

    arma::vec modified(x.n_elem - n, arma::fill::zeros);
    for(size_t i = 0; i < modified.n_elem; ++i) {
        modified(i) = x(i + n);
    }

    return modified;
}

#endif //CGQR_UTILS_HPP