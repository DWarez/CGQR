#ifndef CGQR_QR_FACTORIZATION_HPP
#define CGQR_QR_FACTORIZATION_HPP

#include <iostream>
#include <cassert>
#include <armadillo>

/**
 * Computes the householder reflector of a given vector
 * @param x input vector
 * @return x's reflector
 */
std::tuple<arma::vec, int> compute_householder(const arma::vec &x);

/**
 * Computes the set of householder reflectors from a given matrix.
 * This method is used to compute the QR factorization, hence at each step
 * a column of X is considered and the length of the reflector decreases by 1.
 * @param X input matrix
 * @return set of householder reflectors
 */
std::vector<arma::vec> householder_set(const arma::mat &X);


std::pair<arma::mat, arma::mat> thin_qr(const arma::mat &X);

#endif //CGQR_QR_FACTORIZATION_HPP
