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
 * Computes the thin QR factorization.
 * Notice that the target vector b is required in order to compute the factorization efficiently.
 * @param X input matrix
 * @return Q, R matrices
 */
std::pair<arma::mat, arma::mat> thin_qr(const arma::mat &X, const arma::vec &b);

/**
 * Solve the linear system using the thin QR factorization of the matrix
 * @param Q Q1b of the QR, which is Q*b
 * @param R R of the QR
 * @return vector of weights
 */
arma::vec solve_thin_qr(const arma::mat &Q1b, const arma::mat &R);

#endif //CGQR_QR_FACTORIZATION_HPP
