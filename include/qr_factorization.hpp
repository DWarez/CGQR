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
 * Computes the thin QR factorization
 * @param X input matrix
 * @return Q, R matrices
 */
std::pair<arma::mat, arma::mat> thin_qr(const arma::mat &X);

/**
 * Solve the linear system using the thin QR factorization of the matrix
 * @param Q Q of the QR
 * @param R R of the QR
 * @param b target vector
 * @return vector of weights
 */
arma::vec solve_thin_qr(const arma::mat &Q, const arma::mat &R, const arma::vec &b);

#endif //CGQR_QR_FACTORIZATION_HPP
