#ifndef CGQR_UTILS_HPP
#define CGQR_UTILS_HPP

#include <armadillo>

/**
 * Transforms the original system into the system of normal equations
 * @param X matrix of the system
 * @param b vector of the system
 */
void to_normal_equations(arma::mat &X, arma::vec &b);

/**
 * Takes a matrix and expand its dimensions with the Identity matrix
 * @param X input matrix
 * @param m desired dimension
 * @return expanded matrix
 */
arma::mat expand_matrix(const arma::mat &X, uint m);

#endif //CGQR_UTILS_HPP