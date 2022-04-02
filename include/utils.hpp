#ifndef CGQR_UTILS_HPP
#define CGQR_UTILS_HPP

#include <armadillo>

#define DEFAULT_ML_CUP_PATH "../data/ML-CUP21-TR.csv"

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

/**
 * Returns the ML Cup dataset
 * @return Matrix of data-points and a target vector (11th column)
 */
std::pair<arma::mat, arma::vec> grab_mlcup_dataset();

#endif //CGQR_UTILS_HPP