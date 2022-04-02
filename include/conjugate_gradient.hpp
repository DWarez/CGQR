#ifndef CGQR_CONJUGATE_GRADIENT_HPP
#define CGQR_CONJUGATE_GRADIENT_HPP

#include <armadillo>

/**
 * Given the symmetric matrix X and the target vector b,
 * computes the solution vector w using the conjugate gradient method.
 * @param X positive definite symmetric matrix
 * @param b target vector
 * @param n_iterations number of iterations [default value = 100]
 */
arma::vec conjugate_gradient(const arma::mat &X, const arma::vec &b, uint n_iterations, bool store_results = true);


#endif //CGQR_CONJUGATE_GRADIENT_HPP
