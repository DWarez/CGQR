#ifndef CGQR_CONJUGATE_GRADIENT_HPP
#define CGQR_CONJUGATE_GRADIENT_HPP

#include <armadillo>

#define CONVERGENCE_THRESHOLD 1e-16
/**
 * Given the symmetric matrix X and the target vector b,
 * computes the solution vector w using the conjugate gradient method.
 * @param X positive definite symmetric matrix
 * @param b target vector
 * @param max_iterations number of iterations [default value = 100]
 */
arma::vec conjugate_gradient(const arma::mat &X, const arma::vec &b, uint max_iterations, bool store_results = true,
                             double threshold = CONVERGENCE_THRESHOLD, bool early_stopping = true, uint es_tries = 5);


#endif //CGQR_CONJUGATE_GRADIENT_HPP
