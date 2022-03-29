#include <armadillo>
#include <cassert>
#include "../include/conjugate_gradient.hpp"

arma::vec conjugate_gradient(const arma::mat &X, const arma::vec &b, int n_iterations) {
    assert(X.is_symmetric() && "Error: matrix X must be symmetric");

    arma::vec residual(b);
    arma::vec direction(b);
    arma::vec previous_residual(residual);
    arma::vec w(b.n_elem, arma::fill::zeros);

    for(size_t i = 0; i < n_iterations; ++i) {
        auto alpha = (((arma::dot(residual.t(), residual))/(direction.t() * X * direction)).eval())(0, 0);
        w = w + (alpha * direction);
        residual = residual - (alpha * X * direction);
        auto beta = (arma::dot(residual.t(), residual))/(arma::dot(previous_residual.t(), previous_residual));
        direction = residual + (beta * direction);
    }

    return w;
}