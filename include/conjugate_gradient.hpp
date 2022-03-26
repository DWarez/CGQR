
//
// Created by dwarez on 24/03/22.
//

#ifndef CGQR_CONJUGATE_GRADIENT_HPP
#define CGQR_CONJUGATE_GRADIENT_HPP

#include <armadillo>


void conjugate_gradient(const arma::mat &X, const arma::vec &b, arma::vec &w, int n_iterations = 100) {
    arma::vec residual(b);
    arma::vec direction(b);
    arma::vec previous_residual(residual);

    for(size_t i = 0; i < n_iterations; ++i) {
        auto alpha = (((arma::dot(residual.t(), residual))/(direction.t() * X * direction)).eval())(0, 0);
        w = w + (alpha * direction);
        residual = residual - (alpha * X * direction);
        auto beta = (arma::dot(residual.t(), residual))/(arma::dot(previous_residual.t(), previous_residual));
        direction = residual + (beta * direction);
    }
}

void test(arma::vec &w) {
    w = arma::zeros(w.n_elem);
}

#endif //CGQR_CONJUGATE_GRADIENT_HPP
