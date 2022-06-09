#include <armadillo>
#include <cassert>
#include <sys/stat.h>
#include <chrono>
#include "../include/conjugate_gradient.hpp"

arma::vec conjugate_gradient(const arma::mat &X, const arma::vec &b, uint max_iterations, bool store_results,
                             double threshold, uint es_tries) {
    assert(X.is_symmetric() && "Error: matrix X must be symmetric");

    arma::vec residual(b);
    arma::vec direction(-b);
    arma::vec previous_residual(residual);
    arma::vec w(b.n_elem, arma::fill::zeros);
    double current_distance;
    std::vector<double> history;
    uint i = 0;
    uint tries = 0;

    // starting residual
    current_distance = arma::norm(X*w - b)/arma::norm(b);
    history.push_back(current_distance);

    while(i < max_iterations &&  tries < es_tries) {
        double alpha = - arma::as_scalar((residual.t() * residual)/(direction.t() * X * direction));
        w = w + (alpha * direction);
        residual = residual - (alpha * X * direction);
        double beta = arma::as_scalar((residual.t() * residual)/(previous_residual.t() * previous_residual));
        previous_residual = residual;
        direction = -residual + (beta * direction);
        i++;

        current_distance = arma::norm(X*w - b);
        if(current_distance < threshold) break;
        if(history.back() == current_distance) tries++; else tries = 0;
        history.push_back(current_distance);
    }

    if(store_results) {
        mkdir("../results", S_IRWXU);
        std::ofstream outputfile("../results/cg_run.txt", std::ios::trunc);

        for (auto x: history)
            outputfile << x << " ";

        outputfile << std::endl;
        outputfile.close();
    }

    std::cout << "Number of iterations: " << i - tries << "\n";

    return w;
}