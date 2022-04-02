#include <armadillo>
#include <cassert>
#include <sys/stat.h>
#include <chrono>
#include "../include/conjugate_gradient.hpp"

arma::vec conjugate_gradient(const arma::mat &X, const arma::vec &b, uint n_iterations, bool store_results) {
    assert(X.is_symmetric() && "Error: matrix X must be symmetric");

    arma::vec residual(b);
    arma::vec direction(b);
    arma::vec previous_residual(residual);
    arma::vec w(b.n_elem, arma::fill::ones);
    std::vector<double> history;

    auto start = std::chrono::steady_clock::now();
    std::cout << "Starting iterations\n";
    // starting residual
    history.push_back(arma::norm(X*w - b)/arma::norm(b));

    for(size_t i = 0; i < n_iterations; ++i) {
        auto alpha = (((arma::dot(residual.t(), residual))/(direction.t() * X * direction)).eval())(0, 0);
        w = w + (alpha * direction);
        residual = residual - (alpha * X * direction);
        auto beta = (arma::dot(residual.t(), residual))/(arma::dot(previous_residual.t(), previous_residual));
        direction = residual + (beta * direction);

        history.push_back(arma::norm(X*w - b)/arma::norm(b));
    }

    if(store_results) {
        std::cout << "Storing results\n";
        mkdir("../results", S_IRWXU);
        std::ofstream outputfile("../results/cg_run.txt", std::ios::trunc);

        for(auto x: history)
            outputfile << x << " ";

        outputfile << std::endl;

        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> diff = end - start;

        outputfile << "\nTime of execution: " << diff.count();
    }

    return w;
}