#include <armadillo>
#include <cassert>
#include <sys/stat.h>
#include <chrono>
#include "../include/conjugate_gradient.hpp"

arma::vec conjugate_gradient(const arma::mat &X, const arma::vec &b, uint max_iterations, bool store_results) {
    assert(X.is_symmetric() && "Error: matrix X must be symmetric");

    arma::vec residual(b);
    arma::vec direction(b);
    arma::vec previous_residual(residual);
    arma::vec w(b.n_elem, arma::fill::zeros);
    double current_distance;
    std::vector<double> history;
    uint i = 0;

    auto start = std::chrono::steady_clock::now();
    std::cout << "Starting iterations\n";
    // starting residual
    current_distance = arma::norm(X*w - b)/arma::norm(b);
    history.push_back(current_distance);

    while(i < max_iterations && current_distance >= CONVERGENCE_THRESHOLD) {
        double alpha = (((arma::dot(residual.t(), residual))/(direction.t() * X * direction)).eval())(0, 0);
        w = w + (alpha * direction);
        residual = residual - (alpha * X * direction);
        double beta = (arma::dot(residual.t(), residual))/(arma::dot(previous_residual.t(), previous_residual));
        direction = residual + (beta * direction);
        i++;

        current_distance = arma::norm(X*w - b)/arma::norm(b);
        history.push_back(current_distance);
    }

    if(store_results) {
        std::cout << "Storing results\n";
        mkdir("../results", S_IRWXU);
        std::ofstream outputfile("../results/cg_run.txt", std::ios::trunc);

        for(auto x: history)
            outputfile << x << " ";

        outputfile << std::endl;
        outputfile.close();

        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> diff = end - start;

        std::cout << "Number of iterations: " << i <<"\nTime of execution: " << diff.count() << std::endl;
    }

    return w;
}