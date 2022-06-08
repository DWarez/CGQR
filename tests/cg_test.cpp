#include <iostream>
#include <armadillo>
#include "../include/conjugate_gradient.hpp"
#include "../include/utils.hpp"

bool cg_random_test(uint m, uint n, uint k) {
    std::cout << "\n\nStart of the CG test\n==========================" << std::endl;
    arma::arma_rng::set_seed_random();
    arma::mat X(m, n, arma::fill::randn);
    arma::vec b(X.n_rows, arma::fill::randn);
    arma::vec w = arma::zeros(X.n_cols);

    arma::mat n_X;
    arma::vec n_b;

    arma::vec solution = arma::solve(X, b);

    std::tie(n_X, n_b) = to_normal_equations(X, b);

    auto start = std::chrono::steady_clock::now();

    w = conjugate_gradient(n_X, n_b, k);

    auto end = std::chrono::steady_clock::now();

    std::cout << "Execution time: " <<
              std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()<< " us \n";
    std::cout << "Norm of CG: " << arma::norm(X*w - b)/arma::norm(b)  << "\n";
    std::cout << "Distance from optimal solution: " << arma::norm(w - solution) << "\n";
    std::cout << "Gradient: " << arma::norm(X.t() * X * w - X.t() * b) << "\n";
    std::cout << "==========================\nEnd of the CG experiment" << std::endl;

    return true;
}

int main(int argc, char** argv) {
    if(argc == 3)
        cg_random_test(std::strtoul(argv[1], nullptr, 10), std::strtoul(argv[2], nullptr, 10), 30);
    else if(argc == 4)
        cg_random_test(std::strtoul(argv[1], nullptr, 10), std::strtoul(argv[2], nullptr, 10), std::strtoul(argv[3], nullptr, 10));
    else
        std::cout << "Missing required arguments\nUsage: \n\t ./cg_test N_ROWS N_COLS [OPTIONAL]EARLY_STOPPING_ITERATIONS" << std::endl;
    return 0;
}