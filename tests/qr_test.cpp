#include <iostream>
#include <armadillo>
#include "../include/qr_factorization.hpp"

bool thin_qr_random_test(uint m, uint n) {
    std::cout << "\n\nStart of the QR test\n==========================" << std::endl;
    arma::arma_rng::set_seed_random();
    arma::mat X(m, n, arma::fill::randn);

    arma::mat Q(X.n_rows, X.n_cols);
    arma::mat R(X.n_rows, X.n_cols);

    arma::vec b(X.n_rows, arma::fill::randn);
    arma::vec w(X.n_cols, arma::fill::randn);

    arma::vec solution = arma::solve(X, b);

    auto start = std::chrono::steady_clock::now();

    std::tie(Q, R) = thin_qr(X, b);
    w = solve_thin_qr(Q, R);

    auto end = std::chrono::steady_clock::now();

    std::cout << "Execution time: " <<
              std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()<< " us \n";
    std::cout << "Norm of QR: " << arma::norm(X*w - b)/arma::norm(b)  << "\n";
    std::cout << "Distance from optimal solution: " << arma::norm(w - solution) << "\n";
    std::cout << "Gradient: " << arma::norm(X.t() * X * w - X.t() * b) << "\n";
    std::cout << "==========================\nEnd of the QR experiment" << std::endl;

    return true;
}

int main(int argc, char** argv) {
    if(argc != 3) {
        std::cout << "Wrong parameters.\nUsage:\n\t./qr_test N_ROWS N_COLUMNS" << std::endl;
        return -1;
    }
    thin_qr_random_test(std::strtoul(argv[1], nullptr, 10), std::strtoul(argv[2], nullptr, 10));
    return 0;
}

