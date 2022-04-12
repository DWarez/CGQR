#include <armadillo>
#include "../include/utils.hpp"

std::pair<arma::mat, arma::vec> to_normal_equations(const arma::mat& X, const arma::vec& b) {
    return {X.t() * X, X.t() * b};
}

arma::mat expand_matrix(const arma::mat &X, uint m) {
    arma::mat modified = arma::eye(m, m);

    for(size_t i = m - X.n_rows; i < m; ++i) {
        for(size_t j = m - X.n_cols; j < m; ++j) {
            modified(i,j) = X(i - (m - X.n_rows), j - (m - X.n_cols));
        }
    }

    return modified;
}

std::pair<arma::mat, arma::vec> grab_mlcup_dataset() {
    arma::mat X;
    X.load(DEFAULT_ML_CUP_PATH, arma::csv_ascii);
    // grabbing target vector
    arma::vec b = X.col(X.n_cols - 2);
    // remove index column and target columns
    X.shed_col(0);
    X.shed_cols(X.n_cols - 2, X.n_cols - 1);

    return {X, b};
}

void add_columns(arma::mat &X) {
    X.insert_cols(X.n_cols, arma::log(arma::abs(X.col(3))));
    X.insert_cols(X.n_cols,arma::log(arma::abs(X.col(1)%X.col(6))));
    X.insert_cols(X.n_cols,arma::pow(X.col(2),3));
    X.insert_cols(X.n_cols, X.col(5)%X.col(1));
    X.insert_cols(X.n_cols, X.col(6)%X.col(7));
    X.insert_cols(X.n_cols, arma::pow(X.col(4),2));
    X.insert_cols(X.n_cols, X.col(7)%X.col(7));
    X.insert_cols(X.n_cols, arma::sin(X.col(10)));
    X.insert_cols(X.n_cols, arma::cos(X.col(12)));
    X.insert_cols(X.n_cols, X.col(5)%X.col(9));
}

void problem_properties() {
    arma::mat X, n_X;
    arma::vec b, n_b;
    arma::vec eigval;
    arma::mat eigvec;

    std::tie(X, b) = grab_mlcup_dataset();
    add_columns(X);

    std::cout << "X:\n\tcols: " << X.n_cols << "\trows: " << X.n_rows << "\n";
    std::cout << "b:\n\telements: " << b.n_elem << "\n";

    std::tie(n_X, n_b) = to_normal_equations(X, b);
    arma::vec solution = arma::solve(X, b);
    arma::vec w(X.n_cols, arma::fill::zeros);

    std::cout << "X'X:\n\tcols: " << n_X.n_cols << "\trows: " << n_X.n_rows << "\n";
    arma::eig_sym(eigval, eigvec, n_X);
    std::cout << "X'X:\n\teigenvalues: ";
    for(auto &x: eigval) {
        std::cout << x << "\t";
    }
    std::cout << "\n\tcondition number: " << eigval.back()/eigval.front();
    auto difference = w - solution;
    auto x_norm = arma::sqrt(difference.t() * n_X * difference);
    auto fraction = std::pow(((std::sqrt(eigval.back()/eigval.front()) - 1)/(std::sqrt(eigval.back()/eigval.front()) + 1)), 35);
    std::cout << "\n\tk_th iteration bound: " << 2 * fraction * x_norm << "\n";
}