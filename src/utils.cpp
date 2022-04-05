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
