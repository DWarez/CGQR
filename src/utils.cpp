#include <armadillo>

void to_normal_equations(arma::mat &X, arma::vec &b) {
    X = X * X.t();
    b = X * b;
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
