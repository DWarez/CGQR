#include <armadillo>
#include <cassert>

void to_normal_equations(arma::mat &X, arma::vec &b) {
    X = X * X.t();
    b = X * b;
}

arma::vec trim_head_vector(const arma::vec &x, int n) {
    assert(n >= 0 && "n must be >= 0");

    arma::vec modified(x.n_elem - n, arma::fill::zeros);
    for(size_t i = 0; i < modified.n_elem; ++i) {
        modified(i) = x(i + n);
    }

    return modified;
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
