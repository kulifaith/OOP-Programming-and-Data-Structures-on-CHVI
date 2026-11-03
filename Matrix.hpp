// Matrix.hpp
// ---------------------------------------------------------------------------
// Dense row-major matrix class with the linear-algebra routines needed by the
// CHVI system: multiplication, Gauss-Jordan inverse and a cyclic Jacobi
// eigen-solver for real symmetric matrices (used by FAMD and by the
// Gauss-Hermite quadrature rule).
// ---------------------------------------------------------------------------
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace chvi {

class Matrix {
public:
    Matrix() = default;
    Matrix(std::size_t rows, std::size_t cols, double fill = 0.0)
        : rows_(rows), cols_(cols), data_(rows * cols, fill) {}

    static Matrix identity(std::size_t n) {
        Matrix m(n, n, 0.0);
        for (std::size_t i = 0; i < n; ++i) m(i, i) = 1.0;
        return m;
    }

    std::size_t rows() const { return rows_; }
    std::size_t cols() const { return cols_; }

    double& operator()(std::size_t i, std::size_t j) { return data_[i * cols_ + j]; }
    double operator()(std::size_t i, std::size_t j) const { return data_[i * cols_ + j]; }

    std::vector<double> column(std::size_t j) const {
        std::vector<double> c(rows_);
        for (std::size_t i = 0; i < rows_; ++i) c[i] = (*this)(i, j);
        return c;
    }

    Matrix transpose() const {
        Matrix t(cols_, rows_);
        for (std::size_t i = 0; i < rows_; ++i)
            for (std::size_t j = 0; j < cols_; ++j) t(j, i) = (*this)(i, j);
        return t;
    }

    Matrix multiply(const Matrix& b) const {
        if (cols_ != b.rows_) throw std::invalid_argument("Matrix::multiply: dimension mismatch");
        Matrix r(rows_, b.cols_, 0.0);
        for (std::size_t i = 0; i < rows_; ++i)
            for (std::size_t k = 0; k < cols_; ++k) {
                const double aik = (*this)(i, k);
                for (std::size_t j = 0; j < b.cols_; ++j) r(i, j) += aik * b(k, j);
            }
        return r;
    }

    std::vector<double> multiply(const std::vector<double>& v) const {
        if (cols_ != v.size()) throw std::invalid_argument("Matrix::multiply(vec): dimension mismatch");
        std::vector<double> r(rows_, 0.0);
        for (std::size_t i = 0; i < rows_; ++i)
            for (std::size_t j = 0; j < cols_; ++j) r[i] += (*this)(i, j) * v[j];
        return r;
    }

    /// Gauss-Jordan inverse with partial pivoting. Throws if singular.
    Matrix inverse() const {
        if (rows_ != cols_) throw std::invalid_argument("Matrix::inverse: not square");
        const std::size_t n = rows_;
        Matrix a = *this;
        Matrix inv = identity(n);
        for (std::size_t col = 0; col < n; ++col) {
            std::size_t piv = col;
            for (std::size_t r = col + 1; r < n; ++r)
                if (std::fabs(a(r, col)) > std::fabs(a(piv, col))) piv = r;
            if (std::fabs(a(piv, col)) < 1e-13) throw std::runtime_error("Matrix::inverse: singular matrix");
            if (piv != col)
                for (std::size_t j = 0; j < n; ++j) {
                    std::swap(a(piv, j), a(col, j));
                    std::swap(inv(piv, j), inv(col, j));
                }
            const double d = a(col, col);
            for (std::size_t j = 0; j < n; ++j) {
                a(col, j) /= d;
                inv(col, j) /= d;
            }
            for (std::size_t r = 0; r < n; ++r) {
                if (r == col) continue;
                const double f = a(r, col);
                if (f == 0.0) continue;
                for (std::size_t j = 0; j < n; ++j) {
                    a(r, j) -= f * a(col, j);
                    inv(r, j) -= f * inv(col, j);
                }
            }
        }
        return inv;
    }

    /// Cyclic Jacobi eigen-solver for a real symmetric matrix.
    /// Returns eigenvalues in descending order and the matching eigenvectors
    /// stored as the columns of `vectors`.
    void symmetricEigen(std::vector<double>& values, Matrix& vectors,
                        int maxSweeps = 100) const {
        if (rows_ != cols_) throw std::invalid_argument("symmetricEigen: not square");
        const std::size_t n = rows_;
        Matrix a = *this;
        Matrix v = identity(n);
        for (int sweep = 0; sweep < maxSweeps; ++sweep) {
            double off = 0.0;
            for (std::size_t p = 0; p < n; ++p)
                for (std::size_t q = p + 1; q < n; ++q) off += a(p, q) * a(p, q);
            if (off < 1e-26) break;
            for (std::size_t p = 0; p < n; ++p) {
                for (std::size_t q = p + 1; q < n; ++q) {
                    const double apq = a(p, q);
                    if (std::fabs(apq) < 1e-300) continue;
                    const double theta = (a(q, q) - a(p, p)) / (2.0 * apq);
                    const double t = (theta >= 0 ? 1.0 : -1.0) /
                                     (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
                    const double c = 1.0 / std::sqrt(t * t + 1.0);
                    const double s = t * c;
                    for (std::size_t k = 0; k < n; ++k) {  // A <- A P
                        const double akp = a(k, p), akq = a(k, q);
                        a(k, p) = c * akp - s * akq;
                        a(k, q) = s * akp + c * akq;
                    }
                    for (std::size_t k = 0; k < n; ++k) {  // A <- P^T A
                        const double apk = a(p, k), aqk = a(q, k);
                        a(p, k) = c * apk - s * aqk;
                        a(q, k) = s * apk + c * aqk;
                    }
                    for (std::size_t k = 0; k < n; ++k) {  // V <- V P
                        const double vkp = v(k, p), vkq = v(k, q);
                        v(k, p) = c * vkp - s * vkq;
                        v(k, q) = s * vkp + c * vkq;
                    }
                }
            }
        }
        std::vector<std::size_t> order(n);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(),
                  [&](std::size_t i, std::size_t j) { return a(i, i) > a(j, j); });
        values.assign(n, 0.0);
        vectors = Matrix(n, n);
        for (std::size_t k = 0; k < n; ++k) {
            values[k] = a(order[k], order[k]);
            for (std::size_t i = 0; i < n; ++i) vectors(i, k) = v(i, order[k]);
        }
    }

private:
    std::size_t rows_ = 0, cols_ = 0;
    std::vector<double> data_;
};

}  // namespace chvi
