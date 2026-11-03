// Stats.hpp
// ---------------------------------------------------------------------------
// Statistical helper functions written from first principles (no external
// libraries): descriptive statistics, ranks, Pearson/Spearman correlation,
// normal and chi-square tail probabilities, Gauss-Hermite quadrature, OLS and
// a reproducible random-number generator.
// ---------------------------------------------------------------------------
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

#include "Matrix.hpp"

namespace chvi {

constexpr double kPi = 3.14159265358979323846;

/// Reproducible RNG (Mersenne twister + own Box-Muller so that results are
/// identical on every platform/compiler).
class Rng {
public:
    explicit Rng(std::uint64_t seed = 1) : eng_(seed) {}
    double uniform() { return static_cast<double>(eng_() >> 11) * (1.0 / 9007199254740992.0); }
    double normal() {
        double u1 = uniform();
        if (u1 < 1e-300) u1 = 1e-300;
        const double u2 = uniform();
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * kPi * u2);
    }
    int integer(int n) { return static_cast<int>(uniform() * n) % n; }  // 0..n-1
    bool bernoulli(double p) { return uniform() < p; }
    template <typename T>
    void shuffle(std::vector<T>& v) {
        for (std::size_t i = v.size(); i > 1; --i)
            std::swap(v[i - 1], v[static_cast<std::size_t>(integer(static_cast<int>(i)))]);
    }

private:
    std::mt19937_64 eng_;
};

namespace stats {

inline double mean(const std::vector<double>& v) {
    if (v.empty()) return 0.0;
    double s = 0.0;
    for (double x : v) s += x;
    return s / static_cast<double>(v.size());
}

/// Sample variance (n-1 denominator).
inline double variance(const std::vector<double>& v) {
    if (v.size() < 2) return 0.0;
    const double m = mean(v);
    double s = 0.0;
    for (double x : v) s += (x - m) * (x - m);
    return s / static_cast<double>(v.size() - 1);
}

inline double sd(const std::vector<double>& v) { return std::sqrt(variance(v)); }

inline double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const std::size_t n = v.size();
    return (n % 2 == 1) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

inline double pearson(const std::vector<double>& x, const std::vector<double>& y) {
    const std::size_t n = std::min(x.size(), y.size());
    if (n < 2) return 0.0;
    const double mx = mean(x), my = mean(y);
    double sxy = 0.0, sxx = 0.0, syy = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        sxy += (x[i] - mx) * (y[i] - my);
        sxx += (x[i] - mx) * (x[i] - mx);
        syy += (y[i] - my) * (y[i] - my);
    }
    if (sxx <= 0.0 || syy <= 0.0) return 0.0;
    return sxy / std::sqrt(sxx * syy);
}

/// Mid-ranks (ties receive the average rank), 1-based.
inline std::vector<double> ranks(const std::vector<double>& v) {
    const std::size_t n = v.size();
    std::vector<std::size_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) { return v[a] < v[b]; });
    std::vector<double> r(n);
    std::size_t i = 0;
    while (i < n) {
        std::size_t j = i;
        while (j + 1 < n && v[idx[j + 1]] == v[idx[i]]) ++j;
        const double avg = 0.5 * static_cast<double>(i + j) + 1.0;
        for (std::size_t k = i; k <= j; ++k) r[idx[k]] = avg;
        i = j + 1;
    }
    return r;
}

inline double spearman(const std::vector<double>& x, const std::vector<double>& y) {
    return pearson(ranks(x), ranks(y));
}

inline double skewness(const std::vector<double>& v) {
    const double m = mean(v), s = std::sqrt(variance(v));
    if (s == 0.0) return 0.0;
    double t = 0.0;
    for (double x : v) t += std::pow((x - m) / s, 3);
    return t / static_cast<double>(v.size());
}

inline double excessKurtosis(const std::vector<double>& v) {
    const double m = mean(v), s = std::sqrt(variance(v));
    if (s == 0.0) return 0.0;
    double t = 0.0;
    for (double x : v) t += std::pow((x - m) / s, 4);
    return t / static_cast<double>(v.size()) - 3.0;
}

inline double normalCdf(double z) { return 0.5 * std::erfc(-z / std::sqrt(2.0)); }
inline double normalTwoSidedP(double z) { return std::erfc(std::fabs(z) / std::sqrt(2.0)); }

// --- Regularised incomplete gamma function (Numerical-Recipes style) --------
inline double gammaPSeries(double a, double x) {
    double ap = a, sum = 1.0 / a, del = sum;
    for (int n = 0; n < 2000; ++n) {
        ap += 1.0;
        del *= x / ap;
        sum += del;
        if (std::fabs(del) < std::fabs(sum) * 1e-16) break;
    }
    return sum * std::exp(-x + a * std::log(x) - std::lgamma(a));
}

inline double gammaQContinuedFraction(double a, double x) {
    const double tiny = 1e-300;
    double b = x + 1.0 - a, c = 1.0 / tiny, d = 1.0 / b, h = d;
    for (int i = 1; i < 2000; ++i) {
        const double an = -i * (i - a);
        b += 2.0;
        d = an * d + b;
        if (std::fabs(d) < tiny) d = tiny;
        c = b + an / c;
        if (std::fabs(c) < tiny) c = tiny;
        d = 1.0 / d;
        const double del = d * c;
        h *= del;
        if (std::fabs(del - 1.0) < 1e-15) break;
    }
    return std::exp(-x + a * std::log(x) - std::lgamma(a)) * h;
}

/// Upper regularised incomplete gamma Q(a, x).
inline double gammaQ(double a, double x) {
    if (x <= 0.0) return 1.0;
    if (x < a + 1.0) return 1.0 - gammaPSeries(a, x);
    return gammaQContinuedFraction(a, x);
}

/// Survival function of the chi-square distribution: P(X > x), X ~ chi2(df).
inline double chiSquareSF(double x, double df) { return gammaQ(df / 2.0, x / 2.0); }

// --- Logistic helpers -------------------------------------------------------
inline double sigmoid(double x) {
    if (x >= 0.0) return 1.0 / (1.0 + std::exp(-x));
    const double e = std::exp(x);
    return e / (1.0 + e);
}
/// log(1 + exp(x)) computed without overflow.
inline double log1pExp(double x) {
    if (x > 30.0) return x;
    if (x < -30.0) return std::exp(x);
    return std::log1p(std::exp(x));
}

// --- Gauss-Hermite quadrature (physicists' weight exp(-x^2)) ----------------
struct Quadrature {
    std::vector<double> nodes, weights;
};

/// Golub-Welsch algorithm: eigen-decomposition of the Jacobi matrix.
inline Quadrature gaussHermite(int n) {
    Matrix J(static_cast<std::size_t>(n), static_cast<std::size_t>(n), 0.0);
    for (int i = 0; i + 1 < n; ++i) {
        const double b = std::sqrt((i + 1) / 2.0);
        J(static_cast<std::size_t>(i), static_cast<std::size_t>(i + 1)) = b;
        J(static_cast<std::size_t>(i + 1), static_cast<std::size_t>(i)) = b;
    }
    std::vector<double> vals;
    Matrix vecs;
    J.symmetricEigen(vals, vecs);
    Quadrature q;
    q.nodes = vals;
    q.weights.resize(static_cast<std::size_t>(n));
    for (std::size_t i = 0; i < static_cast<std::size_t>(n); ++i)
        q.weights[i] = std::sqrt(kPi) * vecs(0, i) * vecs(0, i);
    return q;
}

/// Ordinary least squares via the normal equations; X includes any intercept.
inline std::vector<double> ols(const Matrix& X, const std::vector<double>& y) {
    const Matrix Xt = X.transpose();
    const Matrix inv = Xt.multiply(X).inverse();
    std::vector<double> xty(X.cols(), 0.0);
    for (std::size_t j = 0; j < X.cols(); ++j)
        for (std::size_t i = 0; i < X.rows(); ++i) xty[j] += X(i, j) * y[i];
    return inv.multiply(xty);
}

}  // namespace stats
}  // namespace chvi
