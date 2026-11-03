// MixedLogit.hpp
// ---------------------------------------------------------------------------
// Two-level random-intercept logistic regression (Eq. 10):
//     logit P(y_ij = 1 | x_ij, u_j) = x_ij' beta + u_j ,   u_j ~ N(0, sigma^2)
// estimated by maximum likelihood.  The cluster likelihood integral is
// evaluated with ADAPTIVE Gauss-Hermite quadrature (Newton mode-finding per
// cluster); the parameters (beta, log sigma) are maximised with BFGS using
// central-difference gradients; standard errors come from a numerical Hessian.
// ---------------------------------------------------------------------------
#pragma once

#include <cmath>
#include <string>
#include <vector>

#include "Portable.hpp"
#include "Matrix.hpp"
#include "Stats.hpp"

namespace chvi {

struct MixedLogitFit {
    std::vector<std::string> names;
    std::vector<double> beta, se, z, pValue;
    double sigma = 0.0;       // SD of the random intercept
    double logLik = 0.0;
    double aic = 0.0, bic = 0.0;
    double icc = 0.0;         // sigma^2 / (sigma^2 + pi^2/3)
    int nObs = 0, nClusters = 0, iterations = 0;
    bool converged = false;
};

class MixedLogit {
public:
    explicit MixedLogit(int nodes = 15) : quad_(stats::gaussHermite(nodes)) {}

    /// X must contain the intercept column; y in {0,1}; cluster = group ids.
    MixedLogitFit fit(const Matrix& X, const std::vector<int>& y, const std::vector<int>& cluster,
                      const std::vector<std::string>& names) const {
        Problem pr;
        pr.X = &X;
        pr.y = &y;
        // group observations by cluster
        std::vector<int> ids(cluster);
        std::sort(ids.begin(), ids.end());
        ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
        pr.groups.assign(ids.size(), {});
        for (std::size_t i = 0; i < cluster.size(); ++i) {
            const std::size_t g = static_cast<std::size_t>(
                std::lower_bound(ids.begin(), ids.end(), cluster[i]) - ids.begin());
            pr.groups[g].push_back(static_cast<int>(i));
        }
        const std::size_t p = X.cols();

        std::vector<double> theta(p + 1, 0.0);
        double ybar = 0.0;
        for (int v : y) ybar += v;
        ybar = std::min(0.95, std::max(0.05, ybar / static_cast<double>(y.size())));
        theta[0] = std::log(ybar / (1.0 - ybar));
        theta[p] = std::log(0.5);

        MixedLogitFit out;
        out.names = names;
        out.nObs = static_cast<int>(y.size());
        out.nClusters = static_cast<int>(ids.size());
        out.iterations = bfgs(pr, theta, out.converged);

        out.beta.assign(theta.begin(), theta.begin() + static_cast<long>(p));
        out.sigma = std::exp(theta[p]);
        out.logLik = -negLogLik(pr, theta);
        const double k = static_cast<double>(p + 1);
        out.aic = 2.0 * k - 2.0 * out.logLik;
        out.bic = k * std::log(static_cast<double>(out.nObs)) - 2.0 * out.logLik;
        out.icc = out.sigma * out.sigma / (out.sigma * out.sigma + kPi * kPi / 3.0);

        // standard errors from the numerical Hessian (beta-block fallback)
        out.se.assign(p, 0.0);
        out.z.assign(p, 0.0);
        out.pValue.assign(p, 1.0);
        std::vector<double> var = invertedHessianDiag(pr, theta, p, true);
        if (var.empty()) var = invertedHessianDiag(pr, theta, p, false);
        for (std::size_t j = 0; j < p && j < var.size(); ++j) {
            out.se[j] = var[j] > 0 ? std::sqrt(var[j]) : 0.0;
            out.z[j] = out.se[j] > 0 ? out.beta[j] / out.se[j] : 0.0;
            out.pValue[j] = stats::normalTwoSidedP(out.z[j]);
        }
        return out;
    }

private:
    struct Problem {
        const Matrix* X = nullptr;
        const std::vector<int>* y = nullptr;
        std::vector<std::vector<int>> groups;
    };
    stats::Quadrature quad_;

    /// Negative marginal log-likelihood; theta = (beta, log sigma).
    double negLogLik(const Problem& pr, const std::vector<double>& theta) const {
        const std::size_t p = pr.X->cols();
        const double sigma = std::exp(theta[p]);
        double ll = 0.0;
        std::vector<double> eta;
        std::vector<double> terms(quad_.nodes.size());
        for (const auto& g : pr.groups) {
            eta.assign(g.size(), 0.0);
            for (std::size_t a = 0; a < g.size(); ++a) {
                double s = 0.0;
                for (std::size_t j = 0; j < p; ++j) s += (*pr.X)(static_cast<std::size_t>(g[a]), j) * theta[j];
                eta[a] = s;
            }
            // Newton iterations for the mode of h(z) = log f(y|z) - z^2/2
            double z = 0.0, hess = -1.0;
            for (int it = 0; it < 50; ++it) {
                double grad = -z;
                hess = -1.0;
                for (std::size_t a = 0; a < g.size(); ++a) {
                    const double pr1 = stats::sigmoid(eta[a] + sigma * z);
                    grad += sigma * ((*pr.y)[static_cast<std::size_t>(g[a])] - pr1);
                    hess -= sigma * sigma * pr1 * (1.0 - pr1);
                }
                double step = grad / hess;
                step = std::max(-5.0, std::min(5.0, step));
                z -= step;
                if (std::fabs(step) < 1e-10) break;
            }
            hess = -1.0;
            for (std::size_t a = 0; a < g.size(); ++a) {
                const double pr1 = stats::sigmoid(eta[a] + sigma * z);
                hess -= sigma * sigma * pr1 * (1.0 - pr1);
            }
            const double s = 1.0 / std::sqrt(-hess);
            double mx = -1e300;
            for (std::size_t q = 0; q < quad_.nodes.size(); ++q) {
                const double zq = z + std::sqrt(2.0) * s * quad_.nodes[q];
                double h = -0.5 * zq * zq;
                for (std::size_t a = 0; a < g.size(); ++a) {
                    const double e = eta[a] + sigma * zq;
                    h += (*pr.y)[static_cast<std::size_t>(g[a])] * e - stats::log1pExp(e);
                }
                terms[q] = std::log(quad_.weights[q]) + quad_.nodes[q] * quad_.nodes[q] + h;
                mx = std::max(mx, terms[q]);
            }
            double sum = 0.0;
            for (double t : terms) sum += std::exp(t - mx);
            ll += std::log(std::sqrt(2.0) * s) + mx + std::log(sum) - 0.5 * std::log(2.0 * kPi);
        }
        return -ll;
    }

    std::vector<double> gradient(const Problem& pr, std::vector<double> th) const {
        std::vector<double> g(th.size());
        for (std::size_t j = 0; j < th.size(); ++j) {
            const double h = 1e-5 * std::max(1.0, std::fabs(th[j]));
            const double save = th[j];
            th[j] = save + h;
            const double fp = negLogLik(pr, th);
            th[j] = save - h;
            const double fm = negLogLik(pr, th);
            th[j] = save;
            g[j] = (fp - fm) / (2.0 * h);
        }
        return g;
    }

    int bfgs(const Problem& pr, std::vector<double>& th, bool& converged) const {
        const std::size_t n = th.size();
        Matrix H = Matrix::identity(n);
        double f = negLogLik(pr, th);
        std::vector<double> g = gradient(pr, th);
        converged = false;
        int it = 0;
        for (; it < 300; ++it) {
            double gmax = 0.0;
            for (double v : g) gmax = std::max(gmax, std::fabs(v));
            if (gmax < 1e-5) { converged = true; break; }
            std::vector<double> dir = H.multiply(g);
            for (double& d : dir) d = -d;
            double slope = 0.0;
            for (std::size_t j = 0; j < n; ++j) slope += g[j] * dir[j];
            if (slope >= 0.0) {  // not a descent direction: reset
                H = Matrix::identity(n);
                dir = g;
                for (double& d : dir) d = -d;
                slope = 0.0;
                for (std::size_t j = 0; j < n; ++j) slope += g[j] * dir[j];
            }
            double t = 1.0, fNew = f;
            std::vector<double> thNew(n);
            bool ok = false;
            for (int ls = 0; ls < 40; ++ls) {
                for (std::size_t j = 0; j < n; ++j) thNew[j] = th[j] + t * dir[j];
                thNew[n - 1] = std::max(-6.0, std::min(2.5, thNew[n - 1]));
                fNew = negLogLik(pr, thNew);
                if (std::isfinite(fNew) && fNew <= f + 1e-4 * t * slope) { ok = true; break; }
                t *= 0.5;
            }
            if (!ok) { converged = (gmax < 1e-3); break; }
            std::vector<double> gNew = gradient(pr, thNew);
            std::vector<double> s(n), yv(n);
            double sy = 0.0;
            for (std::size_t j = 0; j < n; ++j) {
                s[j] = thNew[j] - th[j];
                yv[j] = gNew[j] - g[j];
                sy += s[j] * yv[j];
            }
            if (sy > 1e-12) {  // BFGS update of the inverse Hessian
                const double rho = 1.0 / sy;
                Matrix Hy = H;
                std::vector<double> Hyv = H.multiply(yv);
                double yHy = 0.0;
                for (std::size_t j = 0; j < n; ++j) yHy += yv[j] * Hyv[j];
                for (std::size_t a = 0; a < n; ++a)
                    for (std::size_t b = 0; b < n; ++b)
                        Hy(a, b) = H(a, b) - rho * (s[a] * Hyv[b] + Hyv[a] * s[b]) +
                                   (rho * rho * yHy + rho) * s[a] * s[b];
                H = Hy;
            }
            const double df = std::fabs(f - fNew);
            th = thNew;
            g = gNew;
            f = fNew;
            if (df < 1e-12 * (1.0 + std::fabs(f))) { converged = true; ++it; break; }
        }
        return it;
    }

    /// Diagonal of the inverse numerical Hessian (empty if not invertible).
    std::vector<double> invertedHessianDiag(const Problem& pr, std::vector<double> th,
                                            std::size_t p, bool full) const {
        const std::size_t m = full ? th.size() : p;
        Matrix Hs(m, m, 0.0);
        for (std::size_t j = 0; j < m; ++j) {
            const double h = 1e-4 * std::max(1.0, std::fabs(th[j]));
            const double save = th[j];
            th[j] = save + h;
            const std::vector<double> gp = gradient(pr, th);
            th[j] = save - h;
            const std::vector<double> gm = gradient(pr, th);
            th[j] = save;
            for (std::size_t i = 0; i < m; ++i) Hs(i, j) = (gp[i] - gm[i]) / (2.0 * h);
        }
        for (std::size_t i = 0; i < m; ++i)
            for (std::size_t j = i + 1; j < m; ++j) {
                const double a = 0.5 * (Hs(i, j) + Hs(j, i));
                Hs(i, j) = Hs(j, i) = a;
            }
        try {
            const Matrix inv = Hs.inverse();
            std::vector<double> d(m);
            for (std::size_t i = 0; i < m; ++i) d[i] = inv(i, i);
            for (std::size_t i = 0; i < p; ++i)
                if (!(d[i] > 0.0)) return {};
            return d;
        } catch (const std::exception&) {
            return {};
        }
    }
};

}  // namespace chvi
