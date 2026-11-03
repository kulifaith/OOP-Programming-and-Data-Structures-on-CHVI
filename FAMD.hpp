// FAMD.hpp
// ---------------------------------------------------------------------------
// Factor Analysis of Mixed Data (Pages, 2004) implemented from first
// principles.  Steps follow Section 3.7.3 of the proposal:
//   1. build the weighted design matrix Z
//        continuous  j : (x - mean)/sd
//        category    c : (1[x = c] - p_c) / sqrt(p_c)      (MCA scaling)
//   2. PCA of Z  ->  eigen-decomposition of C = Z'Z / n  (Jacobi)
//   3. factor scores F = Z V, loadings and Eq.(5) eigenvalue-weighted weights
// ---------------------------------------------------------------------------
#pragma once

#include <cmath>
#include <vector>

#include "Data.hpp"
#include "Matrix.hpp"
#include "Stats.hpp"
#include "Variable.hpp"

namespace chvi {

class FAMDResult {
public:
    const VariableSchema* schema = nullptr;
    const Dataset* data = nullptr;
    std::vector<int> rows;             // household positions used (may repeat)
    Matrix Z;                          // n x J design matrix
    std::vector<double> eigenvalues;   // descending
    Matrix eigenvectors;               // J x J, columns = principal axes
    double totalInertia = 0.0;         // = N1 + sum(q - 1)

    int nDims() const { return static_cast<int>(eigenvalues.size()); }

    /// Kaiser criterion: number of eigenvalues greater than 1.
    int kaiserCount() const {
        int k = 0;
        for (double l : eigenvalues) k += (l > 1.0) ? 1 : 0;
        return k;
    }

    double explainedPct(int k) const {
        return totalInertia > 0 ? 100.0 * eigenvalues[static_cast<std::size_t>(k)] / totalInertia : 0.0;
    }
    double cumulativePct(int K) const {
        double s = 0.0;
        for (int k = 0; k < K; ++k) s += explainedPct(k);
        return s;
    }

    /// Factor coordinates of the fitted rows on the first K axes (n x K).
    Matrix scores(int K) const {
        const std::size_t n = Z.rows(), J = Z.cols();
        Matrix F(n, static_cast<std::size_t>(K), 0.0);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t k = 0; k < static_cast<std::size_t>(K); ++k) {
                double s = 0.0;
                for (std::size_t j = 0; j < J; ++j) s += Z(i, j) * eigenvectors(j, k);
                F(i, k) = s;
            }
        return F;
    }

    /// Variable-level coordinates L(i,k) on the first K axes (nVars x K) in the
    /// FactoMineR convention (`res.famd$var$coord`, used by the R script):
    ///   continuous : squared correlation  r^2(variable, axis k)
    ///   categorical: correlation ratio    eta^2(variable, axis k)
    /// Both lie in [0,1], so categorical and numeric variables are comparable.
    Matrix loadings(int K) const {
        const Matrix F = scores(K);
        const std::size_t n = F.rows(), nv = schema->size();
        Matrix L(nv, static_cast<std::size_t>(K), 0.0);
        for (std::size_t v = 0; v < nv; ++v) {
            const Variable& var = schema->at(v);
            std::vector<double> x(n);
            for (std::size_t i = 0; i < n; ++i)
                x[i] = data->households[static_cast<std::size_t>(rows[i])].x[v];
            for (std::size_t k = 0; k < static_cast<std::size_t>(K); ++k) {
                const std::vector<double> f = F.column(k);
                if (!var.isCategorical()) {
                    { const double r = stats::pearson(x, f); L(v, k) = r * r; }
                } else {
                    const int q = var.numLevels();
                    std::vector<double> sum(static_cast<std::size_t>(q), 0.0), cnt(static_cast<std::size_t>(q), 0.0);
                    double tot = 0.0, sst = 0.0;
                    for (std::size_t i = 0; i < n; ++i) {
                        const std::size_t c = static_cast<std::size_t>(x[i]);
                        sum[c] += f[i];
                        cnt[c] += 1.0;
                        tot += f[i];
                    }
                    const double gm = tot / static_cast<double>(n);
                    for (std::size_t i = 0; i < n; ++i) sst += (f[i] - gm) * (f[i] - gm);
                    double ssb = 0.0;
                    for (std::size_t c = 0; c < static_cast<std::size_t>(q); ++c)
                        if (cnt[c] > 0) {
                            const double m = sum[c] / cnt[c];
                            ssb += cnt[c] * (m - gm) * (m - gm);
                        }
                    L(v, k) = sst > 0 ? std::min(1.0, ssb / sst) : 0.0;
                }
            }
        }
        return L;
    }

    /// Number of non-null dimensions (= J - number of categorical variables).
    int rank() const {
        int r = 0;
        for (double l : eigenvalues) r += (l > 1e-9) ? 1 : 0;
        return r;
    }

    /// Rank-`ncp` reconstruction of the design matrix.  With `regularise` the
    /// retained axes are shrunk by (lambda_k - s2)/lambda_k, s2 being the mean
    /// of the discarded eigenvalues (regularised iterative PCA, Josse & Husson).
    Matrix reconstruct(int ncp, bool regularise) const {
        const std::size_t n = Z.rows(), J = Z.cols();
        double s2 = 0.0;
        const int rk = rank();
        if (regularise && rk > ncp) {
            for (int k = ncp; k < rk; ++k) s2 += eigenvalues[static_cast<std::size_t>(k)];
            s2 /= static_cast<double>(rk - ncp);
        }
        Matrix Zh(n, J, 0.0);
        const Matrix F = scores(ncp);
        for (int k = 0; k < ncp; ++k) {
            const double lam = eigenvalues[static_cast<std::size_t>(k)];
            const double shrink = lam > 0 ? std::max(0.0, (lam - s2) / lam) : 0.0;
            for (std::size_t i = 0; i < n; ++i)
                for (std::size_t j = 0; j < J; ++j)
                    Zh(i, j) += shrink * F(i, static_cast<std::size_t>(k)) * eigenvectors(j, static_cast<std::size_t>(k));
        }
        return Zh;
    }

    /// Eq.(5): W_i = sum_k |L_ik| * lambda_k / sum_k lambda_k  (k = 1..K).
    std::vector<double> weights(int K) const {
        const Matrix L = loadings(K);
        double lamSum = 0.0;
        for (int k = 0; k < K; ++k) lamSum += eigenvalues[static_cast<std::size_t>(k)];
        std::vector<double> w(L.rows(), 0.0);
        for (std::size_t v = 0; v < L.rows(); ++v) {
            double s = 0.0;
            for (int k = 0; k < K; ++k)
                s += std::fabs(L(v, static_cast<std::size_t>(k))) * eigenvalues[static_cast<std::size_t>(k)];
            w[v] = lamSum > 0 ? s / lamSum : 0.0;
        }
        return w;
    }
};

class FAMD {
public:
    explicit FAMD(const VariableSchema& schema) : schema_(schema) {}

    /// Fit FAMD on the given household positions (duplicates allowed, which is
    /// what the cluster bootstrap needs).
    FAMDResult fit(const Dataset& data, const std::vector<int>& rows) const {
        FAMDResult res;
        res.schema = &schema_;
        res.data = &data;
        res.rows = rows;
        const std::size_t n = rows.size();
        const std::size_t J = static_cast<std::size_t>(schema_.numFAMDColumns());
        res.Z = Matrix(n, J, 0.0);

        std::size_t col = 0;
        for (std::size_t v = 0; v < schema_.size(); ++v) {
            const Variable& var = schema_.at(v);
            if (!var.isCategorical()) {
                double m = 0.0;
                for (std::size_t i = 0; i < n; ++i) m += data.households[static_cast<std::size_t>(rows[i])].x[v];
                m /= static_cast<double>(n);
                double ss = 0.0;
                for (std::size_t i = 0; i < n; ++i) {
                    const double d = data.households[static_cast<std::size_t>(rows[i])].x[v] - m;
                    ss += d * d;
                }
                const double s = std::sqrt(ss / static_cast<double>(n));  // population sd
                for (std::size_t i = 0; i < n; ++i)
                    res.Z(i, col) = s > 0 ? (data.households[static_cast<std::size_t>(rows[i])].x[v] - m) / s : 0.0;
                ++col;
            } else {
                const int q = var.numLevels();
                std::vector<double> p(static_cast<std::size_t>(q), 0.0);
                for (std::size_t i = 0; i < n; ++i)
                    p[static_cast<std::size_t>(data.households[static_cast<std::size_t>(rows[i])].x[v])] += 1.0;
                for (double& pc : p) pc /= static_cast<double>(n);
                for (int c = 0; c < q; ++c, ++col) {
                    const double pc = p[static_cast<std::size_t>(c)];
                    if (pc <= 0.0) continue;  // category absent: column stays 0
                    const double sq = std::sqrt(pc);
                    for (std::size_t i = 0; i < n; ++i) {
                        const int code = static_cast<int>(data.households[static_cast<std::size_t>(rows[i])].x[v]);
                        res.Z(i, col) = ((code == c ? 1.0 : 0.0) - pc) / sq;
                    }
                }
            }
        }

        // C = Z'Z / n
        Matrix C(J, J, 0.0);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t a = 0; a < J; ++a) {
                const double za = res.Z(i, a);
                if (za == 0.0) continue;
                for (std::size_t b = a; b < J; ++b) C(a, b) += za * res.Z(i, b);
            }
        for (std::size_t a = 0; a < J; ++a)
            for (std::size_t b = a; b < J; ++b) {
                C(a, b) /= static_cast<double>(n);
                C(b, a) = C(a, b);
            }
        for (std::size_t a = 0; a < J; ++a) res.totalInertia += C(a, a);
        C.symmetricEigen(res.eigenvalues, res.eigenvectors);
        for (double& l : res.eigenvalues)
            if (l < 0.0 && l > -1e-10) l = 0.0;
        return res;
    }

private:
    const VariableSchema& schema_;
};

}  // namespace chvi
