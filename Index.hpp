// Index.hpp
// ---------------------------------------------------------------------------
// Construction of the Climate-Health Vulnerability Index.
//   Eq.(3)/(4) normalisation  -> normalisedMatrix()
//   Eq.(6)     dimension index -> dimensionIndices()
//   Eq.(7)/(8) composite       -> composite()
//   Eq.(9)     z-score         -> build()
// plus the five-level VulnerabilityClassifier (binary search over cut-points).
// ---------------------------------------------------------------------------
#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "Data.hpp"
#include "Matrix.hpp"
#include "Stats.hpp"
#include "Variable.hpp"

namespace chvi {

/// Maps a standardised CHVI score to one of five ordered levels.
/// The cut-points are held in a sorted vector and located by binary search
/// (std::upper_bound), O(log k) per household.
class VulnerabilityClassifier {
public:
    explicit VulnerabilityClassifier(std::vector<double> cuts = {-1.5, -0.5, 0.5, 1.5})
        : cuts_(std::move(cuts)) {
        std::sort(cuts_.begin(), cuts_.end());
    }
    int classify(double z) const {
        return static_cast<int>(std::upper_bound(cuts_.begin(), cuts_.end(), z) - cuts_.begin());
    }
    int numLevels() const { return static_cast<int>(cuts_.size()) + 1; }
    const std::vector<double>& cuts() const { return cuts_; }
    static const char* label(int level) {
        static const char* names[5] = {"Very Low", "Low", "Moderate", "High", "Very High"};
        return (level >= 0 && level < 5) ? names[level] : "?";
    }

private:
    std::vector<double> cuts_;
};

struct DimensionScores {
    std::vector<double> E, S, A;
};

struct CHVIResult {
    DimensionScores dim;          // Exposure, Sensitivity, Adaptive indices (Eq. 6)
    std::vector<double> raw;      // Eq. (8)
    std::vector<double> z;        // Eq. (9) standardised CHVI*
    std::vector<int> level;       // 0..4
    std::vector<double> weights;  // FAMD variable weights used (Eq. 5)
    double mean = 0.0, sd = 1.0;  // of raw scores
};

class IndexBuilder {
public:
    IndexBuilder(const VariableSchema& schema, VulnerabilityClassifier classifier)
        : schema_(schema), classifier_(std::move(classifier)) {}

    const VulnerabilityClassifier& classifier() const { return classifier_; }

    /// n x p matrix of oriented scores Y in [0,1]: min-max (Eq. 3/4) for the
    /// climate z-scores and the explicit category scores for the others.
    /// Categories without a score (NaN) take the column median, as in the R script.
    Matrix normalisedMatrix(const Dataset& d) const {
        Matrix Y(d.numHouseholds(), schema_.size(), 0.0);
        for (std::size_t j = 0; j < schema_.size(); ++j) {
            std::vector<double> col(d.numHouseholds());
            std::vector<double> valid;
            for (std::size_t i = 0; i < d.numHouseholds(); ++i) {
                col[i] = schema_.at(j).normalise(d.households[i].x[j]);
                if (!std::isnan(col[i])) valid.push_back(col[i]);
            }
            const double med = stats::median(valid);
            for (std::size_t i = 0; i < d.numHouseholds(); ++i) Y(i, j) = std::isnan(col[i]) ? med : col[i];
        }
        return Y;
    }

    /// Number of households whose score was missing and replaced by the median.
    std::size_t countMedianImputed(const Dataset& d, std::size_t j) const {
        std::size_t n = 0;
        for (const auto& h : d.households) n += std::isnan(schema_.at(j).normalise(h.x[j])) ? 1 : 0;
        return n;
    }

    /// Eq. (6): weighted average of the normalised variables of each domain.
    DimensionScores dimensionIndices(const Matrix& Y, const std::vector<double>& w) const {
        DimensionScores ds;
        auto one = [&](Domain dom) {
            const std::vector<std::size_t> idx = schema_.indicesOf(dom);
            double wsum = 0.0;
            for (std::size_t j : idx) wsum += w[j];
            std::vector<double> out(Y.rows(), 0.0);
            for (std::size_t i = 0; i < Y.rows(); ++i) {
                double s = 0.0;
                for (std::size_t j : idx) s += (wsum > 0 ? w[j] : 1.0) * Y(i, j);
                out[i] = s / (wsum > 0 ? wsum : static_cast<double>(idx.size()));
            }
            return out;
        };
        ds.E = one(Domain::Exposure);
        ds.S = one(Domain::Sensitivity);
        ds.A = one(Domain::Adaptive);
        return ds;
    }

    /// Eq. (7): alpha*E + beta*S - gamma*A  (Eq. 8 when all weights are 1/3).
    static std::vector<double> composite(const DimensionScores& d, double wE = 1.0 / 3.0,
                                         double wS = 1.0 / 3.0, double wA = 1.0 / 3.0) {
        std::vector<double> raw(d.E.size());
        for (std::size_t i = 0; i < raw.size(); ++i) raw[i] = wE * d.E[i] + wS * d.S[i] - wA * d.A[i];
        return raw;
    }

    /// Full pipeline for one weight vector: indices -> composite -> z -> levels.
    CHVIResult build(const Matrix& Y, const std::vector<double>& w) const {
        CHVIResult r;
        r.weights = w;
        r.dim = dimensionIndices(Y, w);
        r.raw = composite(r.dim);
        r.mean = stats::mean(r.raw);
        r.sd = stats::sd(r.raw);
        r.z.resize(r.raw.size());
        r.level.resize(r.raw.size());
        for (std::size_t i = 0; i < r.raw.size(); ++i) {
            r.z[i] = r.sd > 0 ? (r.raw[i] - r.mean) / r.sd : 0.0;  // Eq. (9)
            r.level[i] = classifier_.classify(r.z[i]);
        }
        return r;
    }

private:
    const VariableSchema& schema_;
    VulnerabilityClassifier classifier_;
};

}  // namespace chvi
