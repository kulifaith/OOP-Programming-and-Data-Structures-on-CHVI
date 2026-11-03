// Robustness.hpp
// ---------------------------------------------------------------------------
// RobustnessAnalyzer: the three sensitivity analyses of Section 3.7.4.
//   1. Component retention : CHVI with K = 4 vs an alternative K (default 12)
//   2. Cluster bootstrap   : B resamples of enumeration areas, FAMD refitted
//                            each time; weights applied to the full sample
//   3. Dimension weighting : 50/50 cluster-level split, OLS of fever on
//                            (E, S, A) in training clusters -> empirical
//                            dimension weights -> regression-weighted CHVI
// All comparisons use Spearman rank correlation.
// ---------------------------------------------------------------------------
#pragma once

#include <algorithm>
#include <unordered_set>
#include <vector>

#include "Portable.hpp"
#include "Data.hpp"
#include "FAMD.hpp"
#include "Index.hpp"
#include "Stats.hpp"

namespace chvi {

struct RobustnessReport {
    // 1. component retention
    int kPrimary = 0, kAlternative = 0, kaiser = 0;
    double rhoComponents = 0.0;
    // 2. cluster bootstrap
    std::vector<double> bootRho;
    double bootMedian = 0.0, bootMin = 0.0, bootMax = 0.0;
    // 3. regression weights
    double wE = 0.0, wS = 0.0, wA = 0.0;
    int trainClusters = 0, testClusters = 0, trainChildren = 0, testHouseholds = 0;
    double rhoRegression = 0.0;
    // repeated random splits (stability of the empirical weights)
    std::vector<double> splitRho;
    double splitMedianRho = 0.0, splitMinRho = 0.0, splitMaxRho = 0.0;
    double wEmin = 0, wEmax = 0, wSmin = 0, wSmax = 0, wAmin = 0, wAmax = 0;
};

class RobustnessAnalyzer {
public:
    RobustnessAnalyzer(const Dataset& data, const VariableSchema& schema, const IndexBuilder& builder,
                       std::uint64_t seed)
        : data_(data), schema_(schema), builder_(builder), seed_(seed) {}

    /// Runs the three checks given the primary FAMD fit and primary CHVI.
    RobustnessReport run(const FAMDResult& primaryFit, const Matrix& Y, const CHVIResult& primary,
                         int kPrimary, int bootstrapB, int altK = 12) const {
        RobustnessReport rep;
        rep.kPrimary = kPrimary;
        componentRetention(primaryFit, Y, primary, rep, altK);
        clusterBootstrap(Y, primary, kPrimary, bootstrapB, rep);
        regressionWeighting(primary, rep);
        return rep;
    }

    // 1. ----------------------------------------------------------------
    void componentRetention(const FAMDResult& fit, const Matrix& Y, const CHVIResult& primary,
                            RobustnessReport& rep, int altK = 12) const {
        rep.kaiser = fit.kaiserCount();
        rep.kAlternative = std::max(1, std::min(altK, fit.rank()));
        const CHVIResult alt = builder_.build(Y, fit.weights(rep.kAlternative));
        rep.rhoComponents = stats::spearman(primary.raw, alt.raw);
    }

    // 2. ----------------------------------------------------------------
    void clusterBootstrap(const Matrix& Y, const CHVIResult& primary, int K, int B,
                          RobustnessReport& rep) const {
        Rng rng(seed_ * 7919ULL + 13ULL);
        const std::vector<int> ids = data_.clusterIds();
        const FAMD famd(schema_);
        rep.bootRho.clear();
        for (int b = 0; b < B; ++b) {
            std::vector<int> rows;
            for (std::size_t c = 0; c < ids.size(); ++c) {
                const int pick = ids[static_cast<std::size_t>(rng.integer(static_cast<int>(ids.size())))];
                const auto& members = data_.clusterIndex.at(pick);
                rows.insert(rows.end(), members.begin(), members.end());
            }
            const FAMDResult f = famd.fit(data_, rows);
            const CHVIResult c = builder_.build(Y, f.weights(K));  // score the FULL sample
            rep.bootRho.push_back(stats::spearman(primary.raw, c.raw));
        }
        if (!rep.bootRho.empty()) {
            rep.bootMedian = stats::median(rep.bootRho);
            rep.bootMin = *std::min_element(rep.bootRho.begin(), rep.bootRho.end());
            rep.bootMax = *std::max_element(rep.bootRho.begin(), rep.bootRho.end());
        }
    }

    // 3. ----------------------------------------------------------------
    /// One 50/50 cluster-level split: OLS of fever on (E,S,A) in the training
    /// clusters, empirical weights |b|/sum|b|, CHVI_Reg on the test households.
    /// The first split is reported in detail; `repeats` further random splits
    /// show how stable the empirical weights are.
    void regressionWeighting(const CHVIResult& primary, RobustnessReport& rep, int repeats = 100) const {
        Rng rng(seed_ * 104729ULL + 7ULL);
        const std::vector<int> ids = data_.clusterIds();
        rep.splitRho.clear();
        std::vector<double> wEs, wSs, wAs;
        for (int r = 0; r < std::max(1, repeats); ++r) {
            std::vector<int> shuffled = ids;
            rng.shuffle(shuffled);
            const std::size_t nTrain = shuffled.size() / 2;
            std::unordered_set<int> train(shuffled.begin(), shuffled.begin() + static_cast<long>(nTrain));

            std::vector<std::size_t> kids;  // training children with observed fever
            for (std::size_t i = 0; i < data_.children.size(); ++i) {
                const Child& c = data_.children[i];
                if (c.hasFever() && train.count(c.cluster)) kids.push_back(i);
            }
            Matrix X(kids.size(), 4, 1.0);
            std::vector<double> y(kids.size());
            for (std::size_t k = 0; k < kids.size(); ++k) {
                const Child& c = data_.children[kids[k]];
                const std::size_t h = static_cast<std::size_t>(c.household);
                X(k, 1) = primary.dim.E[h];
                X(k, 2) = primary.dim.S[h];
                X(k, 3) = primary.dim.A[h];
                y[k] = c.fever;
            }
            const std::vector<double> b = stats::ols(X, y);
            const double tot = std::fabs(b[1]) + std::fabs(b[2]) + std::fabs(b[3]);
            const double wE = tot > 0 ? std::fabs(b[1]) / tot : 1.0 / 3.0;
            const double wS = tot > 0 ? std::fabs(b[2]) / tot : 1.0 / 3.0;
            const double wA = tot > 0 ? std::fabs(b[3]) / tot : 1.0 / 3.0;

            std::vector<double> eq, rg;  // held-out households
            for (std::size_t h = 0; h < data_.households.size(); ++h) {
                if (train.count(data_.households[h].cluster)) continue;
                eq.push_back((primary.dim.E[h] + primary.dim.S[h] - primary.dim.A[h]) / 3.0);
                rg.push_back(wE * primary.dim.E[h] + wS * primary.dim.S[h] - wA * primary.dim.A[h]);
            }
            const double rho = stats::spearman(eq, rg);
            rep.splitRho.push_back(rho);
            wEs.push_back(wE); wSs.push_back(wS); wAs.push_back(wA);
            if (r == 0) {  // detailed report of the first split
                rep.trainClusters = static_cast<int>(nTrain);
                rep.testClusters = static_cast<int>(ids.size() - nTrain);
                rep.trainChildren = static_cast<int>(kids.size());
                rep.testHouseholds = static_cast<int>(eq.size());
                rep.wE = wE; rep.wS = wS; rep.wA = wA;
                rep.rhoRegression = rho;
            }
        }
        rep.splitMedianRho = stats::median(rep.splitRho);
        rep.splitMinRho = *std::min_element(rep.splitRho.begin(), rep.splitRho.end());
        rep.splitMaxRho = *std::max_element(rep.splitRho.begin(), rep.splitRho.end());
        rep.wEmin = *std::min_element(wEs.begin(), wEs.end()); rep.wEmax = *std::max_element(wEs.begin(), wEs.end());
        rep.wSmin = *std::min_element(wSs.begin(), wSs.end()); rep.wSmax = *std::max_element(wSs.begin(), wSs.end());
        rep.wAmin = *std::min_element(wAs.begin(), wAs.end()); rep.wAmax = *std::max_element(wAs.begin(), wAs.end());
    }

private:
    const Dataset& data_;
    const VariableSchema& schema_;
    const IndexBuilder& builder_;
    std::uint64_t seed_;
};

}  // namespace chvi
