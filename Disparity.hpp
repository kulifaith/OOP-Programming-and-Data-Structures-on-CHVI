// Disparity.hpp
// ---------------------------------------------------------------------------
// ChiSquareTest: Pearson chi-square test of independence on an r x c
// contingency table with Cramer's V effect size (Section 3.7.6).
// ---------------------------------------------------------------------------
#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "Stats.hpp"

namespace chvi {

struct ChiSquareResult {
    std::vector<std::vector<double>> observed, expected, colPct;
    double chi2 = 0.0;
    int df = 0;
    double pValue = 1.0;
    double cramersV = 0.0;
    int cellsExpectedBelow5 = 0;
};

class ChiSquareTest {
public:
    static ChiSquareResult run(const std::vector<std::vector<double>>& obs) {
        ChiSquareResult r;
        r.observed = obs;
        const std::size_t R = obs.size(), C = R ? obs[0].size() : 0;
        std::vector<double> rowSum(R, 0.0), colSum(C, 0.0);
        double N = 0.0;
        for (std::size_t i = 0; i < R; ++i)
            for (std::size_t j = 0; j < C; ++j) {
                rowSum[i] += obs[i][j];
                colSum[j] += obs[i][j];
                N += obs[i][j];
            }
        r.expected.assign(R, std::vector<double>(C, 0.0));
        r.colPct.assign(R, std::vector<double>(C, 0.0));
        int usedRows = 0, usedCols = 0;
        for (std::size_t i = 0; i < R; ++i) usedRows += rowSum[i] > 0 ? 1 : 0;
        for (std::size_t j = 0; j < C; ++j) usedCols += colSum[j] > 0 ? 1 : 0;
        for (std::size_t i = 0; i < R; ++i)
            for (std::size_t j = 0; j < C; ++j) {
                r.colPct[i][j] = colSum[j] > 0 ? 100.0 * obs[i][j] / colSum[j] : 0.0;
                if (N <= 0 || rowSum[i] <= 0 || colSum[j] <= 0) continue;
                const double e = rowSum[i] * colSum[j] / N;
                r.expected[i][j] = e;
                r.chi2 += (obs[i][j] - e) * (obs[i][j] - e) / e;
                if (e < 5.0) ++r.cellsExpectedBelow5;
            }
        r.df = std::max(0, (usedRows - 1) * (usedCols - 1));
        r.pValue = r.df > 0 ? stats::chiSquareSF(r.chi2, r.df) : 1.0;
        const int m = std::min(usedRows, usedCols) - 1;
        r.cramersV = (N > 0 && m > 0) ? std::sqrt(r.chi2 / (N * m)) : 0.0;
        return r;
    }
};

}  // namespace chvi
