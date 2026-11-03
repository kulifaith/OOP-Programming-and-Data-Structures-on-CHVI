// Imputer.hpp
// ---------------------------------------------------------------------------
// FAMDImputer: replacement for missMDA::imputeFAMD(hh, ncp = 2) used in the R
// script.  Missing categorical values are filled by the iterative
// regularised-FAMD algorithm:
//   1. start with the modal category,
//   2. fit FAMD, reconstruct the indicator matrix from the first `ncp` axes
//      (with shrinkage of the singular values),
//   3. give every missing cell the category with the largest reconstructed
//      indicator value,
//   4. repeat until no imputed value changes.
// (missMDA uses a very similar scheme; results can differ in a few cells.)
// ---------------------------------------------------------------------------
#pragma once

#include <cmath>
#include <vector>

#include "Portable.hpp"
#include "Data.hpp"
#include "FAMD.hpp"

namespace chvi {

struct ImputationReport {
    int iterations = 0;
    bool converged = false;
    struct Cell { std::size_t household, variable; int code; };
    std::vector<Cell> cells;  // every imputed cell and its final category
};

class FAMDImputer {
public:
    static ImputationReport impute(Dataset& data, const VariableSchema& schema, int ncp = 2, int maxIter = 100) {
        ImputationReport rep;
        const std::size_t n = data.numHouseholds();
        std::vector<int> rows(n);
        for (std::size_t i = 0; i < n; ++i) rows[i] = static_cast<int>(i);

        // first design column of every variable
        std::vector<std::size_t> firstCol(schema.size());
        std::size_t col = 0;
        for (std::size_t v = 0; v < schema.size(); ++v) {
            firstCol[v] = col;
            col += schema.at(v).isCategorical() ? static_cast<std::size_t>(schema.at(v).numLevels()) : 1;
        }

        // missing cells and modal initialisation
        for (std::size_t v = 0; v < schema.size(); ++v) {
            if (!schema.at(v).isCategorical()) continue;
            std::vector<int> cnt(static_cast<std::size_t>(schema.at(v).numLevels()), 0);
            for (const auto& h : data.households)
                if (h.x[v] >= 0) ++cnt[static_cast<std::size_t>(h.x[v])];
            const int mode = static_cast<int>(std::max_element(cnt.begin(), cnt.end()) - cnt.begin());
            for (std::size_t i = 0; i < n; ++i)
                if (data.households[i].x[v] < 0 || std::isnan(data.households[i].x[v])) {
                    rep.cells.push_back({i, v, mode});
                    data.households[i].x[v] = mode;
                }
        }
        if (rep.cells.empty()) { rep.converged = true; return rep; }

        const FAMD famd(schema);
        for (int it = 0; it < maxIter; ++it) {
            const FAMDResult fit = famd.fit(data, rows);
            const Matrix Zh = fit.reconstruct(ncp, true);
            // category proportions of every variable on the current completed data
            std::vector<std::vector<double>> prop(schema.size());
            for (std::size_t v = 0; v < schema.size(); ++v) {
                if (!schema.at(v).isCategorical()) continue;
                prop[v].assign(static_cast<std::size_t>(schema.at(v).numLevels()), 0.0);
                for (const auto& h : data.households) prop[v][static_cast<std::size_t>(h.x[v])] += 1.0;
                for (double& p : prop[v]) p /= static_cast<double>(n);
            }
            int changed = 0;
            for (auto& c : rep.cells) {
                const std::size_t q = static_cast<std::size_t>(schema.at(c.variable).numLevels());
                int best = c.code;
                double bestVal = -1e300;
                for (std::size_t k = 0; k < q; ++k) {
                    const double p = prop[c.variable][k];
                    if (p <= 0.0) continue;
                    const double val = Zh(c.household, firstCol[c.variable] + k) * std::sqrt(p) + p;
                    if (val > bestVal) { bestVal = val; best = static_cast<int>(k); }
                }
                if (best != c.code) { c.code = best; ++changed; }
                data.households[c.household].x[c.variable] = c.code;
            }
            rep.iterations = it + 1;
            if (changed == 0) { rep.converged = true; break; }
        }
        return rep;
    }
};

}  // namespace chvi
