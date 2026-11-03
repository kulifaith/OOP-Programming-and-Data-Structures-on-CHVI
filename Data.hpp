// Data.hpp
// ---------------------------------------------------------------------------
// Data layer: Household and Child records and the Dataset container with a
// hash-map index (cluster -> household positions).
// ---------------------------------------------------------------------------
#pragma once

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "Portable.hpp"
#include "Variable.hpp"

namespace chvi {

/// The 14 UDHS regions (v024 code -> name) used for the climate linkage.
struct RegionInfo {
    int dhsCode;
    const char* name;  // must match region_name in the weather file
};
inline const std::vector<RegionInfo>& regionTable() {
    static const std::vector<RegionInfo> t = {
        {0, "Kampala"},  {1, "Buganda"},  {3, "Busoga"},   {4, "Bukedi"},   {5, "Bugisu"},
        {6, "Teso (Sebei & Teso)"}, {7, "Karamoja"}, {8, "Lango"}, {9, "Acholi"}, {10, "West Nile"},
        {11, "Bunyoro"}, {12, "Tooro"},   {13, "Ankole"},   {14, "Kigezi"}};
    return t;
}
/// Position of a v024 code in regionTable(), or -1.
inline int regionIndex(int dhsCode) {
    const auto& t = regionTable();
    for (std::size_t i = 0; i < t.size(); ++i)
        if (t[i].dhsCode == dhsCode) return static_cast<int>(i);
    return -1;
}

// ---------------------------------------------------------------------------
struct Household {
    int cluster = 0;  // v001: enumeration area (primary sampling unit)
    int number = 0;   // v002: household number within the cluster
    int region = -1;  // index into regionTable()
    int urban = 0;    // v025: 1 = urban, 0 = rural
    /// Values of the 16 indicators in schema order.  Exposure columns hold
    /// the climate z-score; categorical columns hold the category code
    /// (-1 = missing before imputation).
    std::vector<double> x;
};

struct Child {
    int household = 0;       // POSITION of the household in Dataset::households
    int cluster = 0;
    int fever = -1;          // h22: 1 = yes, 0 = no, -1 = missing / don't know
    double ageMonths = 0.0;  // b19
    double ageZ = 0.0;       // standardised age (as scale(age_months) in R)
    int girl = 0;            // b4: 2 = girl
    int birthOrder = 1;      // bord
    int bwCat = 2;           // 0 = normal (>=2.5 kg), 1 = low (<2.5 kg), 2 = not weighed / unknown
    double radio = 0.0;      // v158 frequency of listening to radio
    double tv = 0.0;         // v159 frequency of watching television
    bool hasFever() const { return fever >= 0; }
};

// ---------------------------------------------------------------------------
class Dataset {
public:
    std::vector<Household> households;
    std::vector<Child> children;
    /// Hash-map index: enumeration-area id -> positions of its households.
    std::unordered_map<int, std::vector<int>> clusterIndex;

    bool empty() const { return households.empty(); }
    std::size_t numHouseholds() const { return households.size(); }
    std::size_t numChildren() const { return children.size(); }
    std::size_t numClusters() const { return clusterIndex.size(); }

    void rebuildIndex() {
        clusterIndex.clear();
        for (std::size_t i = 0; i < households.size(); ++i)
            clusterIndex[households[i].cluster].push_back(static_cast<int>(i));
    }

    /// Sorted cluster ids (deterministic iteration order).
    std::vector<int> clusterIds() const {
        std::vector<int> ids;
        ids.reserve(clusterIndex.size());
        for (const auto& kv : clusterIndex) ids.push_back(kv.first);
        std::sort(ids.begin(), ids.end());
        return ids;
    }

    std::vector<double> column(std::size_t j) const {
        std::vector<double> c;
        c.reserve(households.size());
        for (const auto& h : households) c.push_back(h.x[j]);
        return c;
    }
    std::vector<std::vector<double>> allColumns(std::size_t nVars) const {
        std::vector<std::vector<double>> cols;
        for (std::size_t j = 0; j < nVars; ++j) cols.push_back(column(j));
        return cols;
    }

    /// Number of missing entries in column j (NaN, or code < 0 for categorical columns).
    std::size_t countMissing(std::size_t j, bool categorical) const {
        std::size_t n = 0;
        for (const auto& h : households) n += (std::isnan(h.x[j]) || (categorical && h.x[j] < 0)) ? 1 : 0;
        return n;
    }
};

}  // namespace chvi
