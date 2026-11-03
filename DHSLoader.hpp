// DHSLoader.hpp
// ---------------------------------------------------------------------------
// DHSLoader: C++ port of the data-cleaning part of the R script.  Reads the
// UDHS 2022 Kids Recode (.dta), recodes the raw DHS variables exactly as the
// `case_when()` blocks of the script, links the regional climate z-scores,
// keeps one record per household (distinct(v001, v002)) and builds the
// child-level records used by the fever models.
// ---------------------------------------------------------------------------
#pragma once

#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "Portable.hpp"
#include "Climate.hpp"
#include "Data.hpp"
#include "StataReader.hpp"
#include "Stats.hpp"

namespace chvi {

struct LoadReport {
    std::size_t rawRows = 0, households = 0, clusters = 0, children = 0, childrenWithFever = 0;
    std::vector<std::pair<std::string, std::size_t>> missingBefore;  // variable -> # missing households
};

class DHSLoader {
public:
    struct Options {
        /// In this extract v115 is blank for households with water on the
        /// premises.  The R script treats blanks as missing (and imputes
        /// them); set to true to code them as "On premises" instead.
        bool blankWaterTimeIsOnPremises = false;
    };

    // ------------------- recoding rules (R `case_when` blocks) -------------
    static bool in(double v, std::initializer_list<int> set) {
        for (int s : set)
            if (v == s) return true;
        return false;
    }
    static std::string waterJmp(double v113) {
        if (in(v113, {11, 12, 13, 14, 21, 31, 41, 51, 61, 71, 91, 92})) return "Improved";
        if (in(v113, {32, 42})) return "Unimproved";
        if (v113 == 81) return "Surface Water";
        return "";  // "Other" is not a level of the factor -> NA
    }
    static std::string waterTime(double v115, bool blankIsOnPremises) {
        if (std::isnan(v115)) return blankIsOnPremises ? "On premises" : "";
        if (v115 == 1 || v115 == 2) return "On premises";
        if (v115 >= 3 && v115 <= 30) return "<= 30 minutes";
        if (v115 > 30 && v115 < 998) return "> 30 minutes";
        return "";
    }
    static std::string cookingFuel(double v161) {
        if (in(v161, {1, 2, 3, 4})) return "Clean";
        if (v161 == 5) return "Kerosene";
        if (v161 == 6) return "Coal/Lignite";
        if (v161 == 7) return "Charcoal";
        if (v161 == 8) return "Wood";
        if (in(v161, {9, 10, 11})) return "Biomass other";
        if (v161 == 95) return "No cooking";
        return "Other";
    }
    static std::string roof(double v129) {
        if (in(v129, {12, 13})) return "Natural/Traditional";
        if (v129 == 31) return "Iron Sheets";
        if (in(v129, {22, 23, 24, 34, 35, 96})) return "Modern/Other";
        return "";
    }
    static std::string floorMaterial(double v127) {
        if (in(v127, {11, 12})) return "Natural";
        if (v127 == 34) return "Cement";
        if (in(v127, {33, 35, 36, 37, 96})) return "Modern/Other";
        return "";
    }
    static std::string wall(double v128) {
        if (in(v128, {11, 12, 13, 24, 25, 26})) return "Natural/Temporary";
        if (in(v128, {21, 22, 23})) return "Unimproved Traditional";
        if (in(v128, {27, 28, 35, 36, 96})) return "Semi-Durable";
        if (in(v128, {31, 32, 33, 34})) return "Durable/Modern";
        return "";
    }
    static std::string toilet(double v116) {
        if (in(v116, {11, 12, 13, 21, 22, 31})) return "Improved";
        if (in(v116, {14, 15, 23, 41, 51})) return "Unimproved";
        if (v116 == 61) return "Open Defecation";
        if (v116 == 96) return "Other / Unknown";
        return "";
    }
    /// 0 = normal (>= 2.5 kg), 1 = low (< 2.5 kg), 2 = not weighed / unknown.
    static int birthWeightCategory(double m19) {
        if (!std::isnan(m19) && m19 >= 2500 && m19 < 9996) return 0;
        if (!std::isnan(m19) && m19 > 0 && m19 < 2500) return 1;
        return 2;
    }

    // ------------------------------- loader --------------------------------
    static Dataset load(const std::string& dtaPath, const ClimateIndicators& climate, VariableSchema& schema,
                        const Options& opt, LoadReport& report) {
        StataReader dta(dtaPath);
        static const char* needed[] = {"v001", "v002", "v024", "v025", "v113", "v115", "v161", "v129", "v127",
                                       "v128", "v116", "v106", "v169a", "q930a", "v745b", "v190", "h22", "b19",
                                       "b4", "bord", "m19_1", "v158", "v159"};
        std::map<std::string, std::vector<double>> col;
        for (const char* nm : needed) col[nm] = dta.numeric(nm);
        const std::size_t n = dta.numObs();
        report = LoadReport();
        report.rawRows = n;

        // cooking-fuel categories are data driven (sorted labels, like R's factor())
        std::set<std::string> fuelLabels;
        for (std::size_t i = 0; i < n; ++i) fuelLabels.insert(cookingFuel(col["v161"][i]));
        schema.categorical(VariableSchema::kFuel).setLevels(std::vector<std::string>(fuelLabels.begin(), fuelLabels.end()));

        Dataset d;
        std::unordered_map<long long, int> hhPos;  // (v001, v002) -> household position
        std::vector<int> rowHousehold(n, -1);
        for (std::size_t i = 0; i < n; ++i) {
            const long long key = static_cast<long long>(col["v001"][i]) * 1000000LL + static_cast<long long>(col["v002"][i]);
            auto it = hhPos.find(key);
            if (it == hhPos.end()) {
                Household h;
                h.cluster = static_cast<int>(col["v001"][i]);
                h.number = static_cast<int>(col["v002"][i]);
                h.region = regionIndex(static_cast<int>(col["v024"][i]));
                if (h.region < 0) throw std::runtime_error("unknown region code v024 in .dta");
                h.urban = (col["v025"][i] == 1) ? 1 : 0;
                h.x.assign(schema.size(), kNaN);
                const RegionClimate* rc = climate.find(regionTable()[static_cast<std::size_t>(h.region)].name);
                if (!rc) throw std::runtime_error(std::string("no climate data for region ") + regionTable()[static_cast<std::size_t>(h.region)].name);
                for (std::size_t k = 0; k < 4; ++k) h.x[k] = rc->z[k];
                auto setCat = [&](std::size_t v, const std::string& label) {
                    h.x[v] = label.empty() ? -1.0 : static_cast<double>(schema.categorical(v).codeOf(label));
                };
                auto setCode = [&](std::size_t v, double raw, double offset, int maxCode) {
                    const int code = static_cast<int>(raw - offset);
                    h.x[v] = (std::isnan(raw) || code < 0 || code > maxCode) ? -1.0 : code;
                };
                setCat(VariableSchema::kFuel, cookingFuel(col["v161"][i]));
                setCat(VariableSchema::kRoof, roof(col["v129"][i]));
                setCat(VariableSchema::kFloor, floorMaterial(col["v127"][i]));
                setCat(VariableSchema::kWall, wall(col["v128"][i]));
                setCode(VariableSchema::kWealth, col["v190"][i], 1, 4);
                setCode(VariableSchema::kEduc, col["v106"][i], 0, 3);
                setCat(VariableSchema::kWater, waterJmp(col["v113"][i]));
                setCat(VariableSchema::kWaterTime, waterTime(col["v115"][i], opt.blankWaterTimeIsOnPremises));
                setCat(VariableSchema::kToilet, toilet(col["v116"][i]));
                setCode(VariableSchema::kPhone, col["v169a"][i], 0, 1);
                setCode(VariableSchema::kBank, col["q930a"][i], 1, 1);   // 1 = Yes, 2 = No
                setCode(VariableSchema::kLand, col["v745b"][i], 0, 3);
                it = hhPos.emplace(key, static_cast<int>(d.households.size())).first;
                d.households.push_back(std::move(h));
            }
            rowHousehold[i] = it->second;
        }
        // categorical variables with levels that never occur would add empty design columns
        for (std::size_t v = 0; v < schema.size(); ++v)
            if (schema.at(v).isCategorical())
                report.missingBefore.push_back({schema.at(v).name(), d.countMissing(v, true)});

        // child-level records
        d.children.resize(n);
        std::vector<double> ages;
        for (std::size_t i = 0; i < n; ++i) {
            Child& c = d.children[i];
            c.household = rowHousehold[i];
            c.cluster = d.households[static_cast<std::size_t>(c.household)].cluster;
            const double h22 = col["h22"][i];
            c.fever = (h22 == 1) ? 1 : (h22 == 2 ? 0 : -1);
            c.ageMonths = col["b19"][i];
            c.girl = (col["b4"][i] == 2) ? 1 : 0;
            c.birthOrder = static_cast<int>(col["bord"][i]);
            c.bwCat = birthWeightCategory(col["m19_1"][i]);
            c.radio = std::isnan(col["v158"][i]) ? 0.0 : col["v158"][i];
            c.tv = std::isnan(col["v159"][i]) ? 0.0 : col["v159"][i];
            ages.push_back(c.ageMonths);
            report.childrenWithFever += c.hasFever() ? 1 : 0;
        }
        const double am = stats::mean(ages), as = stats::sd(ages);
        for (auto& c : d.children) c.ageZ = as > 0 ? (c.ageMonths - am) / as : 0.0;  // scale(age_months)

        d.rebuildIndex();
        report.households = d.numHouseholds();
        report.clusters = d.numClusters();
        report.children = d.numChildren();
        return d;
    }
};

}  // namespace chvi
