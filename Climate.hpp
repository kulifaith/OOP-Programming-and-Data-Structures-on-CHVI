// Climate.hpp
// ---------------------------------------------------------------------------
// WeatherProcessor: C++ port of the climate part of the R script.
//   daily records  -> drop exact duplicate rows (R: distinct())
//                  -> one row per region-date (mean temp/humidity/wind, SUM
//                     of precipitation)
//                  -> annual regional values (mean temp, total precipitation,
//                     mean humidity, mean wind)
//                  -> 1991-2020 baseline mean and SD, 2021-2022 mean
//                  -> standardised anomaly  z = (recent - baseline)/SD   (Eq. 2)
// ---------------------------------------------------------------------------
#pragma once

#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "Portable.hpp"
#include "Stats.hpp"

namespace chvi {

/// Climate summary of one region.
struct RegionClimate {
    std::string name;
    std::array<double, 4> baselineMean{}, baselineSd{}, recentMean{}, anomaly{}, z{};
    int baselineYears = 0, recentYears = 0;
};

class ClimateIndicators {
public:
    static constexpr int kVars = 4;
    static const char* varLabel(int v) {
        static const char* n[kVars] = {"Temperature (C)", "Precipitation (mm/yr)", "Humidity (%)", "Wind (m/s)"};
        return n[v];
    }

    std::vector<RegionClimate> regions;  // sorted by name
    std::size_t rawRows = 0, duplicateRows = 0, regionDayRows = 0;

    bool empty() const { return regions.empty(); }
    const RegionClimate* find(const std::string& name) const {
        for (const auto& r : regions)
            if (r.name == name) return &r;
        return nullptr;
    }
};

class WeatherProcessor {
public:
    /// Reads weather_Uganda_daily_1991_2022.csv (columns: region_name, date
    /// (m/d/yyyy), temp_mean, humidity_mean, precip_total, wind_mean).
    static ClimateIndicators process(const std::string& csvPath) {
        std::ifstream f(csvPath);
        if (!f) throw std::runtime_error("cannot open weather file: " + csvPath);
        ClimateIndicators out;
        std::string line;
        std::getline(f, line);  // header

        struct DayAcc { double t = 0, h = 0, p = 0, w = 0; int n = 0; };
        std::unordered_map<std::string, DayAcc> daily;   // key: region|yyyymmdd
        std::unordered_set<std::string> seen;            // exact duplicate rows
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            if (!line.empty() && line.back() == '\r') line.pop_back();
            ++out.rawRows;
            if (!seen.insert(line).second) { ++out.duplicateRows; continue; }
            std::vector<std::string> c = split(line);
            if (c.size() < 6) continue;
            int m = 0, d = 0, y = 0;
            if (std::sscanf(c[1].c_str(), "%d/%d/%d", &m, &d, &y) != 3) continue;
            DayAcc& a = daily[c[0] + "|" + chvi::str(y * 10000 + m * 100 + d)];
            a.t += std::atof(c[2].c_str());
            a.h += std::atof(c[3].c_str());
            a.p += std::atof(c[4].c_str());  // precipitation: SUM over duplicates (as in R)
            a.w += std::atof(c[5].c_str());
            ++a.n;
        }
        out.regionDayRows = daily.size();

        // annual aggregates per region-year
        struct YearAcc { double t = 0, p = 0, h = 0, w = 0; int n = 0; };
        std::map<std::pair<std::string, int>, YearAcc> annual;
        for (const auto& kv : daily) {
            const std::size_t bar = kv.first.find('|');
            const std::string region = kv.first.substr(0, bar);
            const int year = std::atoi(kv.first.substr(bar + 1).c_str()) / 10000;
            YearAcc& y = annual[{region, year}];
            y.t += kv.second.t / kv.second.n;
            y.h += kv.second.h / kv.second.n;
            y.w += kv.second.w / kv.second.n;
            y.p += kv.second.p;
            ++y.n;
        }
        // baseline (1991-2020) and recent (2021-2022) summaries
        std::map<std::string, std::array<std::vector<double>, 4>> base, recent;
        for (const auto& kv : annual) {
            const std::string& region = kv.first.first;
            const int year = kv.first.second;
            const std::array<double, 4> v = {kv.second.t / kv.second.n, kv.second.p, kv.second.h / kv.second.n,
                                             kv.second.w / kv.second.n};
            for (int k = 0; k < 4; ++k) {
                if (year >= 1991 && year <= 2020) base[region][static_cast<std::size_t>(k)].push_back(v[static_cast<std::size_t>(k)]);
                if (year == 2021 || year == 2022) recent[region][static_cast<std::size_t>(k)].push_back(v[static_cast<std::size_t>(k)]);
            }
        }
        for (const auto& kv : base) {
            RegionClimate rc;
            rc.name = kv.first;
            rc.baselineYears = static_cast<int>(kv.second[0].size());
            rc.recentYears = static_cast<int>(recent[kv.first][0].size());
            for (std::size_t k = 0; k < 4; ++k) {
                rc.baselineMean[k] = stats::mean(kv.second[k]);
                rc.baselineSd[k] = stats::sd(kv.second[k]);
                rc.recentMean[k] = stats::mean(recent[kv.first][k]);
                rc.anomaly[k] = rc.recentMean[k] - rc.baselineMean[k];
                rc.z[k] = rc.baselineSd[k] > 0 ? rc.anomaly[k] / rc.baselineSd[k] : 0.0;
            }
            out.regions.push_back(rc);
        }
        if (out.regions.empty()) throw std::runtime_error("weather file contained no usable rows");
        return out;
    }

private:
    static std::vector<std::string> split(const std::string& line) {
        std::vector<std::string> out;
        std::stringstream ss(line);
        std::string tok;
        while (std::getline(ss, tok, ',')) out.push_back(tok);
        return out;
    }
};

}  // namespace chvi
