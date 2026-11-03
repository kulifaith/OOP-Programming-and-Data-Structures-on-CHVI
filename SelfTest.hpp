// SelfTest.hpp
// ---------------------------------------------------------------------------
// Built-in verification suite (menu option 14).  Every numerical building
// block is checked against a value that is known analytically or can be
// computed by hand; the real-data pipeline is checked against (a) counts that
// were obtained independently with pandas and (b) the results reported in
// Chapter 4 of the proposal (produced with R / FactoMineR / lme4).
// Returns the number of FAILED checks (0 = all passed).
// ---------------------------------------------------------------------------
#pragma once

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "Portable.hpp"
#include "Climate.hpp"
#include "DHSLoader.hpp"
#include "Data.hpp"
#include "DataStructures.hpp"
#include "Disparity.hpp"
#include "FAMD.hpp"
#include "Index.hpp"
#include "Matrix.hpp"
#include "MixedLogit.hpp"
#include "ResearchSystem.hpp"
#include "StataReader.hpp"
#include "Stats.hpp"
#include "Variable.hpp"

namespace chvi {

class SelfTest {
public:
    SelfTest(std::ostream& os, const Settings& settings) : os_(os), settings_(settings) {}

    int run() {
        os_ << "\n" << std::string(78, '=') << "\n SELF-TEST SUITE\n" << std::string(78, '=') << "\n";
        testLinearAlgebra();
        testStatistics();
        testVariables();
        testRecoding();
        testWeatherProcessor();
        testChiSquare();
        testMixedLogit();
        testDataStructures();
        testRealData();
        os_ << std::string(78, '-') << "\n" << " " << passed_ << " / " << (passed_ + failed_) << " checks passed";
        if (skipped_) os_ << "   (" << skipped_ << " skipped: data files not found)";
        if (failed_) os_ << "   (" << failed_ << " FAILED)";
        os_ << "\n";
        return failed_;
    }

private:
    std::ostream& os_;
    Settings settings_;
    int passed_ = 0, failed_ = 0, skipped_ = 0;

    void check(const std::string& name, bool ok, const std::string& detail = "") {
        (ok ? passed_ : failed_)++;
        os_ << (ok ? "  [PASS] " : "  [FAIL] ") << name;
        if (!detail.empty()) os_ << "  (" << detail << ")";
        os_ << "\n";
    }
    static bool near(double a, double b, double tol) { return std::fabs(a - b) <= tol; }
    static std::string num(double v, int p = 6) {
        std::ostringstream s;
        s << std::setprecision(p) << v;
        return s.str();
    }

    void testLinearAlgebra() {
        os_ << "Linear algebra\n";
        Matrix A(2, 2);
        A(0, 0) = 2; A(0, 1) = 1; A(1, 0) = 1; A(1, 1) = 2;
        std::vector<double> ev;
        Matrix V;
        A.symmetricEigen(ev, V);
        check("Jacobi eigenvalues of [[2,1],[1,2]] are 3 and 1", near(ev[0], 3, 1e-10) && near(ev[1], 1, 1e-10),
              num(ev[0]) + ", " + num(ev[1]));

        Rng rng(5);
        const std::size_t n = 7;
        Matrix S(n, n);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = i; j < n; ++j) S(i, j) = S(j, i) = rng.normal();
        S.symmetricEigen(ev, V);
        Matrix D(n, n, 0.0);
        for (std::size_t i = 0; i < n; ++i) D(i, i) = ev[i];
        const Matrix R = V.multiply(D).multiply(V.transpose());
        double err = 0.0;
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j) err = std::max(err, std::fabs(R(i, j) - S(i, j)));
        check("V D V' reconstructs a random 7x7 symmetric matrix", err < 1e-10, "max err " + num(err, 3));
        const Matrix VtV = V.transpose().multiply(V);
        double orth = 0.0;
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j) orth = std::max(orth, std::fabs(VtV(i, j) - (i == j ? 1.0 : 0.0)));
        check("eigenvectors are orthonormal", orth < 1e-10, "max dev " + num(orth, 3));

        const Matrix Inv = S.inverse();
        const Matrix I = S.multiply(Inv);
        double ie = 0.0;
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j) ie = std::max(ie, std::fabs(I(i, j) - (i == j ? 1.0 : 0.0)));
        check("A * inverse(A) = I", ie < 1e-9, "max dev " + num(ie, 3));
        bool threw = false;
        try { Matrix Z(2, 2, 1.0); Z.inverse(); } catch (const std::runtime_error&) { threw = true; }
        check("singular matrix inverse throws an exception", threw);
    }


    void testStatistics() {
        os_ << "Statistical functions\n";
        const auto q = stats::gaussHermite(15);
        double s0 = 0, s2 = 0, s4 = 0;
        for (std::size_t i = 0; i < q.nodes.size(); ++i) {
            s0 += q.weights[i];
            s2 += q.weights[i] * q.nodes[i] * q.nodes[i];
            s4 += q.weights[i] * std::pow(q.nodes[i], 4);
        }
        check("Gauss-Hermite: sum w = sqrt(pi)", near(s0, std::sqrt(kPi), 1e-10), num(s0));
        check("Gauss-Hermite: sum w x^2 = sqrt(pi)/2", near(s2, std::sqrt(kPi) / 2, 1e-10), num(s2));
        check("Gauss-Hermite: sum w x^4 = 3 sqrt(pi)/4", near(s4, 3 * std::sqrt(kPi) / 4, 1e-9), num(s4));
        check("chi-square SF(3.841459; 1) = 0.05", near(stats::chiSquareSF(3.841459, 1), 0.05, 1e-6));
        check("chi-square SF(5.991465; 2) = 0.05", near(stats::chiSquareSF(5.991465, 2), 0.05, 1e-6));
        check("chi-square SF(9.487729; 4) = 0.05", near(stats::chiSquareSF(9.487729, 4), 0.05, 1e-6));
        check("chi-square SF(279.45; 4) < 2.2e-16 (Chapter 4 magnitude)", stats::chiSquareSF(279.45, 4) < 2.2e-16);
        check("normal two-sided p at z = 1.96 is 0.05", near(stats::normalTwoSidedP(1.959964), 0.05, 1e-6));
        const std::vector<double> x = {1, 2, 3, 4, 5}, y = {5, 6, 7, 8, 7};
        check("Spearman with ties = 8/sqrt(95) = 0.82078", near(stats::spearman(x, y), 8.0 / std::sqrt(95.0), 1e-12),
              num(stats::spearman(x, y)));
        check("Spearman of a monotone transform is 1",
              near(stats::spearman(x, std::vector<double>{1, 8, 27, 64, 125}), 1.0, 1e-12));
        check("Pearson of perfectly negative data is -1",
              near(stats::pearson(x, std::vector<double>{5, 4, 3, 2, 1}), -1.0, 1e-12));
        check("median of {5,1,3,2} is 2.5", near(stats::median({5, 1, 3, 2}), 2.5, 1e-12));
        Matrix X(4, 2, 1.0);
        for (int i = 0; i < 4; ++i) X(static_cast<std::size_t>(i), 1) = i;
        const std::vector<double> b = stats::ols(X, {1, 3, 5, 7});
        check("OLS recovers y = 1 + 2x exactly", near(b[0], 1, 1e-10) && near(b[1], 2, 1e-10));
    }


    void testVariables() {
        os_ << "Variable classes (Eq. 3 / Eq. 4 and category scores)\n";
        ContinuousVariable more("a", "a", Domain::Exposure, true);
        ContinuousVariable less("b", "b", Domain::Adaptive, false);
        more.fit({0, 10});
        less.fit({0, 10});
        check("Eq.(3) min-max: normalise(2.5) = 0.25", near(more.normalise(2.5), 0.25, 1e-12));
        check("Eq.(4) reversed: normalise(2.5) = 0.75", near(less.normalise(2.5), 0.75, 1e-12));
        NominalVariable nv("c", "c", Domain::Sensitivity, {"lo", "mid", "hi"});
        check("ordinal default score = code/(q-1): 0, 0.5, 1",
              near(nv.normalise(0), 0, 1e-12) && near(nv.normalise(1), 0.5, 1e-12) && near(nv.normalise(2), 1, 1e-12));
        NominalVariable sv("d", "d", Domain::Sensitivity, {"x", "y"}, {{"x", 1.0}});
        check("explicit score table; unscored category gives NaN", near(sv.normalise(0), 1.0, 1e-12) && std::isnan(sv.normalise(1)));
        const Variable& base = nv;  // polymorphic call through the base class
        check("polymorphism: kind() through base reference", base.kind() == Kind::Nominal && base.isCategorical());
        VariableSchema sc;
        check("schema has 16 indicators (4 E, 4 S, 8 A)",
              sc.size() == 16 && sc.indicesOf(Domain::Exposure).size() == 4 && sc.indicesOf(Domain::Sensitivity).size() == 4 &&
                  sc.indicesOf(Domain::Adaptive).size() == 8);
        check("4 continuous + 12 categorical indicators", sc.numContinuous() == 4 && sc.numCategorical() == 12);
        IndexBuilder ib(sc, VulnerabilityClassifier());
        DimensionScores ds;
        ds.E = {0.6}; ds.S = {0.9}; ds.A = {0.3};
        check("Eq.(8): (0.6 + 0.9 - 0.3)/3 = 0.4", near(IndexBuilder::composite(ds)[0], 0.4, 1e-12));
        VulnerabilityClassifier cl;
        check("classifier cut-points -1.5,-0.5,0.5,1.5 (left-closed): -2,-1.5,-1,0,1,2 -> 0,1,1,2,3,4",
              cl.classify(-2) == 0 && cl.classify(-1.5) == 1 && cl.classify(-1) == 1 && cl.classify(0) == 2 &&
                  cl.classify(1) == 3 && cl.classify(2) == 4);
    }

    void testRecoding() {
        os_ << "DHS recoding rules (R case_when blocks)\n";
        using L = DHSLoader;
        check("water: 21 -> Improved, 32 -> Unimproved, 81 -> Surface, 96/NaN -> NA",
              L::waterJmp(21) == "Improved" && L::waterJmp(32) == "Unimproved" && L::waterJmp(81) == "Surface Water" &&
                  L::waterJmp(96).empty() && L::waterJmp(kNaN).empty());
        check("water time: 2 -> on premises, 10 -> <=30, 31 -> >30, 998 / NaN -> NA",
              L::waterTime(2, false) == "On premises" && L::waterTime(10, false) == "<= 30 minutes" &&
                  L::waterTime(31, false) == "> 30 minutes" && L::waterTime(998, false).empty() && L::waterTime(kNaN, false).empty());
        check("blank v115 can be coded 'On premises' by option", L::waterTime(kNaN, true) == "On premises");
        check("fuel: 5 Kerosene, 6 Coal/Lignite, 7 Charcoal, 8 Wood, NaN Other",
              L::cookingFuel(5) == "Kerosene" && L::cookingFuel(6) == "Coal/Lignite" && L::cookingFuel(7) == "Charcoal" &&
                  L::cookingFuel(8) == "Wood" && L::cookingFuel(kNaN) == "Other" && L::cookingFuel(2) == "Clean");
        check("roof/floor/wall/toilet recodes",
              L::roof(12) == "Natural/Traditional" && L::roof(31) == "Iron Sheets" && L::floorMaterial(34) == "Cement" &&
                  L::wall(21) == "Unimproved Traditional" && L::wall(33) == "Durable/Modern" && L::toilet(61) == "Open Defecation" &&
                  L::toilet(96) == "Other / Unknown");
        check("birth weight: 3200 normal, 2000 low, 9998/NaN unknown",
              L::birthWeightCategory(3200) == 0 && L::birthWeightCategory(2000) == 1 && L::birthWeightCategory(9998) == 2 &&
                  L::birthWeightCategory(kNaN) == 2);
    }

    void testWeatherProcessor() {
        os_ << "Weather processing (Eq. 2)\n";
        const std::string path = "selftest_weather.csv";
        {
            std::ofstream f(path);
            f << "region_name,date,temp_mean,humidity_mean,precip_total,wind_mean\n";
            for (int y = 1991; y <= 2022; ++y) {
                const double t = (y <= 2020) ? 10.0 + (y % 2) : (y == 2021 ? 13.0 : 15.0);
                f << "R1,1/1/" << y << "," << t << ",50," << 1 << ",3\n";
            }
            f << "R1,1/1/1991,11,50,1,3\n";   // exact duplicate of the 1991 row -> dropped
            f << "R1,1/1/2000,10,50,2,3\n";   // same day, different precip -> precip SUMMED (R behaviour)
        }
        const ClimateIndicators ci = WeatherProcessor::process(path);
        removeFile(path);
        std::vector<double> temps;
        for (int y = 1991; y <= 2020; ++y) temps.push_back(10.0 + (y % 2));
        std::vector<double> prec(30, 1.0);
        prec[2000 - 1991] = 1.0 + 2.0;  // 3
        const RegionClimate* r = ci.find("R1");
        check("exact duplicate row removed", ci.duplicateRows == 1, chvi::str(ci.duplicateRows));
        check("one region, baseline = 30 years, recent = 2 years", ci.regions.size() == 1 && r && r->baselineYears == 30 && r->recentYears == 2);
        check("temperature z = (14 - mean)/sd of 30 annual means",
              r && near(r->z[0], (14.0 - stats::mean(temps)) / stats::sd(temps), 1e-12), r ? num(r->z[0]) : "");
        check("precipitation: duplicate-day values are summed", r && near(r->baselineMean[1], stats::mean(prec), 1e-12),
              r ? num(r->baselineMean[1]) : "");
        check("zero-variance series gives z = 0 (humidity, wind)", r && r->z[2] == 0.0 && r->z[3] == 0.0);
    }

    void testChiSquare() {
        os_ << "Chi-square test of independence\n";
        const ChiSquareResult r = ChiSquareTest::run({{10, 20}, {30, 40}});
        const double hand = 4.0 / 12 + 4.0 / 18 + 4.0 / 28 + 4.0 / 42;
        check("2x2 table [[10,20],[30,40]]: chi2 = 0.79365 (hand calculation)", near(r.chi2, hand, 1e-12), num(r.chi2));
        check("df = 1 and Cramer's V = sqrt(chi2/N)", r.df == 1 && near(r.cramersV, std::sqrt(hand / 100.0), 1e-12));
        check("p-value equals chi-square SF", near(r.pValue, stats::chiSquareSF(hand, 1), 1e-12), num(r.pValue));
        const ChiSquareResult ind = ChiSquareTest::run({{10, 20}, {20, 40}});
        check("independent table gives chi2 = 0 and p = 1", near(ind.chi2, 0, 1e-12) && near(ind.pValue, 1, 1e-9));
    }


    void testMixedLogit() {
        os_ << "Multilevel logistic regression (parameter recovery)\n";
        Rng rng(99);
        const int nC = 400, m = 8;
        const double b0 = -0.5, b1 = 0.8, sg = 0.7;
        Matrix X(static_cast<std::size_t>(nC * m), 2, 1.0);
        std::vector<int> y, cl;
        for (int c = 0; c < nC; ++c) {
            const double u = sg * rng.normal();
            for (int k = 0; k < m; ++k) {
                const double x = rng.normal();
                X(static_cast<std::size_t>(c * m + k), 1) = x;
                y.push_back(rng.bernoulli(stats::sigmoid(b0 + b1 * x + u)) ? 1 : 0);
                cl.push_back(c);
            }
        }
        MixedLogit ml(15);
        const MixedLogitFit f = ml.fit(X, y, cl, {"(Intercept)", "x"});
        check("optimiser converged", f.converged);
        check("beta0 recovered (true -0.5)", near(f.beta[0], b0, 0.15), num(f.beta[0], 4));
        check("beta1 recovered (true 0.8)", near(f.beta[1], b1, 0.15), num(f.beta[1], 4));
        check("sigma recovered (true 0.7)", near(f.sigma, sg, 0.2), num(f.sigma, 4));
        check("standard errors are positive and finite", f.se[0] > 0 && f.se[1] > 0 && std::isfinite(f.se[1]));
        check("true beta1 inside the 95% Wald interval",
              b1 > f.beta[1] - 1.96 * f.se[1] && b1 < f.beta[1] + 1.96 * f.se[1]);
        check("AIC = 2k - 2 logLik with k = 3", near(f.aic, 6.0 - 2.0 * f.logLik, 1e-9));
        check("ICC = sigma^2/(sigma^2 + pi^2/3)",
              near(f.icc, f.sigma * f.sigma / (f.sigma * f.sigma + kPi * kPi / 3.0), 1e-12));
    }


    void testDataStructures() {
        os_ << "Data structures\n";
        TopK<std::pair<double, int>> h(3);
        for (int i = 0; i < 10; ++i) h.push({static_cast<double>((i * 7) % 10), i});
        const auto top = h.sortedDescending();
        check("TopK heap keeps the 3 largest in descending order",
              top.size() == 3 && top[0].first == 9 && top[1].first == 8 && top[2].first == 7);
        AuditLog log(2);
        log.add("a"); log.add("b"); log.add("c");
        check("AuditLog queue keeps only the latest 2 entries", log.entries().size() == 2 && log.entries().front() == "b");
    }


    // ------------------- real data: needs the two input files -------------
    void testRealData() {
        os_ << "Real data (UDHS Kids Recode + weather): independent counts and Chapter 4 replication\n";
        if (!fileExists(settings_.dtaPath) || !fileExists(settings_.weatherPath)) {
            ++skipped_;
            os_ << "  [SKIP] data files not found (" << settings_.dtaPath << ")\n";
            return;
        }
        // --- Stata reader against counts obtained independently with pandas ---
        StataReader dta(settings_.dtaPath);
        check("Stata reader: format 118, 1436 observations, 6356 variables",
              dta.release() == 118 && dta.numObs() == 1436 && dta.numVars() == 6356,
              chvi::str(dta.numVars()) + " vars, " + chvi::str(dta.numObs()) + " obs");
        const std::vector<double> v025 = dta.numeric("v025"), v190 = dta.numeric("v190"), h22 = dta.numeric("h22");
        int urban = 0, w[6] = {0, 0, 0, 0, 0, 0}, f1 = 0, f2 = 0, f8 = 0, fn = 0;
        for (double v : v025) urban += v == 1;
        for (double v : v190) ++w[static_cast<int>(v)];
        for (double v : h22) { f1 += v == 1; f2 += v == 2; f8 += v == 8; fn += std::isnan(v); }
        check("v025 urban = 426 (pandas)", urban == 426);
        check("v190 wealth counts 382/287/268/247/252 (pandas)", w[1] == 382 && w[2] == 287 && w[3] == 268 && w[4] == 247 && w[5] == 252);
        check("h22: 352 yes, 936 no, 16 'don't know', 132 blank (pandas)", f1 == 352 && f2 == 936 && f8 == 16 && fn == 132);
        const std::vector<double> v115 = dta.numeric("v115"), m19 = dta.numeric("m19_1");
        int n115 = 0, n19 = 0;
        for (double v : v115) n115 += std::isnan(v);
        for (double v : m19) n19 += std::isnan(v);
        check("double/float missing values read as NaN: v115 275 blanks, m19_1 248 blanks (pandas)", n115 == 275 && n19 == 248,
              chvi::str(n115) + ", " + chvi::str(n19));

        // --- full pipeline ---
        ResearchSystem sys(settings_);
        sys.settings().quadratureNodes = 1;  // Laplace, as lme4::glmer
        std::ostringstream sink;
        if (!sys.loadData(sink)) { check("data load", false, sink.str()); return; }
        const Dataset& d = sys.data();
        check("995 households in 586 clusters; 1436 children", d.numHouseholds() == 995 && d.numClusters() == 586 && d.numChildren() == 1436);
        int nFever = 0, nNo = 0, nMiss = 0;
        for (const auto& c : d.children) { nFever += c.fever == 1; nNo += c.fever == 0; nMiss += c.fever < 0; }
        check("fever: 352 yes / 936 no / 148 missing (Chapter 4, Table 1)", nFever == 352 && nNo == 936 && nMiss == 148);
        std::size_t missWater = 0, missWt = 0;
        for (const auto& m : sys.loadReport().missingBefore) {
            if (m.first == "water_jmp") missWater = m.second;
            if (m.first == "water_time") missWt = m.second;
        }
        check("missing before imputation: water source 2, water time 211 (pandas)", missWater == 2 && missWt == 211);
        bool complete = true;
        for (const auto& h : d.households)
            for (std::size_t v = 0; v < sys.schema().size(); ++v)
                if (sys.schema().at(v).isCategorical()) complete = complete && h.x[v] >= 0 && h.x[v] < sys.schema().at(v).numLevels();
        check("after FAMD imputation every categorical code is valid", complete && sys.imputation().converged);
        const RegionClimate* kam = sys.climate().find("Kampala");
        const RegionClimate* kar = sys.climate().find("Karamoja");
        check("climate z (Kampala humidity = -3.805, Karamoja temperature = 0.360; pandas)",
              kam && kar && near(kam->z[2], -3.805, 5e-4) && near(kar->z[0], 0.360, 5e-4));

        // --- FAMD invariants ---
        std::ostringstream o2;
        sys.famdAnalysis(o2);
        const FAMDResult& f = sys.famd();
        double sum = 0;
        for (double l : f.eigenvalues) sum += l;
        check("FAMD: J = 46 design columns, 34 dimensions, total inertia 34", sys.schema().numFAMDColumns() == 46 && f.rank() == 34 && near(f.totalInertia, 34.0, 1e-9));
        check("FAMD: eigenvalues sum to total inertia", near(sum, f.totalInertia, 1e-8));
        check("FAMD: 4 components explain 30.6% (Chapter 4: 30.6%)", near(f.cumulativePct(4), 30.6, 0.05), num(f.cumulativePct(4), 5));
        const std::vector<double> W = f.weights(4);
        check("weights: wealth 0.517, cooking fuel 0.377, wall 0.347, floor 0.329 (Chapter 4)",
              near(W[VariableSchema::kWealth], 0.517, 5e-4) && near(W[VariableSchema::kFuel], 0.377, 5e-4) &&
                  near(W[VariableSchema::kWall], 0.347, 5e-4) && near(W[VariableSchema::kFloor], 0.329, 5e-4));
        double wmin = 1, wmax = 0;
        for (std::size_t v = 0; v < 4; ++v) { wmin = std::min(wmin, W[v]); wmax = std::max(wmax, W[v]); }
        check("exposure weights lie between 0.125 and 0.199 (Chapter 4)", near(wmin, 0.125, 5e-4) && near(wmax, 0.199, 5e-4));

        // --- index, validity, disparities ---
        sys.buildIndex(o2);
        const CHVIResult& c = sys.chvi();
        std::vector<int> cnt(5, 0);
        for (int l : c.level) ++cnt[static_cast<std::size_t>(l)];
        check("CHVI classes 77/252/310/307/49 (Chapter 4, Table 2)", cnt[0] == 77 && cnt[1] == 252 && cnt[2] == 310 && cnt[3] == 307 && cnt[4] == 49);
        check("CHVI* has mean 0 and SD 1", near(stats::mean(c.z), 0, 1e-10) && near(stats::sd(c.z), 1, 1e-10));
        check("construct validity r = +0.867 (S), -0.904 (A), +0.557 (E) (Chapter 4, Table 3)",
              near(stats::pearson(c.z, c.dim.S), 0.867, 5e-4) && near(stats::pearson(c.z, c.dim.A), -0.904, 5e-4) &&
                  near(stats::pearson(c.z, c.dim.E), 0.557, 5e-4));
        std::vector<std::vector<double>> tab(5, std::vector<double>(2, 0.0));
        for (std::size_t h = 0; h < d.households.size(); ++h) tab[static_cast<std::size_t>(c.level[h])][d.households[h].urban ? 0 : 1] += 1.0;
        const ChiSquareResult cs = ChiSquareTest::run(tab);
        check("rural-urban chi-square = 279.45, df = 4, p < 2.2e-16 (Chapter 4)", near(cs.chi2, 279.45, 0.01) && cs.df == 4 && cs.pValue < 2.2e-16, num(cs.chi2, 6));

        // --- robustness: K = 4 vs 12 ---
        IndexBuilder ib(sys.schema(), VulnerabilityClassifier());
        const Matrix Y = ib.normalisedMatrix(d);
        const CHVIResult c12 = ib.build(Y, f.weights(12));
        check("Spearman rho (K=4 vs K=12) = 0.998 (Chapter 4)", near(stats::spearman(c.raw, c12.raw), 0.998, 5e-4));

        // --- multilevel models (Laplace approximation = lme4 default) ---
        sys.feverModels(o2);
        const auto& m = sys.models();
        check("AIC M0..M3 = 1488.7 / 1474.7 / 1496.3 / 1483.0 (Chapter 4, Table 4)",
              near(m[0].aic, 1488.7, 0.06) && near(m[1].aic, 1474.7, 0.06) && near(m[2].aic, 1496.3, 0.06) && near(m[3].aic, 1483.0, 0.06),
              num(m[0].aic, 6) + " / " + num(m[1].aic, 6) + " / " + num(m[2].aic, 6) + " / " + num(m[3].aic, 6));
        check("all four models converged; 1288 children", m[0].converged && m[1].converged && m[2].converged && m[3].converged && m[0].nObs == 1288);
        check("CHVI coefficient in M1 is positive and significant (p < 0.001)", m[1].beta[1] > 0 && m[1].pValue[1] < 0.001);
    }
};

}  // namespace chvi
