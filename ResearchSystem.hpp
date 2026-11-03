// ResearchSystem.hpp
// ---------------------------------------------------------------------------
// ResearchSystem: the façade class that owns the data and every analysis
// stage and exposes one method per menu option.  All output goes to a
// std::ostream so the same code prints to the screen and to the report file.
//
//   Stage 0  data         : weather -> climate z-scores, DHS recoding,
//                           imputation, descriptive summaries
//   Objective 1           : FAMD -> weights -> CHVI -> validity -> robustness
//   Objective 2           : multilevel logistic models of child fever
//   Objective 3           : rural-urban chi-square test
// ---------------------------------------------------------------------------
#pragma once

#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "Portable.hpp"
#include "Climate.hpp"
#include "DHSLoader.hpp"
#include "Data.hpp"
#include "DataStructures.hpp"
#include "Disparity.hpp"
#include "FAMD.hpp"
#include "Imputer.hpp"
#include "Index.hpp"
#include "MixedLogit.hpp"
#include "Robustness.hpp"

namespace chvi {

struct Settings {
    std::string dtaPath = "data/UGKR7BFL_10PERCENT.dta";
    std::string weatherPath = "data/weather_Uganda_daily_1991_2022.csv";
    std::string outputDir = "output";
    std::uint64_t seed = 123;     // bootstrap / split seed
    int retainedComponents = 4;   // K retained (scree)
    int alternativeComponents = 12;
    int bootstrapReplicates = 200;
    int topK = 10;
    int quadratureNodes = 1;      // adaptive Gauss-Hermite nodes (1 = Laplace, the glmer default)
    bool blankWaterTimeIsOnPremises = false;
};

class ResearchSystem {
public:
    explicit ResearchSystem(Settings s = Settings()) : settings_(std::move(s)) {}

    Settings& settings() { return settings_; }
    const AuditLog& log() const { return log_; }
    bool hasData() const { return !data_.empty(); }
    Dataset& data() { return data_; }
    VariableSchema& schema() { return schema_; }
    const ClimateIndicators& climate() const { return climate_; }
    const CHVIResult& chvi() const { return chvi_; }
    const FAMDResult& famd() const { return *famd_; }
    const std::vector<MixedLogitFit>& models() const { return models_; }
    const ImputationReport& imputation() const { return imputeReport_; }
    const LoadReport& loadReport() const { return loadReport_; }

    // ======================= Stage 0 : data ================================
    bool loadData(std::ostream& os) {
        try {
            os << "Reading weather file : " << settings_.weatherPath << "\n";
            climate_ = WeatherProcessor::process(settings_.weatherPath);
            os << "  " << climate_.rawRows << " daily rows, " << climate_.duplicateRows << " exact duplicates dropped, "
               << climate_.regionDayRows << " region-days, " << climate_.regions.size() << " regions\n";
            os << "Reading DHS Kids Recode: " << settings_.dtaPath << "\n";
            DHSLoader::Options opt;
            opt.blankWaterTimeIsOnPremises = settings_.blankWaterTimeIsOnPremises;
            schema_ = VariableSchema();
            data_ = DHSLoader::load(settings_.dtaPath, climate_, schema_, opt, loadReport_);
            schema_.fit(data_.allColumns(schema_.size()));
            invalidate();
            std::size_t totalMissing = 0;
            for (const auto& m : loadReport_.missingBefore) totalMissing += m.second;
            imputeReport_ = FAMDImputer::impute(data_, schema_, 2);
            os << "  " << loadReport_.rawRows << " child records -> " << data_.numHouseholds() << " households in "
               << data_.numClusters() << " clusters; " << totalMissing << " missing categorical cells imputed (FAMD, ncp = 2, "
               << imputeReport_.iterations << " iterations" << (imputeReport_.converged ? "" : ", NOT converged") << ")\n";
            log_.add("Loaded UDHS + weather data");
            return true;
        } catch (const std::exception& e) {
            os << "Could not load data: " << e.what() << "\n";
            data_ = Dataset();
            return false;
        }
    }

    void summary(std::ostream& os) {
        ensureData(os);
        title(os, "DATASET SUMMARY  (cf. Table 1 of Chapter 4)");
        int urban = 0;
        for (const auto& h : data_.households) urban += h.urban;
        const int n = static_cast<int>(data_.numHouseholds());
        os << std::fixed << std::setprecision(1);
        os << "Child records (Kids Recode)  : " << data_.numChildren() << "\n"
           << "Households (distinct v001+v002): " << n << "\n"
           << "Enumeration areas (clusters) : " << data_.numClusters() << "\n"
           << "Urban households             : " << urban << " (" << pct(urban, n) << "%)   Rural: " << (n - urban)
           << " (" << pct(n - urban, n) << "%)\n\n";

        std::map<int, int> freq;  // ordered map: value -> frequency
        for (const auto& c : data_.children) ++freq[c.fever];
        const int N = static_cast<int>(data_.numChildren());
        os << "Prevalence of child fever (h22, last 2 weeks)\n"
           << std::left << std::setw(22) << "  Category" << std::right << std::setw(10) << "n" << std::setw(12)
           << "Percent\n";
        const std::pair<int, const char*> cats[3] = {{0, "No"}, {1, "Yes"}, {-1, "Missing / don't know"}};
        for (const auto& c : cats)
            os << std::left << "  " << std::setw(20) << c.second << std::right << std::setw(10) << freq[c.first]
               << std::setw(11) << pct(freq[c.first], N) << "%\n";
        os << std::left << "  " << std::setw(20) << "Total" << std::right << std::setw(10) << N << std::setw(11)
           << "100.0%\n\n";

        os << "Households per region\n";
        std::map<int, int> perRegion;
        for (const auto& h : data_.households) ++perRegion[h.region];
        for (const auto& kv : perRegion)
            os << "  " << std::left << std::setw(22) << regionTable()[static_cast<std::size_t>(kv.first)].name << std::right
               << std::setw(5) << kv.second << "\n";
    }

    void climateAnomalies(std::ostream& os) {
        ensureData(os);
        title(os, "CLIMATE EXPOSURE: STANDARDISED ANOMALIES  (Eq. 2)");
        os << "Weather file: " << climate_.rawRows << " daily rows; " << climate_.duplicateRows
           << " exact duplicates removed; " << climate_.regionDayRows << " region-days.\n"
           << "z = (mean 2021-22 - mean 1991-2020) / SD of the 30 annual values 1991-2020\n"
           << "(annual temperature, humidity, wind = mean of daily values; precipitation = annual total)\n\n"
           << std::left << std::setw(22) << "Region" << std::right;
        for (int v = 0; v < ClimateIndicators::kVars; ++v) os << std::setw(22) << ClimateIndicators::varLabel(v);
        os << "\n" << std::fixed << std::setprecision(3);
        for (const auto& r : climate_.regions) {
            os << std::left << std::setw(22) << r.name << std::right;
            for (std::size_t v = 0; v < 4; ++v) os << std::setw(22) << r.z[v];
            os << "\n";
        }
        os << "\nRaw anomaly (2021-22 minus 1991-2020 mean)\n" << std::left << std::setw(22) << "Region" << std::right;
        for (int v = 0; v < ClimateIndicators::kVars; ++v) os << std::setw(22) << ClimateIndicators::varLabel(v);
        os << "\n" << std::setprecision(2);
        for (const auto& r : climate_.regions) {
            os << std::left << std::setw(22) << r.name << std::right;
            for (std::size_t v = 0; v < 4; ++v) os << std::setw(22) << r.anomaly[v];
            os << "\n";
        }
    }

    /// Distribution of every recoded indicator (households) and the missingness handled.
    void cleaningReport(std::ostream& os) {
        ensureData(os);
        title(os, "RECODING, MISSING DATA AND IMPUTATION");
        os << "Distribution of the recoded categorical indicators over " << data_.numHouseholds()
           << " households (after imputation)\n";
        for (std::size_t v = 0; v < schema_.size(); ++v) {
            if (!schema_.at(v).isCategorical()) continue;
            const CategoricalVariable& cv = schema_.categorical(v);
            std::vector<int> cnt(static_cast<std::size_t>(cv.numLevels()), 0);
            for (const auto& h : data_.households) ++cnt[static_cast<std::size_t>(h.x[v])];
            std::size_t miss = 0;
            for (const auto& m : loadReport_.missingBefore)
                if (m.first == cv.name()) miss = m.second;
            os << "\n  " << cv.label() << "   [missing before imputation: " << miss << "]\n";
            for (int k = 0; k < cv.numLevels(); ++k)
                os << "      " << std::left << std::setw(26) << cv.levelLabel(k) << std::right << std::setw(6)
                   << cnt[static_cast<std::size_t>(k)] << "\n";
        }
        os << "\nImputed cells (FAMD, ncp = 2): " << imputeReport_.cells.size() << " in " << imputeReport_.iterations
           << " iterations\n";
        const IndexBuilder ib = builder();
        for (std::size_t v = 0; v < schema_.size(); ++v) {
            const std::size_t k = ib.countMedianImputed(data_, v);
            if (k > 0)
                os << "  " << schema_.at(v).name() << ": " << k << " households have a category without a score -> column median used\n";
        }
        os << "\nNote: v115 is blank for " << blankWaterTimeBlanks() << " of the households (water piped on the premises). "
           << (settings_.blankWaterTimeIsOnPremises ? "They are coded 'On premises'.\n"
                                                    : "As in the R script they are treated as missing and imputed.\n");
    }

    // ================= Objective 1 : FAMD and the index ====================
    void famdAnalysis(std::ostream& os) {
        ensureFAMD(os);
        title(os, "FACTOR ANALYSIS OF MIXED DATA  (Section 3.7.3)");
        const int K = settings_.retainedComponents;
        os << "Active variables: " << schema_.size() << " (" << schema_.numContinuous() << " continuous, "
           << schema_.numCategorical() << " categorical)   design columns J = " << schema_.numFAMDColumns()
           << "   dimensions = " << famd_->rank() << "\nTotal inertia = " << std::fixed << std::setprecision(3)
           << famd_->totalInertia << "  (= N1 + sum(q-1))\n\n"
           << std::left << std::setw(6) << "Dim" << std::right << std::setw(12) << "Eigenvalue" << std::setw(12)
           << "% Variance" << std::setw(12) << "Cumulative" << "\n";
        const int show = std::min(famd_->nDims(), 14);
        for (int k = 0; k < show; ++k)
            os << std::left << std::setw(6) << (k + 1) << std::right << std::setw(12) << famd_->eigenvalues[static_cast<std::size_t>(k)]
               << std::setw(11) << famd_->explainedPct(k) << "%" << std::setw(11) << famd_->cumulativePct(k + 1) << "%"
               << (k + 1 == K ? "   <- retained" : "") << "\n";
        os << "\nRetained components K = " << K << " explain " << std::setprecision(1) << famd_->cumulativePct(K)
           << "% of total variance.  Kaiser criterion (eigenvalue > 1): " << famd_->kaiserCount() << " components.\n\n";

        const std::vector<double> w = famd_->weights(K);
        std::vector<std::size_t> order(w.size());
        for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
        std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return w[a] > w[b]; });
        os << "Eigenvalue-weighted indicator weights (Eq. 5; coordinates = r^2 or eta^2 as FactoMineR)\n"
           << std::left << std::setw(4) << "#" << std::setw(36) << "Indicator" << std::setw(13) << "Domain" << std::setw(12)
           << "Type" << std::right << std::setw(9) << "Weight" << "\n" << std::setprecision(3);
        int rank = 1;
        for (std::size_t i : order)
            os << std::left << std::setw(4) << rank++ << std::setw(36) << schema_.at(i).label() << std::setw(13)
               << domainName(schema_.at(i).domain()) << std::setw(12) << kindName(schema_.at(i).kind()) << std::right
               << std::setw(9) << w[i] << "\n";
    }

    void buildIndex(std::ostream& os) {
        ensureIndex(os);
        title(os, "CONSTRUCTION AND CLASSIFICATION OF THE CHVI  (Eqs. 6-9)");
        os << "Dimension indices (weighted average of normalised indicators, Eq. 6)\n"
           << std::left << std::setw(24) << "  Dimension" << std::right << std::setw(9) << "Mean" << std::setw(9) << "SD"
           << std::setw(9) << "Min" << std::setw(9) << "Max" << "\n" << std::fixed << std::setprecision(3);
        describeRow(os, "Exposure (E)", chvi_.dim.E);
        describeRow(os, "Sensitivity (S)", chvi_.dim.S);
        describeRow(os, "Adaptive capacity (A)", chvi_.dim.A);
        describeRow(os, "CHVI raw  (E+S-A)/3", chvi_.raw);
        describeRow(os, "CHVI* (z-score, Eq. 9)", chvi_.z);
        os << "\nShape of CHVI*: skewness = " << stats::skewness(chvi_.z) << ", excess kurtosis = " << stats::excessKurtosis(chvi_.z)
           << "\n\n";

        const int n = static_cast<int>(chvi_.z.size());
        std::vector<int> cnt(5, 0);
        for (int l : chvi_.level) ++cnt[static_cast<std::size_t>(l)];
        const IndexBuilder ib = builder();
        os << "Classification (cut-points on CHVI*: ";
        for (double c : ib.classifier().cuts()) os << c << " ";
        os << ")\n" << std::left << std::setw(22) << "  Vulnerability level" << std::right << std::setw(12) << "Households"
           << std::setw(12) << "Percent" << "\n" << std::setprecision(1);
        for (int l = 0; l < 5; ++l)
            os << "  " << std::left << std::setw(20) << VulnerabilityClassifier::label(l) << std::right << std::setw(12)
               << cnt[static_cast<std::size_t>(l)] << std::setw(11) << pct(cnt[static_cast<std::size_t>(l)], n) << "%\n";
        os << "  " << std::left << std::setw(20) << "Total" << std::right << std::setw(12) << n << std::setw(11) << "100.0%\n";
    }

    void constructValidity(std::ostream& os) {
        ensureIndex(os);
        title(os, "CONSTRUCT VALIDITY  (Section 3.7.4)");
        const double rS = stats::pearson(chvi_.z, chvi_.dim.S);
        const double rA = stats::pearson(chvi_.z, chvi_.dim.A);
        const double rE = stats::pearson(chvi_.z, chvi_.dim.E);
        os << "Expected signs under the IPCC framework: S (+), A (-), E (+)\n\n"
           << std::left << std::setw(28) << "Relationship" << std::right << std::setw(10) << "r" << std::setw(18)
           << "Interpretation" << std::setw(12) << "Expected" << "\n" << std::fixed << std::setprecision(3);
        validityRow(os, "CHVI <-> Sensitivity", rS, +1);
        validityRow(os, "CHVI <-> Adaptive Capacity", rA, -1);
        validityRow(os, "CHVI <-> Exposure", rE, +1);
    }

    void robustness(std::ostream& os) {
        ensureIndex(os);
        title(os, "ROBUSTNESS OF THE INDEX  (Section 3.7.4)");
        os << "Running " << settings_.bootstrapReplicates << " cluster-bootstrap replicates ...\n";
        const IndexBuilder ib = builder();  // must outlive the analyzer (held by reference)
        RobustnessAnalyzer ra(data_, schema_, ib, settings_.seed);
        rob_ = ra.run(*famd_, Y_, chvi_, settings_.retainedComponents, settings_.bootstrapReplicates,
                      settings_.alternativeComponents);
        haveRob_ = true;
        const RobustnessReport& r = rob_;
        os << std::fixed << std::setprecision(3) << "\n"
           << "1. Component retention : K=" << r.kPrimary << " vs K=" << r.kAlternative << "   (Kaiser would keep "
           << r.kaiser << ")        Spearman rho = " << r.rhoComponents << "\n"
           << "2. Cluster bootstrap   : B=" << r.bootRho.size() << "  median rho = " << r.bootMedian << "  range [" << r.bootMin
           << ", " << r.bootMax << "]\n"
           << "      (FAMD weights re-estimated on each resample of enumeration areas, then applied to ALL households)\n"
           << "3. Regression weights  : train clusters = " << r.trainClusters << " (" << r.trainChildren
           << " children), test clusters = " << r.testClusters << " (" << r.testHouseholds << " households)\n"
           << "      empirical weights  E = " << r.wE << "  S = " << r.wS << "  A = " << r.wA << "   (equal weights = 0.333 each)\n"
           << "      Spearman rho (equal-weighted vs regression-weighted, test set) = " << r.rhoRegression << "\n"
           << "      Repeated over " << r.splitRho.size() << " random cluster splits: rho median " << r.splitMedianRho << ", range ["
           << r.splitMinRho << ", " << r.splitMaxRho << "]\n"
           << "      empirical-weight ranges over the splits:  E [" << r.wEmin << ", " << r.wEmax << "]  S [" << r.wSmin << ", "
           << r.wSmax << "]  A [" << r.wAmin << ", " << r.wAmax << "]  (unstable -> equal weights preferred)\n\n";
        const bool ok = r.rhoComponents >= 0.80 && r.bootMedian >= 0.80 && r.splitMedianRho >= 0.80;
        os << "Decision rule: the index is 'acceptably robust' when every rho >= 0.80.\n"
           << "H01 (index does not exhibit acceptable robustness): "
           << (ok ? "REJECTED -> the CHVI is robust." : "NOT rejected -> robustness is insufficient.") << "\n";
    }

    // ============== Objective 2 : child fever, multilevel models ===========
    void feverModels(std::ostream& os) {
        ensureIndex(os);
        title(os, "MULTILEVEL LOGISTIC MODELS FOR CHILD FEVER  (Eq. 10)");
        std::vector<std::size_t> kids;
        for (std::size_t i = 0; i < data_.children.size(); ++i)
            if (data_.children[i].hasFever()) kids.push_back(i);
        os << "Children with observed fever status (h22 = 1 or 2): " << kids.size() << "  in " << data_.numClusters()
           << " clusters.  Adaptive Gauss-Hermite quadrature, " << settings_.quadratureNodes
           << (settings_.quadratureNodes == 1 ? " node (= Laplace approximation, as lme4::glmer default).\n\n" : " nodes.\n\n");

        struct Spec { std::string label; std::vector<Cov> covs; };
        const std::vector<Spec> specs = {
            {"M0 Null (random intercept only)", {}},
            {"M1 Contextual (CHVI + Residence + Radio)", {Cov::CHVI, Cov::Urban, Cov::Radio}},
            {"M2 Individual (Age+Sex+BirthOrder+BirthWt)", {Cov::AgeZ, Cov::Girl, Cov::BirthOrder, Cov::LowBW, Cov::BWUnknown}},
            {"M3 Full (Contextual + TV + Individual)",
             {Cov::CHVI, Cov::Urban, Cov::Radio, Cov::TV, Cov::AgeZ, Cov::Girl, Cov::BirthOrder, Cov::LowBW, Cov::BWUnknown}}};

        MixedLogit ml(settings_.quadratureNodes);
        models_.clear();
        modelLabels_.clear();
        for (const Spec& sp : specs) {
            Matrix X(kids.size(), sp.covs.size() + 1, 1.0);
            std::vector<int> y, cl;
            std::vector<std::string> names = {"(Intercept)"};
            for (Cov c : sp.covs) names.push_back(covName(c));
            for (std::size_t r = 0; r < kids.size(); ++r) {
                const Child& ch = data_.children[kids[r]];
                y.push_back(ch.fever);
                cl.push_back(ch.cluster);
                for (std::size_t j = 0; j < sp.covs.size(); ++j) X(r, j + 1) = covValue(sp.covs[j], ch);
            }
            models_.push_back(ml.fit(X, y, cl, names));
            modelLabels_.push_back(sp.label);
        }
        haveModels_ = true;

        os << std::left << std::setw(46) << "Model" << std::right << std::setw(11) << "LogLik" << std::setw(10) << "AIC"
           << std::setw(9) << "dAIC" << std::setw(9) << "sigma" << std::setw(8) << "ICC" << "\n" << std::fixed << std::setprecision(2);
        for (std::size_t m = 0; m < models_.size(); ++m) {
            const MixedLogitFit& f = models_[m];
            os << std::left << std::setw(46) << specs[m].label << std::right << std::setw(11) << f.logLik << std::setw(10) << f.aic;
            if (m == 0) os << std::setw(9) << "--"; else os << std::setw(9) << (f.aic - models_[0].aic);
            os << std::setw(9) << f.sigma << std::setw(8) << f.icc << (f.converged ? "" : "  (!)") << "\n";
        }
        std::size_t best = 0;
        for (std::size_t m = 1; m < models_.size(); ++m)
            if (models_[m].aic < models_[best].aic) best = m;
        os << "\nPreferred specification (lowest AIC): " << specs[best].label << "\n";
        const double lr = 2.0 * (models_[1].logLik - models_[0].logLik);
        os << "Likelihood-ratio test M1 vs M0: chi2(3) = " << lr << ",  p = " << std::scientific << std::setprecision(2)
           << stats::chiSquareSF(lr, 3) << std::fixed << "\n\n";

        for (std::size_t m : {static_cast<std::size_t>(1), best == 1 ? static_cast<std::size_t>(3) : best}) {
            const MixedLogitFit& f = models_[m];
            os << "Fixed effects of " << specs[m].label << "\n" << std::left << std::setw(26) << "  Term" << std::right
               << std::setw(9) << "Beta" << std::setw(9) << "SE" << std::setw(9) << "z" << std::setw(10) << "p" << std::setw(9) << "OR"
               << std::setw(18) << "95% CI (OR)" << "\n" << std::setprecision(3);
            for (std::size_t j = 0; j < f.beta.size(); ++j) {
                std::ostringstream ci;
                ci << std::fixed << std::setprecision(2) << "[" << std::exp(f.beta[j] - 1.96 * f.se[j]) << ", "
                   << std::exp(f.beta[j] + 1.96 * f.se[j]) << "]";
                os << "  " << std::left << std::setw(24) << f.names[j] << std::right << std::setw(9) << f.beta[j] << std::setw(9)
                   << f.se[j] << std::setw(9) << f.z[j] << std::setw(10) << f.pValue[j] << std::setw(9) << std::exp(f.beta[j])
                   << std::setw(18) << ci.str() << "\n";
            }
            os << "\n";
            if (best == 1) break;
        }
        const MixedLogitFit& m1 = models_[1];
        os << "CHVI effect in M1: OR per 1-SD increase = " << std::setprecision(3) << std::exp(m1.beta[1]) << " (p = "
           << std::scientific << std::setprecision(2) << m1.pValue[1] << std::fixed << ")\n"
           << "H02 (CHVI has no association with fever): "
           << (m1.pValue[1] < 0.05 ? "REJECTED -> CHVI is significantly associated with fever." : "NOT rejected at the 5% level.") << "\n";
    }

    // ============== Objective 3 : rural-urban disparities ==================
    void residenceDisparity(std::ostream& os) {
        ensureIndex(os);
        title(os, "RURAL-URBAN DISPARITIES IN THE CHVI  (Section 3.7.6)");
        std::vector<std::vector<double>> tab(5, std::vector<double>(2, 0.0));
        for (std::size_t h = 0; h < data_.households.size(); ++h)
            tab[static_cast<std::size_t>(chvi_.level[h])][data_.households[h].urban ? 0 : 1] += 1.0;
        const ChiSquareResult r = ChiSquareTest::run(tab);
        os << "Distribution of vulnerability level by residence (column %)\n"
           << std::left << std::setw(22) << "  Level" << std::right << std::setw(12) << "Urban (%)" << std::setw(12) << "Rural (%)"
           << std::setw(10) << "n urban" << std::setw(10) << "n rural" << "\n" << std::fixed << std::setprecision(1);
        for (int l = 0; l < 5; ++l)
            os << "  " << std::left << std::setw(20) << VulnerabilityClassifier::label(l) << std::right << std::setw(12)
               << r.colPct[static_cast<std::size_t>(l)][0] << std::setw(12) << r.colPct[static_cast<std::size_t>(l)][1]
               << std::setw(10) << static_cast<int>(tab[static_cast<std::size_t>(l)][0]) << std::setw(10)
               << static_cast<int>(tab[static_cast<std::size_t>(l)][1]) << "\n";
        const double hu = r.colPct[3][0] + r.colPct[4][0], hr = r.colPct[3][1] + r.colPct[4][1];
        const double lu = r.colPct[0][0] + r.colPct[1][0], lr = r.colPct[0][1] + r.colPct[1][1];
        os << "\nHigh + Very High : urban " << hu << "%  vs rural " << hr << "%\n"
           << "Very Low + Low   : urban " << lu << "%  vs rural " << lr << "%\n\n" << std::setprecision(3)
           << "Pearson chi-square = " << r.chi2 << ",  df = " << r.df << ",  p = " << std::scientific << std::setprecision(2)
           << r.pValue << std::fixed << std::setprecision(3) << ",  Cramer's V = " << r.cramersV << "\n";
        if (r.cellsExpectedBelow5 > 0) os << "Note: " << r.cellsExpectedBelow5 << " cell(s) have expected count < 5.\n";
        os << "H03 (no rural-urban disparity): "
           << (r.pValue < 0.05 ? "REJECTED -> significant rural-urban divide." : "NOT rejected at the 5% level.") << "\n";
    }

    // ============== Extra : data-structure demonstrations ==================
    void topVulnerable(std::ostream& os) {
        ensureIndex(os);
        title(os, "MOST VULNERABLE HOUSEHOLDS AND REGIONAL RANKING");
        TopK<std::pair<double, int>> heap(static_cast<std::size_t>(settings_.topK));  // min-heap of size k
        std::unordered_map<int, std::pair<double, int>> agg;                             // region -> (sum, n)
        for (std::size_t h = 0; h < data_.households.size(); ++h) {
            heap.push({chvi_.z[h], static_cast<int>(h)});
            auto& a = agg[data_.households[h].region];
            a.first += chvi_.z[h];
            a.second += 1;
        }
        os << "Top " << settings_.topK << " households by CHVI* (min-heap, O(n log k))\n"
           << std::left << std::setw(6) << "Rank" << std::setw(9) << "Cluster" << std::setw(6) << "HH" << std::setw(22) << "Region"
           << std::setw(8) << "Area" << std::right << std::setw(9) << "CHVI*" << "  " << std::left << "Level" << "\n"
           << std::fixed << std::setprecision(3);
        int rank = 1;
        for (const auto& it : heap.sortedDescending()) {
            const Household& h = data_.households[static_cast<std::size_t>(it.second)];
            os << std::left << std::setw(6) << rank++ << std::setw(9) << h.cluster << std::setw(6) << h.number << std::setw(22)
               << regionTable()[static_cast<std::size_t>(h.region)].name << std::setw(8) << (h.urban ? "Urban" : "Rural")
               << std::right << std::setw(9) << it.first << "  " << std::left
               << VulnerabilityClassifier::label(chvi_.level[static_cast<std::size_t>(it.second)]) << "\n";
        }
        std::vector<std::pair<double, int>> reg;
        for (const auto& kv : agg) reg.push_back({kv.second.first / kv.second.second, kv.first});
        std::sort(reg.begin(), reg.end(), std::greater<std::pair<double, int>>());
        os << "\nRegions ranked by mean CHVI* (hash-map aggregation + sort)\n"
           << std::left << std::setw(6) << "Rank" << std::setw(22) << "Region" << std::right << std::setw(11) << "Mean CHVI*"
           << std::setw(8) << "n" << "\n";
        rank = 1;
        for (const auto& pr : reg)
            os << std::left << std::setw(6) << rank++ << std::setw(22) << regionTable()[static_cast<std::size_t>(pr.second)].name
               << std::right << std::setw(11) << pr.first << std::setw(8) << agg[pr.second].second << "\n";
    }

    /// Household-level results (labels, dimension indices, CHVI) as CSV.
    bool exportHouseholds(const std::string& path, std::ostream& os) {
        ensureIndex(os);
        makeDir(parentDir(path));
        std::ofstream f(path);
        if (!f) { os << "Cannot write " << path << "\n"; return false; }
        f << "cluster,hh,region,urban";
        for (std::size_t v = 0; v < schema_.size(); ++v) f << ',' << schema_.at(v).name();
        f << ",Exposure,Sensitivity,Adaptive,CHVI_raw,CHVI_z,CHVI_level\n" << std::setprecision(10);
        for (std::size_t h = 0; h < data_.households.size(); ++h) {
            const Household& hh = data_.households[h];
            f << hh.cluster << ',' << hh.number << ',' << regionTable()[static_cast<std::size_t>(hh.region)].name << ','
              << hh.urban;
            for (std::size_t v = 0; v < schema_.size(); ++v) {
                if (schema_.at(v).isCategorical()) f << ',' << schema_.at(v).levelLabel(static_cast<int>(hh.x[v]));
                else f << ',' << hh.x[v];
            }
            f << ',' << chvi_.dim.E[h] << ',' << chvi_.dim.S[h] << ',' << chvi_.dim.A[h] << ',' << chvi_.raw[h] << ','
              << chvi_.z[h] << ',' << VulnerabilityClassifier::label(chvi_.level[h]) << '\n';
        }
        os << "Household-level results written to " << path << "\n";
        return true;
    }

    // ============================ Run all ==================================
    void runAll(std::ostream& os) {
        ensureData(os);
        summary(os);
        climateAnomalies(os);
        cleaningReport(os);
        famdAnalysis(os);
        buildIndex(os);
        constructValidity(os);
        robustness(os);
        feverModels(os);
        residenceDisparity(os);
        topVulnerable(os);
        log_.add("Ran complete pipeline");
    }

    /// Discard cached FAMD / index / model results (after a settings change).
    void resetResults() { invalidate(); }

private:
    enum class Cov { CHVI, Urban, Radio, TV, AgeZ, Girl, BirthOrder, LowBW, BWUnknown };

    Settings settings_;
    VariableSchema schema_;
    ClimateIndicators climate_;
    Dataset data_;
    LoadReport loadReport_;
    ImputationReport imputeReport_;
    std::unique_ptr<FAMDResult> famd_;
    Matrix Y_;
    CHVIResult chvi_;
    bool haveIndex_ = false;
    RobustnessReport rob_;
    bool haveRob_ = false;
    std::vector<MixedLogitFit> models_;
    std::vector<std::string> modelLabels_;
    bool haveModels_ = false;
    AuditLog log_;

    void invalidate() {
        famd_.reset();
        haveIndex_ = haveRob_ = haveModels_ = false;
    }

    IndexBuilder builder() const { return IndexBuilder(schema_, VulnerabilityClassifier()); }

    std::size_t blankWaterTimeBlanks() const {
        for (const auto& m : loadReport_.missingBefore)
            if (m.first == "water_time") return m.second;
        return 0;
    }

    void ensureData(std::ostream& os) {
        if (hasData()) return;
        os << "[no data in memory - loading the UDHS and weather files]\n";
        loadData(os);
        if (!hasData()) throw std::runtime_error("data files could not be loaded; check the paths in Settings");
    }
    void ensureFAMD(std::ostream& os) {
        ensureData(os);
        if (famd_) return;
        std::vector<int> rows(data_.numHouseholds());
        for (std::size_t i = 0; i < rows.size(); ++i) rows[i] = static_cast<int>(i);
        famd_ = std::unique_ptr<FAMDResult>(new FAMDResult(FAMD(schema_).fit(data_, rows)));
        log_.add("Fitted FAMD");
    }
    void ensureIndex(std::ostream& os) {
        ensureFAMD(os);
        if (haveIndex_) return;
        const IndexBuilder b = builder();
        Y_ = b.normalisedMatrix(data_);
        chvi_ = b.build(Y_, famd_->weights(settings_.retainedComponents));
        haveIndex_ = true;
        log_.add("Built CHVI (K=" + chvi::str(settings_.retainedComponents) + ")");
    }

    static std::string covName(Cov c) {
        switch (c) {
            case Cov::CHVI: return "CHVI (z)";
            case Cov::Urban: return "Urban (vs rural)";
            case Cov::Radio: return "Radio (v158)";
            case Cov::TV: return "Television (v159)";
            case Cov::AgeZ: return "Age (z)";
            case Cov::Girl: return "Girl (vs boy)";
            case Cov::BirthOrder: return "Birth order";
            case Cov::LowBW: return "Low birth weight";
            case Cov::BWUnknown: return "Birth wt unknown";
        }
        return "?";
    }
    double covValue(Cov c, const Child& ch) const {
        const Household& h = data_.households[static_cast<std::size_t>(ch.household)];
        switch (c) {
            case Cov::CHVI: return chvi_.z[static_cast<std::size_t>(ch.household)];
            case Cov::Urban: return h.urban;
            case Cov::Radio: return ch.radio;
            case Cov::TV: return ch.tv;
            case Cov::AgeZ: return ch.ageZ;
            case Cov::Girl: return ch.girl;
            case Cov::BirthOrder: return ch.birthOrder;
            case Cov::LowBW: return ch.bwCat == 1 ? 1.0 : 0.0;
            case Cov::BWUnknown: return ch.bwCat == 2 ? 1.0 : 0.0;
        }
        return 0.0;
    }

    // ------------------------- printing helpers ----------------------------
    static void title(std::ostream& os, const std::string& t) {
        os << "\n" << std::string(78, '=') << "\n " << t << "\n" << std::string(78, '=') << "\n";
    }
    static double pct(int a, int b) { return b ? 100.0 * a / b : 0.0; }
    static void describeRow(std::ostream& os, const std::string& name, const std::vector<double>& v) {
        const double m = stats::mean(v);
        os << "  " << std::left << std::setw(22) << name << std::right << std::setw(9) << (std::fabs(m) < 5e-4 ? 0.0 : m)
           << std::setw(9) << stats::sd(v) << std::setw(9) << *std::min_element(v.begin(), v.end()) << std::setw(9)
           << *std::max_element(v.begin(), v.end()) << "\n";
    }
    static void validityRow(std::ostream& os, const std::string& name, double r, int expectedSign) {
        const double a = std::fabs(r);
        const char* mag = a >= 0.7 ? "Strong" : (a >= 0.4 ? "Moderate" : "Weak");
        const bool ok = (r > 0 ? 1 : -1) == expectedSign;
        os << std::left << std::setw(28) << name << std::right << std::setw(10) << r << "  " << std::left << std::setw(12)
           << (std::string(mag) + (r > 0 ? " +" : " -")) << std::setw(10) << (expectedSign > 0 ? "positive" : "negative")
           << (ok ? " [OK]" : " [SIGN MISMATCH]") << "\n";
    }
};

}  // namespace chvi
