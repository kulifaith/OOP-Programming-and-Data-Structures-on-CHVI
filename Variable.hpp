// Variable.hpp
// ---------------------------------------------------------------------------
// Class hierarchy describing the 16 CHVI indicators (OOP: abstraction,
// inheritance, polymorphism).
//
//   Variable (abstract)
//     |-- ContinuousVariable            min-max normalisation (Eq. 3 / Eq. 4)
//     |-- CategoricalVariable (abstract) explicit score table per category
//           |-- OrdinalVariable         ordered categories
//           |-- NominalVariable         unordered categories (binary included)
//
// Each categorical variable keeps its list of category labels (position =
// integer code stored in the Household record) and, optionally, an expert
// "vulnerability / capacity score" per label exactly as in the R script
// (e.g. roof: Natural 1.0, Iron sheets 0.5, Modern 0.0).  A score of NaN
// means "no score" and is later replaced by the column median (R script).
// Scores are oriented in the direction of the IPCC domain: higher Exposure /
// Sensitivity score = more vulnerable, higher Adaptive score = more capacity.
// ---------------------------------------------------------------------------
#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace chvi {

enum class Domain { Exposure, Sensitivity, Adaptive };
enum class Kind { Continuous, Ordinal, Nominal };

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

inline const char* domainName(Domain d) {
    switch (d) {
        case Domain::Exposure: return "Exposure";
        case Domain::Sensitivity: return "Sensitivity";
        case Domain::Adaptive: return "Adaptive";
    }
    return "?";
}
inline const char* kindName(Kind k) {
    switch (k) {
        case Kind::Continuous: return "continuous";
        case Kind::Ordinal: return "ordinal";
        case Kind::Nominal: return "nominal";
    }
    return "?";
}

// ---------------------------------------------------------------------------
class Variable {
public:
    Variable(std::string name, std::string label, Domain domain)
        : name_(std::move(name)), label_(std::move(label)), domain_(domain) {}
    virtual ~Variable() = default;

    const std::string& name() const { return name_; }
    const std::string& label() const { return label_; }
    Domain domain() const { return domain_; }

    virtual Kind kind() const = 0;
    bool isCategorical() const { return kind() != Kind::Continuous; }

    /// Number of categories (0 for continuous variables).
    virtual int numLevels() const { return 0; }
    virtual std::string levelLabel(int /*code*/) const { return ""; }

    /// Learn data-dependent constants (e.g. min and max) from a column.
    virtual void fit(const std::vector<double>& /*column*/) {}

    /// Normalised score in [0,1] oriented in the direction of the domain;
    /// NaN when the value has no score.
    virtual double normalise(double raw) const = 0;

private:
    std::string name_, label_;
    Domain domain_;
};

// ---------------------------------------------------------------------------
class ContinuousVariable : public Variable {
public:
    /// `higherIsMore` = true -> Eq. (3);  false -> reversed, Eq. (4).
    ContinuousVariable(std::string name, std::string label, Domain d, bool higherIsMore,
                       std::string unit = "")
        : Variable(std::move(name), std::move(label), d), higherIsMore_(higherIsMore), unit_(std::move(unit)) {}

    Kind kind() const override { return Kind::Continuous; }
    const std::string& unit() const { return unit_; }
    double minValue() const { return min_; }
    double maxValue() const { return max_; }

    void fit(const std::vector<double>& column) override {
        bool any = false;
        for (double v : column) {
            if (std::isnan(v)) continue;
            if (!any) { min_ = max_ = v; any = true; }
            min_ = std::min(min_, v);
            max_ = std::max(max_, v);
        }
    }

    double normalise(double raw) const override {
        const double range = max_ - min_;
        if (std::isnan(raw) || range <= 0.0) return 0.0;
        const double y = higherIsMore_ ? (raw - min_) / range : (max_ - raw) / range;
        return std::min(1.0, std::max(0.0, y));
    }

private:
    bool higherIsMore_;
    std::string unit_;
    double min_ = 0.0, max_ = 1.0;
};

// ---------------------------------------------------------------------------
class CategoricalVariable : public Variable {
public:
    CategoricalVariable(std::string name, std::string label, Domain d, std::vector<std::string> levels,
                        std::map<std::string, double> labelScores = {})
        : Variable(std::move(name), std::move(label), d),
          levels_(std::move(levels)),
          labelScores_(std::move(labelScores)) {}

    int numLevels() const override { return static_cast<int>(levels_.size()); }
    std::string levelLabel(int code) const override {
        return (code >= 0 && code < numLevels()) ? levels_[static_cast<std::size_t>(code)] : "?";
    }
    const std::vector<std::string>& levels() const { return levels_; }
    /// Replace the category list (used when the levels are data driven).
    void setLevels(std::vector<std::string> levels) { levels_ = std::move(levels); }
    /// Code of a label or -1.
    int codeOf(const std::string& label) const {
        const auto it = std::find(levels_.begin(), levels_.end(), label);
        return it == levels_.end() ? -1 : static_cast<int>(it - levels_.begin());
    }

    /// Score of a category: the explicit table when one is supplied,
    /// otherwise min-max scaling of the ordered code, (code)/(q-1).
    double normalise(double raw) const override {
        if (std::isnan(raw) || raw < 0 || raw >= static_cast<double>(levels_.size())) return kNaN;
        if (!labelScores_.empty()) {
            const auto it = labelScores_.find(levels_[static_cast<std::size_t>(raw)]);
            return it == labelScores_.end() ? kNaN : it->second;
        }
        return levels_.size() < 2 ? 0.0 : raw / static_cast<double>(levels_.size() - 1);
    }

private:
    std::vector<std::string> levels_;
    std::map<std::string, double> labelScores_;
};

class OrdinalVariable : public CategoricalVariable {
public:
    using CategoricalVariable::CategoricalVariable;
    Kind kind() const override { return Kind::Ordinal; }
};

class NominalVariable : public CategoricalVariable {
public:
    using CategoricalVariable::CategoricalVariable;
    Kind kind() const override { return Kind::Nominal; }
};

// ---------------------------------------------------------------------------
/// The ordered collection of the 16 indicators of the study.  Column order is
/// the order of `chvi_vars` in the R script.  Adding a new indicator = adding
/// one `add<...>()` line here (plus its recode in DHSLoader).
class VariableSchema {
public:
    static constexpr std::size_t kTemp = 0, kPrecip = 1, kHumid = 2, kWind = 3, kFuel = 4, kRoof = 5,
                                 kFloor = 6, kWall = 7, kWealth = 8, kEduc = 9, kWater = 10, kWaterTime = 11,
                                 kToilet = 12, kPhone = 13, kBank = 14, kLand = 15;

    VariableSchema() { build(); }

    std::size_t size() const { return vars_.size(); }
    const Variable& at(std::size_t i) const { return *vars_[i]; }
    Variable& at(std::size_t i) { return *vars_[i]; }
    CategoricalVariable& categorical(std::size_t i) { return dynamic_cast<CategoricalVariable&>(*vars_[i]); }
    const CategoricalVariable& categorical(std::size_t i) const { return dynamic_cast<const CategoricalVariable&>(*vars_[i]); }

    std::vector<std::size_t> indicesOf(Domain d) const {
        std::vector<std::size_t> r;
        for (std::size_t i = 0; i < vars_.size(); ++i)
            if (vars_[i]->domain() == d) r.push_back(i);
        return r;
    }
    int numContinuous() const {
        int n = 0;
        for (const auto& v : vars_) n += v->isCategorical() ? 0 : 1;
        return n;
    }
    int numCategorical() const { return static_cast<int>(vars_.size()) - numContinuous(); }

    /// FAMD design columns: 1 per continuous variable + 1 dummy per category.
    int numFAMDColumns() const {
        int n = 0;
        for (const auto& v : vars_) n += v->isCategorical() ? v->numLevels() : 1;
        return n;
    }

    void fit(const std::vector<std::vector<double>>& columns) {
        for (std::size_t i = 0; i < vars_.size() && i < columns.size(); ++i) vars_[i]->fit(columns[i]);
    }

private:
    std::vector<std::unique_ptr<Variable>> vars_;

    template <typename V, typename... Args>
    void add(Args&&... args) {
        vars_.push_back(std::unique_ptr<Variable>(new V(std::forward<Args>(args)...)));
    }

    void build() {
        const Domain E = Domain::Exposure, S = Domain::Sensitivity, A = Domain::Adaptive;
        using Lv = std::vector<std::string>;
        using Sc = std::map<std::string, double>;
        // ---- Exposure: standardised climate anomalies (Eq. 2) --------------
        add<ContinuousVariable>("temp_z", "Temperature anomaly (2 m)", E, true, "z");
        add<ContinuousVariable>("precip_z", "Precipitation anomaly", E, true, "z");
        add<ContinuousVariable>("humidity_z", "Relative humidity anomaly", E, true, "z");
        add<ContinuousVariable>("wind_z", "Wind speed anomaly (10 m)", E, true, "z");
        // ---- Sensitivity ----------------------------------------------------
        // cooking_fuel levels are data driven (sorted observed labels, like R's factor())
        add<NominalVariable>("cooking_fuel", "Type of cooking fuel (v161)", S, Lv{},
                             Sc{{"Wood", 1.00}, {"Charcoal", 0.75}, {"Coal/Lignite", 0.75}, {"Kerosene", 0.50}, {"Other", 0.50}});
        add<NominalVariable>("roof", "Main roof material (v129)", S,
                             Lv{"Natural/Traditional", "Iron Sheets", "Modern/Other"},
                             Sc{{"Natural/Traditional", 1.0}, {"Iron Sheets", 0.5}, {"Modern/Other", 0.0}});
        add<NominalVariable>("floor", "Main floor material (v127)", S, Lv{"Natural", "Cement", "Modern/Other"},
                             Sc{{"Natural", 1.0}, {"Cement", 0.5}, {"Modern/Other", 0.0}});
        add<NominalVariable>("wall", "Main wall material (v128)", S,
                             Lv{"Natural/Temporary", "Unimproved Traditional", "Semi-Durable", "Durable/Modern"},
                             Sc{{"Natural/Temporary", 1.0}, {"Unimproved Traditional", 0.67}, {"Semi-Durable", 0.33}, {"Durable/Modern", 0.0}});
        // ---- Adaptive capacity ---------------------------------------------
        add<OrdinalVariable>("v190", "Wealth quintile (v190)", A, Lv{"Poorest", "Poorer", "Middle", "Richer", "Richest"});
        add<OrdinalVariable>("v106", "Education level (v106)", A, Lv{"No education", "Primary", "Secondary", "Higher"});
        add<NominalVariable>("water_jmp", "Drinking water source (JMP, v113)", A,
                             Lv{"Improved", "Unimproved", "Surface Water"},
                             Sc{{"Improved", 1.0}, {"Unimproved", 0.5}, {"Surface Water", 0.0}});
        add<OrdinalVariable>("water_time", "Time to water source (v115)", A,
                             Lv{"On premises", "<= 30 minutes", "> 30 minutes"},
                             Sc{{"On premises", 1.0}, {"<= 30 minutes", 0.5}, {"> 30 minutes", 0.0}});
        add<NominalVariable>("toilet", "Toilet facility (v116)", A,
                             Lv{"Other / Unknown", "Improved", "Unimproved", "Open Defecation"},
                             Sc{{"Improved", 1.0}, {"Unimproved", 0.5}, {"Open Defecation", 0.0}});
        add<NominalVariable>("v169a", "Owns a mobile phone (v169a)", A, Lv{"No", "Yes"});
        add<NominalVariable>("q930a", "Has a bank account (q930a)", A, Lv{"Yes", "No"},
                             Sc{{"Yes", 1.0}, {"No", 0.0}});
        add<OrdinalVariable>("v745b", "Owns land (v745b)", A,
                             Lv{"Does not own", "Alone only", "Jointly only", "Both alone and jointly"});
    }
};

}  // namespace chvi
