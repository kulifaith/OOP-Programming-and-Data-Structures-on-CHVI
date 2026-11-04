// main.cpp
// ---------------------------------------------------------------------------
// Menu-driven front end for the Climate-Health Vulnerability Index system.
//
//   ./chvi                interactive menu
//   ./chvi --auto         run the complete pipeline once and exit
//   ./chvi --selftest     run the built-in verification suite and exit
//   options: --dta FILE --weather FILE --out DIR --seed N --water-blank-on-premises
// ---------------------------------------------------------------------------
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>

#include "Portable.hpp"
#include "ResearchSystem.hpp"
#include "SelfTest.hpp"

using namespace chvi;

namespace {

/// Reads an integer in [lo, hi]; returns false at end-of-input.
bool readChoice(const std::string& prompt, int lo, int hi, int& out) {
    while (true) {
        std::cout << prompt << std::flush;
        std::string line;
        if (!std::getline(std::cin, line)) return false;
        try {
            std::size_t used = 0;
            const int v = std::stoi(line, &used);
            if (used == line.size() && v >= lo && v <= hi) { out = v; return true; }
        } catch (const std::exception&) {}
        std::cout << "  Please enter a whole number between " << lo << " and " << hi << ".\n";
    }
}

void printMenu(const ResearchSystem& sys) {
    std::cout << "\n+------------------------------------------------------------------+\n"
                 "|  CLIMATE-HEALTH VULNERABILITY INDEX (CHVI) FOR UGANDA            |\n"
                 "|  UDHS 2022 + Open-Meteo climate | FAMD | multilevel logit | chi2 |\n"
                 "+------------------------------------------------------------------+\n"
                 "|  DATA                                                            |\n"
                 "|   1. Load UDHS (.dta) and weather (.csv) data                    |\n"
                 "|   2. Dataset summary (fever prevalence, residence)               |\n"
                 "|   3. Climate anomalies by region (Eq. 2)                         |\n"
                 "|   4. Recoding, missing data and imputation                       |\n"
                 "|  OBJECTIVE 1 : CONSTRUCT THE CHVI                                |\n"
                 "|   5. Factor Analysis of Mixed Data and indicator weights         |\n"
                 "|   6. Construct and classify the CHVI                             |\n"
                 "|   7. Construct validity                                          |\n"
                 "|   8. Robustness checks (components, bootstrap, regression)       |\n"
                 "|  OBJECTIVE 2 : ASSOCIATION WITH CHILD FEVER                      |\n"
                 "|   9. Multilevel logistic regression models (M0-M3)               |\n"
                 "|  OBJECTIVE 3 : RESIDENCE DISPARITIES                             |\n"
                 "|  10. Rural-urban chi-square test                                 |\n"
                 "|  TOOLS                                                           |\n"
                 "|  11. Most vulnerable households and regional ranking             |\n"
                 "|  12. Run the complete pipeline                                   |\n"
                 "|  13. Settings (files, components, bootstrap size, top-k)         |\n"
                 "|  14. Run self-tests                                              |\n"
                 "|  15. Export full report to file                                  |\n"
                 "|  16. Export household-level results (CSV)                        |\n"
                 "|   0. Exit                                                        |\n"
                 "+------------------------------------------------------------------+\n";
    if (!sys.log().entries().empty()) {
        std::cout << "Recent actions: ";
        bool first = true;
        for (const auto& e : sys.log().entries()) { std::cout << (first ? "" : " | ") << e; first = false; }
        std::cout << "\n";
    }
}

std::string askText(const std::string& prompt, std::string& target) {
    std::cout << prompt << " [" << target << "]: " << std::flush;
    std::string d;
    if (std::getline(std::cin, d) && !d.empty()) target = d;
    return target;
}

void settingsMenu(ResearchSystem& sys) {
    Settings& s = sys.settings();
    while (true) {
        std::cout << "\nSETTINGS\n  1. UDHS .dta file                  : " << s.dtaPath
                  << "\n  2. Weather .csv file               : " << s.weatherPath
                  << "\n  3. Retained FAMD components (K)    : " << s.retainedComponents
                  << "\n  4. Alternative components (robust.) : " << s.alternativeComponents
                  << "\n  5. Bootstrap replicates (B)        : " << s.bootstrapReplicates
                  << "\n  6. Random seed (bootstrap/split)   : " << s.seed
                  << "\n  7. Top-k households to list        : " << s.topK
                  << "\n  9. Quadrature nodes (1 = Laplace)   : " << s.quadratureNodes
                  << "\n  8. Blank v115 = water on premises  : " << (s.blankWaterTimeIsOnPremises ? "yes" : "no (as in R script)")
                  << "\n  0. Back\n";
        int c, v = 0;
        if (!readChoice("Select: ", 0, 9, c) || c == 0) return;
        switch (c) {
            case 1: askText("UDHS .dta path", s.dtaPath); std::cout << "  Use option 1 to (re)load.\n"; break;
            case 2: askText("Weather .csv path", s.weatherPath); std::cout << "  Use option 1 to (re)load.\n"; break;
            case 3: if (readChoice("Components K (1-30): ", 1, 30, v)) { s.retainedComponents = v; sys.resetResults(); } break;
            case 4: if (readChoice("Alternative K (2-30): ", 2, 30, v)) s.alternativeComponents = v; break;
            case 5: if (readChoice("Bootstrap replicates (10-1000): ", 10, 1000, v)) s.bootstrapReplicates = v; break;
            case 6: if (readChoice("Seed (0-999999): ", 0, 999999, v)) s.seed = static_cast<std::uint64_t>(v); break;
            case 7: if (readChoice("Top-k (1-50): ", 1, 50, v)) s.topK = v; break;
            case 9: if (readChoice("Nodes (1,3,5,...,31): ", 1, 31, v)) { s.quadratureNodes = v; sys.resetResults(); } break;
            case 8: s.blankWaterTimeIsOnPremises = !s.blankWaterTimeIsOnPremises; std::cout << "  Toggled. Use option 1 to reload.\n"; break;
        }
    }
}

void pause() {
    std::cout << "\nPress Enter to continue..." << std::flush;
    std::string dummy;
    std::getline(std::cin, dummy);
}

}  // namespace

int main(int argc, char** argv) {
    Settings settings;
    bool autoRun = false, selfTest = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--auto") autoRun = true;
        else if (a == "--selftest") selfTest = true;
        else if (a == "--seed" && i + 1 < argc) settings.seed = std::strtoull(argv[++i], nullptr, 10);
        else if (a == "--dta" && i + 1 < argc) settings.dtaPath = argv[++i];
        else if (a == "--weather" && i + 1 < argc) settings.weatherPath = argv[++i];
        else if (a == "--nodes" && i + 1 < argc) settings.quadratureNodes = std::atoi(argv[++i]);
        else if (a == "--out" && i + 1 < argc) settings.outputDir = argv[++i];
        else if (a == "--water-blank-on-premises") settings.blankWaterTimeIsOnPremises = true;
        else { std::cerr << "Unknown option: " << a << "\n"; return 2; }
    }
    ResearchSystem sys(settings);

    if (selfTest) return SelfTest(std::cout, settings).run() == 0 ? 0 : 1;
    if (autoRun) {
        try {
            sys.runAll(std::cout);
        } catch (const std::exception& e) {
            std::cerr << "\n[error] " << e.what() << "\n"
                      << "Put UGKR7BFL_10PERCENT.dta and weather_Uganda_daily_1991_2022.csv in the 'data' folder\n"
                      << "(next to the program) or pass --dta FILE --weather FILE.\n";
            return 1;
        }
        return 0;
    }

    std::cout << "\nClimate-Health Vulnerability Index for Uganda\n"
                 "Author : Kuli Faith Rombe (2025/HD06/22009U)\n"
				 "OOP Programming and Data Structures Implementation\n";
    while (true) {
        printMenu(sys);
        int choice;
        if (!readChoice("Enter your choice: ", 0, 16, choice)) break;  // EOF
        try {
            switch (choice) {
                case 0: std::cout << "Goodbye.\n"; return 0;
                case 1: sys.loadData(std::cout); break;
                case 2: sys.summary(std::cout); break;
                case 3: sys.climateAnomalies(std::cout); break;
                case 4: sys.cleaningReport(std::cout); break;
                case 5: sys.famdAnalysis(std::cout); break;
                case 6: sys.buildIndex(std::cout); break;
                case 7: sys.constructValidity(std::cout); break;
                case 8: sys.robustness(std::cout); break;
                case 9: sys.feverModels(std::cout); break;
                case 10: sys.residenceDisparity(std::cout); break;
                case 11: sys.topVulnerable(std::cout); break;
                case 12: sys.runAll(std::cout); break;
                case 13: settingsMenu(sys); continue;
                case 14: SelfTest(std::cout, sys.settings()).run(); break;
                case 15: {
                    const std::string path = sys.settings().outputDir + "/chvi_report.txt";
                    makeDir(sys.settings().outputDir);
                    std::ofstream f(path);
                    if (!f) { std::cout << "Cannot write " << path << "\n"; break; }
                    sys.runAll(f);
                    std::cout << "Full report written to " << path << "\n";
                    break;
                }
                case 16: sys.exportHouseholds(sys.settings().outputDir + "/households_chvi.csv", std::cout); break;
            }
        } catch (const std::exception& e) {
            std::cout << "\n[error] " << e.what() << "\n";
        }
        pause();
    }
    return 0;
}
