// StataReader.hpp
// ---------------------------------------------------------------------------
// Minimal reader for Stata .dta files (format 117/118, little-endian) so that
// the DHS Kids Recode can be read directly by the C++ program.  Only the
// information needed here is parsed: variable names/types and the numeric
// (or fixed-string) data block.  Stata missing values become NaN.
// ---------------------------------------------------------------------------
#pragma once

#include <cmath>
#include "Portable.hpp"
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace chvi {

class StataReader {
public:
    explicit StataReader(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("cannot open Stata file: " + path);
        buf_.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
        parse();
    }

    std::size_t numVars() const { return names_.size(); }
    std::size_t numObs() const { return nObs_; }
    int release() const { return release_; }
    bool hasVar(const std::string& name) const { return index_.count(name) > 0; }

    /// Values of one numeric variable; Stata missing values -> NaN.
    std::vector<double> numeric(const std::string& name) const {
        const auto it = index_.find(name);
        if (it == index_.end()) throw std::runtime_error("variable not found in .dta: " + name);
        const std::size_t v = it->second;
        const std::uint16_t t = types_[v];
        std::vector<double> out(nObs_, std::numeric_limits<double>::quiet_NaN());
        for (std::size_t r = 0; r < nObs_; ++r) {
            const unsigned char* p = ptr(dataStart_ + r * rowLen_ + colOffset_[v]);
            out[r] = decode(t, p);
        }
        return out;
    }

private:
    std::vector<char> buf_;
    int release_ = 0;
    std::size_t nObs_ = 0, rowLen_ = 0, dataStart_ = 0;
    std::vector<std::string> names_;
    std::vector<std::uint16_t> types_;
    std::vector<std::size_t> colOffset_;
    std::unordered_map<std::string, std::size_t> index_;

    const unsigned char* ptr(std::size_t off) const {
        if (off >= buf_.size()) throw std::runtime_error(".dta file truncated or corrupt");
        return reinterpret_cast<const unsigned char*>(buf_.data()) + off;
    }
    template <typename T>
    T read(std::size_t off) const {
        T v;
        if (off + sizeof(T) > buf_.size()) throw std::runtime_error(".dta file truncated or corrupt");
        std::memcpy(&v, buf_.data() + off, sizeof(T));
        return v;
    }
    std::size_t find(const std::string& tag, std::size_t from = 0) const {
        const std::string s(buf_.begin() + static_cast<long>(from), buf_.end());
        const std::size_t p = s.find(tag);
        if (p == std::string::npos) throw std::runtime_error("unsupported .dta layout: missing " + tag);
        return from + p;
    }

    static std::size_t typeSize(std::uint16_t t) {
        if (t >= 1 && t <= 2045) return t;
        switch (t) {
            case 32768: return 8;   // strL
            case 65526: return 8;   // double
            case 65527: return 4;   // float
            case 65528: return 4;   // long
            case 65529: return 2;   // int
            case 65530: return 1;   // byte
        }
        throw std::runtime_error("unknown Stata variable type " + chvi::str(t));
    }

    static double decode(std::uint16_t t, const unsigned char* p) {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        switch (t) {
            case 65530: { const int v = static_cast<std::int8_t>(*p); return v > 100 ? nan : v; }
            case 65529: { std::int16_t v; std::memcpy(&v, p, 2); return v > 32740 ? nan : v; }
            case 65528: { std::int32_t v; std::memcpy(&v, p, 4); return v > 2147483620 ? nan : v; }
            case 65527: { float v; std::memcpy(&v, p, 4); return (std::isnan(v) || v >= 1.7014117331926443e38f) ? nan : v; }
            case 65526: { double v; std::memcpy(&v, p, 8); return (std::isnan(v) || v >= 8.98846567431158e307) ? nan : v; }
            default: return nan;  // strings are not numeric
        }
    }

    void parse() {
        if (buf_.size() < 100 || std::string(buf_.begin(), buf_.begin() + 11) != "<stata_dta>")
            throw std::runtime_error("not a Stata 13+ .dta file (XML-style header missing)");
        const std::size_t rel = find("<release>") + 9;
        release_ = std::atoi(std::string(buf_.begin() + static_cast<long>(rel), buf_.begin() + static_cast<long>(rel) + 3).c_str());
        if (release_ != 117 && release_ != 118)
            throw std::runtime_error("unsupported .dta release " + chvi::str(release_) + " (need 117 or 118)");
        const std::size_t bo = find("<byteorder>") + 11;
        if (std::string(buf_.begin() + static_cast<long>(bo), buf_.begin() + static_cast<long>(bo) + 3) != "LSF")
            throw std::runtime_error("big-endian .dta files are not supported");
        const std::size_t kPos = find("<K>") + 3;
        const std::size_t nPos = find("<N>") + 3;
        const std::size_t K = read<std::uint16_t>(kPos);
        nObs_ = (release_ == 118) ? read<std::uint64_t>(nPos) : read<std::uint32_t>(nPos);

        const std::size_t mapPos = find("<map>") + 5;
        std::uint64_t map[14];
        for (int i = 0; i < 14; ++i) map[i] = read<std::uint64_t>(mapPos + 8u * static_cast<unsigned>(i));
        const std::size_t typesPos = map[2] + std::string("<variable_types>").size();
        const std::size_t namesPos = map[3] + std::string("<varnames>").size();
        dataStart_ = map[9] + std::string("<data>").size();
        const std::size_t nameLen = (release_ == 118) ? 129 : 33;

        types_.resize(K);
        names_.resize(K);
        colOffset_.resize(K);
        std::size_t off = 0;
        for (std::size_t v = 0; v < K; ++v) {
            types_[v] = read<std::uint16_t>(typesPos + 2 * v);
            colOffset_[v] = off;
            off += typeSize(types_[v]);
            const char* s = buf_.data() + namesPos + v * nameLen;
            std::size_t len = 0;
            while (len < nameLen && s[len] != '\0') ++len;
            names_[v] = std::string(s, len);
            index_[names_[v]] = v;
        }
        rowLen_ = off;
        if (dataStart_ + nObs_ * rowLen_ > buf_.size()) throw std::runtime_error(".dta data block exceeds file size");
    }
};

}  // namespace chvi
