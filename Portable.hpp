// Portable.hpp
// ---------------------------------------------------------------------------
// Tiny helpers so that the program compiles with old compilers (e.g. the
// MinGW/TDM-GCC 4.9 shipped with Dev-C++, C++11 mode): no <filesystem>, no
// std::to_string, no std::make_unique.
// ---------------------------------------------------------------------------
#pragma once

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#include <sys/types.h>
#endif

namespace chvi {

/// Number -> string (replacement for std::to_string).
template <typename T>
std::string str(T value) {
    std::ostringstream s;
    s << value;
    return s.str();
}

inline bool fileExists(const std::string& path) {
    std::ifstream f(path.c_str());
    return f.good();
}

inline bool removeFile(const std::string& path) { return std::remove(path.c_str()) == 0; }

/// Create a single directory level; succeeds if it already exists.
inline void makeDir(const std::string& path) {
    if (path.empty()) return;
#ifdef _WIN32
    _mkdir(path.c_str());
#else
    mkdir(path.c_str(), 0755);
#endif
}

/// Directory part of a path ("" if none).
inline std::string parentDir(const std::string& path) {
    const std::size_t p = path.find_last_of("/\\");
    return p == std::string::npos ? std::string() : path.substr(0, p);
}

}  // namespace chvi
