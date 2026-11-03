// DataStructures.hpp
// ---------------------------------------------------------------------------
// Small reusable data-structure classes used by the system.
//   TopK<T>      bounded min-heap that keeps the k largest items seen so far
//                (O(log k) per insertion) -> "most vulnerable households".
//   AuditLog     bounded FIFO queue (std::deque) of recent actions shown in
//                the menu header / report.
// Other structures used elsewhere: unordered_map (cluster index, region
// aggregation), sorted vector + binary search (classifier), std::map
// (ordered frequency tables), vector (matrices, bootstrap replicates).
// ---------------------------------------------------------------------------
#pragma once

#include <algorithm>
#include <deque>
#include <functional>
#include <queue>
#include <string>
#include <vector>

namespace chvi {

template <typename T>
class TopK {
public:
    explicit TopK(std::size_t k) : k_(k) {}

    void push(const T& item) {
        if (k_ == 0) return;
        if (heap_.size() < k_) {
            heap_.push(item);
        } else if (item > heap_.top()) {  // beats the current k-th largest
            heap_.pop();
            heap_.push(item);
        }
    }

    /// Items in descending order (does not modify the heap).
    std::vector<T> sortedDescending() const {
        auto copy = heap_;
        std::vector<T> out;
        while (!copy.empty()) { out.push_back(copy.top()); copy.pop(); }
        std::reverse(out.begin(), out.end());
        return out;
    }
    std::size_t size() const { return heap_.size(); }

private:
    std::size_t k_;
    std::priority_queue<T, std::vector<T>, std::greater<T>> heap_;  // min-heap
};

class AuditLog {
public:
    explicit AuditLog(std::size_t capacity = 8) : cap_(capacity) {}
    void add(const std::string& msg) {
        q_.push_back(msg);
        if (q_.size() > cap_) q_.pop_front();
    }
    const std::deque<std::string>& entries() const { return q_; }

private:
    std::size_t cap_;
    std::deque<std::string> q_;
};

}  // namespace chvi
