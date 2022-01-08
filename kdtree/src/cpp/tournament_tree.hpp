#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <queue>
#include <utility>
#include <vector>

namespace wenda {

namespace kdtree {

namespace detail {
inline std::vector<uint32_t> build_loser_tree_initial(uint32_t n) {
    std::vector<std::pair<uint32_t, uint32_t>> winner_losers(2 * n);

    for (uint32_t i = 0; i < n; ++i) {
        winner_losers[i + n] = {i, i};
    }

    for (uint32_t i = n - 1; i > 0; --i) {
        winner_losers[i] = {
            std::max(winner_losers[2 * i].first, winner_losers[2 * i + 1].first),
            std::min(winner_losers[2 * i].first, winner_losers[2 * i + 1].first)};
    }

    std::vector<uint32_t> result(2 * n);
    std::transform(winner_losers.begin(), winner_losers.end(), result.begin(), [](const auto &p) {
        return p.second;
    });
    return result;
}
} // namespace detail

/** Tournament tree implementation for fast replace-top operations.
 *
 */
template <typename T, typename Cmp = std::less<T>> class TournamentTree {
  protected:
    typedef std::pair<T, uint32_t> value_t;

    std::vector<value_t> data_;
    Cmp cmp_;

    void update_root_from_index(uint32_t idx) {
        value_t winner = data_[idx];

        while (idx > 1) {
            idx /= 2;

            auto &other = data_[idx];

            if (cmp_(winner.first, other.first)) {
                // previous winner lost
                std::swap(winner, other);
            }
        }

        data_[0] = std::move(winner);
    }

  public:
    /** Constructs a tournament tree with the given capacity and value.
     *
     */
    TournamentTree(uint32_t n, T const &val) : data_(2 * n) {
        std::vector<uint32_t> losers = detail::build_loser_tree_initial(n);
        std::transform(losers.begin(), losers.end(), data_.begin(), [&](uint32_t loser) {
            return value_t{val, loser + n};
        });

        data_[0] = value_t{val, 2 * n - 1};
    }

    TournamentTree(TournamentTree const &) = default;
    TournamentTree(TournamentTree &&) = default;

    //! Peek at the top element of the tree.
    T const &top() const { return data_[0].first; }

    //! Replace the top element of the tree with the given value.
    void replace_top(T const &val) {
        std::pair<T, uint32_t> value{val, data_[0].second};
        std::swap(data_[data_[0].second], value);

        update_root_from_index(data_[0].second);
    }

    template <typename OutIt> void copy_values(OutIt it) const {
        std::transform(data_.begin() + data_.size() / 2, data_.end(), it, [](value_t const &v) {
            return v.first;
        });
    }
};

/** An adapter for std::priority_queue which has the same interface as TournamentTree
 *
 */
template <typename T, typename Cmp = std::less<T>>
class PriorityQueue : public std::priority_queue<T, std::vector<T>, Cmp> {
  public:
    PriorityQueue(uint32_t n, T const &val)
        : std::priority_queue<T, std::vector<T>, Cmp>(Cmp{}, std::vector<T>(n, val)) {}

    void replace_top(T const &val) {
        this->pop();
        this->push(val);
    }

    template <typename OutIt> void copy_values(OutIt it) const {
        std::copy(this->c.begin(), this->c.end(), it);
    }
};

} // namespace kdtree

} // namespace wenda
