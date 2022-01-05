#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

namespace wenda {

namespace kdtree {

/** Tournament tree implementation for fast replace-top operations.
 *
 */
template <typename T, typename Cmp = std::less<T> > class TournamentTree {
  private:
    typedef std::pair<T, uint32_t> value_t;

    std::vector<value_t> data_;
    std::pair<T, uint32_t> root_;
    Cmp cmp_;

    void update_root_from_index(uint32_t idx) {
        value_t winner = data_[idx];

        while(idx > 1) {
            auto& other = data_[idx / 2];

            if (cmp_(winner.first, other.first)) {
                // previous winner lost
                std::swap(winner, other);
            }
        }

        root_ = std::move(winner);
    }

  public:
    TournamentTree(size_t n, T const &val) : data_(2 * n) {
        for (size_t i = 0; i < n; ++i) {
            data_[i + n] = {val, i + n};
        }

        std::fill_n(data_.begin(), n, value_t{val, n});
    }

    TournamentTree(TournamentTree const&) = default;

    T const& top() const {
        return *root_.first;
    }

    T replace_top(T const &val) {
        if(cmp_(val, root_.first)) {
            return root_.first;
        }

        std::pair<T, uint32_t> value{val, root_.second};
        std::swap(data_[root_.second], value);

        update_root_from_index(root_.second);

        return value.first;
    }

    template<typename OutIt>
    void copy_values(OutIt it) const {
        std::transform(
            data_.begin() + data_.size() / 2, data_.end(), it,
            [](value_t const& v) { return v.first; });
    }
};

} // namespace kdtree

} // namespace wenda
