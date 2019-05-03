#ifndef CALCULATE_HUBS_H
#define CALCULATE_HUBS_H

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <unordered_map>
#include <vector>

#include "distance.h"

struct HubInfo {
  std::size_t if_hub;
  std::size_t cluster_bound;
  std::size_t index;
};

namespace internal {
std::vector<std::vector<int>>
empirical_null_distribution_mr_f(int n, int m, int rounds, unsigned seed);

template <typename T>
std::vector<std::size_t> partial_sort_indexes(const std::vector<T> &v,
                                              const int k) {
  // initialize original index locations
  std::vector<std::size_t> idx(v.size());
  std::iota(idx.begin(), idx.end(), 0);
  // sort indexes based on comparing values in v
  partial_sort(idx.begin(), idx.begin() + k, idx.end(),
               [&v](std::size_t i1, std::size_t i2) { return v[i1] < v[i2]; });
  return idx;
}

template <typename T>
std::vector<std::size_t>
partial_sort_indexes_decreasing(const std::vector<T> &v, const int k) {
  // initialize original index locations
  std::vector<std::size_t> idx(v.size());
  std::iota(idx.begin(), idx.end(), 0);
  // sort indexes based on comparing values in v
  partial_sort(idx.begin(), idx.begin() + k, idx.end(),
               [&v](std::size_t i1, std::size_t i2) { return v[i1] > v[i2]; });
  return idx;
}

std::vector<int>
calculate_mr(std::size_t tg_id,
             const std::vector<std::unordered_map<std::size_t, std::size_t>>
                 &rank_index);

std::vector<double> calculate_growth_rate(const std::vector<double> &x0,
                                          std::size_t step = 20);

class IntVectorWithMax {
public:
  IntVectorWithMax(const std::size_t s, const int d)
      : s_all(s, d), max(0), max_index(0) {}

  std::vector<int> s_all;
  int max;
  std::size_t max_index;
};

IntVectorWithMax growth_rate_scoring(const std::vector<bool> &is_sig,
                                     int up_score = 1, int down_score = -3,
                                     int delete_top = 10);

std::vector<double> growth_rate_significance(
    const std::vector<double> &tg_growth_rate,
    const std::vector<std::vector<double>> &null_growth_rate,
    std::size_t sig_n);

std::vector<double> calculate_growth_rate_from(std::vector<int> em,
                                               std::size_t step_size);

void update_hub_info(std::size_t sig_n, int hub_sig_cut,
                     const std::vector<double> &tg_growth_rate,
                     const IntVectorWithMax &all, int step_size,
                     HubInfo &hub_info, double tg_growth_rate_threshold = 0.7);
} // namespace internal

template <typename T> class MatrixBase {
public:
  MatrixBase(const MatrixBase &) = delete;
  MatrixBase &operator=(const MatrixBase &) = delete;
  virtual std::size_t row_size() const = 0;
  virtual std::size_t col_size() const = 0;
  virtual const T *operator[](std::size_t index) const = 0;

protected:
  MatrixBase() {}
  MatrixBase(MatrixBase &&) = default;
  MatrixBase &operator=(MatrixBase &&) = default;
  virtual ~MatrixBase() {}
};

template <typename T>
std::vector<HubInfo>
identify_hubs(const std::vector<std::vector<int>> &mr_em,
              const std::vector<std::vector<double>> &null_growth_rate,
              const int step_size = 50, const int sig_n = 375,
              const double p_sig = 0.05, const int hub_sig_cut = 100) {
  std::vector<HubInfo> hub_info_all(mr_em.size());
  auto func = [=](const double &n) { return n < p_sig; };

#pragma omp parallel for
  for (std::size_t i = 0; i < mr_em.size(); i++) {
    auto tg_growth_rate =
        internal::calculate_growth_rate_from(mr_em[i], step_size);
    auto growth_rate_sig = internal::growth_rate_significance(
        tg_growth_rate, null_growth_rate, sig_n);
    std::vector<bool> is_sig(sig_n);
    std::transform(growth_rate_sig.begin(), growth_rate_sig.end(),
                   is_sig.begin(), func);
    const auto all = internal::growth_rate_scoring(is_sig);
    update_hub_info(sig_n, hub_sig_cut, tg_growth_rate, all, step_size,
                    hub_info_all[i]);
  }
  return hub_info_all;
}

template <typename T>
std::vector<std::vector<double>>
get_null_growth_rate(const std::vector<std::vector<int>> &mr_e,
                     const int step_size) {
  auto null_growth_rate = std::vector<std::vector<double>>(
      mr_e.size(), std::vector<double>(mr_e[0].size()));

#pragma omp parallel for
  for (std::size_t i = 0; i < mr_e.size(); i++) {
    null_growth_rate[i] =
        internal::calculate_growth_rate_from(mr_e[i], step_size);
  }
  return null_growth_rate;
}

template <typename T>
std::tuple<std::vector<std::size_t>,
           std::vector<std::unordered_map<std::size_t, std::size_t>>>
get_mr_id(const MatrixBase<T> &data, const double tn_p, const std::size_t k,
          const std::size_t &tn) {

  std::vector<std::unordered_map<std::size_t, std::size_t>> rank_index(
      data.row_size());
  std::vector<std::size_t> mr_id_count(data.row_size(), 0);

  std::vector<ds> ds_data(data.row_size(), ds(0));
#pragma omp parallel for
  for (std::size_t row = 0; row < data.row_size(); row++) {
    ds_data[row] = d(data.col_size(), data[row]);
  }

  for (std::size_t row1 = 0; row1 < data.row_size(); row1++) {
    std::vector<double> cor_cc(data.row_size());
#pragma omp parallel for
    for (std::size_t row2 = 0; row2 < data.row_size(); row2++)
      cor_cc[row2] = ddist(data.col_size(), ds_data[row1], ds_data[row2]);
    cor_cc[row1] = std::numeric_limits<double>::max();
    auto idx = internal::partial_sort_indexes(cor_cc, k);
    // works if Rank_index[i] is vector std::copy(idx.begin(), idx.begin() + k,
    // Rank_index[i].begin());
    for (std::size_t j = 0; j < k; j++)
      rank_index[row1][idx[j]] = j;
    for (std::size_t j = 0; j < k; j++)
      mr_id_count[idx[j]]++;
  }

  std::vector<std::size_t> mr_id(tn);

  if (tn_p < 1) {
    mr_id = internal::partial_sort_indexes_decreasing(
        mr_id_count, tn); // We only need indices of TN maximum values
  } else {
    std::iota(mr_id.begin(), mr_id.end(), 0);
  }

  return std::make_tuple(mr_id, rank_index);
}

template <typename T>
std::vector<std::vector<int>>
get_mr_em(const unsigned tn, const std::vector<std::size_t> &mr_id,
          const std::vector<std::unordered_map<std::size_t, std::size_t>>
              &rank_index) {
  std::vector<std::vector<int>> mr_em(tn);

#pragma omp parallel for
  for (std::size_t i = 0; i < tn; i++)
    mr_em[i] = internal::calculate_mr(mr_id[i], rank_index);

  return mr_em;
}

template <typename T>
std::tuple<std::vector<std::vector<int>>, std::vector<std::size_t>>
get_mr_em(const MatrixBase<T> &data, const double tn_p = 1,
          const std::size_t k = 500L) {

  const auto tn = static_cast<std::size_t>(data.row_size() * tn_p);

  std::vector<std::size_t> mr_id;
  std::vector<std::unordered_map<std::size_t, std::size_t>> rank_index;
  std::tie(mr_id, rank_index) = get_mr_id(data, tn_p, k, tn);

  std::vector<std::vector<int>> mr_em = get_mr_em<T>(tn, mr_id, rank_index);

  return std::make_tuple(mr_em, mr_id);
}

template <typename T>
std::vector<HubInfo>
MRHCA_fast_version(const MatrixBase<T> &data,
                   const std::vector<std::vector<int>> &mr_em,
                   const std::size_t k, const std::size_t step_size,
                   const int round = 1000L, const unsigned seed = 42) {
  const std::vector<std::vector<int>> mr_e =
      internal::empirical_null_distribution_mr_f(data.row_size(), k, round,
                                                 seed);
  const std::vector<std::vector<double>> null_growth_rate =
      get_null_growth_rate<T>(mr_e, step_size);

  return identify_hubs<T>(mr_em, null_growth_rate, step_size,
                          static_cast<int>(k * 0.75));
}

template <typename T>
std::tuple<std::vector<HubInfo>, std::vector<std::size_t>,
           std::vector<std::vector<int>>>
calculate_hubs(const MatrixBase<T> &data, const double tn_p = 1,
               std::size_t k = 500L, std::size_t step_size = 50) {
  k = std::min(k, data.row_size());
  step_size = std::min(step_size, data.row_size());
  std::vector<std::vector<int>> mr_em;
  std::vector<std::size_t> mr_id;
  std::tie(mr_em, mr_id) = get_mr_em<T>(data, tn_p, k);

  auto hub = MRHCA_fast_version<T>(data, mr_em, k, step_size);

  return std::make_tuple(hub, mr_id, mr_em);
}

#endif
