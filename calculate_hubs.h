#ifndef CALCULATE_HUBS_H
#define CALCULATE_HUBS_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "distance.h"

struct HubInfo {
  int if_hub;
  int cluster_bound;
  std::size_t index;
};

namespace Internal {
std::vector<std::vector<int>> test_MR_method1_2(int n, int M, int Rounds);

template <typename T>
std::vector<std::size_t> partial_sort_indexes(const std::vector<T> &v,
                                              const int k) {
  // initialize original index locations
  std::vector<std::size_t> idx(v.size());
  for (std::size_t i = 0; i != idx.size(); ++i)
    idx[i] = i;
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
  for (std::size_t i = 0; i != idx.size(); ++i)
    idx[i] = i;
  // sort indexes based on comparing values in v
  partial_sort(idx.begin(), idx.begin() + k, idx.end(),
               [&v](std::size_t i1, std::size_t i2) { return v[i1] > v[i2]; });
  return idx;
}

std::vector<int>
calculate_MR(std::size_t tg_id,
             const std::vector<std::unordered_map<int, int>> &Rank_index);

std::vector<double> calculate_growth_rate(const std::vector<double> &x0,
                                          int step = 20);

class IntVectorWithMax {
public:
  IntVectorWithMax(std::size_t s, int d) : S_all(s, d), max(0), max_index(0) {}
  std::vector<int> S_all;
  int max;
  std::size_t max_index;
};

IntVectorWithMax growth_rate_scoring(const std::vector<bool> &is_sig,
                                     int up_score = 1, int down_score = -3,
                                     int delete_top = 10);

std::vector<double> growth_rate_significance(
    const std::vector<double> &tg_growth_rate,
    const std::vector<std::vector<double>> &null_growth_rate,
    std::size_t sig_N);

std::vector<double> calcuate_growth_rate_from(std::vector<int> em,
                                              const int step_size);

void update_hub_info(int sig_N, int hub_sig_cut,
                     std::vector<double> tg_growth_rate, IntVectorWithMax all,
                     HubInfo &hub_info);
} // namespace Internal

template <typename T> class IMatrix {
public:
  virtual ~IMatrix() {}
  virtual std::size_t RowSize() const = 0;
  virtual std::size_t ColSize() const = 0;
  virtual const T *operator[](std::size_t index) const = 0;
};

template <typename T>
std::vector<HubInfo>
get_hubs(const std::vector<std::vector<int>> &MR_EM,
         const std::vector<std::vector<double>> &null_growth_rate,
         int step_size = 50) {
  int sig_N = 750;
  double p_sig = 0.01;
  int hub_sig_cut = 100;
  std::vector<HubInfo> hub_info_all(MR_EM.size());
  auto func = [=](const double &n) { return n < p_sig; };

#pragma omp parallel for
  for (int i = 0; i < static_cast<int>(MR_EM.size()); i++) {
    std::vector<double> tg_growth_rate =
        Internal::calcuate_growth_rate_from(MR_EM[i], step_size);
    std::vector<double> growth_rate_sig = Internal::growth_rate_significance(
        tg_growth_rate, null_growth_rate, sig_N);
    std::vector<bool> is_sig;
    is_sig.reserve(sig_N);
    std::transform(growth_rate_sig.begin(), growth_rate_sig.end(),
                   std::back_inserter(is_sig), func);
    Internal::IntVectorWithMax all = Internal::growth_rate_scoring(is_sig);
    Internal::update_hub_info(sig_N, hub_sig_cut, tg_growth_rate, all,
                              hub_info_all[i]);
  }
  return hub_info_all;
}

template <typename T>
std::vector<std::vector<double>> get_null_growth_rate(const IMatrix<T> &data,
                                                      const int k = 1000L,
                                                      int step_size = 50) {
  std::vector<std::vector<int>> ddd2 =
      Internal::test_MR_method1_2(data.RowSize(), k, 1000);

  std::vector<std::vector<double>> null_growth_rate =
      std::vector<std::vector<double>>(ddd2.size(),
                                       std::vector<double>(ddd2[0].size()));

#pragma omp parallel for
  for (int i = 0; i < static_cast<int>(ddd2.size()); i++) {
    null_growth_rate[i] =
        Internal::calcuate_growth_rate_from(ddd2[i], step_size);
  }
  return null_growth_rate;
}

template <typename T>
std::vector<std::vector<int>> get_MR_EM(const IMatrix<T> &data,
                                        const double TN_p = 0.0025,
                                        const int k = 1000L) {
  std::vector<std::unordered_map<int, int>> Rank_index(data.RowSize());
  std::vector<std::size_t> MR_id_count(data.RowSize(), 0);

#pragma omp parallel for
  for (int row1 = 0; row1 < static_cast<int>(data.RowSize()); row1++) {
    std::vector<double> cor_cc(data.RowSize());
    for (int row2 = 0; row2 < row1; row2++)
      cor_cc[row2] = dist(data.ColSize(), data[row1], data[row2]);
    cor_cc[row1] = std::numeric_limits<double>::max();
    for (std::size_t row2 = row1 + 1; row2 < data.RowSize(); row2++)
      cor_cc[row2] = dist(data.ColSize(), data[row1], data[row2]);
    std::vector<std::size_t> idx = Internal::partial_sort_indexes(cor_cc, k);
    // works if Rank_index[i] is vector std::copy(idx.begin(), idx.begin() + k,
    // Rank_index[i].begin());
    for (int j = 0; j < k; j++)
      Rank_index[row1][idx[j]] = j;
    for (int j = 0; j < k; j++)
      MR_id_count[idx[j]]++;
  }
  std::size_t TN = static_cast<std::size_t>(data.RowSize() * TN_p);
  std::vector<std::size_t> MR_id = Internal::partial_sort_indexes_decreasing(
      MR_id_count, TN); // We only need indices of TN maximum values

  std::vector<std::vector<int>> MR_EM(TN);

#pragma omp parallel for
  for (int i = 0; i < static_cast<int>(TN); i++)
    MR_EM[i] = Internal::calculate_MR(MR_id[i], Rank_index);

  return MR_EM;
}

template <typename T>
std::vector<HubInfo> calculate_hubs(const IMatrix<T> &data,
                                    const double TN_p = 0.0025,
                                    const int k = 1000L, int step_size = 50) {
  std::vector<std::vector<int>> MR_EM = get_MR_EM<T>(data, TN_p, k);

  std::vector<std::vector<double>> null_growth_rate =
      get_null_growth_rate<T>(data, k, step_size);

  return get_hubs<T>(MR_EM, null_growth_rate, step_size);
}

#endif
