#include "calculate_hubs.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iterator>
#include <random>
#include <unordered_map>

std::vector<std::vector<int>>
empirical_null_distribution_mr(const int n, const int rounds,
                               const unsigned seed = 42) {
  std::mt19937 generator(seed);
  std::vector<std::vector<int>> ccc(rounds, std::vector<int>(n));

#pragma omp parallel for
  for (std::size_t i = 0; i < ccc.size(); i++)
    for (auto &id : ccc[i]) {
      std::binomial_distribution<int> distribution(
          n - 1, static_cast<double>(rand()) / RAND_MAX);
      const auto bbb1 = distribution(generator);
      const auto bbb2 = distribution(generator);
      id = (bbb1 + 1) * (bbb2 + 1);
    }
  return ccc;
}

std::vector<std::vector<int>> internal::empirical_null_distribution_mr_f(
    const int n, const int m, const int rounds, const unsigned seed = 42) {
  std::mt19937 generator(seed);
  std::vector<std::vector<int>> ccc(rounds, std::vector<int>(n));

#pragma omp parallel for
  for (std::size_t i = 0; i < ccc.size(); i++)
    for (auto &id : ccc[i]) {
      std::binomial_distribution<int> distribution(
          n - 1, static_cast<double>(rand()) / RAND_MAX);
      const auto bbb1 = std::min(distribution(generator), m);
      const auto bbb2 = std::min(distribution(generator), m);
      id = (bbb1 + 1) * (bbb2 + 1);
    }
  return ccc;
}

std::vector<int> internal::calculate_mr(
    const std::size_t tg_id,
    const std::vector<std::unordered_map<std::size_t, std::size_t>>
        &rank_index) {
  const auto n = rank_index.size();
  const auto k = rank_index[0].size();
  std::vector<int> mr_result(n);
  for (std::size_t i = 0; i < n; i++) {
    auto f1 = rank_index[i].find(tg_id);
    auto f2 = rank_index[tg_id].find(i);
    const auto r1 = f1 == rank_index[i].end() ? k : f1->second;
    const auto r2 = f2 == rank_index[tg_id].end() ? k : f2->second;
    mr_result[i] = (r1 + 1) * (r2 + 1);
  }
  mr_result[tg_id] = (k + 1) * (k + 1);
  return mr_result;
}

std::vector<double>
internal::calculate_growth_rate(const std::vector<double> &x0,
                                const std ::size_t step) {
  std::vector<double> gr_all(x0.size());
  for (std::size_t i = 0; i < x0.size(); i++) {
    const auto sid = std::max(0, static_cast<int>(i) - static_cast<int>(step));
    const int eid = std::min(x0.size() - 1, i + step);
    const auto l = eid - sid + 1;
    gr_all[i] = (x0[eid] - x0[sid]) / l;
  }
  return gr_all;
}

internal::IntVectorWithMax
internal::growth_rate_scoring(const std::vector<bool> &is_sig,
                              const int up_score, const int down_score,
                              const int delete_top) {
  assert(down_score < 0);
  int s = 0;
  IntVectorWithMax all(is_sig.size(), 0);
  for (std::size_t i = delete_top + 1; i < is_sig.size(); i++) {
    if (is_sig[i]) {
      s += up_score;
      if (s > all.max) {
        all.max = s;
        all.max_index = i;
      }
    } else
      s += down_score;
    all.s_all[i] = s;
  }
  return all;
}

std::vector<double> internal::growth_rate_significance(
    const std::vector<double> &tg_growth_rate,
    const std::vector<std::vector<double>> &null_growth_rate,
    const std::size_t sig_n) {
  std::vector<int> growth_rate_sig_sum(sig_n);
  std::vector<double> growth_rate_sig(sig_n);
  for (std::size_t i = 0; i < sig_n; i++) {
    for (const auto &j : null_growth_rate) {
      if (tg_growth_rate[i] >= j[i])
        growth_rate_sig_sum[i]++;
    }
    growth_rate_sig[i] =
        static_cast<double>(growth_rate_sig_sum[i]) / null_growth_rate.size();
  }
  return growth_rate_sig;
}

std::vector<double>
internal::calculate_growth_rate_from(std::vector<int> em,
                                     const std::size_t step_size) {
  // Should we use lookup table as
  // https://en.wikibooks.org/wiki/Optimizing_C%2B%2B/General_optimization_techniques/Memoization
  std::vector<double> x0(em.size());
  std::sort(em.begin(), em.end());
  for (std::size_t j = 0; j < em.size(); j++)
    x0[j] = sqrtl(static_cast<double>(em[j]));
  return calculate_growth_rate(x0, step_size);
}

bool all_less_than(const std::vector<double> &tg_growth_rate,
                   const std::size_t step_size,
                   const double tg_growth_rate_threshold) {
  for (std::size_t i = 0; i < step_size; i++) {
    if (tg_growth_rate[i] > tg_growth_rate_threshold)
      return false;
  }
  return true;
}

void internal::update_hub_info(const std::size_t sig_n, const int hub_sig_cut,
                               const std::vector<double> &tg_growth_rate,
                               const IntVectorWithMax &all, const int step_size,
                               HubInfo &hub_info,
                               const double tg_growth_rate_threshold) {
  hub_info.cluster_bound = 0;
  hub_info.if_hub = 0;
  hub_info.index = 0;
  if (all.max > hub_sig_cut &&
      all_less_than(tg_growth_rate, step_size, tg_growth_rate_threshold)) {
    hub_info.cluster_bound = all.max_index + 1;
    hub_info.if_hub = 1;
  }
  if (hub_info.cluster_bound == sig_n) {
    hub_info.cluster_bound = 0;
    for (std::size_t i = 0; i < tg_growth_rate.size(); i++) {
      if (tg_growth_rate[i] > 0) {
        hub_info.index = i + 1;
      }
    }
  }
}
