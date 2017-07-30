#include "calculate_hubs.h"

#include <algorithm>
#include <cassert>
#include <iterator>
#include <random>
#include <unordered_map>

#include "distance.h"

std::vector<std::vector<int>> test_MR_method1_2(int n, int M, int Rounds) {
  unsigned seed_ = 42;
  std::mt19937 generator(seed_);
  std::vector<std::vector<int>> ccc(Rounds, std::vector<int>(n));
  for (size_t i = 0; i < ccc.size(); i++)
    for (size_t j = 0; j < ccc[i].size(); j++) {
      std::binomial_distribution<int> distribution(
          n - 1, static_cast<double>(rand()) / (RAND_MAX));
      int bbb1 = std::min(distribution(generator), M);
      int bbb2 = std::min(distribution(generator), M);
      ccc[i][j] = (bbb1 + 1) * (bbb2 + 1);
    }
  return ccc;
}

template <typename T>
std::vector<size_t> partial_sort_indexes(const std::vector<T> &v, const int k) {
  // initialize original index locations
  std::vector<size_t> idx(v.size());
  for (size_t i = 0; i != idx.size(); ++i)
    idx[i] = i;
  // sort indexes based on comparing values in v
  partial_sort(idx.begin(), idx.begin() + k, idx.end(),
               [&v](size_t i1, size_t i2) { return v[i1] < v[i2]; });
  return idx;
}

template <typename T>
std::vector<size_t> partial_sort_indexes_decreasing(const std::vector<T> &v,
                                                    const int k) {
  // initialize original index locations
  std::vector<size_t> idx(v.size());
  for (size_t i = 0; i != idx.size(); ++i)
    idx[i] = i;
  // sort indexes based on comparing values in v
  partial_sort(idx.begin(), idx.begin() + k, idx.end(),
               [&v](size_t i1, size_t i2) { return v[i1] > v[i2]; });
  return idx;
}

std::vector<int>
calculate_MR(size_t tg_id,
             const std::vector<std::unordered_map<int, int>> &Rank_index) {
  size_t N = Rank_index.size();
  size_t K = Rank_index[0].size();
  std::vector<int> MR_result(N);
  for (size_t i = 0; i < N; i++) {
    auto f1 = Rank_index[i].find(tg_id);
    auto f2 = Rank_index[tg_id].find(i);
    int R1 = (f1 == Rank_index[i].end()) ? K : f1->second;
    int R2 = (f2 == Rank_index[tg_id].end()) ? K : f2->second;
    MR_result[i] = (R1 + 1) * (R2 + 1);
  }
  MR_result[tg_id] = (K + 1) * (K + 1);
  return (MR_result);
}

std::vector<double> calculate_growth_rate(const std::vector<double> &x0,
                                          int step = 20) {
  std::vector<double> gr_all(x0.size());
  for (size_t i = 0; i < x0.size(); i++) {
    int sid = std::max(0, static_cast<int>(i) - step);
    int eid = std::min(x0.size() - 1, i + step);
    // tg_id < -c(sid:eid)
    int l = eid - sid + 1;
    gr_all[i] = (x0[eid] - x0[sid]) / l;
  }
  return (gr_all);
}

class IntVectorWithMax {
public:
  IntVectorWithMax(size_t s, int d) : S_all(s, d), max(0), max_index(0) {}
  std::vector<int> S_all;
  int max;
  size_t max_index;
};

IntVectorWithMax growth_rate_scoring(const std::vector<bool> &is_sig,
                                     int up_score = 1, int down_score = -3,
                                     int delete_top = 10) {
  assert(down_score < 0);
  int S = 0;
  IntVectorWithMax all(is_sig.size(), 0);
  for (size_t i = delete_top + 1; i < is_sig.size(); i++) {
    if (is_sig[i]) {
      S += up_score;
      if (S > all.max) {
        all.max = S;
        all.max_index = i;
      }
    } else
      S += down_score;
    all.S_all[i] = S;
  }
  return (all);
}

std::vector<double> growth_rate_significance(
    const std::vector<double> &tg_growth_rate,
    const std::vector<std::vector<double>> &null_growth_rate, size_t sig_N) {
  std::vector<int> growth_rate_sig_sum(sig_N);
  std::vector<double> growth_rate_sig(sig_N);
  for (size_t i = 0; i < sig_N; i++) {
    for (size_t j = 0; j < null_growth_rate.size(); j++) {
      if (tg_growth_rate[i] >= null_growth_rate[j][i])
        growth_rate_sig_sum[i]++;
    }
    growth_rate_sig[i] =
        static_cast<double>(growth_rate_sig_sum[i]) / null_growth_rate.size();
  }
  return (growth_rate_sig);
}

std::vector<double> calcuate_growth_rate_from(std::vector<int> &em,
                                              const int step_size) {
  // Should we use lookup table as
  // https://en.wikibooks.org/wiki/Optimizing_C%2B%2B/General_optimization_techniques/Memoization
  std::vector<double> x0(em.size());
  std::sort(em.begin(), em.end());
  for (size_t j = 0; j < em.size(); j++)
    x0[j] = sqrtl(static_cast<double>(em[j]));
  return calculate_growth_rate(x0, step_size);
}

void update_hub_info(int sig_N, int hub_sig_cut,
                     std::vector<double> tg_growth_rate, IntVectorWithMax all,
                     HubInfo &hub_info) {
  hub_info.cluster_bound = 0;
  hub_info.if_hub = 0;
  if (all.max > hub_sig_cut) {
    hub_info.cluster_bound = all.max_index + 1;
    hub_info.if_hub = 1;
  }
  if (hub_info.cluster_bound == sig_N) {
    for (size_t i = 0; i < tg_growth_rate.size(); i++) {
      if (tg_growth_rate[i] != 0)
        hub_info.index = i + 1;
    }
  }
}

std::vector<HubInfo>
calculate_hubs(const std::vector<std::vector<double>> &data,
               const double TN_p /*= 0.0025*/, const int k /*= 1000L*/) {
  int step_size = 50;
  std::vector<std::vector<int>> ddd2 = test_MR_method1_2(data.size(), k, 1000);
  std::vector<std::vector<double>> null_growth_rate(
      ddd2.size(), std::vector<double>(ddd2[0].size()));

#pragma omp parallel for
  for (int i = 0; i < static_cast<int>(ddd2.size()); i++) {
    null_growth_rate[i] = calcuate_growth_rate_from(ddd2[i], step_size);
  }

  std::vector<std::unordered_map<int, int>> Rank_index(data.size());
  std::vector<size_t> MR_id_count(data.size(), 0);

#pragma omp parallel for
  for (int i = 0; i < static_cast<int>(data.size()); i++) {
    std::vector<double> cor_cc(data.size());
    for (int j = 0; j < i; j++)
      cor_cc[j] = dist(data[i], data[j]);
    cor_cc[i] = std::numeric_limits<double>::max();
    for (size_t j = i + 1; j < data.size(); j++)
      cor_cc[j] = dist(data[i], data[j]);
    std::vector<size_t> idx = partial_sort_indexes(cor_cc, k);
    // works if Rank_index[i] is vector std::copy(idx.begin(), idx.begin() + k,
    // Rank_index[i].begin());
    for (int j = 0; j < k; j++)
      Rank_index[i][idx[j]] = j;
    for (int j = 0; j < k; j++)
      MR_id_count[idx[j]]++;
  }
  size_t TN = static_cast<size_t>(data.size() * TN_p);
  std::vector<size_t> MR_id = partial_sort_indexes_decreasing(
      MR_id_count, TN); // We only need first TN MR_id
  std::vector<std::vector<int>> MR_EM(TN);

#pragma omp parallel for
  for (int i = 0; i < static_cast<int>(TN); i++)
    MR_EM[i] = calculate_MR(MR_id[i], Rank_index);

  int sig_N = 750;
  double p_sig = 0.01;
  int hub_sig_cut = 100;
  std::vector<HubInfo> hub_info_all(MR_EM.size());
  auto func = [=](const double &n) { return n < p_sig; };

#pragma omp parallel for
  for (int i = 0; i < static_cast<int>(MR_EM.size()); i++) {
    std::vector<double> tg_growth_rate =
        calcuate_growth_rate_from(MR_EM[i], step_size);
    std::vector<double> growth_rate_sig =
        growth_rate_significance(tg_growth_rate, null_growth_rate, sig_N);
    std::vector<bool> is_sig;
    is_sig.reserve(sig_N);
    std::transform(growth_rate_sig.begin(), growth_rate_sig.end(),
                   std::back_inserter(is_sig), func);
    IntVectorWithMax all = growth_rate_scoring(is_sig);
    update_hub_info(sig_N, hub_sig_cut, tg_growth_rate, all, hub_info_all[i]);
  }
  return hub_info_all;
}
