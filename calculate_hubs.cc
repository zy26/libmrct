#include "calculate_hubs.h"

#include <algorithm>
#include <cassert>
#include <iterator>
#include <random>
#include <unordered_map>

std::vector<std::vector<int>> Internal::test_MR_method1_2(int n, int M,
                                                          int Rounds) {
  unsigned seed_ = 42;
  std::mt19937 generator(seed_);
  std::vector<std::vector<int>> ccc(Rounds, std::vector<int>(n));
  for (std::size_t i = 0; i < ccc.size(); i++)
    for (std::size_t j = 0; j < ccc[i].size(); j++) {
      std::binomial_distribution<int> distribution(
          n - 1, static_cast<double>(rand()) / RAND_MAX);
      int bbb1 = std::min(distribution(generator), M);
      int bbb2 = std::min(distribution(generator), M);
      ccc[i][j] = (bbb1 + 1) * (bbb2 + 1);
    }
  return ccc;
}

std::vector<int> Internal::calculate_MR(
    std::size_t tg_id,
    const std::vector<std::unordered_map<int, int>> &Rank_index) {
  std::size_t N = Rank_index.size();
  std::size_t K = Rank_index[0].size();
  std::vector<int> MR_result(N);
  for (std::size_t i = 0; i < N; i++) {
    auto f1 = Rank_index[i].find(tg_id);
    auto f2 = Rank_index[tg_id].find(i);
    int R1 = f1 == Rank_index[i].end() ? K : f1->second;
    int R2 = f2 == Rank_index[tg_id].end() ? K : f2->second;
    MR_result[i] = (R1 + 1) * (R2 + 1);
  }
  MR_result[tg_id] = (K + 1) * (K + 1);
  return MR_result;
}

std::vector<double>
Internal::calculate_growth_rate(const std::vector<double> &x0, int step) {
  std::vector<double> gr_all(x0.size());
  for (std::size_t i = 0; i < x0.size(); i++) {
    int sid = std::max(0, static_cast<int>(i) - step);
    int eid = std::min(x0.size() - 1, i + step);
    // tg_id < -c(sid:eid)
    int l = eid - sid + 1;
    gr_all[i] = (x0[eid] - x0[sid]) / l;
  }
  return gr_all;
}

Internal::IntVectorWithMax
Internal::growth_rate_scoring(const std::vector<bool> &is_sig, int up_score,
                              int down_score, int delete_top) {
  assert(down_score < 0);
  int S = 0;
  IntVectorWithMax all(is_sig.size(), 0);
  for (std::size_t i = delete_top + 1; i < is_sig.size(); i++) {
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
  return all;
}

std::vector<double> Internal::growth_rate_significance(
    const std::vector<double> &tg_growth_rate,
    const std::vector<std::vector<double>> &null_growth_rate,
    std::size_t sig_N) {
  std::vector<int> growth_rate_sig_sum(sig_N);
  std::vector<double> growth_rate_sig(sig_N);
  for (std::size_t i = 0; i < sig_N; i++) {
    for (std::size_t j = 0; j < null_growth_rate.size(); j++) {
      if (tg_growth_rate[i] >= null_growth_rate[j][i])
        growth_rate_sig_sum[i]++;
    }
    growth_rate_sig[i] =
        static_cast<double>(growth_rate_sig_sum[i]) / null_growth_rate.size();
  }
  return growth_rate_sig;
}

std::vector<double> Internal::calcuate_growth_rate_from(std::vector<int> em,
                                                        const int step_size) {
  // Should we use lookup table as
  // https://en.wikibooks.org/wiki/Optimizing_C%2B%2B/General_optimization_techniques/Memoization
  std::vector<double> x0(em.size());
  std::sort(em.begin(), em.end());
  for (std::size_t j = 0; j < em.size(); j++)
    x0[j] = sqrtl(static_cast<double>(em[j]));
  return calculate_growth_rate(x0, step_size);
}

void Internal::update_hub_info(int sig_N, int hub_sig_cut,
                               std::vector<double> tg_growth_rate,
                               IntVectorWithMax all, HubInfo &hub_info) {
  hub_info.cluster_bound = 0;
  hub_info.if_hub = 0;
  if (all.max > hub_sig_cut) {
    hub_info.cluster_bound = all.max_index + 1;
    hub_info.if_hub = 1;
  }
  if (hub_info.cluster_bound == sig_N) {
    for (std::size_t i = 0; i < tg_growth_rate.size(); i++) {
      if (tg_growth_rate[i] != 0)
        hub_info.index = i + 1;
    }
  }
}
