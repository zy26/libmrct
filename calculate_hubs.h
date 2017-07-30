#ifndef CALCULATE_HUBS_H
#define CALCULATE_HUBS_H

#include <cstddef>
#include <cstdint>
#include <vector>

struct HubInfo {
  int if_hub;
  int cluster_bound;
  size_t index;
};

std::vector<HubInfo>
calculate_hubs(const std::vector<std::vector<double>> &data,
               const double TN_p = 0.0025, const int k = 1000L);

#endif
