#ifndef DISTANCE_H
#define DISTANCE_H

#include <cassert> // assert
#include <cstddef> // std::size_t
#include <vector>

struct ds {
  explicit ds(const size_t n) : d(n), s(0) {}
  std::vector<long double> d;
  long double s;
};

/*
@article{welford1962note,
title={Note on a method for calculating corrected sums of squares and products},
author={Welford, BP},
journal={Technometrics},
volume={4},
number={3},
pages={419--420},
year={1962},
publisher={Taylor \& Francis Group}
}
*/

template <typename T> ds d(const std::size_t n, const T &x) {
  ds ret(n);
  long double mx = x[0];
  for (std::size_t i = 1; i < n; i++) {
    const long double r = i / (i + 1.0);
    ret.d[i] = x[i] - mx;
    mx += ret.d[i] / (i + 1.0);
    ret.s += ret.d[i] * ret.d[i] * r;
  }
  return ret;
}

inline double ddist(const std::size_t n, const ds &dx, const ds &dy) {
  long double s = 0.0;

  for (std::size_t i = 1; i < n; i++) {
    const long double r = i / (i + 1.0);
    s += dx.d[i] * dy.d[i] * r;
  }
  // pearson_correlation should be s / (sqrt(sx * sy));
  return static_cast<double>(dx.s / s * dy.s / s);
}

template <typename T> double dist(const std::size_t n, const T &x, const T &y) {
  return ddist(n, d(n, x), d(n, y));
}

template <typename T> double dist(const T &x, const T &y) {
  std::size_t n = x.size();
  assert(n == y.size());
  return dist(n, x, y);
}

template <typename T>
unsigned char dist8(const std::size_t n, const T &x, const T &y) {
  const int result = log2(dist(n, x, y)) * 8;
  return std::max(static_cast<int>(std::numeric_limits<unsigned char>::max()),
                  result);
}

template <typename T> double dist8(const T &x, const T &y) {
  std::size_t n = x.size();
  assert(n == y.size());
  return dist8(n, x, y);
}
#endif
