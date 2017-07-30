#ifndef DISTANCE_H
#define DISTANCE_H

#include <cassert>

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
template <typename T> double dist(const size_t n, const T &x, const T &y) {
  long double sx = 0.0;
  long double sy = 0.0;
  long double s = 0.0;
  long double mx = x[0];
  long double my = y[0];

  for (size_t i = 1; i < n; i++) {
    long double r = i / (i + 1.0);
    long double dx = x[i] - mx;
    long double dy = y[i] - my;
    sx += dx * dx * r;
    sy += dy * dy * r;
    s += dx * dy * r;
    mx += dx / (i + 1.0);
    my += dy / (i + 1.0);
  }
  return sx / s * sy / s; // pearson_correlation should be s / (sqrt(sx * sy));
}

template <typename T> double dist(const T &x, const T &y) {
  size_t n = x.size();
  assert(n == y.size());
  return dist(n, x, y);
}

template <typename T>
unsigned char dist8(const size_t n, const T &x, const T &y) {
  int result = log2(dist(n, x, y)) * 8;
  return std::max(static_cast<int>(std::numeric_limits<unsigned char>::max()),
                  result);
}

template <typename T> double dist8(const T &x, const T &y) {
  size_t n = x.size();
  assert(n == y.size());
  return dist8(n, x, y);
}
#endif