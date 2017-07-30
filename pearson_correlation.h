#ifndef PEARSON_CORRELATION_H
#define PEARSON_CORRELATION_H

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
template <typename T>
double pearson_correlation(const size_t n, const T &x, const T &y) {
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

  return s / (sqrt(sx * sy));
}

template <typename T> double pearson_correlation(const T &x, const T &y) {
  size_t n = x.size();
  assert(n == y.size());
  return pearson_correlation(n, x, y);
}
#endif