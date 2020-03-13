// ccore.cpp - part of peakx

#include <cstdint>
#include <vector>
#include <algorithm>

typedef uint64_t idx_t;
typedef int64_t stride_t;

template <class T> idx_t *schmitt(T const *vec, idx_t count, stride_t stride,
				     T upthr, T downthr, idx_t *nout) {
  int k = 0; // number of transitions found so far
  int res = 1024; // reserved space
  std::vector<idx_t> trans;
  trans.resize(res);
  bool isup = false;
  for (idx_t idx=0; idx<count; idx++) {
    if ((isup) ? (*vec <= downthr) : (*vec >= upthr)) {
      if (k>=res) {
	res *= 2;
	trans.resize(res);
      }
      trans[k++] = idx;
      isup = !isup;
    }
    vec += stride;
  }
  idx_t *out = new idx_t[k];
  std::copy_n(trans.begin(), k, out);
  *nout = k;
  return out;
}

extern "C" {
  void schmitt_free(idx_t *trans) {
    delete [] trans;
  }

  idx_t *schmitt_double(double const *vec, idx_t count, stride_t stride,
                        double upthr, double downthr, idx_t *nout) {
    return schmitt<double>(vec, count, stride, upthr, downthr, nout);
  }

  idx_t *schmitt_float(float const *vec, idx_t count, stride_t stride,
                       float upthr, float downthr, idx_t *nout) {
    return schmitt<float>(vec, count, stride, upthr, downthr, nout);
  }
};
