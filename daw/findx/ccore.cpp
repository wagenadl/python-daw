// ccore.cpp - part of findx

#include <cstdint>
#include <vector>
#include <functional>

typedef uint64_t idx_t;
typedef int64_t stride_t;

template <class T, class OP> idx_t findfirst(T const *vec,
                                             idx_t count, stride_t stride,
                                             OP op, T cf) {
  idx_t idx = 0;
  while (count>0) {
    if (op(*vec, cf))
      return idx;
    idx++;
    count--;
    vec += stride;
  }
  return idx;
}

extern "C" {

#define make_findfirst(name, func, typ)                              \
  idx_t findfirst_##name##_##typ(typ const *vec, idx_t count, stride_t stride, \
                                 typ cf) {                             \
    return findfirst(vec, count, stride, std::func<typ>(), cf); \
  }

#define make_findfirsts(typ) \
  make_findfirst(ge, greater_equal, typ) \
  make_findfirst(gt, greater, typ) \
  make_findfirst(le, less_equal, typ) \
  make_findfirst(lt, less, typ) \
  make_findfirst(eq, equal_to, typ) \
  make_findfirst(ne, not_equal_to, typ)

  make_findfirsts(float)
  make_findfirsts(double)
  make_findfirsts(int64_t)
  make_findfirsts(uint64_t)
  make_findfirsts(int32_t)
  make_findfirsts(uint32_t)
  make_findfirsts(int8_t)
  make_findfirsts(uint8_t)
  make_findfirsts(bool)
  
}
