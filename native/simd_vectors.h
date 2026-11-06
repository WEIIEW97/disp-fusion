#ifndef DISP_FUSION_SIMD_VECTORS_H
#define DISP_FUSION_SIMD_VECTORS_H
/* Optional compiler-vector primitives. The algorithm has a scalar fallback;
 * These operations can be replaced by target-DSP vector intrinsics.
 * memcpy avoids alignment and strict-aliasing assumptions about caller arrays.
 */
#include <cstdint>
#include <cstring>
#include <limits>
namespace dfv {
#ifdef __AVX2__
constexpr int lanes = 8;
#else
constexpr int lanes = 4;
#endif
typedef float F __attribute__((vector_size(lanes * sizeof(float))));
typedef int32_t I __attribute__((vector_size(lanes * sizeof(int32_t))));
typedef uint8_t B __attribute__((vector_size(lanes)));
inline F f(float x) {
  F v;
  for (int i = 0; i < lanes; ++i)
    v[i] = x;
  return v;
}
inline I k(int32_t x) {
  I v;
  for (int i = 0; i < lanes; ++i)
    v[i] = x;
  return v;
}
inline F load(const float *p) {
  F v;
  std::memcpy(&v, p, sizeof(v));
  return v;
}
inline void store(float *p, F v) { std::memcpy(p, &v, sizeof(v)); }
inline I bits(F v) {
  I a;
  std::memcpy(&a, &v, sizeof(v));
  return a;
}
inline F real(I v) {
  F a;
  std::memcpy(&a, &v, sizeof(v));
  return a;
}
inline I bytes(const uint8_t *p) {
  B b;
  std::memcpy(&b, p, sizeof(b));
  return __builtin_convertvector(b, I);
}
inline void store_bytes(uint8_t *p, I v) {
  B b = __builtin_convertvector(v, B);
  std::memcpy(p, &b, sizeof(b));
}
inline F select(I mask, F a, F b) {
  return real((mask & bits(a)) | (~mask & bits(b)));
}
inline F minimum(F a, F b) { return select(b < a, b, a); }
inline F maximum(F a, F b) { return select(a < b, b, a); }
inline F abs(F v) { return real(bits(v) & k(0x7fffffff)); }
inline I finite(F v) { return (bits(v) & k(0x7f800000)) != k(0x7f800000); }
inline int sum(I v) {
  int out = 0;
  for (int i = 0; i < lanes; ++i)
    out += v[i];
  return out;
}

inline size_t initialize(size_t n, const float *stereo, const float *mono,
                         uint8_t *valid, float &lo, float &hi, int &count) {
  const float inf = std::numeric_limits<float>::infinity();
  F low = f(inf), high = f(-inf);
  I counts{};
  size_t i = 0;
  for (; i + lanes <= n; i += lanes) {
    const F s = load(stereo + i), m = load(mono + i);
    const I vs = finite(s) & (s > f(.05f)), vm = finite(m);
    store_bytes(valid + i, (vs & k(1)) | (vm & k(2)));
    counts += vs & k(1);
    low = minimum(low, select(vm, m, f(inf)));
    high = maximum(high, select(vm, m, f(-inf)));
  }
  lo = inf;
  hi = -inf;
  for (int j = 0; j < lanes; ++j) {
    lo = std::min(lo, low[j]);
    hi = std::max(hi, high[j]);
  }
  count = sum(counts);
  return i;
}
inline size_t normalize(size_t n, const float *mono, const uint8_t *valid,
                        float *guide, float lo, float scale) {
  size_t i = 0;
  for (; i + lanes <= n; i += lanes)
    store(guide + i, select((bytes(valid + i) & k(2)) != k(0),
                            (load(mono + i) - f(lo)) * f(scale), f(0)));
  return i;
}

template <bool Protected>
inline int raw_row(int w, size_t row, const float *stereo, const float *guide,
                   const uint8_t *valid, const uint8_t *protect, uint8_t *raw,
                   float mono_edge, float stereo_jump, float anchor_limit) {
  int x = 1;
  for (; x + lanes <= w - 1; x += lanes) {
    const size_t i = row + x;
    const I v = bytes(valid + i), vl = bytes(valid + i - 1),
            vr = bytes(valid + i + 1), vu = bytes(valid + i - w),
            vd = bytes(valid + i + w);
    const F s = load(stereo + i), m = load(guide + i);
    const F sl = load(stereo + i - 1), sr = load(stereo + i + 1),
            su = load(stereo + i - w), sd = load(stereo + i + w);
    const I me = ((((v & vl) & k(2)) != k(0)) &
                  (abs(m - load(guide + i - 1)) > f(mono_edge))) |
                 ((((v & vr) & k(2)) != k(0)) &
                  (abs(m - load(guide + i + 1)) > f(mono_edge))) |
                 ((((v & vu) & k(2)) != k(0)) &
                  (abs(m - load(guide + i - w)) > f(mono_edge))) |
                 ((((v & vd) & k(2)) != k(0)) &
                  (abs(m - load(guide + i + w)) > f(mono_edge)));
    const I se =
        ((((v & vl) & k(1)) != k(0)) & (abs(s - sl) > f(stereo_jump))) |
        ((((v & vr) & k(1)) != k(0)) & (abs(s - sr) > f(stereo_jump))) |
        ((((v & vu) & k(1)) != k(0)) & (abs(s - su) > f(stereo_jump))) |
        ((((v & vd) & k(1)) != k(0)) & (abs(s - sd) > f(stereo_jump)));
    const I count = (vl & k(1)) + (vr & k(1)) + (vu & k(1)) + (vd & k(1));
    const F total = select((vl & k(1)) != k(0), sl, f(0)) +
                    select((vr & k(1)) != k(0), sr, f(0)) +
                    select((vu & k(1)) != k(0), su, f(0)) +
                    select((vd & k(1)) != k(0), sd, f(0));
    const F denominator = maximum(__builtin_convertvector(count, F), f(1));
    const I anchor = (v == k(3)) & (count >= k(2)) &
                     (abs(s - total / denominator) <= f(anchor_limit));
    I frozen{};
    if (Protected)
      frozen = (bytes(protect + i) != k(0)) & k(4);
    store_bytes(raw + i, (me & k(1)) | (se & k(2)) | frozen | (anchor & k(8)));
  }
  return x;
}

struct Progress {
  int end;
  int changes;
};
inline Progress smooth_row(int begin, int end, size_t row, const float *stereo,
                           float *output, const uint8_t *valid,
                           const uint8_t *guard, const uint8_t *removed,
                           const uint8_t *raw, uint8_t *smoothed,
                           float inv_tile, int left, bool last, float lw,
                           float rw, float la, float ra, float lc, float rc,
                           float strength, float cap) {
  int x = begin;
  I changes{};
  F offsets{};
  for (int j = 0; j < lanes; ++j)
    offsets[j] = float(j);
  for (; x + lanes <= end; x += lanes) {
    const size_t i = row + x;
    const I eligible = (bytes(guard + i) == k(0)) & (bytes(valid + i) == k(3)) &
                       (bytes(removed + i) == k(0)) &
                       ((bytes(raw + i) & k(16)) == k(0));
    const F position = (f(float(x)) + offsets + f(.5f)) * f(inv_tile) - f(.5f);
    const F t =
        last ? f(0) : maximum(f(0), minimum(f(1), position - f(float(left))));
    const F support = f(lw) + t * f(rw - lw);
    const F prediction =
        (f(la) + t * f(ra - la)) * position + f(lc) + t * f(rc - lc);
    const F s = load(stereo + i);
    const F proposal = maximum(
        s - f(cap),
        minimum(s + f(cap), s + f(strength) * (prediction - support * s)));
    const I use = eligible & (proposal > f(.05f)) & finite(proposal);
    const F result = select(use, proposal, load(output + i));
    store(output + i, result);
    const I changed = use & (abs(result - s) > f(.05f));
    store_bytes(smoothed + i, changed & k(1));
    changes += changed & k(1);
  }
  return {x, sum(changes)};
}
} // namespace dfv
#endif
