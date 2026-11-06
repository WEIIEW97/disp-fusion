#ifndef DISP_FUSION_EDGE_FIRST_H
#define DISP_FUSION_EDGE_FIRST_H
#include "fast_aligned.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>
#ifdef DF_FAST_VECTOR_TYPES
#include "simd_vectors.h"
#endif

namespace dfe {
inline float median5(float a, float b, float c, float d, float e) {
  const auto swap = [](float &x, float &y) {
    const float lo = std::min(x, y), hi = std::max(x, y);
    x = lo;
    y = hi;
  };
  swap(a, b);
  swap(c, d);
  swap(a, c);
  swap(b, d);
  swap(b, c);
  return std::max(b, std::min(c, e));
}
#ifdef DF_FAST_VECTOR_TYPES
inline dfv::F median5(dfv::F a, dfv::F b, dfv::F c, dfv::F d, dfv::F e) {
  const auto swap = [](dfv::F &x, dfv::F &y) {
    const auto lo = dfv::minimum(x, y), hi = dfv::maximum(x, y);
    x = lo;
    y = hi;
  };
  swap(a, b);
  swap(c, d);
  swap(a, c);
  swap(b, d);
  swap(b, c);
  return dfv::maximum(b, dfv::minimum(c, e));
}
#endif
inline float median3(float a, float b, float c) {
  return std::max(std::min(a, b), std::min(std::max(a, b), c));
}
inline void sort3(float &a, float &b, float &c) {
  float lo = std::min(a, b), hi = std::max(a, b);
  a = lo;
  b = hi;
  lo = std::min(b, c);
  hi = std::max(b, c);
  b = lo;
  c = hi;
  lo = std::min(a, b);
  hi = std::max(a, b);
  a = lo;
  b = hi;
}
inline float median9(std::array<float, 9> v) {
  sort3(v[0], v[1], v[2]);
  sort3(v[3], v[4], v[5]);
  sort3(v[6], v[7], v[8]);
  return median3(std::max(v[0], std::max(v[3], v[6])),
                 median3(v[1], v[4], v[7]),
                 std::min(v[2], std::min(v[5], v[8])));
}
#ifdef DF_FAST_VECTOR_TYPES
inline dfv::F median3(dfv::F a, dfv::F b, dfv::F c) {
  return dfv::maximum(dfv::minimum(a, b), dfv::minimum(dfv::maximum(a, b), c));
}
inline void sort3(dfv::F &a, dfv::F &b, dfv::F &c) {
  auto lo = dfv::minimum(a, b), hi = dfv::maximum(a, b);
  a = lo;
  b = hi;
  lo = dfv::minimum(b, c);
  hi = dfv::maximum(b, c);
  b = lo;
  c = hi;
  lo = dfv::minimum(a, b);
  hi = dfv::maximum(a, b);
  a = lo;
  b = hi;
}
inline dfv::F median9(std::array<dfv::F, 9> v) {
  sort3(v[0], v[1], v[2]);
  sort3(v[3], v[4], v[5]);
  sort3(v[6], v[7], v[8]);
  return median3(dfv::maximum(v[0], dfv::maximum(v[3], v[6])),
                 median3(v[1], v[4], v[7]),
                 dfv::minimum(v[2], dfv::minimum(v[5], v[8])));
}
#endif

/* Edge preparation uses only line rings: binomial Gaussian mono gradients,
 * locally filled separable median-5 gradients and median-consistent anchors.
 * It is intentionally closer to V1 than raw-neighbor guards, but not identical
 * to Gaussian sigma=1 / EDT fill / median-5 in the Python reference. */
class Preparation {
  int w, h;
  std::vector<float> horizontal, horizontal_weight, blur;
  void mono_gradient(int y, const float *center, const float *above,
                     const float *below, uint8_t *raw, const uint8_t *protect,
                     float threshold) {
    const float fy = y == 0 || y == h - 1 ? 1.f : .5f;
    int x = 1;
#ifdef DF_FAST_VECTOR_TYPES
    for (; x + dfv::lanes <= w - 1; x += dfv::lanes) {
      const auto dx =
          (dfv::load(center + x + 1) - dfv::load(center + x - 1)) * dfv::f(.5f);
      const auto dy =
          (dfv::load(below + x) - dfv::load(above + x)) * dfv::f(fy);
      auto flags =
          ((dx * dx + dy * dy) > dfv::f(threshold * threshold)) & dfv::k(1);
      if (protect)
        flags |= (dfv::bytes(protect + x) != dfv::k(0)) & dfv::k(4);
      dfv::store_bytes(raw + x, flags);
    }
#endif
    const auto pixel = [&](int p) {
      const int l = std::max(0, p - 1), r = std::min(w - 1, p + 1);
      const float dx = (center[r] - center[l]) / (r - l),
                  dy = (below[p] - above[p]) * fy;
      raw[p] = (dx * dx + dy * dy > threshold * threshold ? 1 : 0) |
               (protect && protect[p] ? 4 : 0);
    };
    pixel(0);
    for (; x < w - 1; ++x)
      pixel(x);
    pixel(w - 1);
  }
  void stereo_gradient(int y, const float *center, const float *above,
                       const float *below, const float *stereo,
                       const uint8_t *valid, uint8_t *raw, float threshold,
                       float deviation) {
    const float fy = y == 0 || y == h - 1 ? 1.f : .5f;
    int x = 1;
#ifdef DF_FAST_VECTOR_TYPES
    for (; x + dfv::lanes <= w - 1; x += dfv::lanes) {
      const auto dx =
          (dfv::load(center + x + 1) - dfv::load(center + x - 1)) * dfv::f(.5f);
      const auto dy =
          (dfv::load(below + x) - dfv::load(above + x)) * dfv::f(fy);
      const auto anchor =
          (dfv::bytes(valid + x) == dfv::k(3)) &
          (dfv::abs(dfv::load(stereo + x) - dfv::load(center + x)) <=
           dfv::f(deviation));
      const auto flags =
          (((dx * dx + dy * dy) > dfv::f(threshold * threshold)) & dfv::k(2)) |
          (anchor & dfv::k(8));
      dfv::store_bytes(raw + x, dfv::bytes(raw + x) | flags);
    }
#endif
    const auto pixel = [&](int p) {
      const int l = std::max(0, p - 1), r = std::min(w - 1, p + 1);
      const float dx = (center[r] - center[l]) / (r - l),
                  dy = (below[p] - above[p]) * fy;
      raw[p] |=
          (dx * dx + dy * dy > threshold * threshold ? 2 : 0) |
          (valid[p] == 3 && std::abs(stereo[p] - center[p]) <= deviation ? 8
                                                                         : 0);
    };
    pixel(0);
    for (; x < w - 1; ++x)
      pixel(x);
    pixel(w - 1);
  }

public:
  Preparation(int width, int height, bool enabled)
      : w(width), h(height), horizontal(enabled ? 5 * w : 0),
        horizontal_weight(enabled ? 5 * w : 0), blur(enabled ? 3 * w : 0) {}
  size_t bytes() const {
    return (horizontal.capacity() + horizontal_weight.capacity() +
            blur.capacity()) *
           sizeof(float);
  }
  void run(const float *stereo, const float *guide, const uint8_t *valid,
           const uint8_t *protect, uint8_t *raw, const df_fast_config &cfg) {
    for (int v = -2; v <= h + 1; ++v) {
      const int source = std::max(0, std::min(h - 1, v));
      const float *row = guide + static_cast<size_t>(source) * w;
      const uint8_t *validity = valid + static_cast<size_t>(source) * w;
      float *hr = horizontal.data() + ((v + 2) % 5) * w;
      float *hw = horizontal_weight.data() + ((v + 2) % 5) * w;
      int x = 2;
#ifdef DF_FAST_VECTOR_TYPES
      for (; x + dfv::lanes <= w - 2; x += dfv::lanes) {
        dfv::store(
            hr + x,
            (dfv::load(row + x - 2) + dfv::f(4) * dfv::load(row + x - 1) +
             dfv::f(6) * dfv::load(row + x) +
             dfv::f(4) * dfv::load(row + x + 1) + dfv::load(row + x + 2)) *
                dfv::f(1.f / 16));
        const auto a = __builtin_convertvector(
            dfv::bytes(validity + x - 2) & dfv::k(2), dfv::F);
        const auto b = __builtin_convertvector(
            dfv::bytes(validity + x - 1) & dfv::k(2), dfv::F);
        const auto c = __builtin_convertvector(
            dfv::bytes(validity + x) & dfv::k(2), dfv::F);
        const auto d = __builtin_convertvector(
            dfv::bytes(validity + x + 1) & dfv::k(2), dfv::F);
        const auto e = __builtin_convertvector(
            dfv::bytes(validity + x + 2) & dfv::k(2), dfv::F);
        dfv::store(hw + x,
                   (a + dfv::f(4) * b + dfv::f(6) * c + dfv::f(4) * d + e) *
                       dfv::f(1.f / 32));
      }
#endif
      const auto horizontal_pixel = [&](int p) {
        hr[p] = (row[std::max(0, p - 2)] + 4 * row[std::max(0, p - 1)] +
                 6 * row[p] + 4 * row[std::min(w - 1, p + 1)] +
                 row[std::min(w - 1, p + 2)]) /
                16;
        hw[p] =
            ((validity[std::max(0, p - 2)] & 2) +
             4 * (validity[std::max(0, p - 1)] & 2) + 6 * (validity[p] & 2) +
             4 * (validity[std::min(w - 1, p + 1)] & 2) +
             (validity[std::min(w - 1, p + 2)] & 2)) /
            32.f;
      };
      for (int p = 0; p < std::min(2, w); ++p)
        horizontal_pixel(p);
      for (; x < w; ++x)
        horizontal_pixel(x);
      if (v < 2)
        continue;
      const int y = v - 2;
      float *br = blur.data() + (y % 3) * w;
      const float *a = horizontal.data() + ((v - 2) % 5) * w;
      const float *b = horizontal.data() + ((v - 1) % 5) * w;
      const float *c = horizontal.data() + (v % 5) * w;
      const float *d = horizontal.data() + ((v + 1) % 5) * w;
      const float *e = horizontal.data() + ((v + 2) % 5) * w;
      const float *wa = horizontal_weight.data() + ((v - 2) % 5) * w;
      const float *wb = horizontal_weight.data() + ((v - 1) % 5) * w;
      const float *wc = horizontal_weight.data() + (v % 5) * w;
      const float *wd = horizontal_weight.data() + ((v + 1) % 5) * w;
      const float *we = horizontal_weight.data() + ((v + 2) % 5) * w;
      int p = 0;
#ifdef DF_FAST_VECTOR_TYPES
      for (; p + dfv::lanes <= w; p += dfv::lanes)
        dfv::store(br + p, (dfv::load(a + p) + dfv::f(4) * dfv::load(b + p) +
                            dfv::f(6) * dfv::load(c + p) +
                            dfv::f(4) * dfv::load(d + p) + dfv::load(e + p)) /
                               dfv::maximum(dfv::f(1e-6f),
                                            dfv::load(wa + p) +
                                                dfv::f(4) * dfv::load(wb + p) +
                                                dfv::f(6) * dfv::load(wc + p) +
                                                dfv::f(4) * dfv::load(wd + p) +
                                                dfv::load(we + p)));
#endif
      for (; p < w; ++p)
        br[p] =
            (a[p] + 4 * b[p] + 6 * c[p] + 4 * d[p] + e[p]) /
            std::max(1e-6f, wa[p] + 4 * wb[p] + 6 * wc[p] + 4 * wd[p] + we[p]);
      if (y >= 1) {
        const int out = y - 1;
        const float *current = blur.data() + (out % 3) * w;
        mono_gradient(
            out, current, blur.data() + ((out ? out - 1 : out) % 3) * w, br,
            raw + static_cast<size_t>(out) * w,
            protect ? protect + static_cast<size_t>(out) * w : nullptr,
            cfg.mono_edge);
      }
    }
    mono_gradient(
        h - 1, blur.data() + ((h - 1) % 3) * w, blur.data() + ((h - 2) % 3) * w,
        blur.data() + ((h - 1) % 3) * w, raw + static_cast<size_t>(h - 1) * w,
        protect ? protect + static_cast<size_t>(h - 1) * w : nullptr,
        cfg.mono_edge);
    // Separable median-5: fixed small line rings, not a frame-sized EDT.
    for (int v = -2; v <= h + 1; ++v) {
      const int y = std::max(0, std::min(h - 1, v));
      float *mr = horizontal.data() + ((v + 2) % 5) * w;
      int x = 2;
#ifdef DF_FAST_VECTOR_TYPES
      if (y > 0 && y < h - 1)
        for (; x + dfv::lanes <= w - 2; x += dfv::lanes) {
          const size_t i = static_cast<size_t>(y) * w + x;
          auto fill = dfv::f(0);
          for (int offset : {w, -w, 1, -1, 0})
            fill = dfv::select((dfv::bytes(valid + i + offset) & dfv::k(1)) !=
                                   dfv::k(0),
                               dfv::load(stereo + i + offset), fill);
          std::array<dfv::F, 5> taps;
          int k = 0;
          for (int dx = -2; dx <= 2; ++dx) {
            const size_t j = i + dx;
            taps[k++] =
                dfv::select((dfv::bytes(valid + j) & dfv::k(1)) != dfv::k(0),
                            dfv::load(stereo + j), fill);
          }
          dfv::store(mr + x,
                     median5(taps[0], taps[1], taps[2], taps[3], taps[4]));
        }
#endif
      const auto median_pixel = [&](int px) {
        float fill = 0;
        const int yy[5] = {std::min(h - 1, y + 1), std::max(0, y - 1), y, y, y};
        const int xx[5] = {px, px, std::min(w - 1, px + 1), std::max(0, px - 1),
                           px};
        for (int k = 0; k < 5; ++k) {
          const size_t j = static_cast<size_t>(yy[k]) * w + xx[k];
          if (valid[j] & 1)
            fill = stereo[j];
        }
        std::array<float, 5> taps;
        int k = 0;
        for (int dx = -2; dx <= 2; ++dx) {
          const size_t j = static_cast<size_t>(y) * w +
                           std::max(0, std::min(w - 1, px + dx));
          taps[k++] = valid[j] & 1 ? stereo[j] : fill;
        }
        mr[px] = median5(taps[0], taps[1], taps[2], taps[3], taps[4]);
      };
      for (int p = 0; p < std::min(2, w); ++p)
        median_pixel(p);
      for (; x < w; ++x)
        median_pixel(x);
      if (v < 2)
        continue;
      const int filtered = v - 2;
      float *br = blur.data() + (filtered % 3) * w;
      const float *a = horizontal.data() + ((v - 2) % 5) * w;
      const float *b = horizontal.data() + ((v - 1) % 5) * w;
      const float *c = horizontal.data() + (v % 5) * w;
      const float *d = horizontal.data() + ((v + 1) % 5) * w;
      const float *e = horizontal.data() + ((v + 2) % 5) * w;
      int p = 0;
#ifdef DF_FAST_VECTOR_TYPES
      for (; p + dfv::lanes <= w; p += dfv::lanes)
        dfv::store(br + p,
                   median5(dfv::load(a + p), dfv::load(b + p), dfv::load(c + p),
                           dfv::load(d + p), dfv::load(e + p)));
#endif
      for (; p < w; ++p)
        br[p] = median5(a[p], b[p], c[p], d[p], e[p]);
      if (filtered >= 1) {
        const int out = filtered - 1;
        stereo_gradient(out, blur.data() + (out % 3) * w,
                        blur.data() + ((out ? out - 1 : out) % 3) * w, br,
                        stereo + static_cast<size_t>(out) * w,
                        valid + static_cast<size_t>(out) * w,
                        raw + static_cast<size_t>(out) * w, cfg.stereo_jump_px,
                        cfg.anchor_deviation_px);
      }
    }
    stereo_gradient(h - 1, blur.data() + ((h - 1) % 3) * w,
                    blur.data() + ((h - 2) % 3) * w,
                    blur.data() + ((h - 1) % 3) * w,
                    stereo + static_cast<size_t>(h - 1) * w,
                    valid + static_cast<size_t>(h - 1) * w,
                    raw + static_cast<size_t>(h - 1) * w, cfg.stereo_jump_px,
                    cfg.anchor_deviation_px);
  }
};

struct SampleAxis {
  std::array<int, 16> p{};
  int n = 0, lo = 0, hi = 0;
};
struct BlendAxis {
  std::array<int, 4> p{};
  std::array<float, 4> weight{};
  int n = 0;
};
struct EdgeModel {
  float a = 0, b = 0;
  bool accepted = false;
};
struct Row {
  float a = 0, b = 0, weight = 0;
};
struct Segment {
  int begin = 0, end = 0;
};

// A nearby mono value must itself predict the stereo value within tolerance.
// Do not fill the gaps between distinct foreground/background hypotheses.
inline bool mono_match(int w, int h, int px, int py, const float *guide,
                       const uint8_t *valid, float lo, float hi) {
  if (std::isnan(lo) || std::isnan(hi) || lo > hi)
    return true; // Invalid inverse interval: abstain from deletion.
  for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx) {
      const size_t i =
          static_cast<size_t>(std::max(0, std::min(h - 1, py + dy))) * w +
          std::max(0, std::min(w - 1, px + dx));
      if ((valid[i] & 2) && guide[i] >= lo && guide[i] <= hi)
        return true;
    }
  return false;
}
#ifdef DF_FAST_VECTOR_TYPES
inline dfv::I mono_match_vector(int w, int h, int px, int py,
                                const float *guide, const uint8_t *valid,
                                dfv::F lo, dfv::F hi) {
  dfv::I matches = (lo != lo) | (hi != hi) | (lo > hi);
  if (px < 1 || px + dfv::lanes >= w) {
    for (int k = 0; k < dfv::lanes; ++k)
      matches[k] =
          mono_match(w, h, px + k, py, guide, valid, lo[k], hi[k]) ? -1 : 0;
    return matches;
  }
  for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx) {
      const size_t i =
          static_cast<size_t>(std::max(0, std::min(h - 1, py + dy))) * w + px +
          dx;
      const auto value = dfv::load(guide + i);
      matches |= ((dfv::bytes(valid + i) & dfv::k(2)) != dfv::k(0)) &
                 (value >= lo) & (value <= hi);
    }
  return matches;
}
#endif

// At most sixteen distinct covering windows, collected once per row/segment.
struct GuardModels {
  struct Coefficients {
    float a, b;
  };
  std::array<Coefficients, 16> values;
  int n = 0;
  bool agrees(float stereo, float guide, float blended,
              const df_fast_config &cfg) const {
    if (n < cfg.minimum_votes)
      return false;
    int positive = 0, negative = 0;
    for (int k = 0; k < n; ++k) {
      const float p = values[k].a * guide + values[k].b;
      if (std::isfinite(p)) {
        positive += stereo > p + cfg.reject_threshold_px;
        negative += stereo < p - cfg.reject_threshold_px;
      }
    }
    return (stereo < blended - cfg.reject_threshold_px &&
            negative >= cfg.vote_fraction * n) ||
           (stereo > blended + cfg.reject_threshold_px &&
            positive >= cfg.vote_fraction * n);
  }
#ifdef DF_FAST_VECTOR_TYPES
  dfv::I agrees_vector(dfv::F stereo, dfv::F guide, dfv::F blended,
                       const df_fast_config &cfg) const {
    dfv::I positive{}, negative{};
    for (int k = 0; k < n; ++k) {
      const auto p = dfv::f(values[k].a) * guide + dfv::f(values[k].b);
      positive +=
          (dfv::finite(p) & (stereo > p + dfv::f(cfg.reject_threshold_px))) &
          dfv::k(1);
      negative +=
          (dfv::finite(p) & (stereo < p - dfv::f(cfg.reject_threshold_px))) &
          dfv::k(1);
    }
    const auto threshold = dfv::f(cfg.vote_fraction * n);
    return ((stereo < blended - dfv::f(cfg.reject_threshold_px)) &
            (__builtin_convertvector(negative, dfv::F) >= threshold)) |
           ((stereo > blended + dfv::f(cfg.reject_threshold_px)) &
            (__builtin_convertvector(positive, dfv::F) >= threshold));
  }
#endif
};

class Rejection {
  int w, h, gw, gh, side;
  df_fast_config cfg;
  std::vector<SampleAxis> sample_x, sample_y;
  std::vector<BlendAxis> blend_x, blend_y;
  std::vector<EdgeModel> models;
  std::vector<Row> row;
  std::vector<Segment> segments;
  std::vector<int> prefix;
  std::array<float, 256> x{}, y{}, weights{}, residual{}, work{};
  static SampleAxis samples(int center, int length, int radius, int side) {
    SampleAxis a;
    a.lo = std::max(0, center - radius);
    a.hi = std::min(length, center + radius + 1);
    a.n = std::min(side, a.hi - a.lo);
    for (int k = 0; k < a.n; ++k)
      a.p[k] = a.lo + (a.n == 1 ? 0 : k * (a.hi - a.lo - 1) / (a.n - 1));
    return a;
  }
  static BlendAxis blend(int pixel, int count, int radius, int stride) {
    BlendAxis a;
    for (int k = std::max(0, (pixel - radius + stride - 1) / stride);
         k < count && k * stride <= pixel + radius; ++k) {
      a.p[a.n] = k;
      a.weight[a.n++] = std::max(.05f, 1.f - std::abs(pixel - k * stride) /
                                                 float(radius + 1));
    }
    return a;
  }
  float quantile(const float *data, int n, float q) {
    std::copy(data, data + n, work.begin());
    const float at = (n - 1) * q;
    const int k = int(at);
    std::nth_element(work.begin(), work.begin() + k, work.begin() + n);
    const float lo = work[k];
    if (k + 1 >= n)
      return lo;
    const float hi = *std::min_element(work.begin() + k + 1, work.begin() + n);
    return lo + (hi - lo) * (at - k);
  }
  EdgeModel fit(int n) {
    EdgeModel model;
    if (n < cfg.edge_min_samples)
      return model;
    const float span =
        quantile(x.data(), n, .95f) - quantile(x.data(), n, .05f);
    if (span < cfg.edge_min_mono_span)
      return model;
    const float x0 = x[0], y0 = y[0];
    float a = 0, b = 0;
    std::fill(weights.begin(), weights.begin() + n, 1.f);
    for (int iteration = 0; iteration < cfg.edge_iterations; ++iteration) {
      float sw = 0, sx = 0, sy = 0, sxx = 0, sxy = 0;
#ifdef DF_FAST_SIMD
#pragma omp simd reduction(+ : sw, sx, sy, sxx, sxy)
#endif
      for (int i = 0; i < n; ++i) {
        const float vx = x[i] - x0, vy = y[i] - y0, wi = weights[i];
        sw += wi;
        sx += wi * vx;
        sy += wi * vy;
        sxx += wi * vx * vx;
        sxy += wi * vx * vy;
      }
      const float variance = sxx - sx * sx / sw;
      if (!(variance > 1e-10f * sw) || !std::isfinite(variance))
        return model;
      a = (sxy - sx * sy / sw) / variance;
      b = (sy - a * sx) / sw;
      if (!std::isfinite(a) || !std::isfinite(b))
        return model;
#ifdef DF_FAST_SIMD
#pragma omp simd
#endif
      for (int i = 0; i < n; ++i)
        residual[i] = (y[i] - y0) - (a * (x[i] - x0) + b);
      if (iteration + 1 == cfg.edge_iterations)
        break;
      const float center = quantile(residual.data(), n, .5f);
#ifdef DF_FAST_SIMD
#pragma omp simd
#endif
      for (int i = 0; i < n; ++i)
        weights[i] = std::abs(residual[i] - center);
      const float scale = std::max(cfg.edge_scale_floor_px,
                                   1.4826f * quantile(weights.data(), n, .5f));
#ifdef DF_FAST_SIMD
#pragma omp simd
#endif
      for (int i = 0; i < n; ++i)
        weights[i] = std::min(1.f, 1.5f * scale /
                                       std::max(std::abs(residual[i]), 1e-8f));
    }
    if (!(a > 0) || !std::isfinite(a) || !std::isfinite(b))
      return model;
    for (int i = 0; i < n; ++i)
      weights[i] = std::abs(residual[i]);
    if (quantile(weights.data(), n, .8f) > cfg.edge_fit_p80_px)
      return model;
    model.a = a;
    model.b = y0 + b - a * x0;
    model.accepted = std::isfinite(model.b);
    return model;
  }

public:
  Rejection(int width, int height, const df_fast_config &c)
      : w(width), h(height), gw((w + c.edge_stride - 1) / c.edge_stride),
        gh((h + c.edge_stride - 1) / c.edge_stride),
        side(int(std::sqrt(float(c.edge_samples)))), cfg(c),
        sample_x(c.edge_mode ? gw : 0), sample_y(c.edge_mode ? gh : 0),
        blend_x(c.edge_mode ? w : 0), blend_y(c.edge_mode ? h : 0),
        models(c.edge_mode ? gw * gh : 0), row(c.edge_mode ? gw : 0),
        prefix(c.edge_mode ? (gw + 1) * (gh + 1) : 0) {
    if (!cfg.edge_mode)
      return;
    const int radius = cfg.edge_window / 2;
    for (int i = 0; i < gw; ++i)
      sample_x[i] = samples(i * cfg.edge_stride, w, radius, side);
    for (int i = 0; i < gh; ++i)
      sample_y[i] = samples(i * cfg.edge_stride, h, radius, side);
    for (int i = 0; i < w; ++i)
      blend_x[i] = blend(i, gw, radius, cfg.edge_stride);
    for (int begin = 0; begin < w;) {
      int end = begin + 1;
      while (end < w && blend_x[end].n == blend_x[begin].n &&
             blend_x[end].p == blend_x[begin].p)
        ++end;
      segments.push_back({begin, end});
      begin = end;
    }
    for (int i = 0; i < h; ++i)
      blend_y[i] = blend(i, gh, radius, cfg.edge_stride);
  }
  size_t bytes() const {
    return sample_x.capacity() * sizeof(SampleAxis) +
           sample_y.capacity() * sizeof(SampleAxis) +
           (blend_x.capacity() + blend_y.capacity()) * sizeof(BlendAxis) +
           models.capacity() * sizeof(EdgeModel) +
           row.capacity() * sizeof(Row) + prefix.capacity() * sizeof(int) +
           segments.capacity() * sizeof(Segment);
  }
  void run(const float *stereo, const float *guide, const uint8_t *valid,
           uint8_t *raw, uint8_t *guard, const uint8_t *edge,
           const uint8_t *protect, float *output, uint8_t *removed,
           df_fast_stats &stats) {
    std::fill(prefix.begin(), prefix.end(), 0);
    for (int cy = 0; cy < gh; ++cy)
      for (int cx = 0; cx < gw; ++cx) {
        bool found = false;
        for (int py = cy * cfg.edge_stride;
             py < std::min(h, (cy + 1) * cfg.edge_stride) && !found; ++py)
          for (int px = cx * cfg.edge_stride;
               px < std::min(w, (cx + 1) * cfg.edge_stride); ++px) {
            const size_t i = static_cast<size_t>(py) * w + px;
            if (edge[i] && valid[i] == 3 && (!protect || !protect[i])) {
              found = true;
              break;
            }
          }
        const int p = (cy + 1) * (gw + 1) + cx + 1;
        prefix[p] =
            found + prefix[p - 1] + prefix[p - gw - 1] - prefix[p - gw - 2];
      }
    for (int cy = 0; cy < gh; ++cy)
      for (int cx = 0; cx < gw; ++cx) {
        auto &model = models[cy * gw + cx];
        model = {};
        const auto &sx = sample_x[cx];
        const auto &sy = sample_y[cy];
        const int x0 = sx.lo / cfg.edge_stride,
                  x1 = (sx.hi - 1) / cfg.edge_stride + 1,
                  y0 = sy.lo / cfg.edge_stride,
                  y1 = (sy.hi - 1) / cfg.edge_stride + 1;
        if (prefix[y1 * (gw + 1) + x1] - prefix[y0 * (gw + 1) + x1] -
                prefix[y1 * (gw + 1) + x0] + prefix[y0 * (gw + 1) + x0] ==
            0)
          continue;
        ++stats.edge_attempted_windows;
        int n = 0;
        for (int jy = 0; jy < sy.n; ++jy)
          for (int ix = 0; ix < sx.n; ++ix) {
            const size_t i = static_cast<size_t>(sy.p[jy]) * w + sx.p[ix];
            if (valid[i] == 3 && !guard[i] && (raw[i] & 8)) {
              x[n] = guide[i];
              y[n++] = stereo[i];
            }
          }
        if (n >= cfg.edge_min_samples) {
          ++stats.edge_sampled_windows;
          stats.edge_samples_total += n;
          model = fit(n);
        }
        stats.alignment_models += model.accepted;
      }
    for (int py = 0; py < h; ++py) {
      const auto &by = blend_y[py];
      for (int cx = 0; cx < gw; ++cx) {
        Row a;
        for (int k = 0; k < by.n; ++k) {
          const auto &m = models[by.p[k] * gw + cx];
          if (m.accepted) {
            a.a += by.weight[k] * m.a;
            a.b += by.weight[k] * m.b;
            a.weight += by.weight[k];
          }
        }
        row[cx] = a;
      }
      for (const auto &segment : segments) {
        int px = segment.begin;
        GuardModels validators;
        if (cfg.edge_mode == 2) {
          const auto &bx = blend_x[px];
          for (int iy = 0; iy < by.n; ++iy)
            for (int ix = 0; ix < bx.n; ++ix) {
              const auto &model = models[by.p[iy] * gw + bx.p[ix]];
              if (model.accepted)
                validators.values[validators.n++] = {model.a, model.b};
            }
        }
#ifdef DF_FAST_VECTOR_TYPES
        const auto &indices = blend_x[px];
        dfv::F offsets;
        for (int k = 0; k < dfv::lanes; ++k)
          offsets[k] = float(k);
        dfv::I tested{}, covered{}, deleted{};
        for (; px < segment.end && px + dfv::lanes <= w; px += dfv::lanes) {
          const size_t i = static_cast<size_t>(py) * w + px;
          const auto position = dfv::f(float(px)) + offsets;
          auto eligible = (position < dfv::f(float(segment.end))) &
                          (dfv::bytes(edge + i) != dfv::k(0)) &
                          (dfv::bytes(valid + i) == dfv::k(3)) &
                          ((dfv::bytes(raw + i) & dfv::k(4)) == dfv::k(0));
          if (dfv::sum(eligible & dfv::k(1)) == 0)
            continue;
          dfv::F a{}, b{}, weight{};
          for (int k = 0; k < indices.n; ++k) {
            const auto &v = row[indices.p[k]];
            const auto wk = dfv::maximum(
                dfv::f(.05f),
                dfv::f(1.f) -
                    dfv::abs(position -
                             dfv::f(float(indices.p[k] * cfg.edge_stride))) /
                        dfv::f(float(cfg.edge_window / 2 + 1)));
            a += wk * dfv::f(v.a);
            b += wk * dfv::f(v.b);
            weight += wk * dfv::f(v.weight);
          }
          const auto prediction = (a * dfv::load(guide + i) + b) / weight;
          const auto has_model = eligible & (weight > dfv::f(0));
          auto reject = has_model & dfv::finite(prediction) &
                        (dfv::abs(dfv::load(stereo + i) - prediction) >
                         dfv::f(cfg.reject_threshold_px));
          if (cfg.edge_mode == 2) {
            const auto candidate = reject;
            if (validators.n < cfg.minimum_votes)
              reject = dfv::k(0);
            else if (dfv::sum(reject & dfv::k(1)) != 0) {
              const auto s = dfv::load(stereo + i);
              const auto lo =
                  ((s - dfv::f(cfg.reject_threshold_px)) * weight - b) / a;
              const auto hi =
                  ((s + dfv::f(cfg.reject_threshold_px)) * weight - b) / a;
              const auto matches =
                  mono_match_vector(w, h, px, py, guide, valid, lo, hi);
              reject &= (a > dfv::f(0)) & ~matches &
                        validators.agrees_vector(s, dfv::load(guide + i),
                                                 prediction, cfg);
            }
            stats.uncertain_pixels +=
                dfv::sum((candidate & ~reject) & dfv::k(1));
          }
          dfv::store(
              output + i,
              dfv::select(reject,
                          dfv::f(std::numeric_limits<float>::quiet_NaN()),
                          dfv::load(output + i)));
          dfv::store_bytes(removed + i,
                           dfv::bytes(removed + i) | (reject & dfv::k(1)));
          tested += eligible & dfv::k(1);
          covered += has_model & dfv::k(1);
          deleted += reject & dfv::k(1);
        }
        stats.edge_pixels += dfv::sum(tested);
        stats.edge_covered_pixels += dfv::sum(covered);
        stats.uncertain_pixels += dfv::sum(tested) - dfv::sum(covered);
        stats.removed_pixels += dfv::sum(deleted);
#endif
        for (; px < segment.end; ++px) {
          const size_t i = static_cast<size_t>(py) * w + px;
          if (!edge[i] || valid[i] != 3 || (protect && protect[i]))
            continue;
          ++stats.edge_pixels;
          const auto &bx = blend_x[px];
          float a = 0, b = 0, weight = 0;
          for (int k = 0; k < bx.n; ++k) {
            const auto &v = row[bx.p[k]];
            a += bx.weight[k] * v.a;
            b += bx.weight[k] * v.b;
            weight += bx.weight[k] * v.weight;
          }
          if (weight > 0) {
            ++stats.edge_covered_pixels;
            const float prediction = (a * guide[i] + b) / weight;
            bool reject =
                std::isfinite(prediction) &&
                std::abs(stereo[i] - prediction) > cfg.reject_threshold_px;
            if (reject && cfg.edge_mode == 2) {
              const float lo = ((stereo[i] - cfg.reject_threshold_px) * weight -
                                b) /
                               a,
                          hi = ((stereo[i] + cfg.reject_threshold_px) * weight -
                                b) /
                               a;
              reject = a > 0 &&
                       !mono_match(w, h, px, py, guide, valid, lo, hi) &&
                       validators.agrees(stereo[i], guide[i], prediction, cfg);
              stats.uncertain_pixels += !reject;
            }
            if (reject) {
              output[i] = std::numeric_limits<float>::quiet_NaN();
              removed[i] = 1;
              ++stats.removed_pixels;
            }
          } else
            ++stats.uncertain_pixels;
        }
      }
    }
    // All rejection-band samples are excluded from secondary plane anchors and
    // smoothing, including retained points with unavailable/conflicting models.
    for (size_t i = 0; i < static_cast<size_t>(w) * h; ++i)
      guard[i] |= edge[i] || removed[i];
  }
};
} // namespace dfe
#endif
