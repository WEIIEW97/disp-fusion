#include "fast_aligned.h"
#include "edge_first.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <new>
#include <vector>
#ifdef DF_FAST_VECTOR_TYPES
#include "simd_vectors.h"
#endif
#if defined(__GNUC__) || defined(__clang__)
#define DF_RESTRICT __restrict__
#else
#define DF_RESTRICT
#endif
#ifdef DF_FAST_PROFILE
#include <chrono>
#endif

namespace {
constexpr uint8_t VALID_S = 1, VALID_M = 2;
struct Moments {
  int n = 0, barriers = 0, area = 0;
  float d0 = 0, m0 = 0;
  float x = 0, y = 0, xx = 0, xy = 0, yy = 0;
  float d = 0, xd = 0, yd = 0, dd = 0, m = 0, xm = 0, ym = 0, mm = 0, dm = 0;
};
struct Model {
  float a = 0, b = 0, c = 0, affine_a = 0, affine_base = 0, affine_m0 = 0;
  bool plane = false, affine = false;
};
struct Axis {
  int i0, i1;
  float t, position;
};
struct Segment {
  int begin, end, left, right;
};
bool valid_config(const df_fast_config &c) {
  if ((c.tile_size != 8 && c.tile_size != 16 && c.tile_size != 32) ||
      c.plane_radius_cells < 1 || c.plane_radius_cells > 4 ||
      c.alignment_radius_cells < 1 || c.alignment_radius_cells > 8 ||
      c.boundary_guard_px < 0 || c.boundary_guard_px > 16 ||
      c.edge_band_px < c.boundary_guard_px || c.edge_band_px > 32 ||
      c.minimum_anchors < 3 || c.minimum_votes < 1 || c.minimum_votes > 4)
    return false;
  if (c.edge_mode < 0 || c.edge_mode > 2 || c.edge_stride < 8 ||
      c.edge_stride > 64 || c.edge_window < 7 || c.edge_window % 2 != 1 ||
      c.edge_window > 4 * c.edge_stride - 1 || c.edge_samples < 16 ||
      c.edge_samples > 256 || c.edge_iterations < 1 || c.edge_iterations > 8 ||
      c.edge_min_samples < 4 ||
      c.edge_min_samples > int(std::sqrt(float(c.edge_samples))) *
                               int(std::sqrt(float(c.edge_samples))) ||
      c.edge_anchor_guard_px < 0 || c.edge_anchor_guard_px > 16)
    return false;
  const float positive[] = {
      c.mono_edge,          c.stereo_jump_px,      c.anchor_deviation_px,
      c.plane_rms_px,       c.mono_plane_rms,      c.affine_rms_px,
      c.minimum_mono_std,   c.reject_threshold_px, c.edge_fit_p80_px,
      c.edge_min_mono_span, c.edge_scale_floor_px};
  for (float v : positive)
    if (!std::isfinite(v) || v <= 0)
      return false;
  return std::isfinite(c.minimum_coverage) && c.minimum_coverage > 0 &&
         c.minimum_coverage <= 1 && std::isfinite(c.vote_fraction) &&
         c.vote_fraction > .5f && c.vote_fraction <= 1 &&
         std::isfinite(c.smooth_strength) && c.smooth_strength >= 0 &&
         c.smooth_strength <= 1 && std::isfinite(c.max_change_px) &&
         c.max_change_px >= 0;
}
void merge(Moments &a, const Moments &z, float sx, float sy) {
  a.barriers += z.barriers;
  a.area += z.area;
  if (!z.n)
    return;
  if (!a.n) {
    a.d0 = z.d0;
    a.m0 = z.m0;
  }
  const float n = static_cast<float>(z.n), d = z.d0 - a.d0, m = z.m0 - a.m0;
  a.x += z.x + n * sx;
  a.y += z.y + n * sy;
  a.xx += z.xx + 2 * sx * z.x + n * sx * sx;
  a.xy += z.xy + sx * z.y + sy * z.x + n * sx * sy;
  a.yy += z.yy + 2 * sy * z.y + n * sy * sy;
  a.d += z.d + n * d;
  a.m += z.m + n * m;
  a.xd += z.xd + sx * z.d + d * (z.x + n * sx);
  a.yd += z.yd + sy * z.d + d * (z.y + n * sy);
  a.xm += z.xm + sx * z.m + m * (z.x + n * sx);
  a.ym += z.ym + sy * z.m + m * (z.y + n * sy);
  a.dd += z.dd + 2 * d * z.d + n * d * d;
  a.mm += z.mm + 2 * m * z.m + n * m * m;
  a.dm += z.dm + d * z.m + m * z.d + n * d * m;
  a.n += z.n;
}
bool overlap(const void *a, size_t na, const void *b, size_t nb) {
  if (!a || !b)
    return false;
  const uintptr_t aa = reinterpret_cast<uintptr_t>(a),
                  bb = reinterpret_cast<uintptr_t>(b);
  return aa <= bb ? bb - aa < na : aa - bb < nb;
}

template <bool HasProtected>
void raw_interior(int w, int h, const float *DF_RESTRICT stereo,
                  const float *DF_RESTRICT guide,
                  const uint8_t *DF_RESTRICT valid, const uint8_t *protect,
                  uint8_t *DF_RESTRICT raw, const df_fast_config cfg) {
  for (int y = 1; y < h - 1; ++y) {
    const size_t row = static_cast<size_t>(y) * w;
    int begin = 1;
#ifdef DF_FAST_VECTOR_TYPES
    begin = dfv::raw_row<HasProtected>(w, row, stereo, guide, valid, protect,
                                       raw, cfg.mono_edge, cfg.stereo_jump_px,
                                       cfg.anchor_deviation_px);
#endif
#if defined(DF_FAST_SIMD) && !defined(DF_FAST_VECTOR_TYPES)
#pragma omp simd
#endif
    for (int x = begin; x < w - 1; ++x) {
      const size_t i = row + x;
      const uint8_t v = valid[i], vl = valid[i - 1], vr = valid[i + 1],
                    vu = valid[i - w], vd = valid[i + w];
      const float s = stereo[i], m = guide[i];
      const bool me = unsigned(bool(v & VALID_M)) &
                      ((unsigned(bool(vl & VALID_M)) &
                        unsigned(std::abs(m - guide[i - 1]) > cfg.mono_edge)) |
                       (unsigned(bool(vr & VALID_M)) &
                        unsigned(std::abs(m - guide[i + 1]) > cfg.mono_edge)) |
                       (unsigned(bool(vu & VALID_M)) &
                        unsigned(std::abs(m - guide[i - w]) > cfg.mono_edge)) |
                       (unsigned(bool(vd & VALID_M)) &
                        unsigned(std::abs(m - guide[i + w]) > cfg.mono_edge)));
      const bool se =
          unsigned(bool(v & VALID_S)) &
          ((unsigned(bool(vl & VALID_S)) &
            unsigned(std::abs(s - stereo[i - 1]) > cfg.stereo_jump_px)) |
           (unsigned(bool(vr & VALID_S)) &
            unsigned(std::abs(s - stereo[i + 1]) > cfg.stereo_jump_px)) |
           (unsigned(bool(vu & VALID_S)) &
            unsigned(std::abs(s - stereo[i - w]) > cfg.stereo_jump_px)) |
           (unsigned(bool(vd & VALID_S)) &
            unsigned(std::abs(s - stereo[i + w]) > cfg.stereo_jump_px)));
      const int count = !!(vl & VALID_S) + !!(vr & VALID_S) + !!(vu & VALID_S) +
                        !!(vd & VALID_S);
      const float sum = ((vl & VALID_S) ? stereo[i - 1] : 0.f) +
                        ((vr & VALID_S) ? stereo[i + 1] : 0.f) +
                        ((vu & VALID_S) ? stereo[i - w] : 0.f) +
                        ((vd & VALID_S) ? stereo[i + w] : 0.f);
      const bool anchor =
          (v == (VALID_S | VALID_M)) & (count >= 2) &
          (std::abs(s - sum / std::max(count, 1)) <= cfg.anchor_deviation_px);
      const bool frozen = HasProtected ? protect[i] != 0 : false;
      raw[i] =
          (me ? 1 : 0) | (se ? 2 : 0) | (frozen ? 4 : 0) | (anchor ? 8 : 0);
    }
  }
}
} // namespace

struct df_fast_context {
  int w, h, gw, gh;
  size_t n;
  df_fast_config cfg;
  std::vector<float> guide;
  std::vector<uint8_t> valid, raw, temporary, guard, edge;
  std::vector<int> columns;
  std::vector<Moments> cells;
  std::vector<Model> models;
  std::vector<Axis> xs, ys;
  std::vector<Segment> segments;
  dfe::Preparation edge_preparation;
  dfe::Rejection edge_rejection;
  size_t bytes;
  df_fast_context(int width, int height, const df_fast_config &c)
      : w(width), h(height), gw((w + c.tile_size - 1) / c.tile_size),
        gh((h + c.tile_size - 1) / c.tile_size), n(static_cast<size_t>(w) * h),
        cfg(c), guide(n), valid(n), raw(n), temporary(n), guard(n), edge(n),
        columns(w), cells(static_cast<size_t>(gw) * gh), models(cells.size()),
        xs(w), ys(h), edge_preparation(w, h, c.edge_mode != 0),
        edge_rejection(w, h, c) {
    const float inv = 1.f / c.tile_size;
    for (int x = 0; x < w; ++x)
      xs[x] = axis(x, gw, inv);
    for (int y = 0; y < h; ++y)
      ys[y] = axis(y, gh, inv);
    for (int x = 0; x < w;) {
      const int begin = x, left = xs[x].i0, right = xs[x].i1;
      while (x < w && xs[x].i0 == left && xs[x].i1 == right)
        ++x;
      segments.push_back({begin, x, left, right});
    }
    bytes = sizeof(*this) + n * (sizeof(float) + 5 * sizeof(uint8_t)) +
            w * sizeof(int) + cells.size() * (sizeof(Moments) + sizeof(Model)) +
            (w + h) * sizeof(Axis) + segments.capacity() * sizeof(Segment) +
            edge_preparation.bytes() + edge_rejection.bytes();
  }
  static Axis axis(int p, int count, float inv) {
    const float position = (p + .5f) * inv - .5f;
    const float g =
        std::max(0.f, std::min(static_cast<float>(count - 1), position));
    const int lo = static_cast<int>(g);
    return {lo, std::min(lo + 1, count - 1), g - lo, position};
  }
  template <int R> void dilate_fixed(uint8_t mask, std::vector<uint8_t> &dest) {
    const int width = w, height = h;
    const uint8_t *source = raw.data();
    uint8_t *scratch = temporary.data();
    uint8_t *target = dest.data();
    for (int y = 0; y < height; ++y) {
      const size_t row = static_cast<size_t>(y) * width;
      const uint8_t *src = source + row;
      uint8_t *tmp = scratch + row;
      for (int x = 0; x < std::min(R, width); ++x) {
        uint8_t v = 0;
        for (int k = std::max(0, x - R); k <= std::min(width - 1, x + R); ++k)
          v |= src[k];
        tmp[x] = (v & mask) != 0;
      }
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC ivdep
#endif
      for (int x = R; x < width - R; ++x) {
        uint8_t v = 0;
        for (int k = -R; k <= R; ++k)
          v |= src[x + k];
        tmp[x] = (v & mask) != 0;
      }
      for (int x = std::max(R, width - R); x < width; ++x) {
        uint8_t v = 0;
        for (int k = std::max(0, x - R); k <= std::min(width - 1, x + R); ++k)
          v |= src[k];
        tmp[x] = (v & mask) != 0;
      }
    }
    for (int y = 0; y < height; ++y) {
      const uint8_t *rows[2 * R + 1];
      for (int k = -R; k <= R; ++k)
        rows[k + R] = scratch + static_cast<size_t>(
                                    std::max(0, std::min(height - 1, y + k))) *
                                    width;
      uint8_t *out = target + static_cast<size_t>(y) * width;
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC ivdep
#endif
      for (int x = 0; x < width; ++x) {
        uint8_t v = 0;
        for (int k = 0; k < 2 * R + 1; ++k)
          v |= rows[k][x];
        out[x] = v;
      }
    }
  }
  void dilate(uint8_t mask, int radius, std::vector<uint8_t> &dest) {
    if (radius == 3) {
      dilate_fixed<3>(mask, dest);
      return;
    }
    if (radius == 6) {
      dilate_fixed<6>(mask, dest);
      return;
    }
    if (radius == 2) {
      dilate_fixed<2>(mask, dest);
      return;
    }
    if (radius == 4) {
      dilate_fixed<4>(mask, dest);
      return;
    }
    for (int y = 0; y < h; ++y) {
      const size_t row = static_cast<size_t>(y) * w;
      int sum = 0;
      for (int x = 0; x <= std::min(radius, w - 1); ++x)
        sum += (raw[row + x] & mask) != 0;
      for (int x = 0; x < w; ++x) {
        temporary[row + x] = sum > 0;
        if (x - radius >= 0)
          sum -= (raw[row + x - radius] & mask) != 0;
        if (x + radius + 1 < w)
          sum += (raw[row + x + radius + 1] & mask) != 0;
      }
    }
    std::fill(columns.begin(), columns.end(), 0);
    for (int y = 0; y <= std::min(radius, h - 1); ++y)
      for (int x = 0; x < w; ++x)
        columns[x] += temporary[static_cast<size_t>(y) * w + x];
    for (int y = 0; y < h; ++y) {
      const size_t row = static_cast<size_t>(y) * w;
      for (int x = 0; x < w; ++x)
        dest[row + x] = columns[x] > 0;
      if (y - radius >= 0)
        for (int x = 0; x < w; ++x)
          columns[x] -= temporary[static_cast<size_t>(y - radius) * w + x];
      if (y + radius + 1 < h)
        for (int x = 0; x < w; ++x)
          columns[x] += temporary[static_cast<size_t>(y + radius + 1) * w + x];
    }
  }
  Moments window(int cx, int cy, int radius) const {
    Moments a;
    for (int y = std::max(0, cy - radius); y <= std::min(gh - 1, cy + radius);
         ++y)
      for (int x = std::max(0, cx - radius); x <= std::min(gw - 1, cx + radius);
           ++x)
        merge(a, cells[static_cast<size_t>(y) * gw + x],
              static_cast<float>(x - cx), static_cast<float>(y - cy));
    return a;
  }
  bool enough(const Moments &a) const {
    return a.n >= cfg.minimum_anchors && a.n >= cfg.minimum_coverage * a.area;
  }
};

extern "C" uint32_t df_fast_abi_version(void) { return 2; }
extern "C" uint32_t df_fast_build_features(void) {
  uint32_t flags = 16 | 32; // Guard support, with discrete-neighbor hypotheses.
#ifdef DF_FAST_SIMD
  flags |= 1;
#endif
#ifdef DF_FAST_VECTOR_TYPES
  flags |= 2;
#endif
#ifdef __AVX2__
  flags |= 4;
#endif
#ifdef DF_FAST_PROFILE
  flags |= 8;
#endif
  return flags;
}
extern "C" void df_fast_default_config(df_fast_config *c) {
  if (c)
    *c = {16,   1,    2,     2,   6,     64,  3,   .7f,   .012f, 2.f,
          .75f, .35f, .002f, 1.f, .005f, 4.f, .8f, .85f,  .75f,  1,
          49,   16,   64,    3,   12,    3,   2.f, .025f, .15f};
}
extern "C" df_fast_context *df_fast_create(int32_t w, int32_t h,
                                           const df_fast_config *c) {
  if (!c || !valid_config(*c) || w < 2 || h < 2 || w > 8192 || h > 8192)
    return nullptr;
  try {
    return new df_fast_context(w, h, *c);
  } catch (const std::bad_alloc &) {
    return nullptr;
  }
}
extern "C" void df_fast_destroy(df_fast_context *c) { delete c; }
extern "C" size_t df_fast_workspace_bytes(const df_fast_context *c) {
  return c ? c->bytes : 0;
}

extern "C" int df_fast_process(df_fast_context *c, const float *stereo,
                               const float *mono, const uint8_t *protected_mask,
                               float *out, uint8_t *removed, uint8_t *smoothed,
                               df_fast_stats *stats) {
  if (!c || !stereo || !mono || !out || !removed || !smoothed)
    return 1;
  const size_t n = c->n, nf = n * sizeof(float);
  if (overlap(out, nf, stereo, nf) || overlap(out, nf, mono, nf) ||
      overlap(out, nf, removed, n) || overlap(out, nf, smoothed, n) ||
      overlap(out, nf, protected_mask, n) || overlap(removed, n, smoothed, n) ||
      overlap(removed, n, stereo, nf) || overlap(removed, n, mono, nf) ||
      overlap(removed, n, protected_mask, n) ||
      overlap(smoothed, n, stereo, nf) || overlap(smoothed, n, mono, nf) ||
      overlap(smoothed, n, protected_mask, n))
    return 1;
  df_fast_stats report{};
  report.workspace_bytes = c->bytes;
#ifdef DF_FAST_PROFILE
  auto stamp = std::chrono::steady_clock::now();
  const auto phase = [&](int i) {
    const auto now = std::chrono::steady_clock::now();
    report.phase_ms[i] =
        std::chrono::duration<double, std::milli>(now - stamp).count();
    stamp = now;
  };
#else
  const auto phase = [](int) {};
#endif
  const auto cfg = c->cfg;
  const int w = c->w, h = c->h, tile = cfg.tile_size;
  const float inv_tile = 1.f / tile;
  float lo = std::numeric_limits<float>::infinity(), hi = -lo;
  std::memcpy(out, stereo, nf);
  std::memset(removed, 0, n);
  std::memset(smoothed, 0, n);
  uint8_t *DF_RESTRICT validity = c->valid.data();
  float *DF_RESTRICT normalized_guide = c->guide.data();
  int valid_count = 0;
  const float infinity = std::numeric_limits<float>::infinity();
  size_t begin = 0;
#ifdef DF_FAST_VECTOR_TYPES
  begin = dfv::initialize(n, stereo, mono, validity, lo, hi, valid_count);
#endif
#if defined(DF_FAST_SIMD) && !defined(DF_FAST_VECTOR_TYPES)
#pragma omp simd reduction(min:lo) reduction(max:hi) reduction(+:valid_count)
#endif
  for (size_t i = begin; i < n; ++i) {
    const bool vs = std::isfinite(stereo[i]) && stereo[i] > .05f,
               vm = std::isfinite(mono[i]);
    validity[i] = (vs ? VALID_S : 0) | (vm ? VALID_M : 0);
    valid_count += vs;
    lo = std::min(lo, vm ? mono[i] : infinity);
    hi = std::max(hi, vm ? mono[i] : -infinity);
  }
  report.valid_pixels = valid_count;
  if (!std::isfinite(lo)) {
    if (stats)
      *stats = report;
    return 0;
  }
  report.mono_min = lo;
  report.mono_max = hi;
  if (!std::isfinite(hi - lo)) {
    if (stats)
      *stats = report;
    return 0;
  }
  const float inv_range = 1.f / std::max(hi - lo, 1e-6f);
  begin = 0;
#ifdef DF_FAST_VECTOR_TYPES
  begin = dfv::normalize(n, mono, validity, normalized_guide, lo, inv_range);
#endif
#if defined(DF_FAST_SIMD) && !defined(DF_FAST_VECTOR_TYPES)
#pragma omp simd
#endif
  for (size_t i = begin; i < n; ++i)
    normalized_guide[i] =
        validity[i] & VALID_M ? (mono[i] - lo) * inv_range : 0.f;
  phase(0);
  if (cfg.edge_mode != 0) {
    c->edge_preparation.run(stereo, c->guide.data(), c->valid.data(),
                            protected_mask, c->raw.data(), cfg);
  } else {
    const auto boundary_pixel = [&](int y, int x) {
      const size_t i = static_cast<size_t>(y) * w + x;
      uint8_t flags = protected_mask && protected_mask[i] ? 4 : 0;
      float neighbor_sum = 0;
      int neighbor_count = 0;
      const int offsets[4] = {x ? -1 : 0, x + 1 < w ? 1 : 0, y ? -w : 0,
                              y + 1 < h ? w : 0};
      for (int offset : offsets) {
        const size_t j =
            static_cast<size_t>(static_cast<ptrdiff_t>(i) + offset);
        if ((c->valid[i] & c->valid[j] & VALID_M) &&
            std::abs(c->guide[i] - c->guide[j]) > cfg.mono_edge)
          flags |= 1;
        if ((c->valid[i] & c->valid[j] & VALID_S) &&
            std::abs(stereo[i] - stereo[j]) > cfg.stereo_jump_px)
          flags |= 2;
        if (offset && (c->valid[j] & VALID_S)) {
          neighbor_sum += stereo[j];
          ++neighbor_count;
        }
      }
      if (c->valid[i] == (VALID_S | VALID_M) && neighbor_count >= 2 &&
          std::abs(stereo[i] - neighbor_sum / neighbor_count) <=
              cfg.anchor_deviation_px)
        flags |= 8;
      c->raw[i] = flags;
    };
    for (int x = 0; x < w; ++x) {
      boundary_pixel(0, x);
      boundary_pixel(h - 1, x);
    }
    for (int y = 1; y < h - 1; ++y) {
      boundary_pixel(y, 0);
      boundary_pixel(y, w - 1);
    }
    if (protected_mask)
      raw_interior<true>(w, h, stereo, c->guide.data(), c->valid.data(),
                         protected_mask, c->raw.data(), cfg);
    else
      raw_interior<false>(w, h, stereo, c->guide.data(), c->valid.data(),
                          nullptr, c->raw.data(), cfg);
  }
  phase(1);
  c->dilate(
      7, cfg.edge_mode != 0 ? cfg.edge_anchor_guard_px : cfg.boundary_guard_px,
      c->guard);
  c->dilate(3, cfg.edge_band_px, c->edge);
  phase(2);
  if (cfg.edge_mode != 0) {
    c->edge_rejection.run(stereo, c->guide.data(), c->valid.data(),
                          c->raw.data(), c->guard.data(), c->edge.data(),
                          protected_mask, out, removed, report);
    phase(6);
    if (cfg.smooth_strength == 0 || cfg.max_change_px == 0) {
      for (size_t i = 0; i < n; ++i)
        report.boundary_pixels += c->guard[i] != 0;
      if (stats)
        *stats = report;
      return 0;
    }
  }
  for (int cy = 0; cy < c->gh; ++cy)
    for (int cx = 0; cx < c->gw; ++cx) {
      Moments a;
      const int x0 = cx * tile, x1 = std::min(w, x0 + tile), y0 = cy * tile,
                y1 = std::min(h, y0 + tile);
      a.area = (x1 - x0) * (y1 - y0);
      bool found = false, any_barrier = false;
      for (int y = y0; y < y1 && !found; ++y)
        for (int x = x0; x < x1; ++x) {
          const size_t i = static_cast<size_t>(y) * w + x;
          any_barrier |= c->guard[i] != 0;
          if (!c->guard[i] && (c->raw[i] & 8)) {
            a.d0 = stereo[i];
            a.m0 = c->guide[i];
            found = true;
            break;
          }
        }
      if (!found) {
        a.barriers = any_barrier;
        c->cells[static_cast<size_t>(cy) * c->gw + cx] = a;
        continue;
      }
      bool full = (x1 - x0 == tile && y1 - y0 == tile);
      const uint8_t *cell_guard = c->guard.data();
      const uint8_t *cell_raw = c->raw.data();
      for (int y = y0; y < y1 && full; ++y) {
        int rejected = 0;
        const size_t row = static_cast<size_t>(y) * w;
#ifdef DF_FAST_SIMD
#pragma omp simd reduction(+ : rejected)
#endif
        for (int x = x0; x < x1; ++x)
          rejected += (cell_guard[row + x] != 0) || !(cell_raw[row + x] & 8);
        full = rejected == 0;
      }
      if (full) {
        float sd = 0, sxd = 0, syd = 0, sdd = 0, sm = 0, sxm = 0, sym = 0,
              smm = 0, sdm = 0;
        const float d0 = a.d0, m0 = a.m0;
        const float *normalized = c->guide.data();
        for (int y = y0; y < y1; ++y) {
          const float py = (y - y0 + .5f) * inv_tile - .5f;
          const size_t row = static_cast<size_t>(y) * w;
#ifdef DF_FAST_SIMD
#pragma omp simd reduction(+ : sd, sxd, syd, sdd, sm, sxm, sym, smm, sdm)
#endif
          for (int x = x0; x < x1; ++x) {
            const float px = (x - x0 + .5f) * inv_tile - .5f,
                        d = stereo[row + x] - d0, m = normalized[row + x] - m0;
            sd += d;
            sxd += px * d;
            syd += py * d;
            sdd += d * d;
            sm += m;
            sxm += px * m;
            sym += py * m;
            smm += m * m;
            sdm += d * m;
          }
        }
        a.n = a.area;
        a.xx = a.yy = (tile * tile - 1) / 12.f;
        a.d = sd;
        a.xd = sxd;
        a.yd = syd;
        a.dd = sdd;
        a.m = sm;
        a.xm = sxm;
        a.ym = sym;
        a.mm = smm;
        a.dm = sdm;
        report.anchor_pixels += a.n;
        c->cells[static_cast<size_t>(cy) * c->gw + cx] = a;
        continue;
      }
      float sx = 0, sy = 0, sxx = 0, sxy = 0, syy = 0, sd = 0, sxd = 0, syd = 0,
            sdd = 0, sm = 0, sxm = 0, sym = 0, smm = 0, sdm = 0;
      int count = 0, barriers = 0;
      const float d0 = a.d0, m0 = a.m0;
      const float *normalized = c->guide.data();
      const uint8_t *guard = c->guard.data();
      const uint8_t *raw = c->raw.data();
      for (int y = y0; y < y1; ++y) {
        const size_t row = static_cast<size_t>(y) * w;
        const float py = (y - y0 + .5f) * inv_tile - .5f;
#ifdef DF_FAST_SIMD
#pragma omp simd reduction(+:count,barriers,sx,sy,sxx,sxy,syy,sd,sxd,syd,sdd,sm,sxm,sym,smm,sdm)
#endif
        for (int x = x0; x < x1; ++x) {
          const size_t i = row + x;
          const bool ok = !guard[i] && (raw[i] & 8);
          const float weight = ok ? 1.f : 0.f,
                      px = (x - x0 + .5f) * inv_tile - .5f;
          const float d = ok ? stereo[i] - d0 : 0.f,
                      m = ok ? normalized[i] - m0 : 0.f;
          count += ok;
          barriers += guard[i] != 0;
          sx += weight * px;
          sy += weight * py;
          sxx += weight * px * px;
          sxy += weight * px * py;
          syy += weight * py * py;
          sd += d;
          sxd += px * d;
          syd += py * d;
          sdd += d * d;
          sm += m;
          sxm += px * m;
          sym += py * m;
          smm += m * m;
          sdm += d * m;
        }
      }
      a.n = count;
      a.barriers = barriers;
      a.x = sx;
      a.y = sy;
      a.xx = sxx;
      a.xy = sxy;
      a.yy = syy;
      a.d = sd;
      a.xd = sxd;
      a.yd = syd;
      a.dd = sdd;
      a.m = sm;
      a.xm = sxm;
      a.ym = sym;
      a.mm = smm;
      a.dm = sdm;
      report.anchor_pixels += a.n;
      c->cells[static_cast<size_t>(cy) * c->gw + cx] = a;
    }
  phase(3);
  for (int cy = 0; cy < c->gh; ++cy)
    for (int cx = 0; cx < c->gw; ++cx) {
      Model model;
      const Moments p = c->window(cx, cy, cfg.plane_radius_cells);
      if (!p.barriers && c->enough(p)) {
        const float ni = 1.f / p.n, xx = p.xx - p.x * p.x * ni,
                    xy = p.xy - p.x * p.y * ni, yy = p.yy - p.y * p.y * ni;
        const float det = xx * yy - xy * xy;
        if (xx > .01f * p.n && yy > .01f * p.n && det > 1e-4f * xx * yy) {
          const float xd = p.xd - p.x * p.d * ni, yd = p.yd - p.y * p.d * ni,
                      xm = p.xm - p.x * p.m * ni, ym = p.ym - p.y * p.m * ni;
          const float ax = (xd * yy - yd * xy) / det,
                      by = (yd * xx - xd * xy) / det;
          const float ma = (xm * yy - ym * xy) / det,
                      mb = (ym * xx - xm * xy) / det;
          const float de = std::max(0.f,
                                    p.dd - p.d * p.d * ni - ax * xd - by * yd),
                      me = std::max(0.f,
                                    p.mm - p.m * p.m * ni - ma * xm - mb * ym);
          if (de <= cfg.plane_rms_px * cfg.plane_rms_px * p.n &&
              me <= cfg.mono_plane_rms * cfg.mono_plane_rms * p.n) {
            model.a = ax;
            model.b = by;
            model.c = p.d0 + (p.d - ax * p.x - by * p.y) * ni;
            model.plane = std::isfinite(model.c) && std::isfinite(ax) &&
                          std::isfinite(by);
          }
        }
      }
      const Moments a = cfg.alignment_radius_cells == cfg.plane_radius_cells
                            ? p
                            : c->window(cx, cy, cfg.alignment_radius_cells);
      if (cfg.edge_mode == 0 && c->enough(a)) {
        const float ni = 1.f / a.n, vm = a.mm - a.m * a.m * ni,
                    vd = a.dd - a.d * a.d * ni, cov = a.dm - a.d * a.m * ni;
        if (vm > cfg.minimum_mono_std * cfg.minimum_mono_std * a.n && cov > 0) {
          const float scale = cov / vm, error = std::max(0.f, vd - scale * cov);
          if (error <= cfg.affine_rms_px * cfg.affine_rms_px * a.n) {
            model.affine_a = scale;
            model.affine_base = a.d0 + (a.d - scale * a.m) * ni;
            model.affine_m0 = a.m0;
            model.affine =
                std::isfinite(scale) && std::isfinite(model.affine_base);
          }
        }
      }
      if (!model.plane)
        model.a = model.b = model.c = 0;
      if (!model.affine)
        model.affine_a = model.affine_base = model.affine_m0 = 0;
      report.plane_models += model.plane;
      if (cfg.edge_mode == 0)
        report.alignment_models += model.affine;
      c->models[static_cast<size_t>(cy) * c->gw + cx] = model;
    }
  phase(4);
  const uint8_t *boundaries = c->guard.data();
  uint64_t boundary_count = 0;
#ifdef DF_FAST_SIMD
#pragma omp simd reduction(+ : boundary_count)
#endif
  for (size_t i = 0; i < n; ++i)
    boundary_count += boundaries[i] != 0;
  report.boundary_pixels = boundary_count;
  if (cfg.edge_mode == 0)
    for (int y = 0; y < h; ++y) {
      const Axis &ay = c->ys[y];
      for (const Segment &seg : c->segments) {
        const Model p00 =
            c->models[static_cast<size_t>(ay.i0) * c->gw + seg.left];
        const Model p01 =
            c->models[static_cast<size_t>(ay.i0) * c->gw + seg.right];
        const Model p10 =
            c->models[static_cast<size_t>(ay.i1) * c->gw + seg.left];
        const Model p11 =
            c->models[static_cast<size_t>(ay.i1) * c->gw + seg.right];
        const int v00 = p00.affine, v01 = p01.affine,
                  v10 = p10.affine && (ay.t > 0),
                  v11 = p11.affine && (ay.t > 0);
        if (!(v00 || v01 || v10 || v11))
          continue;
        const uint8_t *DF_RESTRICT valid = c->valid.data();
        const uint8_t *DF_RESTRICT edge = c->edge.data();
        uint8_t *DF_RESTRICT raw = c->raw.data();
        const float *DF_RESTRICT mono = c->guide.data();
        uint64_t deletes = 0, uncertain = 0;
        for (int x = seg.begin; x < seg.end; ++x) {
          const size_t i = static_cast<size_t>(y) * w + x;
          const bool eligible = (valid[i] == (VALID_S | VALID_M)) &
                                bool(edge[i]) &
                                (!(protected_mask && protected_mask[i]));
          if (!eligible)
            continue;
          const int rx = (seg.left != seg.right) &
                         (((x + .5f) * inv_tile - .5f) > seg.left);
          const float s = stereo[i], m = mono[i];
          const float d00 =
              s - (p00.affine_base + p00.affine_a * (m - p00.affine_m0));
          const float d01 =
              s - (p01.affine_base + p01.affine_a * (m - p01.affine_m0));
          const float d10 =
              s - (p10.affine_base + p10.affine_a * (m - p10.affine_m0));
          const float d11 =
              s - (p11.affine_base + p11.affine_a * (m - p11.affine_m0));
          const int votes = v00 + v10 + rx * (v01 + v11);
          const int positive = (v00 & (d00 > cfg.reject_threshold_px)) +
                               (v10 & (d10 > cfg.reject_threshold_px)) +
                               rx * ((v01 & (d01 > cfg.reject_threshold_px)) +
                                     (v11 & (d11 > cfg.reject_threshold_px)));
          const int negative = (v00 & (d00 < -cfg.reject_threshold_px)) +
                               (v10 & (d10 < -cfg.reject_threshold_px)) +
                               rx * ((v01 & (d01 < -cfg.reject_threshold_px)) +
                                     (v11 & (d11 < -cfg.reject_threshold_px)));
          const bool drop =
              eligible & (votes >= cfg.minimum_votes) &
              (std::max(positive, negative) >= cfg.vote_fraction * votes);
          const bool freeze = eligible & (!drop) & bool(positive || negative);
          out[i] = drop ? std::numeric_limits<float>::quiet_NaN() : out[i];
          removed[i] = drop;
          raw[i] |= freeze ? 16 : 0;
          deletes += drop;
          uncertain += freeze;
        }
        report.removed_pixels += deletes;
        report.uncertain_pixels += uncertain;
      }
    }
  if (cfg.smooth_strength > 0 && cfg.max_change_px > 0)
    for (int y = 0; y < h; ++y) {
      const Axis &ay = c->ys[y];
      for (const Segment &seg : c->segments) {
        const Model &p00 =
            c->models[static_cast<size_t>(ay.i0) * c->gw + seg.left];
        const Model &p01 =
            c->models[static_cast<size_t>(ay.i0) * c->gw + seg.right];
        const Model &p10 =
            c->models[static_cast<size_t>(ay.i1) * c->gw + seg.left];
        const Model &p11 =
            c->models[static_cast<size_t>(ay.i1) * c->gw + seg.right];
        const float w00 = (1 - ay.t) * p00.plane, w01 = (1 - ay.t) * p01.plane,
                    w10 = ay.t * p10.plane, w11 = ay.t * p11.plane;
        const float lw = w00 + w10, rw = w01 + w11;
        if (lw + rw == 0)
          continue;
        const float la = w00 * p00.a + w10 * p10.a,
                    ra = w01 * p01.a + w11 * p11.a;
        const float lc =
            w00 * (p00.c - p00.a * seg.left + p00.b * (ay.position - ay.i0)) +
            w10 * (p10.c - p10.a * seg.left + p10.b * (ay.position - ay.i1));
        const float rc =
            w01 * (p01.c - p01.a * seg.right + p01.b * (ay.position - ay.i0)) +
            w11 * (p11.c - p11.a * seg.right + p11.b * (ay.position - ay.i1));
        const uint8_t *DF_RESTRICT guard = c->guard.data();
        const uint8_t *DF_RESTRICT valid = c->valid.data();
        const uint8_t *DF_RESTRICT raw = c->raw.data();
        uint64_t changes = 0;
        int begin = seg.begin;
#ifdef DF_FAST_VECTOR_TYPES
        const auto progress =
            dfv::smooth_row(begin, seg.end, static_cast<size_t>(y) * w, stereo,
                            out, valid, guard, removed, raw, smoothed, inv_tile,
                            seg.left, seg.left == seg.right, lw, rw, la, ra, lc,
                            rc, cfg.smooth_strength, cfg.max_change_px);
        begin = progress.end;
        changes = progress.changes;
#endif
#if defined(DF_FAST_SIMD) && !defined(DF_FAST_VECTOR_TYPES)
#pragma omp simd reduction(+ : changes)
#endif
        for (int x = begin; x < seg.end; ++x) {
          const size_t i = static_cast<size_t>(y) * w + x;
          const bool eligible = (!guard[i]) &
                                (valid[i] == (VALID_S | VALID_M)) &
                                (!removed[i]) & (!(raw[i] & 16));
          const float position = (x + .5f) * inv_tile - .5f;
          const float t =
              seg.left == seg.right
                  ? 0.f
                  : std::max(0.f, std::min(1.f, position - seg.left));
          const float support = lw + t * (rw - lw);
          const float prediction =
              (la + t * (ra - la)) * position + lc + t * (rc - lc);
          const float proposal = std::max(
              stereo[i] - cfg.max_change_px,
              std::min(stereo[i] + cfg.max_change_px,
                       stereo[i] + cfg.smooth_strength *
                                       (prediction - support * stereo[i])));
          const bool use =
              eligible & (proposal > .05f) & std::isfinite(proposal);
          out[i] = use ? proposal : out[i];
          const bool changed = use & (std::abs(out[i] - stereo[i]) > .05f);
          smoothed[i] = changed;
          changes += changed;
        }
        report.smoothed_pixels += changes;
      }
    }
  phase(5);
  if (stats)
    *stats = report;
  return 0;
}
