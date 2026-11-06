#ifndef DISP_FUSION_FAST_ALIGNED_H
#define DISP_FUSION_FAST_ALIGNED_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Experimental fixed-cost refinement of spatially aligned float32 images.
 * Stereo: physical pixel disparity. Mono: larger means nearer. Nonfinite mono
 * freezes that sample. No allocation in process; one context per stream.
 */
typedef struct {
  int32_t tile_size, plane_radius_cells, alignment_radius_cells;
  int32_t boundary_guard_px, edge_band_px, minimum_anchors, minimum_votes;
  float minimum_coverage, mono_edge, stereo_jump_px, anchor_deviation_px;
  float plane_rms_px, mono_plane_rms, affine_rms_px, minimum_mono_std;
  float reject_threshold_px, vote_fraction, smooth_strength, max_change_px;
  /* Mode 1: sampled robust V1-style rejection. Mode 2: mode 1 plus a fixed
   * one-pixel discrete-mono-match and minimum_votes/vote_fraction veto.
   * Mode 0 preserves the earlier tile-affine approximation for comparison. */
  int32_t edge_mode, edge_window, edge_stride, edge_samples, edge_iterations;
  int32_t edge_min_samples, edge_anchor_guard_px;
  float edge_fit_p80_px, edge_min_mono_span, edge_scale_floor_px;
} df_fast_config;
typedef struct {
  uint64_t valid_pixels, anchor_pixels, boundary_pixels, plane_models;
  uint64_t alignment_models, removed_pixels, smoothed_pixels, uncertain_pixels;
  uint64_t workspace_bytes;
  uint64_t edge_pixels, edge_covered_pixels, edge_attempted_windows;
  uint64_t edge_sampled_windows, edge_samples_total;
  float mono_min, mono_max;
  double phase_ms[7]; /* index 6 = robust edge stage; zero without profiling. */
} df_fast_stats;
typedef struct df_fast_context df_fast_context;
uint32_t df_fast_abi_version(void);
uint32_t
df_fast_build_features(void); /* bits 0 SIMD, 1 vectors, 2 AVX2, 3 profiling, 4
                                 guard, 5 discrete guard */
void df_fast_default_config(df_fast_config *config);
df_fast_context *df_fast_create(int32_t width, int32_t height,
                                const df_fast_config *config);
void df_fast_destroy(df_fast_context *context);
size_t df_fast_workspace_bytes(const df_fast_context *context);
/* Contiguous row-major arrays. No output may overlap any input or another
 * output. protected_mask may be NULL; nonzero freezes rejection AND smoothing.
 * Boundary guards freeze smoothing, not supported invalidation at an edge.
 * Stats may be NULL. Returns 0 on success, 1 on invalid arguments or overlap.
 */
int df_fast_process(df_fast_context *context, const float *stereo,
                    const float *mono, const uint8_t *protected_mask,
                    float *output, uint8_t *removed_mask,
                    uint8_t *smoothed_mask, df_fast_stats *stats);
#ifdef __cplusplus
}
#endif
#endif
