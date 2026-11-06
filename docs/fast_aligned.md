# Edge-first aligned refinement

For a complete stage-by-stage mathematical specification of the current edge-first implementation, see [the C++ algorithm pipeline](cpp_algorithm_pipeline.md).

This optional C++ backend takes already aligned stereo disparity and relative mono prediction. The active performance target is **640×360, approximately 5 ms/frame on a DSP**. The original-resolution sample evaluation also covers 1280×720. Registration, neural inference and device transfer are outside its scope. The current implementation does **not** meet that timing target on the measured desktop CPU. No DSP timing has been measured.

V1/V2 remain available through `DepthRefiner`, their configuration and CLI. Their core files and saved results are unchanged. The native backend is an approximation with a separate API, not an equivalent port of either reference.

See [C++ versus V1/V2](cpp_vs_v1_v2.md) for the algorithm differences, confidence tradeoffs and known comparison-input defect.

Optional **`edge_mode=2`** adds a one-pixel discrete mono-neighbor match and directional consensus veto to the mode-1 candidate. It reduces over-removal at uncertain boundaries while preserving the existing interior plane results. See the [mathematical guard specification](edge_rejection_guard.md) and [20-frame comparison](../outputs/cpp_edge_guard_640x360_sample20_20261104/index.html). Default mode 1, legacy mode 0 and original V1/V2 remain available. Mode 2 can preserve a thin wrong fringe and is not a guarantee of physical edge accuracy.

## Processing order

The default `edge_mode=1` runs:

1. Validate and normalize the two input fields.
2. Detect mono/stereo boundaries and screen stereo anchors.
3. Fit bounded-budget robust local mono-to-stereo models.
4. Reject discrepant stereo samples within the edge band.
5. Freeze the **entire edge band**, including retained and unsupported samples.
6. Fit stereo planes using surviving, unguarded interior anchors, then apply bounded smoothing only to eligible interior samples.

Edge decisions are finished before plane moments are accumulated. Plane fitting cannot consume rejected samples. Changing `tile_size`, disabling smoothing, or setting its change bound to zero cannot change the default-mode rejection mask. No stage fills original invalid samples or newly rejected samples.

## Inputs and normalization

Inputs must be native-endian, contiguous, aligned `float32` arrays on the same pixel grid:

- Stereo $s_i$: disparity in physical pixels; finite values greater than 0.05 are valid.
- Mono $m_i$: relative inverse-depth-like prediction; larger means nearer. Convert metric mono depth to inverse depth **before** calling this backend.
- Optional byte protection mask: nonzero freezes rejection and smoothing at that sample. Nearby model support is guarded too.

For finite mono samples,

$$
g_i=\frac{m_i-m_{\min}}{\max(m_{\max}-m_{\min},10^{-6})}.
$$

Missing mono freezes the corresponding output. If no usable mono exists, or its global range overflows float32, processing returns the original field. Mono is guidance, not metric ground truth.

## Boundary preparation and anchors

Mono gradients use a separable binomial Gaussian kernel $[1,4,6,4,1]/16$. Filtering is validity-weighted, so missing mono is excluded from the numerator and denominator rather than treated as zero depth. Stereo uses an auxiliary locally filled horizontal median-5 followed by vertical median-5. This separable filter is an approximation to V1's full 2D median/nearest-fill preparation. Filling is restricted to auxiliary edge detection; it never creates output depth.

Central differences, with one-sided image-border differences, define

$$
B_i=[\|\nabla\widetilde g_i\|_2>\tau_g]
\lor[\|\nabla\widetilde s_i\|_2>\tau_s].
$$

By default $\tau_g=0.012$, $\tau_s=2$ disparity pixels. Square dilation gives anchor guard $G=D_3(B\lor P)$ and rejection band $E=D_6(B)$, where $P$ is explicit protection. Protection also enters the smoothing guard.

Candidate anchors satisfy valid stereo and mono, $i\notin G$, and

$$
|s_i-\widetilde s_i|\leq 0.75\;\text{pixels}.
$$

Median screening reduces isolated corrupt anchors; it cannot certify depth accuracy.

## Robust local alignment and rejection

The edge grid has stride 16 and clipped 49×49 windows. Its grid is independent of the plane tile grid. The default lattice samples at most **8×8=64 locations/window**, requiring at least 12 surviving anchors. Windows with no eligible nearby edge pixels are skipped. The mono Q95−Q5 span must reach 0.025.

Each model estimates $s\simeq a_k g+b_k$ using three bounded Huber IRLS fits. Centered weighted moments solve the two-parameter least-squares system without SVD. Between fits,

$$
r_j=s_j-(a_k g_j+b_k),\qquad
\sigma=\max\!\left(0.15,1.4826\operatorname{median}_j|r_j-\operatorname{median}(r)|\right),
$$

$$
w_j=\min\!\left(1,\frac{1.5\sigma}{\max(|r_j|,10^{-8})}\right).
$$

The factor 1.4826 scales MAD to the standard deviation of Gaussian noise. Accept positive finite slopes with sampled Q80 absolute residual at most 2 disparity pixels. This sampled training gate does not independently validate the model.

Accepted overlapping windows supply triangle weights $t_{ik}$, with each one-dimensional factor floored at 0.05. At most four grid centers cover each axis (up to sixteen windows per pixel). Compute

$$
\widehat s_i=
\frac{\sum_{k\ni i}t_{ik}(a_k g_i+b_k)}{\sum_{k\ni i}t_{ik}},\qquad
R_i=V_i\land E_i\land[W_i>0]\land[|s_i-\widehat s_i|>\tau_r],
$$

where $V_i$ requires valid stereo/mono and no explicit protection. The prediction must be finite. Default $\tau_r=4$ pixels. Set $s_i^{(e)}=\mathrm{NaN}$ for $R_i$, otherwise retain $s_i$ exactly. Unsupported pixels retain their original value. This is **V1-style blended-residual rejection**, without the earlier fast path's restrictive three-vote consensus veto. It is not the full V1 implementation.

The 64-sample default trades local evidence density for bounded work. `edge_samples=256, edge_min_samples=24` selects a denser 16×16 lattice. This changes decisions; it is not a numerically equivalent optimization. The comparison records this ablation explicitly. Small sample counts, local model mismatch and mono bias can cause incorrect deletions. More rejection is not evidence of higher accuracy.

## Interior plane smoothing

After rejection, the plane guard becomes $G'=G\lor E\lor R$. Thus even a retained edge pixel with a confident model cannot move during smoothing.

Default plane tiles are 16×16. Tile moments are merged across 3×3 tiles, giving a nominal 48×48 window. A plane needs at least 64 anchors, 70% coverage, adequate coordinate spread and no guarded samples in its tile support. The mono and stereo training RMS limits are respectively 0.002 normalized units and 0.35 disparity pixels.

For centered coordinates $q_i=(x_i,y_i)$, estimate stereo slopes through

$$
C=\sum_i(q_i-\bar q)(q_i-\bar q)^\top,\quad
v=\sum_i(q_i-\bar q)(s_i-\bar s),\quad
[a,b]^\top=C^{-1}v,\quad c=\bar s-a\bar x-b\bar y.
$$

Local reference offsets reduce float32 cancellation. Mono uses the same coordinate system to gate planarity. Stereo determines plane values and slopes; mono values do not directly overwrite metric disparity.

For up to four neighboring accepted plane models, use bilinear weights $u_{ik}$ without renormalizing missing support:

$$
\delta_i=\sum_k u_{ik}I_k(\widehat p_k(i)-s_i),\qquad
s'_i=\operatorname{clip}(s_i+0.85\delta_i,s_i-0.75,s_i+0.75).
$$

Apply this only to retained valid guided samples outside $G'$. Discard nonfinite or invalid proposals. Fitted stereo means anchor local geometry, but overlap and clipping do not guarantee zero global bias. Training RMS and boundary guards do not prove physical planarity, distinguish all disconnected surfaces, or repair a bad mono boundary.

Setting `smooth_strength=0` skips plane statistics, fitting and blending entirely after rejection. A coarser `tile_size=32` changes only smoothing support and coverage in the default mode; it cannot weaken or strengthen rejection.

## Build and use

```bash
cmake -S native -B build/native -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=clang++ -DDF_FAST_NATIVE_TUNING=ON
cmake --build build/native -j 2
build/native/disp_fusion_fast_bench 640 360 200
```

Host tuning is for desktop measurements. For a DSP use its toolchain, target flags and a separate build directory. `DF_FAST_VECTOR_TYPES=OFF` and `DF_FAST_SIMD=OFF` provide scalar fallback. SIMD does not create worker threads. Fast-math is disabled; floating-point reduction order can still vary across builds. `DF_FAST_PROFILE=ON` enables seven phase timers; index 6 is robust edge fitting/rejection.

**The current C/Python ABI is version 2. Rebuild old libraries and C clients.** The Python wrapper rejects ABI 1 rather than interpreting its smaller structures.

Mode 2 is an ABI-2 extension advertised by feature mask `0x20`; rebuild the native library before selecting it. An older ABI-2 binary still works for modes 0/1, and the wrapper rejects unsupported mode 2 explicitly. To benchmark the guard, run `build/native/disp_fusion_fast_bench 640 360 200 16 2` (tile size 16, mode 2).

```python
import numpy as np
from disp_fusion import FastAlignedConfig, FastAlignedRefiner

# Already-aligned native float32 stereo/mono arrays, shape (360, 640).
config = FastAlignedConfig(edge_mode=1, tile_size=16)
with FastAlignedRefiner(640, 360, config) as engine:
    output = np.empty((360, 640), np.float32)
    removed = np.empty((360, 640), np.uint8)
    smoothed = np.empty((360, 640), np.uint8)
    stats = engine.process_into(stereo, mono, output, removed, smoothed)
```

Configuration defaults use physical disparity pixels; they do not automatically scale with image size. The saved comparisons use the earlier half-width input preparation and its explicitly halved pixel thresholds: rejection 2, fit P80 1, anchor deviation 0.375, plane RMS 0.175 and maximum smoothing change 0.375. Their recorded configs specify the exact values.

`refine()` allocates result buffers for convenience; use `process_into()` or the [C ABI](../native/fast_aligned.h) for reusable buffers. No output may overlap an input or another output. A context is not reentrant; use one per stream. Frame processing makes no C++ heap allocations. The context still contains full-frame guidance/masks as well as line rings, tile statistics and bounded fit workspaces; this is not a fully line-buffered or fixed-point DSP port. At 640×360, tile16 context payload is 2,285,380 bytes (about 2.18 MiB), excluding allocator overhead and caller inputs/outputs (another 14 bytes/pixel for two float inputs, one float output and two masks).

`edge_mode=0` retains the previous native raw-edge/tile-affine consensus method for reproducibility. Reproduce old outputs with their saved configuration, including `edge_band_px=4`; merely selecting mode 0 with the new defaults is insufficient. `boundary_guard_px`, `alignment_radius_cells`, `affine_rms_px`, `minimum_mono_std`, `minimum_votes` and `vote_fraction` govern that legacy path. Default-mode edge models instead use the separate `edge_*` parameters above. V1/V2 remain separate reference algorithms.

Mode 2 reuses `minimum_votes` (default 3) and `vote_fraction` (default 0.8). The mono tolerance radius is fixed at one input pixel. Its `uncertain_pixels` counter includes unsupported edge samples plus original mode-1 candidates retained by the veto. It adds no full-frame buffers; all other fitting and plane parameters are shared with mode 1.

## Evidence and limitations

The [latest gallery](../outputs/cpp_edge_first_samples_native_640x360_20261104/index.html), [report](../outputs/cpp_edge_first_samples_native_640x360_20261104/REPORT.md) and [measurements](../outputs/cpp_edge_first_samples_native_640x360_20261104/aggregate.json) cover three existing frames for both DAM and ZipDepth at **1280×720 and 640×360**. The resized target was corrected on 2026-11-04 to preserve the original 16:9 aspect ratio. These twelve cases reuse existing caches with corrected bilinear mono gathering and a separate depth-consistency visibility gate. The fusion kernel is unchanged; no inference is rerun. Mean per-case CPU median latency is 27.16 ms at original size and 6.67 ms at 640×360.

The [20-frame 640×360 comparison](../outputs/v1_v2_cpp_640x360_sample20_20261104/index.html) includes both the original calibrated public V1/V2 pipeline and their unchanged refinement cores on the same aligned inputs as C++. These are separate evaluation groups: the public workflow includes registration/support differences, while the shared-input group compares refinement behavior. Exact configurations, checks and aggregate diagnostics are in its [report](../outputs/v1_v2_cpp_640x360_sample20_20261104/README.md). That historical evaluation uses mode 1. The subsequent [guarded comparison](../outputs/cpp_edge_guard_640x360_sample20_20261104/REPORT.md) evaluates optional mode 2 against exactly those saved inputs and outputs.

The [historical 640×400 gallery](../outputs/fast_aligned_edge_first_640x400_20261103/index.html) remains available, but its prepared mono inputs contain artificial holes from winner-only back-mapping. This defect reduced algorithm support; its results are not a controlled quality baseline for the latest run. See the [defect analysis](cpp_vs_v1_v2.md#known-comparison-input-defect). The new visibility gate also remains a stereo-based heuristic, not independent visibility truth. Roughness and rejection counts do not establish physical accuracy. Historical outputs and the former native binary/source snapshot remain under [the earlier evaluation](../outputs/fast_aligned_sample3_20261103/).

The report separates edge-only and full latency, including normalization, preparation, guards, fitting, output writes and Python validation/ctypes/stats overhead. CPU medians and tails do not establish DSP feasibility. Target memory layout, vector instructions, fixed-point precision, DMA, input-dependent work and worst-case deadline compliance remain unmeasured.
