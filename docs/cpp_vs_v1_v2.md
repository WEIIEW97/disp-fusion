# C++ aligned refinement versus V1 and V2

For a complete stage-by-stage mathematical specification of the current edge-first implementation, see [the C++ algorithm pipeline](cpp_algorithm_pipeline.md).

The current C++ backend is a separate algorithm approximation, rather than a direct translation of V1 or V2. Its default `edge_mode=1` uses V1-style rejection followed by bounded local stereo-plane smoothing. V1/V2 remain unchanged and available as reference implementations.

This comparison describes the current edge-first mode. The older native `edge_mode=0` uses a different tile-affine consensus approximation and should not be confused with V2.

Optional `edge_mode=2` adds a one-pixel discrete mono-neighbor match and directional consensus veto to mode 1; see its [definition and evaluation](edge_rejection_guard.md). It retains C++ sampling, robust fitting and plane smoothing, so it is still a different algorithm from V2. The table below continues to describe preserved mode 1.

## Main differences

| Aspect | V1 | V2 | Current C++ (`edge_mode=1`) |
|---|---|---|---|
| Edge rejection | Threshold stereo disagreement with locally aligned mono | V1 rejection plus agreement across multiple windows | V1-style disagreement rejection using lower-budget local models |
| Alignment samples per window | Up to 1,200 anchors | Same as V1 | At most 64 lattice locations by default; fewer survive anchor screening |
| Robust affine fitting | 8 Huber IRLS iterations, float64 least squares | Same as V1 | 3 Huber IRLS fits, float32 centered closed-form solves |
| Minimum anchor support | 60 anchors | Same as V1 | 12 sampled anchors by default |
| Consensus requirement | None | At least 3 models, at least 80% same-sign agreement | None |
| Smoothing | None; retained disparity stays exact | Bounded mono/stereo-guided neighbor smoothing | Bounded blending of fitted stereo planes |
| Stronger plane smoothing | None | Optional multiscale robust plane stage | Single-scale plane stage |
| Invalid points | Never filled | Never filled | Never filled |

These are algorithm defaults and mechanisms, not a claim that every spatial setting is identical. The existing 640×400 comparisons use explicitly configured 49×49 alignment windows with stride 16 for both reference and native paths. The reference library's general alignment defaults are 97×97 and stride 32. Pixel-valued thresholds in the comparison are scaled with the prepared disparity; exact configurations are recorded with its results.

## Rejection and confidence

All three methods fit local positive-slope affine relationships between normalized mono guidance $g$ and stereo disparity $s$:

$$
s_i \simeq a_k g_i+b_k.
$$

Accepted overlapping models provide a triangle-weighted prediction

$$
\widehat s_i=\frac{\sum_{k\ni i}t_{ik}(a_k g_i+b_k)}{\sum_{k\ni i}t_{ik}}.
$$

For valid, covered edge samples, V1 rejects when

$$
R_1(i)=[|s_i-\widehat s_i|>\tau].
$$

V2 adds a consensus veto. Let $N_i$ count accepted covering models, and let $N_i^+$ and $N_i^-$ count models whose signed discrepancy exceeds $+\tau$ or falls below $-\tau$, respectively. With default settings,

$$
R_2(i)=R_1(i)\land[N_i\geq3]\land
\left[\frac{\max(N_i^+,N_i^-)}{N_i}\geq0.8\right].
$$

Thus V2's rejected pixels are a subset of V1's when they share the same alignment evidence.

The default C++ path uses V1-style blended-discrepancy rejection without the consensus veto. Its anchor selection, sample density, robust fitting and boundary preparation differ, so its prediction is generally different from V1's. Its rejection mask is not guaranteed to be a subset of either reference mask.

Removing the consensus veto restores visible edge cleanup, but lower anchor support and fewer robust iterations can make a local alignment less reliable. The C++ path can reject correct stereo samples if mono geometry or local alignment is wrong. More deletions do not establish better accuracy. A denser native configuration is available with `edge_samples=256, edge_min_samples=24`; it changes decisions and is not numerically equivalent to the 64-sample default.

## Smoothing mechanisms

V1 only invalidates selected samples. Every retained disparity remains exactly equal to its original stereo value.

V2 smooths the retained field through a data-anchored four-neighbor graph:

$$
E(d)=\frac12\sum_i(d_i-s_i)^2+
\frac{\lambda}{2}\sum_{(i,j)}w_{ij}(d_i-d_j)^2.
$$

The fixed weights depend on both mono and stereo differences. Iterative updates are bounded relative to retained stereo, and invalid or unsafe samples never move. This mechanism does not assume that an entire local surface is planar and can accommodate gently curved surfaces.

The C++ path instead fits stereo disparity planes

$$
p_k(x,y)=a_kx+b_ky+c_k
$$

from merged tile moments. It accepts models only when stereo and mono pass local planarity gates, then applies bounded corrections toward neighboring accepted stereo planes. This directly targets planar wall stripes. Mono controls boundaries and suitability; stereo supplies plane values and slopes. Curved or poorly supported surfaces may receive no smoothing.

Both V2 and C++ finish rejection before smoothing. The current C++ path additionally freezes its entire rejection band, including retained samples with unavailable models. Rejected samples cannot enter plane fitting or become valid again. Changing plane tile size, disabling smoothing, or setting the smoothing change bound to zero cannot change its rejection mask.

## Optional enhanced V2 versus native planes

The optional `InteriorSmoothConfig(enabled=True)` stage is separate from standard V2 graph smoothing. It uses multiscale windows, robust plane fitting, held-out P90 residual checks, connected-component restrictions and a median correction offset.

Native planes use one grid scale, moment-based least squares and training RMS gates. They omit those multiscale, robust, held-out and component-specific checks. This reduces computation but also reduces model validation. A low training residual cannot certify correct surface geometry or separation of disconnected surfaces.

## Other preprocessing and implementation differences

Reference alignment normalizes mono using its 2nd and 98th percentiles with clipping. C++ uses finite global min/max, making normalization more sensitive to extreme values.

Reference edge preparation uses Gaussian filtering, nearest-valid auxiliary stereo filling, a full 2D median-5 filter and elliptical dilation. C++ uses validity-weighted binomial Gaussian filtering, local auxiliary filling, a separable median-5 approximation and square dilation. These change anchors and boundary masks independently of implementation language. Auxiliary filling never creates output disparity.

The full reference API projects stereo into RGB, compares it with native RGB mono, and maps decisions/guidance back to stereo coordinates. C++ accepts already-aligned arrays and performs no registration or inference. Its speed measurements therefore exclude those stages. The prepared aligned-input comparisons also exclude reference registration from the timed algorithm comparison.

Native processing uses reusable buffers, optional SIMD and no frame-loop C++ heap allocation. This implementation work does not make its outputs equivalent to V1/V2 or establish the approximately 5 ms/frame DSP target. The active resized target is 640×360, with original 1280×720 also evaluated on samples; actual DSP performance remains unmeasured.

## Known comparison-input defect

The prepared aligned mono arrays used in the historical 640×400 gallery contain artificial black grids. Preparation copied mono guidance back only to z-buffer-winning stereo pixels. Multiple stereo pixels projecting to the same rounded RGB pixel caused the other source pixels to lose guidance, even on continuous surfaces. Nonfinite guidance appears black in the colormap.

For DAM frame `000286`, the original 1280×720 mono cache is entirely finite. The prepared 640×400 field has 41,648 missing samples (16.3% of the image), including 17,336 valid stereo samples (7.5% of valid stereo). This affects algorithm support, not only visualization.

The [new original-resolution and 640×360 samples](../outputs/cpp_edge_first_samples_native_640x360_20261104/REPORT.md) use bilinear mono gathering at each valid stereo pixel's continuous projected RGB coordinate, with a separate depth-consistency visibility gate. Original invalid stereo remains invalid, and no inference is rerun. That gate remains a heuristic.

The subsequent [20-frame comparison](../outputs/v1_v2_cpp_640x360_sample20_20261104/README.md) runs original public V1/V2 at 640×360 and separately runs their unchanged refinement cores on the same aligned inputs as C++. The public-pipeline and shared-input results are labeled separately. The latter controls the input arrays when comparing refinement behavior; roughness or removal rates still do not establish physical accuracy. The defective historical inputs should not be used for a quality ranking.

## Related implementation and documentation

- [Native algorithm, equations, configuration and build instructions](fast_aligned.md)
- [Reference local alignment and V1 rejection](../disp_fusion/alignment.py)
- [V2 consensus and processing pipeline](../disp_fusion/pipeline.py)
- [Reference graph and optional interior smoothing](../disp_fusion/smoothing.py)
- [Native edge preparation and robust rejection](../native/edge_first.h)
- [Native plane fitting and processing order](../native/fast_aligned.cpp)
