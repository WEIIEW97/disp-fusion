# Optional Boundary-Frozen Multi-Scale Interior Smoothing

## Purpose and scope

This optional stage extends V2 **after** consensus rejection and bounded graph smoothing. Its purpose is to reduce stripe-like or medium-scale variation inside continuous, approximately planar regions without changing the established rejection decisions or moving boundary samples.

The implementation is part of [smoothing.py](../disp_fusion/smoothing.py), not a standalone algorithm script. It is disabled by default. V1 and the default V2 retain their original behavior. Existing native-RGB mono caches are reused; no new network inference or training is needed.

Mono does not supply absolute depth. Stereo determines each local disparity trend, while mono helps reject regions that are not locally well approximated by an affine surface. A smooth mono prediction alone is not treated as proof of a planar physical surface.

## 1. Inputs

- Original stereo disparity $s$, in stereo-reference pixel units.
- Baseline V2 result $b$, after its unchanged rejection and graph-smoothing stages.
- Normalized mono guidance $g$, sampled back through the existing stereo-to-RGB source correspondence.
- The existing safe mask and an explicit protection mask.

All arrays use the original stereo grid. No image-orientation conversion is introduced.

## 2. Frozen samples and barriers

The enhancement freezes samples that are invalid, lack reliable local guidance apart from the narrow sampling-gap allowance below, belong to the existing mono-edge protection band, or were candidates for removal under V1. In particular, V1 candidates retained by V2 consensus remain uncertain and cannot be smoothed by the new stage.

Additional stereo barriers mark neighboring original valid disparities differing by more than 2px; these barriers are dilated by 3 pixels. The model windows cannot cross a protected sample or these stereo barriers.

An original or newly invalid sample remains invalid. Small stereo holes may lie inside a model window, but are never used as anchors or output targets. Local four-connected valid components are computed when holes occur, and a model can use only the component containing its window center. It cannot combine samples from disconnected sides of an invalid gap.

Missing mono guidance is different from missing stereo. An originally valid, unprotected stereo sample within 1.5 pixels of reliable mapped guidance can receive the same accepted local trend correction as its neighbors. It is never an anchor, and no mono value is synthesized for it. This narrowly scoped rule avoids a sampling-grid imprint when sparse Z-buffer visibility leaves thin gaps inside an otherwise continuous wall. Wider unseen regions remain frozen. Set `guide_hole_radius` to zero to disable this allowance. Finite-guide samples explicitly excluded by the safe mask remain frozen.

These are numerical guardrails, not semantic wall segmentation. Incorrect or undetected geometry can still pass the guards.

## 3. Multi-scale model windows

The default window sizes are

$$
\mathcal W=\{31,63,127\}\text{ pixels}.
$$

Window-center stride is $\operatorname{round}(0.5w)$ for size $w$. Windows are clipped at image borders; very narrow clipped windows are skipped.

Anchor requirements:

- valid, safe stereo with finite mapped mono;
- no protection or stereo-jump barrier;
- original stereo differs from its auxiliary $5\times5$ median by at most 0.75px;
- at least 128 anchors;
- anchors cover at least 70% of the clipped window area.

The median is used only to screen anchors. It is not exported as depth and does not fill output holes.

## 4. Robust affine fitting and holdout checks

Using normalized local coordinates $x,y$, fit

$$
\widehat s_k(x,y)=a_kx+b_ky+c_k
$$

to original stereo anchors. A separate affine model is fitted to normalized mono:

$$
\widehat g_k(x,y)=\alpha_kx+\beta_ky+\gamma_k.
$$

Spatially interleaved anchors satisfying $(3v+5u)\bmod7=0$ are reserved for validation. They are never used to fit coefficients. At least 16 training and 16 validation samples must be available; at most 512 of each are selected per window. Each model uses five robust IRLS iterations. Rank-deficient fits are rejected. The later median-centering step uses all eligible anchors, including holdout anchors, so the final correction is not claimed to be an independent held-out prediction.

A window is accepted only when its held-out residuals satisfy

$$
Q_{90}(|s-\widehat s_k|)\le0.5\text{ px},
\qquad
Q_{90}(|g-\widehat g_k|)\le0.003.
$$

The mono threshold is in globally percentile-normalized mono units, not metric depth. These tests reduce, but do not eliminate, the risk of flattening curved surfaces. Smaller windows can still approximate a gently curved surface locally; this is not a global plane constraint.

## 5. Overlap blending and metric anchoring

For each accepted window, compute the trend correction relative to baseline V2 and center it on the median anchor correction:

$$
o_k=\operatorname{median}_{i\in A_k}\bigl(\widehat s_k(i)-b_i\bigr),
\qquad
\delta_k(i)=\widehat s_k(i)-b_i-o_k.
$$

This discourages arbitrary local offsets. It does not guarantee exact preservation of the global median or slope, so both are measured during evaluation.

Accepted corrections are combined with separable triangular spatial weights, zero at the outer model-window boundary, and a scale factor $(w/31)^2$. Larger accepted windows therefore contribute more strongly to reducing broader variation. All fit and correction targets remain in the same local valid component. The normalization denominator has a unit support floor so a lone low-weight model fades into the unchanged baseline rather than creating a hard coverage seam.

$$
\delta(i)=\frac{\sum_k\omega_k(i)\delta_k(i)}{\max(1,\sum_k\omega_k(i))},
\qquad
d_i=\operatorname{clip}\bigl(b_i+0.85\delta(i),\ s_i-0.75,\ s_i+0.75\bigr).
$$

No accepted model means no additional change. Unsafe or invalid samples never receive a correction. A proposal that would become invalid is discarded.

The bound is relative to **original stereo**, not baseline V2. The two smoothing stages do not each receive an independent 0.75px budget. Because they may choose opposite corrections, the difference between baseline V2 and enhanced V2 can exceed 0.75px even though both remain within the permitted original-stereo interval.

## 6. API and configuration

```python
from disp_fusion import DepthRefiner, RefinementConfig, InteriorSmoothConfig

config = RefinementConfig(
    interior=InteriorSmoothConfig(enabled=True),
    max_change_px=0.75,
)
result = DepthRefiner(calibration, config).refine(stereo, mono, version="v2")
```

The existing CLI accepts nested settings through `--config`:

```json
{
  "interior": {
    "enabled": true,
    "windows": [31, 63, 127],
    "strength": 0.85
  }
}
```

No additional algorithm-specific CLI or project experiment script is required. Results remain V2 results, with the exact `interior` settings recorded in the run configuration and an `interior_smoothing` diagnostic record.

The stage can also operate on an existing V2 array through `multiscale_interior_smooth`, provided its guidance, safety/protection masks, original stereo, and total modification bound are supplied consistently.

## 7. Evaluation and interpretation

The one-off evaluation script is outside the project at `/tmp/evaluate_interior_smoothing_20261029.py`. It reuses the existing 20-frame ZipDepth and DAMv2-L caches, verifies V1 output and V2 rejection-only output against saved results, and writes new results without overwriting the baselines.

Evaluation includes:

- identical V2 validity and deletion masks;
- bit-exact frozen and protected pixels;
- total disparity change bounded by original stereo ±0.75px;
- shared-mask roughness statistics;
- fixed-ROI detrended residuals, slopes, and floating-point row/column profiles;
- optional cached matching diagnostics, never passed into the algorithm.

The robust plane subtracted for residual figures is fitted to original stereo; it is **not independent ground truth**. The fixed rectangles are diagnostic ROIs and are not guaranteed wall labels in every frame. A reduction in residual MAD can coexist with persistent or increased broad structured variation, so profiles and directional statistics must be examined alongside colormaps.

Default V1/V2 behavior remains elementwise identical to retained reference outputs: the final test run passed 60 basic tests and 40 cached real-data regression cases. Stronger smoothing is an optional geometric assumption and does not guarantee improved absolute depth accuracy.

### Example: frame 000286

Residual MAD is measured after subtracting the same robust plane fitted to original stereo on identical retained pixels. It is a smoothness diagnostic, not a ground-truth error.

| Model / diagnostic ROI | Baseline V2 MAD / px | Enhanced MAD / px | Reduction |
|---|---:|---:|---:|
| ZipDepth / upper wall | 0.088735 | 0.053097 | 40.16% |
| DAMv2-L / upper wall | 0.088732 | 0.038210 | 56.94% |
| ZipDepth / right wall | 0.071668 | 0.026932 | 62.42% |
| DAMv2-L / right wall | 0.071667 | 0.026241 | 63.38% |

This does **not** establish complete stripe removal. The upper-wall column-mean residual standard deviation rises from 0.101932px to 0.112577px with ZipDepth and 0.107633px with DAMv2-L. Right-wall row-mean variation decreases, while its column-mean variation increases. The method suppresses local variation but does not uniformly reduce every scale or direction.

For this example, ROI median disparities shift by approximately -0.030/-0.066px on the upper wall and +0.046/+0.041px on the right wall (ZipDepth/DAMv2-L). The largest change in the fitted slope components is approximately 0.000482 disparity pixels per image pixel. These quantities describe stability relative to the baseline, not physical accuracy.

The feature therefore remains optional. Prefer inspecting both residual images and floating-point profiles rather than selecting it solely by the lower MAD.

### Full 20-frame evaluation

Both existing mono caches were reused for the same 20 frames. All 40 model/frame outputs preserve baseline V2 rejection masks and protected samples. Original invalid samples and newly rejected samples remain invalid. Default V1 and rejection-only V2 arrays were also checked against their saved references during evaluation. Total changes remain within original stereo ±0.75px.

| Model | Baseline common-stencil roughness / px | Enhanced roughness / px | Reduction | Cached-IR MAE before / px | Cached-IR MAE after / px |
|---|---:|---:|---:|---:|---:|
| ZipDepth | 0.106202 | 0.102096 | 3.87% | 3.344567 | 3.348675 |
| DAMv2-L | 0.106719 | 0.102414 | 4.03% | 3.344442 | 3.349059 |

The roughness calculation uses common retained five-point stencils. The optional IR audit uses identical retained, checked locations before and after enhancement; cached IR is not physical ground truth. Its MAE becomes slightly worse rather than better: ZipDepth has 10,682 points improving by more than 0.25px versus 18,304 worsening, and DAMv2-L has 13,698 improving versus 22,731 worsening. Smoother output must not be presented as demonstrated accuracy improvement.

The extra interior stage took approximately 120.17 seconds for ZipDepth and 120.73 seconds for DAMv2-L, around 6 seconds per frame per model on this machine, excluding shared preprocessing, graph smoothing, plotting, and export. This is an indicative runtime, not a controlled performance benchmark.

**2026-11-03 implementation update:** equivalent CPU acceleration reduces the interior stage to approximately 0.56 seconds per frame and the complete cached-input refinement to approximately 1.9 seconds per frame on the same machine. The original algorithm parameters and saved output effects are retained. First-use compilation, inference, plotting and export are excluded; see [the performance report](performance.md) for equivalence checks and execution controls. The historical timings above describe the earlier implementation, not the current accelerated path.

- [All 20 comparison frames](../outputs/interior_multiscale_sample20_20261029/index.html)
- [Evaluation notes and aggregate roughness](../outputs/interior_multiscale_sample20_20261029/README.md)
- [Exact configuration and source hashes](../outputs/interior_multiscale_sample20_20261029/run.json)
- [000286 upper-wall residual comparison](../outputs/interior_multiscale_sample20_20261029/000286/upper_wall_residual_comparison.png)
- [000286 right-wall residual comparison](../outputs/interior_multiscale_sample20_20261029/000286/right_wall_residual_comparison.png)
- [000286 upper-wall profiles](../outputs/interior_multiscale_sample20_20261029/000286/upper_wall_profiles.png)
- [000286 right-wall profiles](../outputs/interior_multiscale_sample20_20261029/000286/right_wall_profiles.png)

[Back to the full pipeline](algorithm_pipeline.md) · [Project README](../README.md)
