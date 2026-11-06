# Guarded edge rejection

Optional native **`edge_mode=2`** reduces excessive boundary invalidation using two vetoes: a match to an actual neighboring mono prediction, then directional agreement across local models. Default mode 1 and original V1/V2 remain unchanged. The fitting, anchors, edge bands and plane smoother are shared with mode 1; their complete equations remain in [the pipeline specification](cpp_algorithm_pipeline.md).

## 1. Failure being addressed

Near a depth discontinuity, a one-pixel displacement between mono and stereo can compare different surfaces. Mode 1's threshold on the blended prediction can then remove a correct stereo sample. Changes of covering-window support can also make the deletion contour uneven. Hard invalidation amplifies the ordinary raster staircase at 640×360 into an eroded edge.

Voting alone cannot address a positional error shared by every window. The additional neighborhood test tolerates one input pixel of displacement. It reduces extra erosion; it does not fit a continuous straight contour or eliminate rasterization.

## 2. Discrete neighboring hypotheses

Let $s_i$ denote stereo disparity, $g_i$ the same normalized, unfiltered guidance as mode 1, and $V_m$ its validity mask. The finite neighborhood is

$$
\mathcal N_i=\{j\in\Omega:\|j-i\|_\infty\leq1,\ V_m(j)\}.
$$

Borders are replicated. The eligible center has finite mono, so this set is nonempty; repeated border samples do not alter the existence test below. The radius is fixed at **one input-image pixel** rather than automatically rescaled with resolution.

Let $\mathcal K_i$ contain distinct accepted windows covering $i$. Their positive-slope affine models and original triangle weights give

$$
p_{ik}=a_kg_i+b_k,\qquad a_k>0,\qquad w_{ik}>0,
$$

$$
A_i=\sum_{k\in\mathcal K_i}w_{ik}a_k,\quad
B_i=\sum_{k\in\mathcal K_i}w_{ik}b_k,\quad
Z_i=\sum_{k\in\mathcal K_i}w_{ik},\quad
\bar p_i=\frac{A_i g_i+B_i}{Z_i}.
$$

For each actual neighboring guidance value, evaluate the **same blended model at the target pixel**:

$$
\mathcal H_i=\left\{\frac{A_i g_j+B_i}{Z_i}:j\in\mathcal N_i\right\},\qquad
M_i=\left[\min_{h\in\mathcal H_i}|s_i-h|\leq\tau\right].
$$

Here $\tau$ is the existing disparity rejection threshold. A match $M_i$ vetoes deletion. The model coefficients and weights stay attached to $i$, not $j$. No hypothesis is copied into the output.

Crucially, $\mathcal H_i$ is a set of at most nine sampled predictions, **not the filled interval between its extrema**. If the neighborhood predicts only foreground and background, an intermediate flying depth receives no support unless it is within $\tau$ of an observed hypothesis. This distinction fixes the failure of the [archived interval prototype](../outputs/cpp_edge_guard_interval_experiment_640x360_20261104/REPORT.md).

For $A_i,Z_i>0$, the implementation avoids nine affine evaluations by inverting the tolerance test:

$$
M_i=\left[\exists j\in\mathcal N_i:\ell_i\leq g_j\leq u_i\right],\qquad
\ell_i=\frac{Z_i(s_i-\tau)-B_i}{A_i},\quad
u_i=\frac{Z_i(s_i+\tau)-B_i}{A_i}.
$$

This is a query interval around the stereo value, tested against actual mono samples. It does not fill gaps between hypotheses. The algebraic equivalence is in real arithmetic; the implementation uses float32, inclusive match comparisons and the original candidate mask explicitly. A NaN or reversed inverse interval causes abstention. A nonpositive $A_i$ also prevents deletion.

## 3. Directional model consensus

Each accepted model votes at the **central** mono sample, not at neighborhood extrema:

$$
v_i^-=\sum_{k\in\mathcal K_i}[p_{ik}\text{ finite}][s_i<p_{ik}-\tau],\qquad
v_i^+=\sum_{k\in\mathcal K_i}[p_{ik}\text{ finite}][s_i>p_{ik}+\tau].
$$

These are unweighted counts. Their denominator $n_i=|\mathcal K_i|$ includes all accepted covering models, so a nonfinite prediction cannot increase the agreement fraction. Duplicated border window indices are not counted. At most sixteen windows cover a pixel with the validated sampling geometry.

Consensus must agree with the direction of the original blended residual:

$$
C_i=[n_i\geq n_{\min}]\land\left(
([s_i<\bar p_i-\tau]\land[v_i^-\geq\rho n_i])\lor
([s_i>\bar p_i+\tau]\land[v_i^+\geq\rho n_i])
\right).
$$

Defaults are `minimum_votes` $n_{\min}=3$ and `vote_fraction` $\rho=0.8$. Three or four models require unanimity; five require at least four votes. A majority predicting below stereo cannot justify deletion when the weighted blend predicts above stereo.

Let $R_1$ be the original mode-1 candidate mask on exactly the same inputs and configuration. The final rule is

$$
\boxed{R_2(i)=R_1(i)\land[A_i>0]\land\neg M_i\land C_i.}
$$

The explicit $R_1$ condition guarantees $R_2\subseteq R_1$, including floating-point decision margins. All other candidates are kept at their original stereo value. Missing or conflicting evidence produces abstention rather than a correction.

## 4. Smoothing and validity invariants

The entire original edge band $E$ stays frozen before plane fitting. With anchor/protection guard $G$,

$$
H=G\lor E\lor R_2=G\lor E,
$$

because $R_2\subseteq R_1\subseteq E$. Thus the plane support, original-stereo training values and eligible smoothing targets are unchanged.

- Every mode-2 deletion was also a mode-1 deletion.
- Every value retained by mode 1 has exactly the same output under mode 2, including interior plane corrections, for the same build and inputs.
- Samples in $R_1\setminus R_2$ keep their original stereo disparity exactly, inside the frozen edge band.
- Original invalid stereo, explicit protection and missing center guidance remain unchanged.

All forty real cases were checked for these properties. Preserving an originally valid sample is not completion of an original invalid hole, and does not prove that the sample is physically correct.

## 5. Evidence and limitations

The [20-frame gallery](../outputs/cpp_edge_guard_640x360_sample20_20261104/index.html), [report](../outputs/cpp_edge_guard_640x360_sample20_20261104/REPORT.md) and [exact measurements](../outputs/cpp_edge_guard_640x360_sample20_20261104/aggregate.json) use existing corrected 640×360 inputs and both cached mono models. No inference is rerun.

Across these frames, mode 2 reduces removals from 308,538 to 118,752 with ZipDepth (61.5%) and from 279,559 to 133,343 with DAM (52.3%). These are deletion statistics, not accuracy improvements against ground truth.

In the known-depth slanted-step tests, either a +1 or −1 pixel mono offset causes mode 1 to delete 360 correct stereo samples. Mode 2 deletes zero correct samples and still removes all 43 injected spikes. The fixed radius does not cover arbitrary registration error; the offset sweep in the report includes ±2 pixels. In a separate test, mode 2 rejects all 360 intermediate flying depths with zero false deletions, whereas the archived interval prototype retained those wrong depths.

There remains an unavoidable local ambiguity: a one-pixel wrong-side foreground halo can match an actual neighboring foreground prediction, just as a one-pixel registration offset can. In the halo test, mode 1 removes all 360 wrong samples and mode 2 retains all 360. The guard chooses preservation under this uncertainty, so it cannot guarantee both zero false deletions and zero retained outliers. This is why it is an explicit option rather than a silent replacement.

The method performs no semantic segmentation, line fitting, hole filling or subpixel contour reconstruction. Coherent mono bias and model errors remain possible, and ordinary raster stairs remain visible when enlarged.

## 6. Usage, cost and compatibility

```python
from dataclasses import replace
from disp_fusion import FastAlignedConfig, FastAlignedRefiner

base = FastAlignedConfig()  # Supply disparity-unit scaling for your inputs.
guarded = replace(base, edge_mode=2)
with FastAlignedRefiner(640, 360, guarded) as engine:
    result = engine.refine(stereo, mono)
```

The saved comparison keeps its existing 2 px rejection threshold and 0.375 px smoothing bound. Mode 2 does not alter either value or other fitting/plane settings. It uses `minimum_votes` and `vote_fraction`, which mode 1 ignores. The one-pixel radius is fixed rather than a new configuration field.

C/Python structure layouts and ABI version 2 are unchanged. Feature mask **`0x20` (bit index 5)** advertises the final discrete guard; bit 4 alone identifies the older interval prototype. The wrapper explicitly rejects mode 2 on either an older baseline ABI-2 binary or the interval-only binary and requests a rebuild. Modes 0/1 remain usable on those binaries.

```bash
cmake --build build/native -j 2
build/native/disp_fusion_fast_bench 640 360 200 16 2
```

The final arguments select plane tile size and edge mode. Row/segment collection holds at most sixteen coefficient pairs on the stack. Neighborhood scans and votes run only for blocks containing candidates. There are no new dense frame buffers or frame-loop heap allocations; existing context payload is unchanged.

On the Intel i7-6850K CPU with Clang/AVX2 and one native thread, the mean of six case medians is 6.47 ms for the archived mode-1 build, 6.54 ms for mode 1 in the extended build, and 7.38 ms for guarded mode 2. The guard adds about 0.84 ms relative to current mode 1. These measurements exclude setup, registration, inference and I/O. They do not establish DSP timing or a hard deadline; the 5 ms target remains unmet.

`uncertain_pixels` in mode 2 counts eligible edge pixels without an accepted model plus original mode-1 candidates kept by the veto. These disjoint sets measure abstention, not calibrated confidence. Other edge/fit statistics retain their definitions.

The previous binary/source are preserved in the run's `baseline_native/` directory. Original V1/V2 sources and all historical outputs remain unchanged. The mode-1 equation audit predates this extension; its hashes identify that archived implementation. The rejected interval experiment also remains archived with its source and outputs for reproducibility.
