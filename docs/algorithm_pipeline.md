# Current Stereo–Mono Refinement Pipeline

Implementation reviewed: 2026-10-29. All numerical settings below are the current defaults unless stated otherwise.

The current algorithm is **stereo-driven rejection and bounded smoothing**, guided by monocular structure. It is not a weighted average of stereo depth and mono depth.

It contains two versions:

- **V1:** locally align mono to stereo, detect disagreements near edges, and invalidate selected stereo samples. Retained disparities are unchanged.
- **V2:** require multi-window agreement before invalidating a sample, then smooth reliable retained regions using mono-guided neighborhood weights.

Both versions preserve the original stereo coordinate system and never fill originally invalid pixels.

## 1. Overall workflow

```text
Original stereo disparity + camera calibration
                  │
                  ▼
      Project valid stereo samples into RGB
      Record visibility and source-pixel indices
                  │
                  ├──────── Cached native-RGB mono prediction
                  │                         │
                  ▼                         ▼
        Stereo edge estimation     Mono normalization and edges
                  │                         │
                  └────────────┬────────────┘
                               ▼
                 Select stable stereo anchors
                               ▼
              Robust mono → stereo local alignment
                               ▼
                  RGB-frame discrepancy evidence
                       ┌───────┴────────┐
                       ▼                ▼
                      V1               V2
                 Edge rejection   Consensus rejection
                       │                │
                       ▼                ▼
               Map removals back to stereo coordinates
                       │                │
                       │       Map mono guidance back
                       │                ▼
                       │       Bounded graph smoothing
                       ▼                ▼
                  V1 output         V2 output
```

**Coordinate projection and mono-value alignment are different operations:**

1. Calibration establishes which stereo samples correspond to RGB pixels.
2. Local affine fitting makes mono values numerically comparable with stereo.

Implementation: [pipeline.py](../disp_fusion/pipeline.py).

## 2. Inputs and assumptions

The algorithm takes:

| Input | Meaning |
|---|---|
| $d_s(p)$ | Original stereo disparity in reference-camera pixels |
| $M(q)$ | Cached monocular prediction in native RGB coordinates |
| $K_s,K_r$ | Stereo-reference and RGB intrinsics |
| $R,t$ | Stereo-reference-to-RGB rigid transform |
| RGB distortion | RGB projection distortion coefficients |
| $B$ | Stereo baseline |

The default assumptions are:

- Inputs already use normal calibrated camera coordinates.
- No image-orientation correction is part of the refinement pipeline.
- Stereo disparity corresponds to the supplied reference-camera calibration.
- Mono dimensions match the RGB calibration.
- The array API receives floating-point disparity, not undecoded Q6 values.
- A larger mono value should correspond to a larger disparity after a positive affine mapping.

The algorithm does **not** automatically determine whether an arbitrary model produces depth or inverse depth. A model with the opposite value convention needs an explicit input conversion.

A stereo sample is considered valid when

$$
V_s(p)=
\mathbf1\!\left[d_s(p)\text{ is finite}\;\land\;d_s(p)>0.05\right].
$$

For Q6 input, the reader decodes

$$
d_s=\frac{d_{\mathrm{Q6}}}{64}.
$$

The refinement library reads existing mono predictions; it does not run or train a neural network.

## 3. Project stereo into RGB

### 3.1 Recover stereo-reference 3D points

For a valid stereo pixel $p=(u_s,v_s)$,

$$
Z_s(p)=\frac{f_{x,s}B}{d_s(p)}.
$$

Then

$$
\mathbf X_s(p)=
Z_s(p)
\begin{bmatrix}
(u_s-c_{x,s})/f_{x,s}\\
(v_s-c_{y,s})/f_{y,s}\\
1
\end{bmatrix}.
$$

With the current calibration format, $B$, $Z_s$, and translation are expressed in millimeters.

### 3.2 Transform into the RGB camera

$$
\mathbf X_r(p)=R\mathbf X_s(p)+t.
$$

Writing $\mathbf X_r=(X_r,Y_r,Z_r)^\top$, the normalized RGB coordinates are

$$
x_r=\frac{X_r}{Z_r},
\qquad
y_r=\frac{Y_r}{Z_r}.
$$

Apply the calibrated RGB distortion model, then RGB intrinsics:

$$
(u_r,v_r)=
\Pi_{K_r,\mathrm{distortion}}(\mathbf X_r).
$$

The implementation rounds projected coordinates to the nearest RGB pixel. Samples outside the RGB image or behind its camera are excluded.

### 3.3 Resolve visibility with a Z-buffer

Multiple stereo samples may project to the same RGB pixel. The algorithm keeps the sample with the smallest positive $Z_r$.

For each occupied RGB pixel $q$, it stores:

- RGB-camera depth $Z_r(q)$;
- the winning original stereo source index $I(q)$;
- equivalent disparity

$$
s_r(q)=\frac{f_{x,s}B}{Z_r(q)}.
$$

Notice that the numerator remains $f_{x,s}B$. This is an **equivalent inverse-depth quantity**, not a new stereo measurement made by the RGB camera.

Implementation: [calibration.py](../disp_fusion/calibration.py).

## 4. Normalize mono and construct edge evidence

### 4.1 Normalize mono values

Let $Q_2(M)$ and $Q_{98}(M)$ be the second and ninety-eighth percentiles. Define

$$
m(q)=
\operatorname{clip}
\left(
\frac{M(q)-Q_2(M)}
{\max(Q_{98}(M)-Q_2(M),10^{-6})},
0,1
\right).
$$

This removes the raw output scale for subsequent structural processing. It does not establish metric depth.

### 4.2 Detect mono edges

The normalized mono is Gaussian-smoothed with $\sigma=1$:

$$
m_g=G_{\sigma=1}*m.
$$

The mono edge mask is

$$
E_m(q)=
\mathbf1\!\left[\|\nabla m_g(q)\|>0.012\right].
$$

The Gaussian-smoothed mono is used for edge detection. The affine fitting and later guidance use the normalized mono $m$.

### 4.3 Detect stereo edges

The projected stereo map contains holes. For **edge estimation only**, the algorithm:

1. temporarily fills holes using the nearest valid projected stereo sample;
2. applies a $5\times5$ median filter.

Let the resulting auxiliary map be $s_{\mathrm{med}}$. Then

$$
E_s(q)=
\mathbf1\!\left[\|\nabla s_{\mathrm{med}}(q)\|>2\right].
$$

**This temporary fill never becomes output depth.** It is only an auxiliary map for edge detection and anchor selection.

### 4.4 Construct the comparison band

$$
E(q)=\operatorname{Dilate}_{6}\big(E_m\lor E_s\big).
$$

Only samples inside this edge band are eligible for disagreement-based removal.

Consequently, the rejection stage does not inspect every interior pixel as a potential outlier.

## 5. Select stereo anchors

Local mono alignment must be anchored to stereo values that appear locally stable.

An anchor must:

1. contain valid projected stereo;
2. lie outside a 3-pixel dilation of the combined edge mask;
3. differ from the auxiliary median stereo by at most $0.75$ equivalent-disparity pixels.

Formally,

$$
A(q)=
V_r(q)
\land
\neg\operatorname{Dilate}_{3}(E_m\lor E_s)(q)
\land
\left[|s_r(q)-s_{\mathrm{med}}(q)|\le0.75\right].
$$

Here $V_r$ denotes valid projected stereo in RGB coordinates.

These are **heuristically stable anchors**, not verified ground truth. A smooth but incorrect stereo region can still satisfy these conditions.

## 6. Robust local mono-to-stereo alignment

### 6.1 Overlapping windows

The algorithm uses:

- window size: $97\times97$;
- stride: 32 pixels;
- image-border windows clipped to the image.

A window is processed only if it contains a valid stereo sample inside the comparison band.

### 6.2 Fit acceptance requirements

For each window:

- at least 60 anchors must be available;
- at most 1,200 anchors are sampled for fitting;
- normalized mono must have sufficient variation:

$$
Q_{95}(m_A)-Q_5(m_A)\ge0.025.
$$

The last condition rejects windows where mono variation is too small to estimate a stable slope.

### 6.3 Fit a robust affine mapping

For window $k$,

$$
\widehat s_k(q)=a_km(q)+b_k.
$$

The coefficients are estimated with eight iterations of iteratively reweighted least squares.

At each iteration:

$$
(a_k,b_k)
=
\arg\min_{a,b}
\sum_{q\in A_k}
w_q\big[s_r(q)-am(q)-b\big]^2.
$$

With residuals $r_q=s_r(q)-a_km(q)-b_k$, the robust scale is

$$
\sigma=
\max\left(
0.15,\;
1.4826\,\operatorname{median}
|r_q-\operatorname{median}(r)|
\right),
$$

and weights are updated as

$$
w_q=
\min\left(
1,\frac{1.5\sigma}{\max(|r_q|,10^{-8})}
\right).
$$

The final fit is rejected if

$$
a_k\le0
$$

or

$$
Q_{80}(|r_q|)>2\;\text{pixels}.
$$

Thus, a failed or inconsistent fit contributes no local prediction.

### 6.4 Combine accepted windows

Each accepted window predicts mono in stereo-equivalent units. Overlapping predictions are combined with separable center-weighted triangular weights:

$$
\widehat s(q)=
\frac{\sum_k\omega_k(q)\widehat s_k(q)}
{\sum_k\omega_k(q)}.
$$

The RGB-frame discrepancy is

$$
r(q)=|s_r(q)-\widehat s(q)|.
$$

If no accepted window covers a location, there is no valid residual-based removal decision there.

**Implementation nuance:** failure to obtain an affine fit disables rejection there, but does not necessarily disable V2 smoothing. Smoothing uses visible normalized mono guidance, not affine-fit coverage.

Implementation of Steps 4–6: [alignment.py](../disp_fusion/alignment.py).

## 7. V1: edge-band rejection

The default discrepancy threshold is

$$
\tau=4.
$$

A projected sample is marked for removal when

$$
D_{1,r}(q)=
V_r(q)
\land E(q)
\land[\widehat s(q)\text{ is available}]
\land[r(q)>\tau].
$$

The threshold is measured in **RGB-frame equivalent-disparity pixels**, not raw mono units or millimeters.

The removal decision is mapped back through $I(q)$, the stored original stereo source index.

The output is

$$
d_1(p)=
\begin{cases}
\mathrm{invalid},&p\in D_1,\\
d_s(p),&\text{otherwise}.
\end{cases}
$$

V1 therefore has two important properties:

- retained valid disparity values are exactly unchanged;
- originally invalid samples are never filled.

Stereo samples without a winning RGB projection cannot be removed through this RGB comparison.

## 8. V2: multi-window consensus rejection

V2 starts with the same V1 removal condition but adds an agreement requirement.

For every accepted window covering a valid projected sample,

$$
r_k(q)=s_r(q)-\widehat s_k(q).
$$

Count

$$
N(q)=\text{number of accepted covering windows},
$$

$$
N_+(q)=\sum_k\mathbf1[r_k(q)>\tau],
\qquad
N_-(q)=\sum_k\mathbf1[r_k(q)<-\tau].
$$

Define same-direction agreement:

$$
P(q)=
\frac{\max(N_+(q),N_-(q))}
{\max(N(q),1)}.
$$

V2 removes a sample only when

$$
D_{2,r}(q)=
D_{1,r}(q)
\land[N(q)\ge3]
\land[P(q)\ge0.8].
$$

All accepted covering windows count in the denominator, including those whose residual does not exceed the threshold.

Therefore,

$$
D_2\subseteq D_1.
$$

V2 removes fewer or equal samples than V1. This reduces some unstable deletions, but can also retain more genuinely incorrect samples. Overlapping windows are not independent measurements.

The V2 removal mask is then mapped back to original stereo coordinates.

## 9. Return mapping and smoothing eligibility

The return operation is **source-index mapping**, not dense back-projection of a synthesized RGB depth image.

For each winning correspondence $I(q)=p$:

- transfer the RGB removal decision to stereo pixel $p$;
- sample normalized mono guidance back to $p$:

$$
g(p)=m(q).
$$

Unobserved stereo samples have no mapped mono guidance.

A stereo sample is eligible for V2 smoothing only if it:

1. remains valid after V2 rejection;
2. has mapped mono guidance;
3. is outside the mapped 3-pixel mono-edge protection band;
4. was **not** a V1 removal candidate.

The last restriction is deliberate:

> A sample rejected by V1 but retained by V2 voting is considered uncertain. It keeps its original value and does not participate in smoothing.

## 10. V2: bounded mono-guided graph smoothing

This stage operates on the **original stereo pixel grid**, not the RGB grid.

Let $s_i$ be the retained original stereo disparity and $g_i$ its mapped normalized mono guide.

### 10.1 Build a four-neighbor graph

Only horizontal and vertical neighbors are connected.

A pair receives zero weight if either endpoint is unsafe or invalid, or if

$$
|s_i-s_j|>2\;\text{pixels}.
$$

Otherwise,

$$
w_{ij}=
\exp\left[
-\frac{(g_i-g_j)^2}{2(0.015)^2}
-\frac{(s_i-s_j)^2}{2(1.0)^2}
\right].
$$

The weights are fixed before iteration.

Mono therefore controls **which neighbors can be smoothed together**, rather than providing the absolute target disparity.

### 10.2 Data-anchored smoothing objective

The corresponding graph energy is

$$
E(d)=
\frac12\sum_i(d_i-s_i)^2
+
\frac{\alpha}{2}
\sum_{\{i,j\}}w_{ij}(d_i-d_j)^2,
\qquad \alpha=4.
$$

The first term anchors the result to stereo. The second suppresses local disparity variation across permitted connections.

### 10.3 Apply bounded Jacobi updates

Initialize

$$
d_i^{(0)}=s_i.
$$

Perform 16 iterations:

$$
d_i^{(t+1)}
=
\operatorname{clip}
\left(
\frac{s_i+\alpha\sum_jw_{ij}d_j^{(t)}}
{1+\alpha\sum_jw_{ij}},
\;
s_i-0.75,
\;
s_i+0.75
\right).
$$

The bound is always relative to the **original retained stereo value**, not the previous iteration. It does not accumulate across iterations.

Unsafe samples and samples without graph connections remain unchanged. Invalid samples are never graph nodes and remain invalid.

This is a fixed-iteration approximate optimization, not an exact global solution.

Implementation: [smoothing.py](../disp_fusion/smoothing.py).

### 10.4 Optional stronger interior smoothing

With `InteriorSmoothConfig(enabled=True)`, V2 applies an additional multi-scale affine-disparity stage after the graph smoother. The default workflow above is unchanged when the option is disabled.

The extension preserves rejection masks and protected pixels, fits local stereo trends at 31/63/127-pixel scales, and uses mono planarity checks to gate those fits. It does not use mono absolute depth. Corrections remain bounded relative to original stereo, rather than receiving an additional modification budget. See [the complete optional-stage specification](interior_smoothing.md) for component handling, holdout checks, overlap blending, and limitations.

## 11. Outputs and guarantees

The API returns:

| Field | Meaning |
|---|---|
| `disparity` | Final V1 or V2 floating-point disparity |
| `removed_mask` | Originally valid samples invalidated by the algorithm |
| `rejected_disparity` | Rejection-only result, before V2 smoothing |
| `smoothed_mask` | Retained samples changed by more than 0.05px |
| `diagnostics` | Alignment and smoothing statistics |

`smoothed_mask` is a reporting threshold: changes smaller than or equal to 0.05px can exist without being marked.

CLI exports include floating-point arrays, Q6 disparity, masks, statistics, and colormaps. Q6 invalid values are encoded as $-1$.

The implementation checks that:

- no originally invalid sample becomes valid;
- smoothing does not change the post-rejection validity mask;
- V1 retained values remain unchanged;
- V2 smoothing changes do not exceed the configured disparity bound.

All output coordinates and dimensions match the stereo input.

## 12. What this algorithm does—and does not—achieve

The current method uses mono in two specific ways:

1. **Locally aligned mono values identify edge-near stereo disagreements.**
2. **Mono gradients and differences protect boundaries and guide local smoothing.**

It does not perform semantic object segmentation, enforce straight-line edges, fit entire walls to planes, or reconstruct large incorrect regions from mono.

Important limitations are:

- Incorrect stereo can produce incorrect RGB projections.
- Stable-looking stereo anchors can still be wrong.
- The affine mono–stereo relationship can fail locally.
- Rejection can remove correct samples.
- Interior errors outside the comparison band can remain.
- Boundary protection reduces smoothing across detected edges; it does not prove that those edges are geometrically correct.
- The $0.75$px bound is a disparity bound, not a fixed metric-depth bound:

$$
Z=\frac{f_{x,s}B}{d},
\qquad
\Delta Z=
-\frac{f_{x,s}B\,\Delta d}{d(d+\Delta d)}.
$$

The same disparity adjustment can therefore cause a larger depth change at smaller disparity.

**In short: V1 preserves stereo values and removes selected disagreements; V2 makes rejection more conservative and reduces small-scale jitter. Neither version guarantees correct absolute depth or reconstructs mono's complete geometry.**

## Source reference

- [Shared pipeline and return mapping](../disp_fusion/pipeline.py)
- [Calibration and stereo-to-RGB projection](../disp_fusion/calibration.py)
- [Edges, anchor selection, robust alignment, and V1 rejection](../disp_fusion/alignment.py)
- [Bounded graph smoothing](../disp_fusion/smoothing.py)
- [Default parameters and configuration validation](../disp_fusion/config.py)
- [Input decoding and Q6 export](../disp_fusion/io.py)
- [Mono cache provenance checks](../disp_fusion/cache.py)
- [Unified command-line interface](../disp_fusion/cli.py)

[Back to the project README](../README.md)
