# Equivalent CPU Acceleration

This document describes equivalent acceleration of the retained V1/V2 algorithm. The later [experimental aligned-input backend](fast_aligned.md) is a different, fixed-cost approximation for a DSP-oriented budget; its timing and quality claims must not be confused with the equivalence checks below.

## Scope

The 2026-11-03 optimization changes execution, not the refinement policy. It applies to the existing V1/V2 library and its optional stronger interior stage. The public array API, CLI and algorithm configuration remain unchanged. Enabling native execution does not enable the optional interior stage.

No input downsampling, fewer windows, fewer anchors, fewer IRLS/Jacobi iterations, relaxed gates, new hole filling, or neural inference is used. The original metric anchoring, boundary protection, topology checks, holdout split and total disparity bound are retained.

## Implementation

1. Optional Numba kernels compile robust SVD least-squares IRLS, median/MAD calculations, interior-window setup and correction accumulation. `fastmath=False` prevents unsafe floating-point reassociation. The weighted least-squares method is still SVD, not a normal-equation approximation. Its rank cutoff explicitly matches NumPy's `rcond=None` definition.
2. Anchor selection and window topology are prepared once per frame. Training/holdout samples retain NumPy's original integer-linspace selection, including endpoint rounding.
3. At most four CPU workers evaluate independent interior-window models. They read immutable frame arrays and do not write overlapping outputs. Results are consumed in the original scale/row/column order, and correction accumulation remains serial and deterministic.
4. Scalar native accumulation avoids per-window prediction grids and weight temporaries. A vectorized NumPy implementation is retained when native execution is disabled.
5. Percentile gates explicitly use NumPy's linear-interpolation arithmetic. Numba's built-in percentile uses a slightly different interpolation expression, which could otherwise move a strict decision by a floating-point ULP.

These kernels are shared implementation details, not a new algorithm version or an independent experiment script. The benchmark script is outside the project: `/tmp/benchmark_disp_fusion_acceleration_20261103.py`.

## Measurement

Hardware: Intel Core i7-6850K, six physical cores / twelve logical CPUs. Software: Python 3.11, NumPy 1.24.4, OpenCV 4.11.0 and Numba 0.57.1. The interior fit pool uses four workers; OpenCV reports twelve threads. The refinement implementation uses no GPU.

Inputs are 1280×720 stereo disparity and native-RGB mono arrays. Both existing mono caches are reused for the same twenty retained frames, giving forty enhanced refinement runs. One warm-up is excluded. Three selected frames per model (`000286`, `000450`, `000900`) are additionally evaluated with a snapshot of the original code for fresh paired runtime and full-array equivalence comparisons.

Approximate steady-state costs on this machine:

| Scope | Before | Accelerated |
|---|---:|---:|
| Complete pipeline with stronger interior smoothing | 8.0 s/frame | 1.9 s/frame |
| Additional interior stage | 5.7 s/frame | 0.56 s/frame |
| Estimated 907-frame sequential batch, one mono source | About two hours | About 29 minutes |

The 907-frame figure is an extrapolation, not a completed 907-frame benchmark. Times exclude mono inference, disk input, colormap generation, metric/report generation and output export. First-use compilation and process startup are also excluded. Frame content, memory allocation and other machine load affect timings. The optimized code is still not real-time video processing.

With a deliberately empty compilation cache, the first complete frame took 15.31 seconds, including compilation and processing. Following frames use the approximately 1.9-second steady-state path. Warm up once during application startup and keep a writable persistent code cache; do not interpret the steady-state figure as cold-start single-frame latency.

Exact per-frame times, paired reference times, comparison checks and source SHA-256 hashes are in [the benchmark record](performance_benchmark_20261103.json). The original-code snapshot is temporarily retained at `/tmp/disp-fusion-speed-reference-zS2TFi/disp_fusion`; it is not an installed fallback algorithm or permanent backup.

| Mono cache | Enhanced pipeline mean, 20 frames | Interior mean | Original-code mean, 3 paired frames | Accelerated mean, same 3 frames | Paired speedup |
|---|---:|---:|---:|---:|---:|
| ZipDepth | 1.915 s | 0.559 s | 7.996 s | 1.836 s | 4.36× |
| DAMv2-L | 1.938 s | 0.555 s | 8.042 s | 1.904 s | 4.22× |

The two different frame populations are stated explicitly: speedup uses the matched three-frame subset, not the twenty-frame accelerated mean divided by an unmatched reference mean.

## Equivalence checks

The forty model/frame runs are compared against the original saved enhanced outputs. The benchmark checks valid disparity differences, validity masks, Q6 rounding, accepted/attempted window counts, accepted counts per scale, modeled coverage, guidance-gap targets, protected/frozen changes and changed-pixel counts. The six fresh original-code comparisons also check complete output arrays and removal/smoothing masks. Evaluated outputs have zero disparity differences; no saved colormaps need regeneration.

Unit tests additionally cover robust-fit agreement with the original NumPy SVD IRLS calculation, rank deficiency, exact sampling, exact percentile interpolation, serial-versus-parallel output identity and explicit fallback selection. Existing real-data V1/V2 regression tests remain in place, with six additional enhanced-output/mask regression cases.

Validation completed with 118 passing cases (72 basic and 46 cached real-data cases). Explicit NumPy-only execution also passed all 72 basic cases; the 46 opt-in real-data cases were skipped in that separate fallback run. The normal CLI help command was checked without a cache-directory override, including the read-only-cache fallback.

This is measured equivalence on the tested data and platform, not a proof of bitwise identity for every floating-point input, BLAS library or future compiler. Keeping numerical precision, SVD, sampling and ordered reduction reduces that risk; strict boundary cases should remain part of deployment validation.

## Usage and fallback

Install the optional extra with `pip install -e '.[acceleration]'` if Numba is unavailable. It is already installed on the evaluated machine. No changes to the normal `DepthRefiner` call or CLI are required.

- `DISP_FUSION_ACCELERATION=0`: disable compilation before importing the library; run the NumPy implementation with one fit worker.
- `DISP_FUSION_WORKERS=1`: retain native kernels but use a single interior-fit worker. A positive integer can override the default worker count. Avoid oversized pools, particularly when processing multiple frames concurrently.
- `NUMBA_CACHE_DIR=/path/to/writable/cache`: relocate compiled-code caches when required by installation permissions. A missing or invalidated cache adds compilation overhead on the next first call. If no writable cache locator exists, the library can still import and compile in memory, but that compilation cannot be reused by a later process.

The algorithm's accuracy limitations are unchanged: stronger smoothness is not proof of physical depth accuracy. See [the interior-smoothing specification](interior_smoothing.md) and [the full pipeline](algorithm_pipeline.md).
