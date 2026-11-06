# Stereo–Mono Refinement Library

This library retains the two classical refinement pipelines confirmed on 2026-10-28. Monocular predictions are read exclusively from existing native-RGB caches:

- **V1**: robust local alignment → discrepancy testing in edge bands → invalidation. Retained disparities are unchanged.
- **V2**: veto V1 removal candidates unless multiple windows agree, then apply bounded mono-guided smoothing. Defaults require at least 3 windows and 80% agreement on a same-sign, above-threshold residual; each disparity changes by at most 0.75px.

Neither version fills originally invalid pixels; V2 smoothing also leaves newly invalidated pixels empty. Mono guides structural decisions and neighborhood weights, without directly overwriting absolute depth. There is no network training, downloading, or inference, no sphere template, and no legacy-version fallback chain. Comparison uses the locally aligned mono prediction at the same RGB pixel.

For the complete workflow, mathematical definitions, default parameters, and implementation limitations, see [Current Stereo–Mono Refinement Pipeline](docs/algorithm_pipeline.md).

## 1. Python library

```python
from pathlib import Path
import numpy as np
from disp_fusion import DepthRefiner, RefinementConfig, load_depth_rgb_calibration
from disp_fusion.io import read_stereo_bin

cal = load_depth_rgb_calibration(Path("/path/to/raw_calibration_list.json"),
                                 (1280, 720), (1280, 720))
stereo = read_stereo_bin(Path("/path/to/000286_disp.bin"), 720, 1280, scale=64)
mono = np.load("/path/to/native_rgb_mono/000286.npy")

refiner = DepthRefiner(cal, RefinementConfig(threshold_px=4, max_change_px=.75))
results = refiner.refine_many(stereo, mono, versions=("v1", "v2"))
v1 = results["v1"].disparity
v2 = results["v2"].disparity
v2_vote_only = results["v2"].rejected_disparity

# Running a single version uses the same pipeline:
result = refiner.refine(stereo, mono, version="v1")
```

For a given threshold, `refine_many()` performs the shared projection and window alignment only once. The array API neither reads dataset paths nor modifies its inputs; it returns removal masks, smoothing masks, and diagnostics. Stereo input must be float32/float64 disparity in native pixel units, not undecoded Q6 values or depth in millimeters. Inputs are assumed to use normal calibrated camera coordinates; outputs preserve the stereo input's orientation and pixel grid.

## 2. One CLI for all versions and mono sources

Run directly from the project root, or optionally install with `pip install -e . --no-deps --no-build-isolation` to use the `disp-fusion` command. The example paths below represent normally oriented data: disparity must match the stereo reference calibration, and cached mono must match the RGB images and RGB calibration.

```bash
python -m disp_fusion run \
  --data-root /path/to/dataset \
  --stereo-root /path/to/stereo_disparity \
  --mono-cache /path/to/dam_rgb_cache \
  --all --versions v1 v2 --thresholds 4 \
  --output outputs/library_dam_v1_v2
```

To switch to ZipDepth, change only the cache and output directories, not the algorithm code:

```bash
python -m disp_fusion run \
  --data-root /path/to/dataset \
  --stereo-root /path/to/stereo_disparity \
  --mono-cache /path/to/zipdepth_rgb_cache \
  --all --versions v1 v2 --thresholds 4 \
  --allow-legacy-unhashed-cache \
  --output outputs/library_zip_v1_v2
```

- `--frames 000286`: selected frames; `--sample20`: the retained 20-frame sample; `--all`: all frames in the cache. These options are mutually exclusive.
- `--versions v1` or `v2`: run only the specified version.
- `--thresholds 2 4 8`: sweep thresholds through the same entry point, without additional algorithm scripts.
- `--config settings.json`: override `RefinementConfig` fields; window settings belong in the nested `alignment` object.
- `--stereo-run /path/to/run`: read the original stereo from an existing result manifest instead of `--stereo-root`. This never uses a previous fused output as the stereo input.
- Nonempty output directories are rejected by default. Even `--overwrite` only permits overwriting result directories containing this library's `run.json`.

The legacy Zip cache lacks input hashes and requires explicit permission to use. Recording current input hashes cannot establish byte-for-byte identity with historical inputs. A DAM cache input-hash mismatch raises an error. The CLI does not invoke a GPU or neural network.

## 3. Shared export and comparison

The common result format includes `run.json`, per-frame floating-point disparity/Q6, removal/smoothing masks, `summary.json`, `aggregate.json`, and full-frame colormaps. New exports also save the original stereo for standalone comparisons; invalid Q6 values are -1. V2's vote-only output is available as the `v2_vote_tau4` ablation.

```bash
python -m disp_fusion compare \
  --input DAM outputs/library_dam_v1_v2 v2_tau4 \
  --input Zip outputs/library_zip_v1_v2 v2_tau4 \
  --output outputs/library_model_comparison
```

Use the variant names listed under `variants` in `run.json`. Compare existing results by changing input paths/variants, without writing dedicated scripts. Roughness on commonly retained stencils and removal rates are reported separately; smoother output does not imply greater accuracy against physical ground truth. The library does not depend on the removed legacy IR matching cache.

## 4. Retained experiment results

The original results have been supplemented with the common `run.json`. Arrays, historical statistics, and gallery images are unchanged; documentation and gallery text have been translated into English.

| Model / version | Directory | Main variant |
|---|---|---|
| DAM V1 | `outputs/simple_edge_rejection_dam518_sample20` | `v1_tau4`, also tau2/tau8 |
| Zip V1 | `outputs/simple_edge_rejection_zipdepth_sample20` | `v1_tau4`, also tau2/tau8 |
| DAM V2 | `outputs/edge_consensus_smooth_v2_dam518_sample20` | `v2_tau4`, `v2_vote_tau4` |
| Zip V2 | `outputs/edge_consensus_smooth_v2_zipdepth_sample20` | `v2_tau4`, `v2_vote_tau4` |

The two retained native input caches are `outputs/damv2l_rgb_native_s518_sample20` (20 frames) and `outputs/zipdepth_rgb_native_all907` (907 frames). Both model/version comparison galleries are also retained.

The galleries' `RESULTS.md` and JSON records describe historical experiments and may reference original scripts that have since been archived. Their recorded dataset-specific import conventions are historical provenance, not steps in the default pipeline. Existing manifests retain those conventions for reading saved results. For the current workflow on normally oriented inputs, use this README and `python -m disp_fusion --help`. The original evidence for historical IR statistics is also archived, rather than required at runtime.

## 5. Coordinate conventions

The default workflow is **stereo reference → RGB comparison → stereo reference**. It assumes normally oriented inputs that already match their camera calibration. It does not apply image-orientation corrections or infer a reference camera from filenames.

For a stereo pixel $p=(u,v)$ with disparity $d_s(p)$, reconstruct its reference-camera point and transform it into RGB:

$$
Z_s(p)=\frac{f_sB}{d_s(p)},\qquad
\mathbf X_s(p)=Z_s(p)K_s^{-1}[u,v,1]^\top,
\qquad \mathbf X_r(p)=R_{s\to r}\mathbf X_s(p)+\mathbf t_{s\to r}.
$$

Project $\mathbf X_r$ through the RGB intrinsics and distortion model. A nearest-depth Z-buffer resolves collisions and records the original stereo source index for each occupied RGB pixel. V1/V2 compare locally aligned mono with the projected stereo equivalent disparity $f_sB/Z_r$ in RGB coordinates.

The return mapping uses these recorded source indices: RGB rejection decisions and mono guidance are mapped back only to the corresponding original stereo samples. V1 leaves retained disparities unchanged; V2 applies bounded smoothing on the original stereo grid using the mapped guidance. This is a sparse correspondence-based return mapping, not dense back-projection of a newly synthesized RGB depth map. Unobserved source pixels are left unchanged, and neither original holes nor newly removed points are filled. Output dimensions, orientation, and reference camera always match the stereo input.

## 6. File organization

```text
disp_fusion/
  config.py          # Configuration shared by all versions
  calibration.py     # Calibration and a single forward projection
  alignment.py       # Shared window alignment, edge tests, reliable anchors
  pipeline.py        # One API, with stages selected by v1/v2
  smoothing.py       # Optional bounded graph smoothing
  _acceleration.py   # Optional native compilation; unchanged numerical policy
  fast_aligned.py    # Experimental already-aligned native backend wrapper
  cache.py           # Read-only mono cache and input identity checks
  io.py              # Data loading and Q6 export
  metrics.py         # Post-hoc diagnostics, not decision inputs
  visualization.py   # Shared colormaps, grids, and galleries
  artifacts.py       # Common result reading and writing
  cli.py             # Single entry point for run / compare
native/              # Standalone C++ kernel, C ABI, optional SIMD and benchmark
```

## 7. Validation

```bash
python -m pytest -q
DISP_FUSION_REAL_DATA=1 python -m pytest -q tests/test_saved_outputs.py
```

The second command uses the two retained caches to perform elementwise regression checks of V1, V2, and V2 vote-only arrays on 20 frames per model (DAM/Zip), plus enhanced-output/mask checks on three frames per model, without rerunning either model. Local real-data tests are skipped by default. All 173 tests passed on this machine: 72 basic reference tests, 55 optional native cases across GCC/Clang/scalar configurations, and 46 cached real-data regressions. Saved reference outputs remain elementwise identical. The NumPy-only reference fallback previously passed all 72 basic cases. Coordinate tests cover asymmetric input orientation, identity and translated stereo-to-RGB projection, return mapping to source pixels, and agreement between the CLI and array API. Optional-interior tests cover stripes on sloped planes, curvature rejection, disconnected regions, true invalid holes versus sparse guidance gaps, coverage-edge tapering, and the shared modification bound. Acceleration tests cover numerical fits, exact sampling/percentiles, deterministic worker order, explicit fallback and read-only cache handling. Native tests require CMake and an available C++17 compiler; unavailable compiler cases are skipped. Historical import metadata remains supported without changing the default orientation of normal inputs. The unified CLI and wheel packaging were previously verified. Historical V7 was removed with the old algorithms and is no longer part of the current test suite.

## 8. Cleanup and recovery

Legacy algorithms, standalone experiment scripts, old tests/reports, FoundationStereo-specific dependencies, and old generated data were moved out of the project. Raw data, external model repositories, and checkpoints were untouched. User configuration such as `.git` and `.vscode` was retained.

- Manifest: `CLEANUP_MANIFEST.json`, 172 entries, approximately 23.1 GiB.
- Temporary recovery directory: `/tmp/disp-fusion-legacy-20261028-KhlYhg`, preserving original relative paths.
- **This is not a permanent backup; the system may clean `/tmp`.** Use the manifest to recover individual files, and do not overwrite the current library wholesale.
- The temporary archive is on the same filesystem, so this space has not been freed; the files were only removed from the active project. The archive can be cleaned up separately once it is no longer needed.

## 9. Optional stronger interior smoothing

V2 can additionally apply boundary-frozen, multi-scale affine-disparity smoothing. It uses stereo to determine local surface trends and mono to gate suitable regions. The stage is disabled by default; it changes neither V1 nor V2 rejection decisions, fills no holes, and preserves protected pixels. The final disparity remains within the original stereo ±0.75px by default.

```python
from disp_fusion import DepthRefiner, RefinementConfig, InteriorSmoothConfig

config = RefinementConfig(interior=InteriorSmoothConfig(enabled=True))
result = DepthRefiner(cal, config).refine(stereo, mono, version="v2")
```

For the existing CLI, pass `--config` with `{"interior": {"enabled": true}}`. See [the mathematical specification and limitations](docs/interior_smoothing.md). This is stronger local smoothing, not a metric-depth prediction or guaranteed wall reconstruction. The experiment script remains outside the project in `/tmp`.

The 20-frame ZipDepth/DAM comparison is available in the [interior-smoothing gallery](outputs/interior_multiscale_sample20_20261029/index.html), with [evaluation notes](outputs/interior_multiscale_sample20_20261029/README.md). Local wall residuals improve, but broad directional variation is not uniformly reduced and cached IR-consistency MAE increases slightly. The enhancement therefore remains disabled by default.

## 10. Equivalent CPU acceleration

When Numba is installed, the existing API and CLI automatically compile the robust fitting and interior-window kernels. Independent interior fits use up to four CPU workers, but overlapping corrections are still accumulated serially in their original order. No window sizes, samples, iterations, thresholds, validity rules, or disparity bounds are reduced. The optional interior stage remains disabled unless explicitly enabled.

```bash
pip install -e '.[acceleration]'
```

Numba is already available on the evaluated machine; no installation or command change is needed there. Without Numba, the same library uses its NumPy implementation. Set `DISP_FUSION_ACCELERATION=0` before starting Python to explicitly disable native compilation, or `DISP_FUSION_WORKERS=1` for single-worker fitting. `DISP_FUSION_WORKERS` otherwise overrides the default of at most four workers. The first call can incur compilation overhead; machine code is cached. Set `NUMBA_CACHE_DIR` to a writable cache directory when running from a read-only installation.

The complete cached-input pipeline with stronger interior smoothing now takes approximately **1.9 seconds per 1280×720 frame**, versus roughly **8 seconds** for the original implementation on the i7-6850K CPU. The interior stage alone takes about **0.56 seconds**. These steady-state timings exclude inference, plotting, export and input loading. The first frame with an empty compilation cache took 15.31 seconds, including compilation. See [performance and equivalence validation](docs/performance.md) for the benchmark scope and limitations. No neural inference was rerun and existing result files were not overwritten.

## 11. Edge-first aligned-input backend

`FastAlignedRefiner` is an optional native approximation for already-aligned inputs. The active resized target is **640×360**, preserving the original 16:9 aspect ratio. The latest sample evaluation also processes original **1280×720** inputs. Registration and mono inference are outside the kernel; V1/V2 remain available as references.

The default order is **robust local edge rejection → freeze the entire edge band → bounded interior stereo-plane smoothing**. Rejected samples never enter plane fitting or get filled. Retained edge samples stay unchanged. Switching off smoothing or changing its tile size cannot alter rejection decisions.

For excessive edge erosion, optional **`edge_mode=2`** adds a one-pixel tolerance by matching actual neighboring mono predictions, followed by directional model consensus before deletion. It preserves every value retained by mode 1 and keeps vetoed candidates at their original stereo disparity. The [guard definition and tradeoffs](docs/edge_rejection_guard.md), [20-frame old/new gallery](outputs/cpp_edge_guard_640x360_sample20_20261104/index.html), and [report](outputs/cpp_edge_guard_640x360_sample20_20261104/REPORT.md) include known-depth tests and timing. Mode 1 remains the default. The guard can retain a thin wrong fringe; it does not guarantee correct edge geometry.

```python
from disp_fusion import FastAlignedConfig, FastAlignedRefiner

# Already-aligned float32 disparity and inverse-depth-like mono arrays.
with FastAlignedRefiner(640, 360, FastAlignedConfig(edge_mode=1)) as refiner:
    result = refiner.refine(stereo, mono)
```

Defaults use physical disparity pixels and do not automatically rescale with resolution. The saved half-size sample run explicitly halves disparity-valued tolerances. Exact configurations are recorded in its manifest.

Use `process_into()` with preallocated buffers for frame processing. The native ABI is 2. See the [complete mathematical pipeline](docs/cpp_algorithm_pipeline.md), [build/API guide](docs/fast_aligned.md) and [comparison with V1/V2](docs/cpp_vs_v1_v2.md).

The [original-resolution sample gallery](outputs/cpp_edge_first_samples_native_640x360_20261104/index.html), [report](outputs/cpp_edge_first_samples_native_640x360_20261104/REPORT.md), and [measurements](outputs/cpp_edge_first_samples_native_640x360_20261104/aggregate.json) contain frames `000286`, `000450`, `000900` with DAM Large/518 and ZipDepth at both resolutions: twelve cases. Existing mono caches are reused with corrected bilinear guidance sampling and a separate depth-consistency visibility gate. The fusion kernel is unchanged. All cases preserve original holes and unsupported samples, obey the retained-change bound, and produce identical rejection masks with smoothing disabled.

The [20-frame V1/V2 and C++ gallery](outputs/v1_v2_cpp_640x360_sample20_20261104/index.html) runs the original public V1/V2 pipeline at 640×360 on the exact frame list from the earlier comparison. Its main layout remains Original / ZipDepth V1 above ZipDepth V2 / DAMv2-L V2. Expandable sections add C++ and a separate comparison of the unchanged reference refinement cores on the same aligned inputs. See its [report](outputs/v1_v2_cpp_640x360_sample20_20261104/README.md) and [configuration](outputs/v1_v2_cpp_640x360_sample20_20261104/run.json). No inference is rerun.

| Resolution | Mean of six case CPU medians |
|---|---:|
| 1280×720 | 27.16 ms |
| 640×360 | 6.67 ms |

These CPU measurements exclude registration, inference and setup, and do not establish DSP or worst-case performance. The approximately 5 ms/frame target remains unmet. The 20-frame evaluation separates the original RGB workflow from the shared-aligned-input refinement comparison; no independent physical depth truth is available.

The [historical 640×400 results](outputs/fast_aligned_edge_first_640x400_20261103/index.html) retain their original winner-only guidance defect and should not be used as a controlled quality baseline for this run. Older outputs, the previous native binary/source snapshot and the `edge_mode=0` implementation remain available under [the earlier evaluation](outputs/fast_aligned_sample3_20261103/).
