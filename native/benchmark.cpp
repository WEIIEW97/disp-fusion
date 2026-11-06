#include "fast_aligned.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <new>
#include <vector>
namespace {
bool track_allocations = false;
size_t frame_allocations = 0;
} // namespace
void *operator new(std::size_t size) {
  if (track_allocations)
    ++frame_allocations;
  if (void *p = std::malloc(size ? size : 1))
    return p;
  throw std::bad_alloc();
}
void *operator new[](std::size_t size) { return ::operator new(size); }
void operator delete(void *p) noexcept { std::free(p); }
void operator delete[](void *p) noexcept { std::free(p); }
void operator delete(void *p, std::size_t) noexcept { std::free(p); }
void operator delete[](void *p, std::size_t) noexcept { std::free(p); }
int main(int argc, char **argv) {
  const int w = argc > 1 ? std::atoi(argv[1]) : 640,
            h = argc > 2 ? std::atoi(argv[2]) : 360;
  const int iterations = argc > 3 ? std::atoi(argv[3]) : 100;
  if (w < 2 || h < 2 || iterations < 1)
    return 1;
  const size_t n = static_cast<size_t>(w) * h;
  std::vector<float> s(n), m(n), out(n), samples(iterations);
  std::vector<uint8_t> removed(n), smoothed(n);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      const size_t i = static_cast<size_t>(y) * w + x;
      const float step = x >= w / 2 ? 25.f : 0.f;
      s[i] = 30.f + .003f * x + .002f * y + step + .12f * std::sin(x * .28f);
      m[i] = .15f + .0001f * x + .0001f * y + step * .02f;
      if (x % 91 == 0 && y % 83 == 0)
        s[i] = std::numeric_limits<float>::quiet_NaN();
    }
  df_fast_config cfg;
  df_fast_default_config(&cfg);
  if (argc > 4)
    cfg.tile_size = std::atoi(argv[4]);
  if (argc > 5)
    cfg.edge_mode = std::atoi(argv[5]);
  auto *context = df_fast_create(w, h, &cfg);
  if (!context)
    return 2;
  df_fast_stats stats{};
  track_allocations = true;
  for (int i = 0; i < 5; ++i)
    df_fast_process(context, s.data(), m.data(), nullptr, out.data(),
                    removed.data(), smoothed.data(), &stats);
  for (int k = 0; k < iterations; ++k) {
    const auto begin = std::chrono::steady_clock::now();
    const int status =
        df_fast_process(context, s.data(), m.data(), nullptr, out.data(),
                        removed.data(), smoothed.data(), &stats);
    const auto end = std::chrono::steady_clock::now();
    if (status)
      return 3;
    samples[k] = std::chrono::duration<float, std::milli>(end - begin).count();
  }
  track_allocations = false;
  std::sort(samples.begin(), samples.end());
  std::cout << "{\"width\":" << w << ",\"height\":" << h
            << ",\"tile_size\":" << cfg.tile_size
            << ",\"edge_mode\":" << cfg.edge_mode
            << ",\"iterations\":" << iterations
            << ",\"median_ms\":" << samples[iterations / 2]
            << ",\"p90_ms\":" << samples[(iterations - 1) * 9 / 10]
            << ",\"min_ms\":" << samples.front()
            << ",\"max_ms\":" << samples.back()
            << ",\"workspace_bytes\":" << stats.workspace_bytes
            << ",\"plane_models\":" << stats.plane_models
            << ",\"alignment_models\":" << stats.alignment_models
            << ",\"frame_heap_allocations\":" << frame_allocations
            << ",\"native_threads\":1,\"build_features\":"
            << df_fast_build_features() << ",\"last_phase_ms\":[";
  for (int i = 0; i < 7; ++i)
    std::cout << (i ? "," : "") << stats.phase_ms[i];
  std::cout << "],\"dsp_measurement\":false}" << std::endl;
  df_fast_destroy(context);
  return 0;
}
