# Picker Benchmark Results

Benchmarks measure the per-keystroke hot path: feature normalization and NN inference
across 1K, 10K, and 100K file repositories with realistic usage data (full recency
list, transition history, trigrams, open buffers).

Run with `just benchmark`. See `benchmarks/picker_benchmark.lua` for methodology.

## Batch Norm Fusion (548ca15)

Fuses batch normalization parameters into weight matrices at load time so
`calculate_score()` runs a tight scalar loop with zero table allocations,
replacing the general-purpose `forward_pass()` at inference.

### Before (69d7b15)

```
--- 1,000 files ---
Static features:        3.55ms total |  3.551us/item
Per-keystroke:          7.05ms total |  7.047us/item
  normalize:            0.22ms total |  0.223us/item
  nn_inference:         6.28ms total |  6.284us/item

--- 10,000 files ---
Static features:       39.29ms total |  3.929us/item
Per-keystroke:         91.66ms total |  9.166us/item
  normalize:            3.32ms total |  0.332us/item
  nn_inference:        74.12ms total |  7.412us/item

--- 100,000 files ---
Static features:      632.26ms total |  6.323us/item
Per-keystroke:       1154.95ms total | 11.549us/item
  normalize:           61.84ms total |  0.618us/item
  nn_inference:      1039.04ms total | 10.390us/item
```

### After (548ca15)

```
--- 1,000 files ---
Static features:        3.37ms total |  3.372us/item
Per-keystroke:          0.92ms total |  0.925us/item
  normalize:            0.07ms total |  0.073us/item
  nn_inference:         0.84ms total |  0.837us/item

--- 10,000 files ---
Static features:       41.00ms total |  4.100us/item
Per-keystroke:          9.74ms total |  0.974us/item
  normalize:            1.34ms total |  0.134us/item
  nn_inference:         8.43ms total |  0.843us/item

--- 100,000 files ---
Static features:      640.96ms total |  6.410us/item
Per-keystroke:        127.35ms total |  1.273us/item
  normalize:           38.83ms total |  0.388us/item
  nn_inference:        95.77ms total |  0.958us/item
```

### Speedup Summary

| Metric | 1K files | 10K files | 100K files |
|--------|----------|-----------|------------|
| NN inference (per item) | 6.28 → 0.84 us (**7.5x**) | 7.41 → 0.84 us (**8.8x**) | 10.39 → 0.96 us (**10.8x**) |
| Per-keystroke (per item) | 7.05 → 0.93 us (**7.6x**) | 9.17 → 0.97 us (**9.4x**) | 11.55 → 1.27 us (**9.1x**) |
| Per-keystroke total | 7.05 → 0.92 ms | 91.66 → 9.74 ms | 1154.95 → 127.35 ms |

Key observations:
- **7.5-10.8x faster** NN inference from batch norm fusion
- Speedup increases with repo size due to reduced GC pressure from zero allocations
- Static feature computation unchanged (not affected by the optimization)
- At 100K files, per-keystroke drops from 1.15s to 127ms — well under interactive thresholds
