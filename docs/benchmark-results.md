# Picker Benchmark Results

Benchmarks measure the per-keystroke hot path: feature normalization and NN inference
across 1K, 10K, and 100K file repositories with realistic usage data (full recency
list, transition history, trigrams, open buffers).

Run with `just benchmark`. See `benchmarks/picker_benchmark.lua` for methodology.

## Baseline (69d7b15)

Initial implementation before any performance optimizations.

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

## Batch Norm Fusion (548ca15)

Fuses batch normalization parameters into weight matrices at load time so
`calculate_score()` runs a tight scalar loop with zero table allocations,
replacing the general-purpose `forward_pass()` at inference.

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

| Metric | 1K files | 10K files | 100K files |
|--------|----------|-----------|------------|
| NN inference (per item) | 6.28 → 0.84 us (**7.5x**) | 7.41 → 0.84 us (**8.8x**) | 10.39 → 0.96 us (**10.8x**) |
| Per-keystroke (per item) | 7.05 → 0.93 us (**7.6x**) | 9.17 → 0.97 us (**9.4x**) | 11.55 → 1.27 us (**9.1x**) |

- **7.5-10.8x faster** NN inference from batch norm fusion
- Speedup increases with repo size due to reduced GC pressure from zero allocations
- Static feature computation unchanged (not affected by the optimization)
- At 100K files, per-keystroke drops from 1.15s to 127ms — well under interactive thresholds

## Static Feature Optimizations

Eliminates redundant per-item work in the static feature computation hot path
(transform phase):

- Hoist `require()` to module level (eliminates per-item `package.loaded` lookup)
- Remove redundant `normalize_path()` calls (paths already normalized upstream)
- Replace `string.sub` project check with `string.find` (avoids string allocation)
- Replace `vim.fn.fnamemodify` with pure Lua string matching (eliminates Vimscript bridge)
- Precompute current file directory and depth once per session (eliminates repeated `get_directory()` + `vim.split()`)
- Rewrite `calculate_proximity()` with zero-allocation byte scanning (eliminates per-item `vim.split()` table allocations)

```
--- 1,000 files ---
Static features:        0.66ms total |  0.658us/item
Per-keystroke:          0.97ms total |  0.967us/item
  normalize:            0.03ms total |  0.033us/item
  nn_inference:         0.86ms total |  0.857us/item

--- 10,000 files ---
Static features:        8.19ms total |  0.819us/item
Per-keystroke:          9.90ms total |  0.990us/item
  normalize:            0.40ms total |  0.040us/item
  nn_inference:         9.02ms total |  0.902us/item

--- 100,000 files ---
Static features:      176.94ms total |  1.769us/item
Per-keystroke:        111.96ms total |  1.120us/item
  normalize:           11.27ms total |  0.113us/item
  nn_inference:        90.74ms total |  0.907us/item
```

| Metric | 1K files | 10K files | 100K files |
|--------|----------|-----------|------------|
| Static features (per item) | 3.37 → 0.66 us (**5.1x**) | 4.10 → 0.82 us (**5.0x**) | 6.41 → 1.77 us (**3.6x**) |
| Static features total | 3.37 → 0.66 ms | 41.00 → 8.19 ms | 640.96 → 176.94 ms |

- **3.6-5.1x faster** static feature computation from eliminating redundant work
- Per-keystroke path unchanged (optimizations target the one-time transform phase)
- At 100K files, static features drop from 641ms to 177ms — a 464ms reduction in picker open time
- Largest gains from eliminating `vim.split()` table allocations and `normalize_path()` string operations

## Cumulative Impact

End-to-end improvement from baseline to current state at 100K files:

| Metric | Baseline (69d7b15) | Current | Speedup |
|--------|-------------------|---------|---------|
| Static features | 632ms (6.32 us/item) | 177ms (1.77 us/item) | **3.6x** |
| Per-keystroke | 1155ms (11.55 us/item) | 112ms (1.12 us/item) | **10.3x** |
| NN inference | 1039ms (10.39 us/item) | 91ms (0.91 us/item) | **11.4x** |
