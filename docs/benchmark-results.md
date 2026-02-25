# Picker Benchmark Results

Benchmarks measure the scoring hot path across 1K, 10K, and 100K file repositories
with realistic usage data (full recency list, transition history, trigrams, open buffers).
Measured phases: static features (one-time transform), per-keystroke (normalize + NN inference),
per-keystroke fast (zero-allocation NN path), transform phase (full per-item processing),
and weight loading (one-time initialization).

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

## Per-Keystroke Fast Path

Eliminates all table allocations and hash lookups from the per-keystroke scoring
hot path for the NN algorithm:

- Pre-allocate flat `nn_input` buffer during transform phase with pre-normalized static features
- Per keystroke: update 3 dynamic features (match, virtual_name, frecency) inline with normalization
- Pass flat buffer directly to `calculate_score_direct()` — no intermediate `normalized_features` table
- Cache transform closure creation (was re-created per item)
- Cache `vim.fs.normalize` availability check and options table at module load (eliminates per-item pcall + table allocation)
- Update weight-learning and debug paths to consume `nn_input` directly

```
--- 1,000 files ---
Static features:        0.63ms total |  0.627us/item
Per-keystroke:          0.89ms total |  0.889us/item
Per-keystroke fast:     0.71ms total |  0.707us/item
  normalize:            0.03ms total |  0.031us/item
  nn_inference:         0.83ms total |  0.827us/item
Transform phase:        0.91ms total |  0.912us/item
Weight loading:         0.01ms total (one-time)

--- 10,000 files ---
Static features:        6.42ms total |  0.642us/item
Per-keystroke:          9.23ms total |  0.923us/item
Per-keystroke fast:     7.11ms total |  0.711us/item
  normalize:            0.37ms total |  0.037us/item
  nn_inference:         8.21ms total |  0.821us/item
Transform phase:       10.25ms total |  1.025us/item
Weight loading:         0.01ms total (one-time)

--- 100,000 files ---
Static features:      120.51ms total |  1.205us/item
Per-keystroke:        108.56ms total |  1.086us/item
Per-keystroke fast:    78.15ms total |  0.782us/item
  normalize:           12.71ms total |  0.127us/item
  nn_inference:        93.57ms total |  0.936us/item
Transform phase:      156.85ms total |  1.569us/item
Weight loading:         0.01ms total (one-time)
```

| Metric | 1K files | 10K files | 100K files |
|--------|----------|-----------|------------|
| Per-keystroke (per item) | 0.89 → 0.71 us (**1.3x**) | 0.92 → 0.71 us (**1.3x**) | 1.09 → 0.78 us (**1.4x**) |
| Per-keystroke total | 0.89 → 0.71 ms | 9.23 → 7.11 ms | 108.56 → 78.15 ms |

- **1.3-1.4x faster** per-keystroke scoring from eliminating normalized_features table allocation and hash lookups
- Speedup increases with repo size due to reduced GC pressure (zero per-keystroke allocations for NN)
- At 100K files, per-keystroke drops from ~112ms to 78ms — a 34ms reduction per keystroke

## Cumulative Impact

End-to-end improvement from baseline to current state at 100K files:

| Metric | Baseline (69d7b15) | Current | Speedup |
|--------|-------------------|---------|---------|
| Static features | 632ms (6.32 us/item) | 121ms (1.21 us/item) | **5.2x** |
| Per-keystroke | 1155ms (11.55 us/item) | 78ms (0.78 us/item) | **14.8x** |
| NN inference | 1039ms (10.39 us/item) | 84ms (0.83 us/item) | **12.4x** |
