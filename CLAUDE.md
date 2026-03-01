# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Neovim file picker plugin (neural-open.nvim) that learns how you navigate. A neural network trains on your file selections to rank results, combining fuzzy matching with contextual signals like buffer state, directory proximity, frecency, and file-to-file transition patterns. Built for Snacks.nvim.

## Development Commands
Use the @justfile commands.

Always make sure all `just precommit` checks pass before checking in code.

When making changes to the scoring hot path (`scorer.lua`, `item_scorer.lua`, `nn.lua` inference, `source.lua` transform, `item_source.lua` transform), run
`just benchmark` before and after to measure per-keystroke latency impact.
Results are documented in `docs/benchmark-results.md`.

## Architecture

### Core Components

- **`plugin/neural-open.lua`**: Neovim plugin file for native lazy loading, commands, `<Plug>` mappings, and autocmds (BufEnter for recency tracking, VimLeavePre for recency flush)
- **`init.lua`**: Main plugin entry point, configuration management, and Snacks.nvim integration
- **`source.lua`**: Snacks picker source implementation with file discovery and async processing
- **`scorer.lua`**: Multi-factor scoring algorithm (fuzzy matching, frecency, proximity, buffers). Owns `FEATURE_NAMES` (the canonical feature ordering for `input_buf`), `input_buf_to_features()` for file picker feature conversion, and `get_item_identity(item)` shared by all algorithms for unified file/item identity
- **`weights.lua`**: Self-learning weight adjustment system that adapts to user preferences
- **`recent.lua`**: File-path-based recency tracking with pending touches and debounced persistence. BufEnter events add paths to an in-memory `pending_touches` list (no disk I/O); a 5s debounce timer or VimLeavePre merges pending touches with the on-disk list and writes back
- **`item_tracking.lua`**: Generic frecency, recency, and transition tracking for non-file picker items, keyed by picker name + item identity. Supports global and CWD-scoped tracking with deadline-based exponential decay (30-day half-life). Includes item-to-item transition frecency (CWD-scoped source tracking). Data persisted under `item_tracking` key in each picker's tracking file (`<picker_name>.tracking.json`). Stateless module: every operation reads from disk, modifies, and writes back immediately (no module-level cache or debounce). Public functions accept an optional pre-loaded `store` parameter to avoid redundant disk reads
- **`frecency.lua`**: Shared deadline-based frecency math. Pure functions with no I/O or mutable state: `deadline_to_score`/`score_to_deadline` conversion, `bump` for increment-and-store, `normalize_transition` for [0,1] mapping, and `prune_map`/`prune_nested` for in-place size-limiting of frecency tables. Used by `transitions.lua`, `item_tracking.lua`, and `debug.lua`
- **`path.lua`**: Shared path normalization utility. Caches `vim.fs.normalize` availability at module load and provides a single `normalize(path)` function used by `source.lua`, `transitions.lua`, and `recent.lua`
- **`item_scorer.lua`**: 8-feature scoring pipeline for non-file item pickers. Owns `ITEM_FEATURE_NAMES` (canonical feature ordering: match, frecency, cwd_frecency, recency, cwd_recency, text_length_inv, not_last_selected, transition), normalization functions, and allocation-free `on_match_handler`. Parallel to `scorer.lua`
- **`item_source.lua`**: Context capture and per-item transform for non-file item pickers. Loads item tracking data, initializes algorithm via `registry.get_algorithm_for_picker`, computes static features. Parallel to `source.lua`
- **`db.lua`**: Per-picker JSON file storage with atomic writes. Each picker uses two files: `<weights_dir>/<picker_name>.json` (model weights) and `<weights_dir>/<picker_name>.tracking.json` (tracking data: recency list, transition frecency, item tracking). Auto-migrates legacy single-file layout on first `get_tracking()` call
- **`types.lua`**: LuaCATS type definitions for the `nos` field structure and other plugin types

### Data Structure

See @lua/neural-open/types.lua for critical types.

IMPORTANT: ensure that types are always up to date. Try to make types non-optional when possible.

The plugin uses a dedicated `nos` field on picker items to encapsulate all neural-open specific data. File pickers use `NosItemData` (11 features), item pickers use `NosItemPickerData` (8 features). See `types.lua` for full definitions.

**File picker `nos` field** (`NosItemData`):
- **input_buf**: Pre-allocated flat array of 11 normalized features. Static features filled at transform time; dynamic features (match, virtual_name, frecency) updated inline per keystroke. Feature order is defined by `FEATURE_NAMES` in `scorer.lua`.
- **raw_features**, **neural_score**, **normalized_path**, **virtual_name**, **is_open_buffer**, **is_alternate**, **recent_rank**, **ctx** (NosContext)

**Item picker `nos` field** (`NosItemPickerData`):
- **input_buf**: Pre-allocated flat array of 8 normalized features. Only match is dynamic (updated per keystroke). Feature order defined by `ITEM_FEATURE_NAMES` in `item_scorer.lua`.
- **raw_features**, **neural_score**, **item_id**, **ctx** (NosItemContext)

This structure provides clean separation between plugin-specific data and native Snacks picker fields. All algorithms use a unified scoring pipeline: raw_features → input_buf (flat pre-allocated buffer with static features pre-normalized at transform time, dynamic features updated inline per keystroke) → `calculate_score(input_buf)` → neural_score. For the NN algorithm, inference uses a pre-computed fused cache (batch norm folded into weights at load time) for zero-allocation scoring. Classic pre-computes a positional weight array at weight-load time for a dot-product hot path. `scorer.normalize_features()` is retained as a utility for debug views and weight learning.

### Scoring System

The plugin uses a multi-factor scoring algorithm that combines:
- Fuzzy path matching (fzf/fzy algorithms)
- Virtual name matching (special handling for index files)
- Open buffer prioritization
- Directory proximity scoring
- Project-relative bonus
- Frecency (frequency + recency with exponential decay)
- Recent file ranking (persistent recency list with linear decay scoring)
- Trigram similarity (character-level similarity to current file)
- File transition frecency (learns common file-to-file navigation patterns with exponential decay)
- Not-current file indicator (binary signal that deprioritizes the currently open file)

### Self-Learning Mechanism

The plugin supports multiple learning algorithms:

#### Classic Algorithm
When users select files that weren't ranked #1, the system:
1. Analyzes scoring component differences between selected and higher-ranked items
2. Adjusts weights using a learning rate (default 0.6)
3. Persists learned weights asynchronously to prevent UI blocking
4. Uses atomic file writes (temp file + rename) for data integrity

No-op when the selected item is already rank #1 (nothing to adjust).

#### Neural Network Algorithm
Trains on every selection, including rank #1. Uses a neural network with pairwise hinge loss to learn file ranking patterns:
- **Architecture**: Multi-layer perceptron with batch normalization and dropout
  - **Activation**: Leaky ReLU (α=0.01) for all hidden layers to prevent dying neurons and improve gradient flow
- **Training**: Pairwise ranking with hinge loss (directly optimizes relative ordering)
  - **Loss Function**: L = max(0, margin - (score_pos - score_neg))
  - **Margin**: Default 1.0 (configurable) - minimum score difference required between positive and negative items
  - **Pairs**: Constructs (selected item, non-selected item) pairs for each user selection
  - **Hard Negatives**: Uses top-10 ranked items as negatives (focuses on most competitive items where fine distinctions matter)
- **Match Dropout**: Randomly drops out match features (and virtual_name for file pickers) during training (default 25%)
  - Applied consistently to both positive and negative items in each pair (same dropout decision)
  - For item pickers, only the match feature is dropped (virtual_name does not exist in the 8-feature pipeline)
  - Forces network to learn from non-search features (frecency, proximity, etc.)
  - Improves ranking before any search query is typed
  - Only applied during training, not during inference
- **Optimizers**: Supports SGD and AdamW optimizers
  - **SGD**: Simple gradient descent with weight decay (L2 regularization)
  - **AdamW**: Adaptive learning rate optimizer with decoupled weight decay
    - More robust to hyperparameter choices
    - Better convergence on diverse datasets
    - Recommended for most users
    - Includes learning rate warmup (10 steps by default) for training stability
- **State Persistence**: Network weights, optimizer state (moments for AdamW), training history, ranking accuracy metrics, and format version
- **Ranking Accuracy Metrics**: Tracks percentage of correctly ranked pairs over time
  - **Correct Ranking**: Percentage of pairs where score_pos > score_neg
  - **Margin-Correct Ranking**: Percentage of pairs where score_pos - score_neg >= margin
  - **Time Windows**: Displayed for last 1, 10, 100, 1000 batches
  - **Display Format**: XX.XX% (YY.YY%) where X is correct%, Y is margin-correct%
  - **Purpose**: Helps understand if network is improving at satisfying the ranking margin
- **Learning Rate**: Default 0.001 for AdamW (adaptive to gradient distribution changes)
- **Warmup**: Optional learning rate warmup to stabilize early training
  - Linearly increases learning rate from 10% to 100% over first N steps
  - Enabled by default for AdamW (10 steps)
  - Helps mitigate bias correction amplification in AdamW
- **Inference Cache**: `prepare_inference_cache()` fuses batch norm parameters into weight matrices once per weight load, so `calculate_score(input_buf)` runs a tight loop with zero table allocations. `calculate_score(input_buf)` is the per-keystroke hot path, taking a pre-allocated flat array (`input_buf`). The cache (`st.inference_cache`) is invalidated and rebuilt whenever weights reload or training updates the network. Inside `nn.lua`, always call `prepare_inference_cache(st)` after modifying `st.weights`, `st.biases`, `st.gammas`, `st.betas`, `st.running_means`, or `st.running_vars` on a state table.
- **Input-Size Migration**: Automatically expands the first layer when new features are added, using Xavier initialization for new rows, resetting first-layer optimizer moments, and backfilling training history using a feature-name-driven defaults table (`MIGRATION_DEFAULTS` in `nn.lua`). Binary `not_*` features default to 1.0; all other new features default to 0.0. Works for any picker type (e.g., file pickers 10→11, item pickers 7→8). Users are notified of the upgrade.

### Data Persistence

- **Weights directory**: Per-picker data in the configured `weights_path` directory (default `~/.local/share/nvim/neural-open/`). Each picker uses two files:
  - `<picker_name>.json` — model weights only (e.g., `files.json`, `just_recipes.json`):
    - Classic algorithm: Feature weights
    - Neural network algorithm: Network weights, biases, batch norm parameters, optimizer state (timestep and moments for AdamW), training history (pairwise format with `normalized_path` for file pickers or `item_id` for item pickers as positive_file/negative_file), ranking accuracy metrics, and format version
  - `<picker_name>.tracking.json` — feature tracking data (e.g., `files.tracking.json`):
    - Transition frecency: Nested map of file-to-file navigation patterns with exponential decay (30-day half-life, deadline-based storage, shared between algorithms)
    - Recency list: Ordered array of recently accessed file paths (default 100 entries). BufEnter adds paths to an in-memory `pending_touches` list; a 5s debounce timer or VimLeavePre merges with the on-disk list and writes back
    - Item tracking (non-file pickers): `item_tracking` key holding global frecency, CWD-scoped frecency, global recency list, CWD-scoped recency lists, and item-to-item transition frecency. Deadline-based exponential decay (30-day half-life). Written immediately to disk on each selection (stateless read-modify-write)
- **Auto-migration**: On first run after upgrading, `weights.json` is automatically renamed to `files.json` with a `weights.json.bak` backup. If `weights_path` is configured as a `.json` file path, a deprecation warning is logged and the parent directory is used. On first `get_tracking()` call, if `<picker>.tracking.json` does not exist, tracking keys (`recency_list`, `transition_frecency`, `item_tracking`) are extracted from `<picker>.json` into the new tracking file and removed from the weight file. Legacy `transition_history` key is also cleaned up during this migration
- **Atomic writes**: Uses temp file + rename pattern to prevent data corruption
- **Async updates**: Weight learning happens in background without blocking UI
- **Format Versioning**: Neural network weights include version field (v2.0-hinge for pairwise format)

### Integration Points

- Registers as `neural_open` source in Snacks.nvim picker system (lazy, on first `open()`)
- Uses Snacks.nvim's async file discovery and database utilities
- Leverages Snacks picker actions and preview system
- Creates `:NeuralOpen` user command with subcommands (`algorithm`, `pick`, `reset`)
- Provides `<Plug>(NeuralOpen)` mapping for easy keybinding
- `setup()` is optional - plugin works with sensible defaults

### Public Picker API

The plugin exposes a generic `pick()` API for creating arbitrary item pickers with neural scoring. Each picker trains independently.

- **`M.register_picker(name, config)`**: Stores a picker config in the registry for later use. `config.type` defaults to `"item"`.
- **`M.pick(name, opts)`**: Opens a picker. Registers inline if not previously registered; merges `opts` over existing registration. Builds and registers a Snacks source (`neural_open_<name>`), then opens it.
  - `type = "item"` (default): Uses 8-feature item scoring pipeline (`item_source.lua` + `item_scorer.lua`). Confirm handler records selection via `item_tracking` and triggers weight learning (NN trains on every selection; classic only trains when rank > 1).
  - `type = "file"`: Uses 11-feature file scoring pipeline (`source.lua` + `scorer.lua`) with per-picker weight isolation. Custom finder replaces the default multi-finder.
- **`M.open()`**: Unchanged — opens the default file picker.
- **`:NeuralOpen pick <name>`**: Opens a registered picker by name.

Configurable `file_sources` (default `{"buffers", "recent", "files", "git_files"}`) controls which Snacks sources the default file picker uses.

## Testing

The project uses Busted for testing with comprehensive test coverage including:
- Scorer algorithm validation with raw features and flat input buffer scoring
- Weight learning system testing using input buffers and scorer utilities
- Database operations and persistence
- Mock-based unit testing for external dependencies
- Type safety validation for the simplified `nos` field structure
- End-to-end multi-picker pipeline validation (item/file transforms, scoring correctness, weight learning isolation, auto-migration)
- Per-picker state isolation regression tests (NN architecture independence, classic weight independence, weight save routing per picker_name)
- Item picker debug preview validation (correct 8-feature names, type-specific sections, file-only sections suppressed, item transition sections displayed)

**Test Isolation**: Tests run in complete isolation using temporary XDG directories to protect your real Neovim environment. Always use `just test` to ensure proper isolation.

### Test Configuration Best Practices

When writing tests, follow DRY principles by reusing configuration from `init.lua` rather than duplicating values:

**Helper Functions** (available in `tests.helpers`):

1. `get_default_config()` - Returns a deep copy of the default configuration
   - Use for: Tests that need to inspect or compare against default values
   - Example: Verifying config merging behavior

2. `create_test_config(overrides)` - Creates full config with selective overrides
   - Use for: Tests modifying top-level config (algorithm, weights_path, debug.preview)
   - Example: Integration tests with multiple config changes

3. `create_algorithm_config(algorithm_name, overrides)` - Creates config with algorithm-specific overrides
   - Use for: Algorithm tests that only need to modify one algorithm's settings (most common)
   - Example: Testing classic algorithm with custom learning rate

**Patterns by Test Type:**

**Pure Function Tests** (no config needed):
```lua
describe("matrix operations", function()
  local nn_core = require("neural-open.algorithms.nn_core")

  it("multiplies matrices", function()
    local result = nn_core.matmul(a, b)
    assert.equals(expected, result)
  end)
end)
```

**Algorithm Tests with Default Config (instance API):**
```lua
local helpers = require("tests.helpers")
local classic = require("neural-open.algorithms.classic")

local config = helpers.create_algorithm_config("classic")
config.algorithm_config.classic.picker_name = "test"
local instance = classic.create_instance(config.algorithm_config.classic)
instance.load_weights()
-- Use instance.calculate_score(), instance.update_weights(), etc.
```

**Algorithm Tests with Overrides (instance API):**
```lua
local helpers = require("tests.helpers")
local nn = require("neural-open.algorithms.nn")

local config = helpers.create_algorithm_config("nn", {
  architecture = { 9, 4, 1 },  -- Smaller network for testing
  batch_size = 4,              -- Faster test execution
})
config.algorithm_config.nn.picker_name = "test"
local instance = nn.create_instance(config.algorithm_config.nn)
instance.load_weights()
-- Use instance.calculate_score(), instance.update_weights(), etc.
```

**Config Merging Tests:**
```lua
describe("setup", function()
  before_each(function()
    package.loaded["neural-open"] = nil  -- Reset for clean test
  end)

  it("merges user config with defaults", function()
    local helpers = require("tests.helpers")
    local neural_open = require("neural-open")
    local defaults = helpers.get_default_config()

    neural_open.setup({ algorithm = "nn" })

    assert.equals("nn", neural_open.config.algorithm)
    assert.equals(defaults.weights_path, neural_open.config.weights_path)
  end)
end)
```

**Integration Tests with Complex Config:**
```lua
local helpers = require("tests.helpers")

-- Create full config with multiple overrides for integration scenario
local config = helpers.create_test_config({
  algorithm = "classic",
  algorithm_config = {
    classic = {
      default_weights = {
        trigram = 50,  -- Emphasize trigram for this test
        match = 100,
      }
    }
  }
})
```

**Mocking Dependencies While Preserving Config:**
```lua
local helpers = require("tests.helpers")
local default_weights = helpers.get_default_config().algorithm_config.classic.default_weights

-- Mock returns real defaults, not hardcoded values
local mock_weights = {
  get_weights = function(algo, _picker_name)
    return vim.deepcopy(default_weights)
  end,
  save_weights = function(_algo, _weights, _latency_ctx, _picker_name) end,
}
package.loaded["neural-open.weights"] = mock_weights
```

**Mocking Weights Module (Simplified):**
```lua
local helpers = require("tests.helpers")

-- Use the helper to create a weights mock
local weights_mock = helpers.create_weights_mock()
package.loaded["neural-open.weights"] = weights_mock.mock

-- Later in test, access saved weights
local saved = weights_mock.get_saved()
assert.is_not_nil(saved)
```

**Guidelines:**

1. Always use `vim.deepcopy()` when getting config to prevent test interference
2. Only override values that differ from defaults for your specific test
3. Use helpers to get current defaults rather than hardcoding values
4. For integration tests, prefer `create_test_config()` for complex scenarios
5. When mocking modules, make mocks return real config values via helpers

## File Structure

- `plugin/neural-open.lua`: Commands, `<Plug>` mappings, and autocmds (native lazy loading)
- `lua/neural-open/`: Core plugin modules
- `tests/`: Busted test specifications
- `benchmarks/`: Hot-path benchmark scripts (run with `just benchmark`)
- `docs/`: Benchmark results and supplementary documentation
- `examples/`: Real-world picker examples (just recipes, make targets, vim commands) — copy-paste templates for creating custom pickers

## Configuration

The plugin is highly configurable with settings for:
- **Scoring algorithms**: Classic (weighted features), Naive (simple matching), Neural Network (ML-based)
- **Neural Network optimizer**: SGD or AdamW
  - SGD: Simple, predictable, requires tuning learning rate
  - AdamW: Adaptive, robust, recommended for most users
- **Matching algorithms**: fzf/fzy for fuzzy matching
- **Learning rate adjustments**: Per-algorithm learning rate configuration
- **Database location and persistence**: Configurable storage directory for per-picker weight files (`weights_path`)
- **Scoring weights and factors**: Customizable feature weights for Classic algorithm
- **Item picker algorithm config**: Separate algorithm configurations for non-file pickers via `item_algorithm_config` (8-feature pipeline: match, frecency, cwd_frecency, recency, cwd_recency, text_length_inv, not_last_selected, transition). Default NN architecture is `{8, 16, 8, 1}`
- **Regularization**: Weight decay, dropout rates, match_dropout, layer-specific decay multipliers for Neural Network
- **Training stability**: Learning rate warmup for Neural Network (recommended for AdamW)
- **Ranking margin**: Configurable margin for pairwise hinge loss (default 1.0) - controls minimum score difference between selected and non-selected items
- **Transition frecency**: File-to-file navigation patterns tracked with exponential decay (30-day half-life), automatically pruned
- **Recency list**: Persistent ordered list of recently accessed files, updated on BufEnter with debounced disk writes (default 100 entries, configurable via `recency_list_size`)
- **Performance debugging**: Latency tracking to diagnose performance issues (disabled by default for zero overhead)

### Debug Configuration

All debug settings are consolidated under the `debug` table:

```lua
require("neural-open").setup({
  debug = {
    preview = false,              -- Show detailed score breakdown in preview
    latency = false,              -- Enable detailed latency tracking
    latency_file = nil,           -- Optional path for persistent logging (e.g., "/tmp/neuralopen-latency.log")
    latency_threshold_ms = 100,   -- Only log operations exceeding this duration (default 100ms)
    latency_auto_clipboard = false, -- Auto-copy timing reports to clipboard
    snacks_scores = false,         -- Show Snacks.nvim debug scores in picker
  }
})
```

### Performance Debugging with debug.latency

The plugin includes a comprehensive latency tracking system to diagnose performance issues:

**Usage Example**:
```lua
require("neural-open").setup({
  debug = {
    latency = true,
    latency_threshold_ms = 100,  -- Only log slow operations
  }
})
```

**View Last Timing Report**:
```vim
:lua require("neural-open.latency").show_last()
```

**Architecture**: The latency module uses a context-based threading model that is safe across `vim.schedule` boundaries and has zero overhead when disabled (function replacement pattern eliminates all branching).

**Output Format**: Hierarchical tree showing operation nesting, timing, data sizes, and actionable suggestions:
```
File: src/components/Header.tsx
=== NeuralOpen Latency Report ===
Total: 150.30ms
Status: ⚠️ SLOW

async.weight_update      150.30ms
  ├─ nn.pair_construction  5.20ms [pairs_created=10, selected_rank=5]
  ├─ nn.batch_construction  2.10ms [num_batches=3, history_size=45]
  ├─ nn.training  100.30ms [num_batches=3, total_pairs=60, avg_forward_ms=25.2, avg_backward_ms=50.1, avg_update_ms=15.3, avg_loss=0.234, optimizer=adamw, core_matmul_count=240, core_matmul_ops=2400000, memory_delta_kb=512]
  │   ├─ core.matmul  45.20ms
  │   ├─ core.element_wise  30.10ms
  │   └─ core.batch_normalize  5.30ms
  └─ nn.save_weights  42.70ms

Suggestions:
  - Slow training (100.30ms). Backward pass is bottleneck (avg 50.1ms) - consider reducing network depth/width.
```

**Performance Characteristics**:
- When disabled: <0.001ms overhead per operation (zero-cost abstraction)
- When enabled: ~0.01ms overhead per operation
- Memory: ~1KB per timing context (10-20 operations)
- Thread-safe: Each file selection gets isolated timing context
