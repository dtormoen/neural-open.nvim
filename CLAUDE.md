# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Neovim file picker plugin (neural-open.nvim) that learns how you navigate. A neural network trains on your file selections to rank results, combining fuzzy matching with contextual signals like buffer state, directory proximity, frecency, and file-to-file transition patterns. Built for Snacks.nvim.

## Development Commands
Use the @justfile commands.

Always make sure all `just precommit` checks pass before checking in code.

When making changes to the scoring hot path (`scorer.lua`, `nn.lua` inference), run
`just benchmark` before and after to measure per-keystroke latency impact.
Results are documented in `docs/benchmark-results.md`.

## Architecture

### Core Components

- **`plugin/neural-open.lua`**: Neovim plugin file for native lazy loading, commands, `<Plug>` mappings, and autocmds (BufEnter for recency tracking, VimLeavePre for recency flush)
- **`init.lua`**: Main plugin entry point, configuration management, and Snacks.nvim integration
- **`source.lua`**: Snacks picker source implementation with file discovery and async processing
- **`scorer.lua`**: Multi-factor scoring algorithm (fuzzy matching, frecency, proximity, buffers)
- **`weights.lua`**: Self-learning weight adjustment system that adapts to user preferences
- **`recent.lua`**: Persistent recency tracking with in-memory cache and debounced disk writes
- **`db.lua`**: JSON file storage with atomic writes for persistent weight storage
- **`types.lua`**: LuaCATS type definitions for the `nos` field structure and other plugin types

### Data Structure

See @lua/neural-open/types.lua for critical types.

IMPORTANT: ensure that types are always up to date. Try to make types non-optional when possible.

The plugin uses a dedicated `nos` field on picker items to encapsulate all neural-open specific data:
- **raw_features**: Original computed feature scores in their native units
- **normalized_features**: Features normalized to [0,1] range for consistent weighting
- **neural_score**: Total weighted score combining all features
- **normalized_path**: Cached normalized absolute path for consistent file comparison
- **is_open_buffer**, **is_alternate**: Buffer state flags
- **recent_rank**: Position in persistent recency list (1-based)
- **virtual_name**: Cached virtual name for special files (e.g., index.js -> parent/index.js)
- **ctx**: Reference to shared session context (contains cwd, current_file, current_file_trigrams, recent_files, alternate_buf)

This structure provides clean separation between plugin-specific data and native Snacks picker fields. The scoring pipeline follows: raw_features → normalized_features → (apply weights) → neural_score. For the NN algorithm, inference uses a pre-computed fused cache (batch norm folded into weights at load time) for zero-allocation scoring. Component scores are calculated on-the-fly from normalized features and weights when needed using `weights.calculate_components()`.

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
- File transition tracking (learns common file-to-file navigation patterns)

### Self-Learning Mechanism

The plugin supports multiple learning algorithms:

#### Classic Algorithm
When users select files that weren't ranked #1, the system:
1. Analyzes scoring component differences between selected and higher-ranked items
2. Adjusts weights using a learning rate (default 0.6)
3. Persists learned weights asynchronously to prevent UI blocking
4. Uses atomic file writes (temp file + rename) for data integrity

#### Neural Network Algorithm
Uses a neural network with pairwise hinge loss to learn file ranking patterns:
- **Architecture**: Multi-layer perceptron with batch normalization and dropout
  - **Activation**: Leaky ReLU (α=0.01) for all hidden layers to prevent dying neurons and improve gradient flow
- **Training**: Pairwise ranking with hinge loss (directly optimizes relative ordering)
  - **Loss Function**: L = max(0, margin - (score_pos - score_neg))
  - **Margin**: Default 1.0 (configurable) - minimum score difference required between positive and negative items
  - **Pairs**: Constructs (selected item, non-selected item) pairs for each user selection
  - **Hard Negatives**: Uses top-10 ranked items as negatives (focuses on most competitive items where fine distinctions matter)
- **Match Dropout**: Randomly drops out match and virtual_name features during training (default 25%)
  - Applied consistently to both positive and negative items in each pair (same dropout decision)
  - Forces network to learn from non-search features (frecency, proximity, etc.)
  - Improves file ranking before any search query is typed
  - Only applied during training, not during inference
- **Optimizers**: Supports SGD and AdamW optimizers
  - **SGD**: Simple gradient descent with weight decay (L2 regularization)
  - **AdamW**: Adaptive learning rate optimizer with decoupled weight decay
    - More robust to hyperparameter choices
    - Better convergence on diverse datasets
    - Recommended for most users
    - Includes learning rate warmup (100 steps by default) for training stability
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
  - Enabled by default for AdamW (100 steps)
  - Helps mitigate bias correction amplification in AdamW
- **Inference Cache**: `prepare_inference_cache()` fuses batch norm parameters into weight matrices once per weight load, so `calculate_score()` runs a tight loop with zero table allocations. The cache (`state.inference_cache`) is invalidated and rebuilt whenever weights reload or training updates the network. Always call `prepare_inference_cache()` after modifying `state.weights`, `state.biases`, `state.gammas`, `state.betas`, `state.running_means`, or `state.running_vars`.
- **Migration**: Automatically migrates from older BCE-based format (v1.0) to pairwise hinge loss format (v2.0)
  - Preserves network weights while resetting optimizer state and training history
  - Users are notified of the upgrade

### Data Persistence

- **Weights file**: JSON file storing learned weight adjustments per algorithm
  - Classic algorithm: Feature weights
  - Neural network algorithm: Network weights, biases, batch norm parameters, optimizer state (timestep and moments for AdamW), training history (pairwise format), and format version
  - Transition history: Ring buffer of file-to-file navigation patterns (shared between algorithms, default 200 entries)
  - Recency list: Ordered array of recently accessed file paths (default 100 entries), updated on BufEnter with debounced persistence
- **Atomic writes**: Uses temp file + rename pattern to prevent data corruption
- **Async updates**: Weight learning happens in background without blocking UI
- **Format Versioning**: Neural network weights include version field (v2.0-hinge for pairwise format)
- **Automatic Migration**: When loading older BCE-based weights (v1.0), automatically migrates to pairwise hinge loss format (v2.0)
  - Preserves network weights (all learned features retained)
  - Resets optimizer state (gradient statistics become invalid with new loss function)
  - Clears training history (old format incompatible with pairwise structure)
  - Provides user notification explaining the upgrade

### Integration Points

- Registers as `neural_open` source in Snacks.nvim picker system (lazy, on first `open()`)
- Uses Snacks.nvim's async file discovery and database utilities
- Leverages Snacks picker actions and preview system
- Creates `:NeuralOpen` user command with subcommands (`algorithm`, `reset`)
- Provides `<Plug>(NeuralOpen)` mapping for easy keybinding
- `setup()` is optional - plugin works with sensible defaults

## Testing

The project uses Busted for testing with comprehensive test coverage including:
- Scorer algorithm validation with raw and normalized feature separation
- Weight learning system testing using normalized features directly (no stored components)
- Database operations and persistence
- Mock-based unit testing for external dependencies
- Type safety validation for the simplified `nos` field structure

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

**Algorithm Tests with Default Config:**
```lua
local helpers = require("tests.helpers")
local classic = require("neural-open.algorithms.classic")

local config = helpers.get_default_config()
classic.init(config.algorithm_config.classic)
classic.load_weights()
```

**Algorithm Tests with Overrides:**
```lua
local helpers = require("tests.helpers")

local config = helpers.create_algorithm_config("nn", {
  architecture = { 9, 4, 1 },  -- Smaller network for testing
  batch_size = 4,              -- Faster test execution
})
nn.init(config.algorithm_config.nn)
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
  get_weights = function(algo)
    return vim.deepcopy(default_weights)
  end,
  save_weights = function() end,
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

## Configuration

The plugin is highly configurable with settings for:
- **Scoring algorithms**: Classic (weighted features), Naive (simple matching), Neural Network (ML-based)
- **Neural Network optimizer**: SGD or AdamW
  - SGD: Simple, predictable, requires tuning learning rate
  - AdamW: Adaptive, robust, recommended for most users
- **Matching algorithms**: fzf/fzy for fuzzy matching
- **Learning rate adjustments**: Per-algorithm learning rate configuration
- **Database location and persistence**: Configurable storage path for weights
- **Scoring weights and factors**: Customizable feature weights for Classic algorithm
- **File ignore patterns**: Exclude files from picker
- **Performance limits**: max_results to control picker size
- **Regularization**: Weight decay, dropout rates, match_dropout, layer-specific decay multipliers for Neural Network
- **Training stability**: Learning rate warmup for Neural Network (recommended for AdamW)
- **Ranking margin**: Configurable margin for pairwise hinge loss (default 1.0) - controls minimum score difference between selected and non-selected items
- **Transition history**: Configurable ring buffer size for tracking file-to-file navigation patterns (default 200)
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
