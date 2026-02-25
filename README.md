# Neural Open for Snacks.nvim

A Neovim file picker that learns how you navigate. A neural network trains on your file selections to rank results by what you're most likely to open next, combining fuzzy matching with contextual signals like buffer state, directory proximity, frecency, and file-to-file transition patterns.

Inspired by [smart-open.nvim](https://github.com/danielfalk/smart-open.nvim), built for [Snacks.nvim](https://github.com/folke/snacks.nvim).

## Features

- **Neural Network Ranking**: An MLP trains online from your selections using pairwise hinge loss, learning to rank files by relative preference
- **Pre-trained Defaults**: Ships with default weights trained on over 16k samples so ranking is useful from the first launch
- **10 Scoring Features**: Fuzzy match, virtual name, open/alternate buffer, directory proximity, project scope, frecency, recency, trigram similarity, and file transition history
- **Self-Learning**: Adapts to your navigation patterns over time and persists learned weights across sessions
- **Score Preview**: Enable `debug.preview` to watch the model's score breakdowns and training in real time
- **Multiple Algorithms**: Neural network (default), classic weighted sum based on smart-open.nvim, or naive baseline
- **Fast Performance**: Leverages Snacks.nvim's async file discovery and picker infrastructure

## Requirements

- Neovim >= 0.11.0
- [Snacks.nvim](https://github.com/folke/snacks.nvim) installed and configured

## Installation

Using [lazy.nvim](https://github.com/folke/lazy.nvim):

```lua
{
  "dtormoen/neural-open.nvim",
  dependencies = {
    "folke/snacks.nvim",
  },
  keys = {
    { "<leader><leader>", "<Plug>(NeuralOpen)", desc = "Neural Open Files" },
  },
  -- opts are optional. NeuralOpen will automatically use the defaults below.
  opts = {
    algorithm = "nn",
  },
}
```

## Configuration

```lua
require("neural-open").setup({
  -- Scoring algorithm: "nn" (neural network), "classic" (weighted features), or "naive"
  algorithm = "nn",

  -- Algorithm-specific configurations
  algorithm_config = {
    -- Neural network algorithm settings (default)
    nn = {
      architecture = { 10, 16, 16, 8, 1 }, -- Input → Hidden1 → Hidden2 → Hidden3 → Output
      optimizer = "adamw",                 -- "sgd" or "adamw"
      learning_rate = 0.001,               -- Learning rate for gradient descent
      batch_size = 128,                    -- Number of samples per training batch
      history_size = 2000,                 -- Maximum stored historical selections
      batches_per_update = 5,              -- Number of batches per weight update
      weight_decay = 0.0001,               -- L2 regularization to prevent overfitting
      layer_decay_multipliers = nil,       -- Optional per-layer decay rates
      dropout_rates = { 0, 0.25, 0.25 },   -- Dropout rates for hidden layers
      warmup_steps = 10,                   -- Learning rate warmup steps (recommended for AdamW)
      warmup_start_factor = 0.1,           -- Start at 10% of learning rate
      adam_beta1 = 0.9,                    -- AdamW first moment decay
      adam_beta2 = 0.999,                  -- AdamW second moment decay
      adam_epsilon = 1e-8,                 -- AdamW numerical stability
      match_dropout = 0.25,                -- Dropout rate for match/virtual_name during training
      margin = 1.0,                        -- Margin for pairwise hinge loss
    },
    -- Classic algorithm settings (weighted feature scoring)
    classic = {
      learning_rate = 0.6,  -- Learning rate for weight adjustments (0.0 to 1.0)
      default_weights = {
        match = 140,        -- Snacks fuzzy matching
        virtual_name = 131, -- Virtual name matching
        open = 3,           -- Open buffer bonus
        alt = 4,            -- Alternate buffer bonus
        proximity = 13,     -- Directory proximity
        project = 10,       -- Project (cwd) bonus
        frecency = 17,      -- Frecency score
        recency = 9,        -- Recency score
        trigram = 10,       -- Trigram similarity
        transition = 5,     -- File transition tracking
      },
    },
    naive = {
      -- No configuration needed
    },
  },

  -- Path to JSON file storing learned weights
  weights_path = vim.fn.stdpath("data") .. "/neural-open/weights.json",

  -- Ring buffer size for file transition history
  transition_history_size = 200,

  -- Maximum number of files in persistent recency list
  recency_list_size = 100,

  -- Debug settings
  debug = {
    preview = false,                -- Show detailed score breakdown in preview
    snacks_scores = false,          -- Show Snacks.nvim debug scores in picker
    latency = false,                -- Enable detailed latency tracking
    latency_file = nil,             -- Optional file path for persistent latency logging
    latency_threshold_ms = 100,     -- Only log operations exceeding this duration
    latency_auto_clipboard = false, -- Copy timing report to clipboard
  },

  -- Special files that include parent directory in virtual name
  special_files = {
    ["__init__.py"] = true,
    ["index.js"] = true,
    ["index.jsx"] = true,
    ["index.ts"] = true,
    ["index.tsx"] = true,
    ["init.lua"] = true,
    ["init.vim"] = true,
    ["mod.rs"] = true,
  },
})
```

## Usage

### Open Neural Picker

```lua
-- Via command
:NeuralOpen

-- Via Lua
require("neural-open").open()

-- With custom options
require("neural-open").open({
  cwd = "/path/to/project",
  prompt = "Neural Open> ",
})

-- Using Snacks.nvim directly
require("snacks").picker.pick("neural_open")
```

### Switch Algorithm

Change the scoring algorithm at runtime:

```vim
" Show current algorithm
:NeuralOpen algorithm

" Switch to neural network algorithm (default)
:NeuralOpen algorithm nn

" Switch to classic weighted algorithm
:NeuralOpen algorithm classic

" Switch to naive algorithm
:NeuralOpen algorithm naive
```

### Reset Learned Weights

If the learned weights aren't producing good results, reset them:

```lua
require("neural-open").reset_weights()
```

Or via command:

```vim
" Reset current algorithm's weights
:NeuralOpen reset

" Reset specific algorithm's weights
:NeuralOpen reset nn
:NeuralOpen reset classic
```

## How It Works

For each file in the picker, the plugin computes a set of normalized features capturing context like fuzzy match quality, buffer state, directory proximity, and usage history. These features are fed into one of three scoring algorithms to produce a final ranking. All algorithms learn from your file selections and persist their parameters to disk.

### Scoring Features

Each file receives a score based on 10 features, all normalized to [0,1]:

1. **Match**: Fuzzy path matching score from Snacks.nvim's matcher, normalized with a sigmoid
2. **Virtual Name**: Fuzzy match against a virtual name that includes the parent directory for index/init files (e.g., `components/index.js` matches "components"), normalized with a sigmoid
3. **Open**: Binary bonus for files currently open in a buffer
4. **Alternate**: Binary bonus for the alternate buffer (`#` file)
5. **Proximity**: Ratio of shared directory depth between the current file and the candidate, where same-directory files score 1.0
6. **Project**: Binary bonus for files under the current working directory
7. **Frecency**: Frequency + recency score from Snacks.nvim's built-in frecency tracking, normalized with `1 - 1/(1 + x/8)`
8. **Recency**: Position in a persistent most-recently-accessed list (updated on BufEnter), scored with linear decay: `(max - rank + 1) / max`
9. **Trigram**: Character-level similarity between the candidate's virtual name and the current filename using Dice coefficient over 3-character trigram sets
10. **Transition**: Learned file-to-file navigation frequency from a ring buffer of past selections, scored with `1 - 1/(1 + count)`

### Neural Network Algorithm (default)

A multi-layer perceptron that takes the 10 normalized features as input and outputs a ranking score. Trained online using pairwise hinge loss: when you select a file, the network learns from (selected, non-selected) pairs constructed from the top-ranked candidates. Uses batch normalization and Leaky ReLU activations during training; at inference time, batch normalization is fused into the weight matrices so scoring runs with zero allocations per keystroke. Match/virtual_name features are randomly dropped during training to force the network to learn from contextual features (frecency, proximity, etc.), improving ranking before any query is typed. Supports AdamW (default) and SGD optimizers with optional learning rate warmup.

### Classic Algorithm

Based on [smart-open.nvim](https://github.com/danielfalk/smart-open.nvim)'s ranking approach, adapted for Snacks.nvim, and extended with trigram and transition features. Computes a weighted sum of the normalized features. When you select a file that wasn't ranked #1, the algorithm compares feature values between your selection and higher-ranked items and adjusts the weights using a configurable learning rate.

### Naive Algorithm

Simple unweighted sum of all normalized features. No learning. Useful for testing and as a baseline.

## API Reference

### Functions

- `setup(opts)` - Initialize the plugin with configuration
- `open(opts)` - Open the neural picker
- `reset_weights(algorithm_name?)` - Reset learned weights to defaults (optional algorithm name)
- `set_algorithm(name?)` - Set or display current algorithm ("classic", "naive", "nn")

### Mappings

- `<Plug>(NeuralOpen)` - Open the neural picker

### Commands

- `:NeuralOpen` - Open the neural picker
- `:NeuralOpen algorithm [name]` - Show or set scoring algorithm
- `:NeuralOpen reset [algorithm]` - Reset weights for current or specified algorithm

### Picker Source

The plugin registers as `neural_open` source in Snacks.nvim:

```lua
require("snacks").picker.sources.neural_open
```

## Development

### Testing

```bash
# Install dependencies
just setup

# Run tests (isolated from your real Neovim environment)
just test
```

Tests run in complete isolation using temporary XDG directories to protect your real configuration.

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## License

MIT License - See LICENSE file for details

## Acknowledgements

- [smart-open.nvim](https://github.com/danielfalk/smart-open.nvim) - Original implementation and reference for many scoring features and the classic algorithm.
- [Snacks.nvim](https://github.com/folke/snacks.nvim) - Modern and extensible fuzzy-finder.
