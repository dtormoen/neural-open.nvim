---@meta

--- Available algorithm names
---@alias AlgorithmName "classic" | "naive" | "nn"

--- Base algorithm interface for scoring algorithms.
--- All scoring algorithms should implement these methods.
---@class Algorithm
---@field calculate_score fun(normalized_features: table<string, number>): number Calculates the total score from normalized features using the algorithm's approach
---@field update_weights fun(selected_item: NeuralOpenItem, ranked_items: NeuralOpenItem[], latency_ctx?: table): nil Updates internal weights based on user selection for learning algorithms. Called when user selects an item that wasn't ranked #1. Optional latency_ctx for performance tracking
---@field debug_view fun(item: NeuralOpenItem, all_items?: NeuralOpenItem[]): string[] Generates debug information specific to this algorithm
---@field get_name fun(): AlgorithmName Returns the display name of the algorithm
---@field init fun(config: table): nil Initialize the algorithm with configuration
---@field load_weights fun(): nil Load the latest weights from the weights module

---@class NosRawFeatures
---@field match number Raw fuzzy match score from matcher (typically 0-200+)
---@field virtual_name number Raw virtual name match score (typically 0-200+)
---@field frecency number Raw frecency value from database (0-∞)
---@field open number Binary: 1 if open buffer, 0 otherwise
---@field alt number Binary: 1 if alternate buffer, 0 otherwise
---@field proximity number Raw proximity score (0-1 based on directory depth)
---@field project number Binary: 1 if in project/cwd, 0 otherwise
---@field recency number Raw recency rank (1-based position in persistent recency list)
---@field trigram number Raw trigram similarity score (0-1 Dice coefficient)
---@field transition number Raw transition score (0-1 based on transition count)

---@class NosNormalizedFeatures
---@field match number Normalized to [0,1] using sigmoid
---@field virtual_name number Normalized to [0,1] using sigmoid
---@field frecency number Normalized to [0,1] using 1 - 1/(1+x)
---@field open number Already [0,1] binary
---@field alt number Already [0,1] binary
---@field proximity number Already [0,1] from calculation
---@field project number Already [0,1] binary
---@field recency number Normalized to [0,1] using (max - rank + 1) / max (linear decay)
---@field trigram number Already [0,1] from Dice coefficient
---@field transition number Already [0,1] from calculation (1-1/(1+count))

---@class NosContext
---@field recent_files table<string, {recent_rank: number}> Recent files mapping
---@field alternate_buf string? Path to alternate buffer file
---@field cwd string Current working directory
---@field current_file string Current file path
---@field current_file_trigrams table<string, boolean>? Trigrams of current file
---@field algorithm Algorithm The scoring algorithm instance for this session
---@field transition_scores table<string, number>? Precomputed transition scores map

---@class NosItemData
---@field raw_features NosRawFeatures Raw computed feature scores
---@field normalized_features NosNormalizedFeatures Normalized [0,1] features
---@field neural_score number Total weighted score
---@field virtual_name string? Cached virtual name for special files
---@field normalized_path string Cached normalized absolute path
---@field is_open_buffer boolean File is open in a buffer
---@field is_alternate boolean File is the alternate buffer
---@field recent_rank number? Position in recent files (1-based)
---@field ctx NosContext Reference to shared session context

---@class NeuralOpenItem: snacks.picker.Item
---@field file string Original file path
---@field nos NosItemData Neural-open specific data
---@field buf? number Buffer number if open
---@field cwd? string Current working directory
---@field neural_rank? number Used during weight learning

---@class NosClassicConfig
---@field learning_rate number Learning rate for weight adjustments (0-1)
---@field default_weights table<string, number> Default weights for scoring features

---@class NosNaiveConfig
--- No configuration needed for naive algorithm

---@alias OptimizerType "sgd" | "adamw"

---@class NosNNConfig
---@field architecture number[] Neural network layer sizes [input, hidden1, ..., output]
---@field learning_rate number Learning rate for gradient descent (0-1)
---@field batch_size number Number of samples per training batch
---@field history_size number Maximum number of historical selections to store
---@field batches_per_update number Number of batches before weight update
---@field weight_decay number L2 regularization coefficient to prevent overfitting (default 0.0001)
---@field layer_decay_multipliers? number[] Optional per-layer decay rate multipliers
---@field dropout_rates? number[] Dropout rates for each hidden layer (0-1, not applied to output layer)
---@field optimizer? OptimizerType Optimizer algorithm (default "sgd")
---@field adam_beta1? number AdamW first moment decay (default 0.9)
---@field adam_beta2? number AdamW second moment decay (default 0.999)
---@field adam_epsilon? number AdamW numerical stability (default 1e-8)
---@field warmup_steps? number Number of steps to warm up learning rate (default 0, disabled)
---@field warmup_start_factor? number Starting learning rate factor for warmup (default 0.1)
---@field match_dropout? number Dropout rate for match/virtual_name features during training (default 0.25)
---@field margin? number Margin for pairwise hinge loss (default 1.0)

---@class NosAlgorithmConfig
---@field classic NosClassicConfig Configuration for classic algorithm
---@field naive NosNaiveConfig Configuration for naive algorithm
---@field nn NosNNConfig Configuration for neural network algorithm

---@class NosDebugConfig
---@field preview boolean Show detailed score breakdown in preview
---@field latency boolean Log detailed latency metrics for performance debugging
---@field latency_file string? Optional file path for persistent latency logging
---@field latency_threshold_ms number Only log operations exceeding this duration (default 100)
---@field latency_auto_clipboard boolean Copy timing report to clipboard
---@field snacks_scores boolean Show Snacks.nvim debug scores in picker

---@class NosConfig
---@field algorithm AlgorithmName Active algorithm: "classic" | "naive" | "nn"
---@field algorithm_config NosAlgorithmConfig Algorithm-specific configurations
---@field weights_path string Path to store learned weights
---@field special_files table<string, boolean> Special files requiring virtual name handling
---@field transition_history_size number Ring buffer size for transition history (default 200)
---@field recency_list_size number Maximum number of files in persistent recency list (default 100)
---@field debug NosDebugConfig Debug settings
