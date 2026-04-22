---@meta

--- Available algorithm names
---@alias AlgorithmName "classic" | "naive" | "nn"

--- Base algorithm interface for scoring algorithms.
--- All scoring algorithms should implement these methods.
---@class Algorithm
---@field calculate_score fun(input_buf: number[]): number Calculates the total score from a flat input buffer of normalized features
---@field update_weights fun(selected_item: NeuralOpenItem, ranked_items: NeuralOpenItem[], latency_ctx?: table): nil Updates internal weights based on user selection for learning algorithms. Called when user selects an item that wasn't ranked #1. Optional latency_ctx for performance tracking
---@field debug_view fun(item: NeuralOpenItem, all_items?: NeuralOpenItem[]): string[], table[]? Generates debug information specific to this algorithm. Optional second return value is highlight specs {row, col, end_col, group}
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
---@field transition number Raw transition frecency score (exponentially-decayed visit score, 0-∞)
---@field not_current number Binary: 1 if NOT the current buffer file, 0 if it is

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
---@field transition number Already [0,1] from calculation (1-1/(1+score/4))
---@field not_current number Already [0,1] binary

---@class NosContext
---@field recent_files table<string, {recent_rank: number}> Recent files mapping
---@field alternate_buf string? Path to alternate buffer file
---@field cwd string Current working directory
---@field current_file string Current file path
---@field current_file_dir string? Directory portion of current_file (up to and including last /)
---@field current_file_depth number Number of /-separated segments in current_file_dir (excluding leading /)
---@field current_file_trigrams table<number, boolean>? Trigrams of current file (packed integer keys: b1*65536 + b2*256 + b3)
---@field current_file_trigrams_size number Number of unique trigrams in current file (for dice_coefficient_direct)
---@field algorithm Algorithm The scoring algorithm instance for this session
---@field transition_scores table<string, number>? Precomputed transition scores map

---@class NosItemData
---@field raw_features NosRawFeatures Raw computed feature scores
---@field neural_score number Total weighted score
---@field virtual_name string? Cached virtual name for special files
---@field normalized_path string Cached normalized absolute path
---@field is_open_buffer boolean File is open in a buffer
---@field is_alternate boolean File is the alternate buffer
---@field recent_rank number? Position in recent files (1-based)
---@field input_buf number[] Pre-allocated flat array of 11 normalized features for all algorithms
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
---@field picker_name? string Picker name for weight file isolation (set by registry)
---@field feature_names? string[] Feature name ordering for positional weight mapping (set by item_source for item pickers)

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
---@field optimizer? OptimizerType Optimizer algorithm (default "adamw")
---@field adam_beta1? number AdamW first moment decay (default 0.9)
---@field adam_beta2? number AdamW second moment decay (default 0.999)
---@field adam_epsilon? number AdamW numerical stability (default 1e-8)
---@field warmup_steps? number Number of steps to warm up learning rate (default 0, disabled)
---@field warmup_start_factor? number Starting learning rate factor for warmup (default 0.1)
---@field match_dropout? number Dropout rate for match/virtual_name features during training (default 0.25)
---@field margin? number Margin for pairwise hinge loss (default 1.0)
---@field picker_name? string Picker name for weight file isolation (set by registry)

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

--- Persisted item tracking storage format (stored under "item_tracking" key in picker tracking file)
---@class NosItemTrackingStore
---@field frecency table<string, number> Global item frecency: item_id -> deadline_timestamp
---@field cwd_frecency table<string, table<string, number>> CWD-scoped frecency: cwd -> { item_id -> deadline_timestamp }
---@field recency_list string[] Global ordered list of recent item selections (index 1 = most recent)
---@field cwd_recency table<string, string[]> CWD-scoped recency lists: cwd -> ordered list
---@field transition_frecency table<string, table<string, number>> Item-to-item transition frecency: source_id -> { dest_id -> deadline_timestamp } (empty table if not yet tracked)

--- Computed tracking data returned by get_tracking_data for feature computation
---@class NosItemTrackingData
---@field frecency table<string, number> Global frecency scores: item_id -> computed score
---@field cwd_frecency table<string, number> CWD-scoped frecency scores: item_id -> computed score
---@field recency_rank table<string, number> Global recency ranks: item_id -> 1-based position
---@field cwd_recency_rank table<string, number> CWD-scoped recency ranks: item_id -> 1-based position
---@field last_selected string? Most recently selected item id (global)
---@field last_cwd_selected string? Most recently selected item id in current CWD

--- Raw feature scores for item pickers (8 features)
---@class NosItemRawFeatures
---@field match number Raw fuzzy match score from matcher (typically 0-200+)
---@field frecency number Raw global frecency score from item_tracking (0-∞)
---@field cwd_frecency number Raw CWD-scoped frecency score from item_tracking (0-∞)
---@field recency number Raw global recency rank (1-based position in recency list, 0 if unranked)
---@field cwd_recency number Raw CWD-scoped recency rank (1-based, 0 if unranked)
---@field text_length_inv number Raw text length (used to compute inverse normalization)
---@field not_last_selected number Binary: 1 if NOT the most recently selected item, 0 if it is
---@field transition number Transition frecency score from item_tracking, already [0,1] via 1-1/(1+raw/4)

--- Normalized feature scores for item pickers (8 features, all in [0,1])
---@class NosItemNormalizedFeatures
---@field match number Normalized to [0,1] using sigmoid
---@field frecency number Normalized to [0,1] using 1 - 1/(1+x/8)
---@field cwd_frecency number Normalized to [0,1] using 1 - 1/(1+x/8)
---@field recency number Normalized to [0,1] using linear decay (max - rank + 1) / max
---@field cwd_recency number Normalized to [0,1] using linear decay (max - rank + 1) / max
---@field text_length_inv number Normalized to [0,1] using 1/(1+len*0.1)
---@field not_last_selected number Already [0,1] binary
---@field transition number Already [0,1] from calculation (1-1/(1+score/4))

--- Session context for item pickers
---@class NosItemContext
---@field cwd string Current working directory
---@field algorithm Algorithm The scoring algorithm instance for this session
---@field tracking_data NosItemTrackingData Preloaded tracking data for feature computation
---@field picker_name string Name of the picker (for weight isolation)
---@field transition_scores table<string, number>? Precomputed transition scores map (dest_id -> normalized score)

--- Neural-open data attached to item picker items (the item.nos field)
---@class NosItemPickerData
---@field raw_features NosItemRawFeatures Raw computed feature scores (8 features)
---@field neural_score number Total weighted score
---@field item_id string Item identity (item.value or item.text)
---@field input_buf number[] Pre-allocated flat array of 8 normalized features
---@field ctx NosItemContext Reference to shared session context

--- Item picker item with neural-open data
---@class NosItemPickerItem: snacks.picker.Item
---@field text string Display text for the item
---@field value string? Optional distinct value (identity defaults to text if not set)
---@field desc string? Optional description
---@field nos NosItemPickerData Neural-open specific data
---@field neural_rank? number Used during weight learning

--- Configuration for a registered picker (passed to pick() or register_picker())
---@class NosPickerConfig
---@field type "file"|"item" Picker type: "file" uses 11-feature pipeline, "item" uses 8-feature pipeline (default "item")
---@field title string? Picker window title
---@field finder fun(opts: table, ctx: table): any Snacks finder function
---@field items table[]? Static items (alternative to finder for item pickers)
---@field format string|function|nil Format function or named format
---@field preview string|function|nil Preview function or named preview
---@field confirm fun(picker: table, item: table)? Action on selection
---@field actions table<string, fun(picker: table, item: table)>? Snacks picker actions, forwarded to source config
---@field win table? Snacks picker window config (e.g. win.input.keys), forwarded to source config
---@field algorithm AlgorithmName? Algorithm override (default from global config)
---@field algorithm_config table? Algorithm config overrides

---@class NosConfig
---@field algorithm AlgorithmName Active algorithm: "classic" | "naive" | "nn"
---@field algorithm_config NosAlgorithmConfig Algorithm-specific configurations for file pickers
---@field item_algorithm_config NosAlgorithmConfig Algorithm-specific configurations for item pickers (8-feature pipeline)
---@field weights_path string Path to the default file picker's weight file (dirname used as fallback weights_dir)
---@field weights_dir? string Directory for all picker weight files (overrides dirname of weights_path)
---@field special_files table<string, boolean> Special files requiring virtual name handling
---@field recency_list_size number Maximum number of files in persistent recency list (default 100)
---@field file_sources string[] File sources for the default file picker (default {"buffers", "neural_recent", "files", "git_files"})
---@field debug NosDebugConfig Debug settings
