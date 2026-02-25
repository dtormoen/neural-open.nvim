--- Latency tracking module for performance debugging
--- Provides thread-safe, context-based timing with zero overhead when disabled
---@module 'neural-open.latency'

local M = {}

--- Store the last timing context for manual inspection
M.last_timing = nil

--- Create a timing context for a single operation flow
--- This context is passed through sync and async boundaries to maintain thread safety
---@return table|nil context nil if disabled, context table otherwise
local function create_context_impl()
  return {
    timings = {}, -- Map of operation_name -> timing data
    start_time = vim.loop.hrtime(),
  }
end

--- Start timing an operation within a context
---@param ctx table|nil Timing context (nil if disabled)
---@param operation_name string Name of the operation
---@param parent_name? string Optional parent operation for hierarchy
local function start_impl(ctx, operation_name, parent_name)
  if not ctx then
    return
  end
  ctx.timings[operation_name] = {
    start = vim.loop.hrtime(),
    parent = parent_name,
  }
end

--- Finish timing an operation
---@param ctx table|nil Timing context
---@param operation_name string Name of the operation
local function finish_impl(ctx, operation_name)
  if not ctx or not ctx.timings[operation_name] then
    return
  end
  local timing = ctx.timings[operation_name]

  -- Calculate duration
  local duration_ns = vim.loop.hrtime() - timing.start
  timing.duration_ms = duration_ns / 1e6
  timing.start = nil -- Clear start time, keep duration
end

--- Add metadata to an operation (e.g., data sizes, counts)
---@param ctx table|nil Timing context
---@param operation_name string Name of the operation
---@param metadata table Arbitrary metadata about the operation
local function add_metadata_impl(ctx, operation_name, metadata)
  if not ctx or not ctx.timings[operation_name] then
    return
  end
  ctx.timings[operation_name].metadata = metadata
end

--- Measure a synchronous operation with automatic timing and error handling
---@param ctx table|nil Timing context
---@param operation_name string Name of the operation
---@param fn function Function to measure
---@param parent_name? string Optional parent operation
---@return any result Result from function (or error object if pcall failed)
---@return boolean success True if function succeeded
local function measure_impl(ctx, operation_name, fn, parent_name)
  if not ctx then
    return fn(), true
  end

  start_impl(ctx, operation_name, parent_name)

  local success, result = pcall(fn)
  if not success then
    ctx.timings[operation_name].error = tostring(result)
  end

  finish_impl(ctx, operation_name)
  return result, success
end

--- Format a number with thousands separators
---@param num number
---@return string
local function format_number(num)
  local formatted = tostring(num)
  local k
  while true do
    formatted, k = string.gsub(formatted, "^(-?%d+)(%d%d%d)", "%1,%2")
    if k == 0 then
      break
    end
  end
  return formatted
end

--- Format bytes to human-readable size
---@param bytes number
---@return string
local function format_bytes(bytes)
  if bytes < 1024 then
    return bytes .. "B"
  elseif bytes < 1024 * 1024 then
    return string.format("%.1fKB", bytes / 1024)
  else
    return string.format("%.1fMB", bytes / (1024 * 1024))
  end
end

--- Build a tree structure from timing data
---@param timings table<string, table> Timing data
---@return table tree Root nodes
local function build_tree(timings)
  local nodes = {}
  local children = {}

  -- First pass: create all nodes
  for name, timing in pairs(timings) do
    nodes[name] = {
      name = name,
      timing = timing,
      children = {},
    }
  end

  -- Second pass: build parent-child relationships
  for name, node in pairs(nodes) do
    local parent_name = node.timing.parent
    if parent_name and nodes[parent_name] then
      table.insert(nodes[parent_name].children, node)
      children[name] = true
    end
  end

  -- Third pass: collect root nodes (those without parents or with missing parents)
  local roots = {}
  for name, node in pairs(nodes) do
    if not children[name] then
      table.insert(roots, node)
    end
  end

  -- Sort roots by name for consistent output
  table.sort(roots, function(a, b)
    return a.name < b.name
  end)

  return roots
end

--- Format timing tree as lines with indentation
---@param node table Tree node
---@param indent string Current indentation
---@param lines table Accumulator for lines
---@param threshold_ms number Threshold for warnings
local function format_node(node, indent, lines, threshold_ms)
  local timing = node.timing
  local duration = timing.duration_ms or 0
  local name = node.name

  -- Build the line
  local line = indent

  -- Add tree branch characters
  if indent ~= "" then
    line = line .. "└─ "
  end

  line = line .. name

  -- Add duration
  if timing.duration_ms then
    line = string.format("%s  %8.2fms", line, duration)
  else
    line = line .. "  ⏱ incomplete"
  end

  -- Add warning symbols based on duration
  if timing.duration_ms then
    if duration >= 500 then
      line = line .. " 🔥"
    elseif duration >= threshold_ms then
      line = line .. " ⚠️"
    end
  end

  -- Add error marker
  if timing.error then
    line = line .. " ❌ " .. timing.error
  end

  -- Add metadata if present
  if timing.metadata then
    local meta_parts = {}
    for k, v in pairs(timing.metadata) do
      if k == "bytes" then
        table.insert(meta_parts, format_bytes(v))
      else
        table.insert(meta_parts, k .. "=" .. format_number(v))
      end
    end
    if #meta_parts > 0 then
      line = line .. " [" .. table.concat(meta_parts, ", ") .. "]"
    end
  end

  table.insert(lines, line)

  -- Sort children by duration (descending) for easier bottleneck identification
  table.sort(node.children, function(a, b)
    local a_dur = a.timing.duration_ms or 0
    local b_dur = b.timing.duration_ms or 0
    return a_dur > b_dur
  end)

  -- Recursively format children
  local child_indent
  if indent == "" then
    child_indent = "  "
  else
    child_indent = indent:gsub("└─", "  ") .. "  "
  end

  for _, child in ipairs(node.children) do
    format_node(child, child_indent, lines, threshold_ms)
  end
end

--- Generate suggestions based on timing patterns
---@param ctx table Timing context
---@param threshold_ms number Threshold for warnings
---@return string[] suggestions
local function generate_suggestions(ctx, threshold_ms)
  local suggestions = {}

  -- Check for slow json_encode operations
  for name, timing in pairs(ctx.timings) do
    if name:match("json_encode") and timing.duration_ms and timing.duration_ms > threshold_ms then
      if timing.metadata and timing.metadata.bytes then
        table.insert(
          suggestions,
          string.format(
            "Large JSON encoding detected (%s in %.2fms). Consider reducing history_size or batches_per_update.",
            format_bytes(timing.metadata.bytes),
            timing.duration_ms
          )
        )
      else
        table.insert(
          suggestions,
          string.format("Slow JSON encoding (%.2fms). Check NN history_size or state complexity.", timing.duration_ms)
        )
      end
    end
  end

  -- Check for slow file operations
  for name, timing in pairs(ctx.timings) do
    if name:match("file_write") and timing.duration_ms and timing.duration_ms > 50 then
      table.insert(
        suggestions,
        string.format(
          "Slow file write (%.2fms). Check filesystem type (network drive?) or move weights_path to faster storage.",
          timing.duration_ms
        )
      )
    end
  end

  -- Check for slow training with phase-specific guidance
  for name, timing in pairs(ctx.timings) do
    if name:match("training") and timing.duration_ms and timing.duration_ms > 20 then
      local meta = timing.metadata or {}
      local suggestion = string.format("Slow training (%.2fms)", timing.duration_ms)

      -- Provide specific guidance based on bottleneck phase
      if meta.avg_backward_ms and meta.avg_forward_ms and meta.avg_update_ms then
        local fwd = meta.avg_forward_ms
        local bwd = meta.avg_backward_ms
        local upd = meta.avg_update_ms

        -- Identify bottleneck (allow 20% margin for noise)
        if bwd > fwd * 1.2 and bwd > upd * 1.2 then
          suggestion = suggestion
            .. ". Backward pass is bottleneck (avg "
            .. string.format("%.1fms", bwd)
            .. "ms) - consider reducing network depth/width."
        elseif upd > fwd * 1.2 and upd > bwd * 1.2 then
          suggestion = suggestion .. ". Parameter updates are slow (avg " .. string.format("%.1fms", upd) .. "ms)"
          if meta.optimizer == "adamw" then
            suggestion = suggestion .. " - AdamW has higher overhead than SGD."
          else
            suggestion = suggestion .. " - check optimizer complexity."
          end
        elseif fwd > bwd * 1.2 and fwd > upd * 1.2 then
          suggestion = suggestion
            .. ". Forward pass is bottleneck (avg "
            .. string.format("%.1fms", fwd)
            .. "ms) - consider reducing batch_size or input features."
        else
          -- No clear bottleneck, all phases similar
          suggestion = suggestion
            .. ". All phases similar - consider reducing batch_size, architecture, or batches_per_update."
        end
      else
        -- No metadata available, provide generic guidance
        suggestion = suggestion .. ". Consider reducing batch_size, network architecture, or batches_per_update."
      end

      table.insert(suggestions, suggestion)
    end
  end

  return suggestions
end

--- Format timing context as a tree structure for display
---@param ctx table Timing context
---@return string[] lines Formatted output lines
local function format_timing_tree(ctx)
  local config = require("neural-open").config
  local threshold_ms = config.debug.latency_threshold_ms or 100

  local lines = {}

  -- Calculate total duration
  local total_ms = (vim.loop.hrtime() - ctx.start_time) / 1e6

  -- Header
  table.insert(lines, "=== NeuralOpen Latency Report ===")
  table.insert(lines, string.format("Total: %.2fms", total_ms))

  -- Add warning for slow operations
  if total_ms >= 500 then
    table.insert(lines, "Status: 🔥 VERY SLOW")
  elseif total_ms >= threshold_ms then
    table.insert(lines, "Status: ⚠️ SLOW")
  else
    table.insert(lines, "Status: ✓ OK")
  end

  table.insert(lines, "")

  -- Build and format tree
  local tree = build_tree(ctx.timings)
  for _, root in ipairs(tree) do
    format_node(root, "", lines, threshold_ms)
  end

  -- Add legend
  table.insert(lines, "")
  table.insert(lines, "Flags: ⚠️ >" .. threshold_ms .. "ms, 🔥 >500ms, ⏱ incomplete, ❌ error")

  -- Add suggestions
  local suggestions = generate_suggestions(ctx, threshold_ms)
  if #suggestions > 0 then
    table.insert(lines, "")
    table.insert(lines, "Suggestions:")
    for _, suggestion in ipairs(suggestions) do
      table.insert(lines, "  - " .. suggestion)
    end
  end

  return lines
end

--- Log the timing context
---@param ctx table|nil Timing context
---@param file_name string Name of file that was selected
local function log_context_impl(ctx, file_name)
  if not ctx then
    return
  end

  local total_ms = (vim.loop.hrtime() - ctx.start_time) / 1e6
  local config = require("neural-open").config

  -- Check threshold
  if config.debug.latency_threshold_ms and total_ms < config.debug.latency_threshold_ms then
    return
  end

  local lines = format_timing_tree(ctx)

  -- Add file name to report
  table.insert(lines, 1, "File: " .. (file_name or "unknown"))
  table.insert(lines, 2, "=== NeuralOpen Latency Report ===")
  table.remove(lines, 3) -- Remove duplicate header

  -- Always store for inspection
  M.last_timing = ctx

  -- Notify
  local formatted = table.concat(lines, "\n")
  vim.notify(formatted, vim.log.levels.INFO)

  -- Optional: append to file
  if config.debug.latency_file then
    local file = io.open(config.debug.latency_file, "a")
    if file then
      file:write("\n" .. string.rep("=", 60) .. "\n")
      file:write("Timestamp: " .. os.date("%Y-%m-%d %H:%M:%S") .. "\n")
      file:write(formatted .. "\n")
      file:close()
    end
  end

  -- Optional: copy to clipboard
  if config.debug.latency_auto_clipboard then
    vim.fn.setreg("+", formatted)
  end
end

-- No-op implementations for disabled state
local noop = function() end
local noop_ctx = function()
  return nil
end
local noop_measure = function(_, _, fn)
  return fn(), true
end

--- Enable or disable latency tracking
---@param enable boolean
function M.set_enabled(enable)
  if enable then
    -- Use real implementations
    M.create_context = create_context_impl
    M.start = start_impl
    M.finish = finish_impl
    M.measure = measure_impl
    M.add_metadata = add_metadata_impl
    M.log_context = log_context_impl
  else
    -- Use no-op implementations (zero overhead)
    M.create_context = noop_ctx
    M.start = noop
    M.finish = noop
    M.measure = noop_measure
    M.add_metadata = noop
    M.log_context = noop
  end
end

--- Show last timing (for debugging in Lua console)
function M.show_last()
  if M.last_timing then
    local lines = format_timing_tree(M.last_timing)
    print(table.concat(lines, "\n"))
  else
    print("No timing data available. Enable debug.latency and select a file first.")
  end
end

-- Initialize with disabled state
M.set_enabled(false)

return M
