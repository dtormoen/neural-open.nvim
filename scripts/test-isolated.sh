#!/usr/bin/env bash
# Test isolation wrapper for neural-open.nvim
# This script ensures tests run in isolated XDG directories to protect the user's real Neovim environment

set -euo pipefail

# Create a unique temporary directory for this test run
TEST_DIR=$(mktemp -d -t neural-open-test-XXXXXX)

# Ensure cleanup on exit (success or failure)
trap "rm -rf '$TEST_DIR'" EXIT INT TERM

# Set up isolated XDG directories
export XDG_DATA_HOME="$TEST_DIR/data"
export XDG_CONFIG_HOME="$TEST_DIR/config"
export XDG_CACHE_HOME="$TEST_DIR/cache"
export XDG_STATE_HOME="$TEST_DIR/state"

# Set testing flag so tests can verify isolation
export NEURAL_OPEN_TESTING=1

# Create the directories to avoid any issues
mkdir -p "$XDG_DATA_HOME"
mkdir -p "$XDG_CONFIG_HOME"
mkdir -p "$XDG_CACHE_HOME"
mkdir -p "$XDG_STATE_HOME"

# Show isolation info if verbose
if [[ "${VERBOSE:-0}" == "1" ]]; then
    echo "Test isolation active:"
    echo "  XDG_DATA_HOME=$XDG_DATA_HOME"
    echo "  XDG_CONFIG_HOME=$XDG_CONFIG_HOME"
    echo "  XDG_CACHE_HOME=$XDG_CACHE_HOME"
    echo "  XDG_STATE_HOME=$XDG_STATE_HOME"
    echo ""
fi

# Execute the test command with all arguments
exec "$@"