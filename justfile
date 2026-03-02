list:
    just --list

# Run tests with isolation (safe for real environment)
test *args:
    ./scripts/test-isolated.sh busted {{args}} tests/

# Run tests with verbose output
test-verbose:
    VERBOSE=1 ./scripts/test-isolated.sh busted -v tests/

# Format code with stylua
format:
    stylua lua/ tests/ benchmarks/

# Lint code with luacheck
lint:
    luacheck lua/ tests/ benchmarks/

# Type check with lua-language-server
typecheck:
    VIMRUNTIME="`nvim --clean --headless --cmd 'lua io.write(os.getenv("VIMRUNTIME"))' --cmd 'quit'`" llscheck ./lua --checklevel Information

# Install development dependencies
setup:
    luarocks make neural-open-scm-1.rockspec --local
    luarocks install LuaFileSystem --local
    luarocks install busted --local
    luarocks install luacheck --local
    luarocks install llscheck --local
    luarocks install luassert --local
    luarocks install nlua --local

# Clean up temporary files
clean:
    rm -rf luarocks
    rm -rf .luarocks
    rm -f .luarc.log
    find . -name "*.rock" -delete

# Run picker hot-path benchmark
benchmark:
    ./scripts/test-isolated.sh nlua benchmarks/picker_benchmark.lua

# Update bundled default NN weights from trained weight files and re-capture regression tests
update-default-weights dir="~/.local/share/nvim/neural-open":
    ./scripts/update-default-weights.py {{dir}}
    REGRESSION_CAPTURE=1 ./scripts/test-isolated.sh busted tests/regression_spec.lua

# Run all checks required before commiting
precommit: format lint typecheck test
    echo "Everything passed!"
