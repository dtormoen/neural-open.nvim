rockspec_format = '3.0'
package = "neural-open"
version = "scm-1"
source = {
  url = "git+https://github.com/dtormoen/neural-open.nvim"
}
description = {
  summary = "Neural file picker for Neovim using Snacks.nvim with machine learning ranking",
  detailed = "A Neovim file picker that learns how you navigate. A neural network trains on your file selections to rank results, combining fuzzy matching with contextual signals like buffer state, directory proximity, frecency, and file-to-file transition patterns. Built for Snacks.nvim.",
  homepage = "https://github.com/dtormoen/neural-open.nvim",
  license = "MIT"
}
dependencies = {
  "snacks.nvim >= 2.20.0"
}
test_dependencies = {
  "LuaFileSystem",
  "busted",
  "llscheck",
  "luajit >= 2.1.0",
  "luassert",
  "nlua",
  "snacks.nvim >= 2.20.0"
}
build = {
  type = "builtin",
  copy_directories = {
  },
}
