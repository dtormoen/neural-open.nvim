-- vim: ft=lua tw=80

stds.nvim = {
  globals = {
    "vim",
  },
  read_globals = {
    "jit",
    "os",
    "bit",
  },
}

std = "lua51+nvim"

globals = {
  "vim.g",
  "vim.b",
  "vim.w",
  "vim.o",
  "vim.bo",
  "vim.wo",
  "vim.go",
  "vim.env",
}

read_globals = {
  "vim",
}

ignore = {
  "631", -- Line is too long
  "212", -- Unused argument
  "122", -- Setting a read-only field of a global variable
}

files = {
  ["lua/**/*.lua"] = {
    std = "lua51+nvim",
  },
  ["tests/**/*_spec.lua"] = {
    std = "lua51+nvim+busted",
    read_globals = {
      "describe",
      "it",
      "before_each",
      "after_each",
      "setup",
      "teardown",
      "pending",
      "assert",
      "spy",
      "mock",
    },
  },
}