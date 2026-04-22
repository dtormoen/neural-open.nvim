# Changelog

## [0.1.5](https://github.com/dtormoen/neural-open.nvim/compare/v0.1.4...v0.1.5) (2026-04-22)


### Features

* expose known item_id queries and forward picker actions/win ([83af87f](https://github.com/dtormoen/neural-open.nvim/commit/83af87f096104ea3ddd66d9c4b798a90d3ee4f2b))
* update default weights ([de224d7](https://github.com/dtormoen/neural-open.nvim/commit/de224d761476eaaa19a1cf9d6c766fffea9875c6))


### Bug Fixes

* propagate pinned cwd through ctx so git_files proc runs from git_root ([ec7fa0c](https://github.com/dtormoen/neural-open.nvim/commit/ec7fa0c4b0e1a2cc1799ea79de6bb3ce28585951))

## [0.1.4](https://github.com/dtormoen/neural-open.nvim/compare/v0.1.3...v0.1.4) (2026-04-21)


### Features

* add neural_recent file source combining recency and frecency ([36c2826](https://github.com/dtormoen/neural-open.nvim/commit/36c2826e9e8a955dd75b916d2953dad55a386270))


### Bug Fixes

* prevent git_files from clobbering picker cwd in subdirectories ([bb1923f](https://github.com/dtormoen/neural-open.nvim/commit/bb1923f3010fbb7f07b0c0cb5efa7ce8a5e0eaf3))

## [0.1.3](https://github.com/dtormoen/neural-open.nvim/compare/v0.1.2...v0.1.3) (2026-03-09)


### Bug Fixes

* use git rev-parse for more robust git repo detection ([facebc6](https://github.com/dtormoen/neural-open.nvim/commit/facebc6da1acf55c0cff5d11e6f5a807e4c5248b))

## [0.1.2](https://github.com/dtormoen/neural-open.nvim/compare/v0.1.1...v0.1.2) (2026-03-04)

This release adds **custom item pickers** — neural-open can now learn from any
picker, not just files. Use `pick()` and `register_picker()` to create pickers
(e.g., just recipes, make targets, vim commands) that each train their own
neural network. An 8-feature scoring pipeline handles frecency, recency,
transitions, and more for non-file items.

Also new: `weights_dir` config for explicit control over where per-picker weight files
are stored (backward-compatible with `weights_path`), and bundled default weights for
item pickers so ranking works well out of the box.

### Features

* add bundled item picker default weights and multi-picker weight export ([dcaa6ba](https://github.com/dtormoen/neural-open.nvim/commit/dcaa6ba3677769970dbb58c39ea47864d659ad98))
* add item picker scoring pipeline with 7-feature architecture ([47c5f87](https://github.com/dtormoen/neural-open.nvim/commit/47c5f8723a4c6149d1a7d29ab7c5e607be4fb2de))
* add item tracking infrastructure for non-file pickers ([94b0d94](https://github.com/dtormoen/neural-open.nvim/commit/94b0d94059ec208d555d67b9a155a96694b10611))
* add item-to-item transition tracking for item pickers ([1a477b2](https://github.com/dtormoen/neural-open.nvim/commit/1a477b25399bc682e229f236fc9040940c5f9af0))
* add per-picker state infrastructure and create_instance to nn.lua ([865408d](https://github.com/dtormoen/neural-open.nvim/commit/865408d59985c599d966592e60ca1103281295bf))
* add public picker API with pick(), register_picker(), and learning integration ([0381ec6](https://github.com/dtormoen/neural-open.nvim/commit/0381ec62c21fe83b1cf783a085804b423254fa56))
* add registry isolation tests and update registry for create_instance API ([ecaa442](https://github.com/dtormoen/neural-open.nvim/commit/ecaa4426ec60f97b598869e6b6192c70fbf069d7))
* add types, end-to-end tests, examples, and documentation for multi-picker support ([ba84a11](https://github.com/dtormoen/neural-open.nvim/commit/ba84a11aa3d5cdbcfc45bb25ce90b4539e039e99))
* add weights_dir config with backward-compatible weights_path ([08eb4bb](https://github.com/dtormoen/neural-open.nvim/commit/08eb4bbf2fcef1fabc2ca52c75854634a4d44e14))
* consolidate debug preview for file and item pickers ([c98ead1](https://github.com/dtormoen/neural-open.nvim/commit/c98ead14022189e4f303725be75944eea8e13a04))
* migrate nn tests to instance API and remove legacy module-level API ([c8c091c](https://github.com/dtormoen/neural-open.nvim/commit/c8c091cbebd1bc92120f10960f0caf5a4ec041a1))
* refactor classic.lua to per-picker state isolation via create_instance ([52297c2](https://github.com/dtormoen/neural-open.nvim/commit/52297c2fce7cca8c99f48f9bd2b59573e27eada6))
* refactor weight storage for per-picker file separation ([0c89120](https://github.com/dtormoen/neural-open.nvim/commit/0c891204ce54b06d32942392f1cec74762a5c166))
* update default weights ([c395f22](https://github.com/dtormoen/neural-open.nvim/commit/c395f22f6b844d935e16403bb0c4619ab01e9f8b))


### Bug Fixes

* robust NN input-size migration with feature-name-driven defaults ([ca2df99](https://github.com/dtormoen/neural-open.nvim/commit/ca2df993b175985d739ebc5cdd86cf49ffb30462))
* suppress spurious adamw optimizer warnings in benchmark ([cd4611b](https://github.com/dtormoen/neural-open.nvim/commit/cd4611b12ce3e453306c6d5d96397793b3ed6e72))
* unify item frecency normalization with file picker formula ([5d6ec75](https://github.com/dtormoen/neural-open.nvim/commit/5d6ec75e220f1f5cc50f68423fddfc7c75ef8274))

## [0.1.1](https://github.com/dtormoen/neural-open.nvim/compare/v0.1.0...v0.1.1) (2026-03-04)


### Bug Fixes

* skip git_files source when not inside a git repository ([07f74df](https://github.com/dtormoen/neural-open.nvim/commit/07f74dff874b5d3cf7600602ae029acf39fa088c))

## [0.1.0](https://github.com/dtormoen/neural-open.nvim/compare/v0.0.1...v0.1.0) (2026-02-26)


### Features

* add not_current binary feature to all scoring algorithms ([0cba6ea](https://github.com/dtormoen/neural-open.nvim/commit/0cba6ea870af5140b506ba940b6176be5149f120))
* add picker hot-path benchmark framework ([f174c49](https://github.com/dtormoen/neural-open.nvim/commit/f174c49b62c5171d4e4c3c1bdd5005560a733c90))
* replace transition ring buffer with frecency-based scoring ([3814f6b](https://github.com/dtormoen/neural-open.nvim/commit/3814f6b4b9c1be700ac97855f4eff883fd4ca835))
* update default weights ([64ce23f](https://github.com/dtormoen/neural-open.nvim/commit/64ce23fcbab440b3a36218b4e3c0a0860e79ff82))


### Bug Fixes

* consolidate path normalization and fix training history path inconsistency ([cb80c2d](https://github.com/dtormoen/neural-open.nvim/commit/cb80c2d42e0d177b4c9e068f21b8f85e025d2c0c))
* defer BufEnter recency tracking to avoid blocking startup ([b4b709b](https://github.com/dtormoen/neural-open.nvim/commit/b4b709bffe3f99effe0fb0e2797bbf90482a9222))
* make nn_batch integration test deterministic ([79b092b](https://github.com/dtormoen/neural-open.nvim/commit/79b092b28bb1647940a74dab96a35c3e4a2afb20))
* respect batch_size when constructing first training batch ([437a487](https://github.com/dtormoen/neural-open.nvim/commit/437a487b955715e21a4687203882fac06bcf501e))


### Performance Improvements

* optimize inference hot path, zero-allocation trigrams, and benchmark sub-tests ([bf92324](https://github.com/dtormoen/neural-open.nvim/commit/bf92324d1e67cd91c821c5c095983c9fe98bc57b))
* optimize NN picker inference with batch norm fusion ([ebd3e93](https://github.com/dtormoen/neural-open.nvim/commit/ebd3e939d2bb6be37f273800d25341a59a3ceb30))
* optimize static feature computation with zero-allocation scoring ([4d5619d](https://github.com/dtormoen/neural-open.nvim/commit/4d5619dc8436db3a29da19e925678a140cea35ed))
* zero-allocation NN per-keystroke scoring with fused normalize+inference ([b5e7aba](https://github.com/dtormoen/neural-open.nvim/commit/b5e7aba85f53195852a098cc7e2b0470eb768419))
* zero-allocation trigram scoring and transform phase optimizations ([f073e82](https://github.com/dtormoen/neural-open.nvim/commit/f073e824cf3512cb2c36cf390e472d60da96588b))


### Miscellaneous Chores

* release 0.1.0 ([42ea225](https://github.com/dtormoen/neural-open.nvim/commit/42ea225571c3ac1b348b27e99121a65951823880))
