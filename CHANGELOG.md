# Changelog

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
