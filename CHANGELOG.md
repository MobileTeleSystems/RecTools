# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## Unreleased
### Added
- Added `MRR (Mean Reciprocal Rank)` to `metrics` ([#29](https://github.com/MobileTeleSystems/RecTools/pull/29))
- Added `F1beta`, `MCC (Matthew correlation coefficient)` to `metrics` ([#32](https://github.com/MobileTeleSystems/RecTools/pull/32))
- Added `LastNSplitter` to `model_selection` ([#33](https://github.com/MobileTeleSystems/RecTools/pull/32))
- Added random `KFoldSplitter` to `model_selection` ([#31](https://github.com/MobileTeleSystems/RecTools/pull/31))
- Support for `Python 3.10` ([#47](https://github.com/MobileTeleSystems/RecTools/pull/47))

### Changed
- Bumped `implicit` version to 0.7.1 ([#45](https://github.com/MobileTeleSystems/RecTools/pull/45))
- Bumped `poetry` version to 1.4.0 for github workflows ([#43](https://github.com/MobileTeleSystems/RecTools/pull/43))
- Updated dependencies ([#43](https://github.com/MobileTeleSystems/RecTools/pull/43))
- Moved `nmslib` from main dependencies to extras ([#36](https://github.com/MobileTeleSystems/RecTools/pull/36))
- Added base `Splitter` class to construct data splitters ([#31](https://github.com/MobileTeleSystems/RecTools/pull/31))
- Updated notebooks in `examples` ([#44](https://github.com/MobileTeleSystems/RecTools/pull/44))
- Moved `lightfm` to extras ([#51](https://github.com/MobileTeleSystems/RecTools/pull/51))
- Renamed `nn` extra to `torch` ([#51](https://github.com/MobileTeleSystems/RecTools/pull/51))
- Optimized inference for vector models with COSINE and DOT distances using `implicit` library topk method ([#52](https://github.com/MobileTeleSystems/RecTools/pull/52))

### Fixed
- Fixed bugs with new version of `pytorch_lightning` ([#43](https://github.com/MobileTeleSystems/RecTools/pull/43))
- Fixed `pylint` config for new version ([#43](https://github.com/MobileTeleSystems/RecTools/pull/43))
- Fixed CI  ([#40](https://github.com/MobileTeleSystems/RecTools/pull/40)) ([#34](https://github.com/MobileTeleSystems/RecTools/pull/34))
- Fixed cyclic imports ([#45](https://github.com/MobileTeleSystems/RecTools/pull/45))

### Removed
- 
