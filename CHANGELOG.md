# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## Unreleased

### Added
- VisualApp and ItemToItemVisualApp widgets for visual comparison of recommendations ([#80](https://github.com/MobileTeleSystems/RecTools/pull/80), [#82](https://github.com/MobileTeleSystems/RecTools/pull/82)) (Need examples and readme update)
- Methods for conversion `Interactions` to raw form and for getting raw interactions from `Dataset` ([#69](https://github.com/MobileTeleSystems/RecTools/pull/69))
- `AvgRecPopularity (Average Recommendation Popularity)` to `metrics` ([#81](https://github.com/MobileTeleSystems/RecTools/pull/81))
- Added `normalized` parameter to `AvgRecPopularity` metric ([#89](https://github.com/MobileTeleSystems/RecTools/pull/89))

### Changed
- Loosened `pandas`, `torch` and `torch-light` versions for `python >= 3.8` ([#58](https://github.com/MobileTeleSystems/RecTools/pull/58))

### Fixed
- Bug in `Interactions.from_raw` method ([#58](https://github.com/MobileTeleSystems/RecTools/pull/58))


## [0.4.2] - 01.12.2023

### Added
- Ability to pass internal ids to `recommend` and `recommend_to_items` methods and get internal ids back ([#70](https://github.com/MobileTeleSystems/RecTools/pull/70))
- `rectools.model_selection.cross_validate` function ([#71](https://github.com/MobileTeleSystems/RecTools/pull/71), [#73](https://github.com/MobileTeleSystems/RecTools/pull/73))

### Changed
- Loosened `lightfm` version, now it's possible to use 1.16 and 1.17 ([#72](https://github.com/MobileTeleSystems/RecTools/pull/72))

### Fixed
- Small bug in `LastNSplitter` with incorrect `i_split` in info ([#70](https://github.com/MobileTeleSystems/RecTools/pull/70))


## [0.4.1] - 31.10.2023

### Added
- LightFM wrapper inference speed benchmark ([#60](https://github.com/MobileTeleSystems/RecTools/pull/60))
- iALS with features quality benchmark ([#60](https://github.com/MobileTeleSystems/RecTools/pull/60))

### Changed
- Updated attrs version ([#56](https://github.com/MobileTeleSystems/RecTools/pull/56))
- Optimized inference for vector models with EUCLIDEAN distance using `implicit` library topk method ([#57](https://github.com/MobileTeleSystems/RecTools/pull/57))
- Changed features processing example ([#60](https://github.com/MobileTeleSystems/RecTools/pull/60))


## [0.4.0] - 27.10.2023

### Added
- `MRR (Mean Reciprocal Rank)` to `metrics` ([#29](https://github.com/MobileTeleSystems/RecTools/pull/29))
- `F1beta`, `MCC (Matthew correlation coefficient)` to `metrics` ([#32](https://github.com/MobileTeleSystems/RecTools/pull/32))
- Base `Splitter` class to construct data splitters ([#31](https://github.com/MobileTeleSystems/RecTools/pull/31))
- `RandomSplitter` to `model_selection` ([#31](https://github.com/MobileTeleSystems/RecTools/pull/31))
- `LastNSplitter` to `model_selection` ([#33](https://github.com/MobileTeleSystems/RecTools/pull/32))
- Support for `Python 3.10` ([#47](https://github.com/MobileTeleSystems/RecTools/pull/47))

### Changed
- Bumped `implicit` version to `0.7.1` ([#45](https://github.com/MobileTeleSystems/RecTools/pull/45))
- Bumped `lightfm` version to `1.17` ([#43](https://github.com/MobileTeleSystems/RecTools/pull/43))
- Bumped `pylint` version to `2.17.6` ([#43](https://github.com/MobileTeleSystems/RecTools/pull/43)) 
- Moved `nmslib` from main dependencies to extras ([#36](https://github.com/MobileTeleSystems/RecTools/pull/36))
- Moved `lightfm` to extras ([#51](https://github.com/MobileTeleSystems/RecTools/pull/51))
- Renamed `nn` extra to `torch` ([#51](https://github.com/MobileTeleSystems/RecTools/pull/51))
- Optimized inference for vector models with COSINE and DOT distances using `implicit` library topk method ([#52](https://github.com/MobileTeleSystems/RecTools/pull/52))
- Changed initialization of `TimeRangeSplitter` (instead of `date_range` argument, use `test_size` and `n_splits`) ([#53](https://github.com/MobileTeleSystems/RecTools/pull/51))
- Changed split infos key names in splitters ([#53](https://github.com/MobileTeleSystems/RecTools/pull/51))

### Fixed
- Bugs with new version of `pytorch_lightning` ([#43](https://github.com/MobileTeleSystems/RecTools/pull/43))
- `pylint` config for new version ([#43](https://github.com/MobileTeleSystems/RecTools/pull/43))
- Cyclic imports ([#45](https://github.com/MobileTeleSystems/RecTools/pull/45))

### Removed
- `Markdown` dependancy ([#54](https://github.com/MobileTeleSystems/RecTools/pull/54))
