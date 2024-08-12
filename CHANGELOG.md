# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## Unreleased

### Added
- `Debias` mechanism for classification, ranking and auc metrics. New parameter `is_debiased` to `calc_from_confusion_df`, `calc_per_user_from_confusion_df` methods of classification metrics, `calc_from_fitted`, `calc_per_user_from_fitted` methods of auc and rankning (`MAP`) metrics, `calc_from_merged`, `calc_per_user_from_merged` methods of ranking (`NDCG`, `MRR`) metrics. ([#152](https://github.com/MobileTeleSystems/RecTools/pull/152))
- `nbformat >= 4.2.0` dependency to `[visuals]` extra ([#169](https://github.com/MobileTeleSystems/RecTools/pull/169))
- `filter_on_interactions_df_row_indexes` method of `Dataset` ([#177](https://github.com/MobileTeleSystems/RecTools/pull/177))
- `on_unsupported_targets` parameter to  `recommend` and `recommend_to_items` model methods ([#177](https://github.com/MobileTeleSystems/RecTools/pull/177))

### Fixed
- `display()` method in `MetricsApp` ([#169](https://github.com/MobileTeleSystems/RecTools/pull/169))
- `IntraListDiversity` metric computation in `cross_validate` ([#177](https://github.com/MobileTeleSystems/RecTools/pull/177))

### Removed
- [Breaking] `assume_external_ids` parameter in `recommend` and `recommend_to_items` model methods ([#177](https://github.com/MobileTeleSystems/RecTools/pull/177))


### Fixed
- `IntraListDiversity` metric computation in `cross_validate` ([#177](https://github.com/MobileTeleSystems/RecTools/pull/177))


## [0.7.0] - 29.07.2024

### Added
- Extended Theory&Practice RecSys baselines tutorial ([#139](https://github.com/MobileTeleSystems/RecTools/pull/139))
- `MetricsApp` to create plotly scatterplot widgets for metric-to-metric trade-off analysis ([#140](https://github.com/MobileTeleSystems/RecTools/pull/140), [#154](https://github.com/MobileTeleSystems/RecTools/pull/154))
- `Intersection` metric ([#148](https://github.com/MobileTeleSystems/RecTools/pull/148))
- `PartialAUC` and `PAP` metrics  ([#149](https://github.com/MobileTeleSystems/RecTools/pull/149))
- New params (`tol`, `maxiter`, `random_state`) to the `PureSVD` model ([#130](https://github.com/MobileTeleSystems/RecTools/pull/130))
- Recommendations data quality metrics: `SufficientReco`, `UnrepeatedReco`, `CoveredUsers` ([#155](https://github.com/MobileTeleSystems/RecTools/pull/155))
- `r_precision` parameter to `Precision` metric ([#155](https://github.com/MobileTeleSystems/RecTools/pull/155))

### Fixed
- Used `rectools-lightfm` instead of pure `lightfm` that allowed to install it using `poetry>=1.5.0` ([#165](https://github.com/MobileTeleSystems/RecTools/pull/165))
- Added restriction to `pytorch` version for MacOSX + x86_64 that allows to install it on such platforms ([#142](https://github.com/MobileTeleSystems/RecTools/pull/142))
- `PopularInCategoryModel` fitting for multiple times, `cross_validate` compatibility, behaviour with empty category interactions ([#163](https://github.com/MobileTeleSystems/RecTools/pull/163))


## [0.6.0] - 13.05.2024

### Added 
- Warm users/items support in `Dataset` ([#77](https://github.com/MobileTeleSystems/RecTools/pull/77))
- Warm and cold users/items support in `ModelBase` and all possible models ([#77](https://github.com/MobileTeleSystems/RecTools/pull/77), [#120](https://github.com/MobileTeleSystems/RecTools/pull/120), [#122](https://github.com/MobileTeleSystems/RecTools/pull/122))
- Warm and cold users/items support in `cross_validate` ([#77](https://github.com/MobileTeleSystems/RecTools/pull/77))
- [Breaking] Default value for train dataset type and params for user and item dataset types in `DSSMModel` ([#122](https://github.com/MobileTeleSystems/RecTools/pull/122))
- [Breaking] `n_factors` and `deterministic` params to `DSSMModel` ([#122](https://github.com/MobileTeleSystems/RecTools/pull/122))
- Hit Rate metric ([#124](https://github.com/MobileTeleSystems/RecTools/pull/124))
- Python `3.11` support (without `nmslib`) ([#126](https://github.com/MobileTeleSystems/RecTools/pull/126))
- Python `3.12` support (without `nmslib` and `lightfm`) ([#126](https://github.com/MobileTeleSystems/RecTools/pull/126))

### Changed
- Changed the logic of choosing random sampler for `RandomModel` and increased the sampling speed ([#120](https://github.com/MobileTeleSystems/RecTools/pull/120))
- [Breaking] Changed the logic of `RandomModel`: now the recommendations are different for repeated calls of recommend methods ([#120](https://github.com/MobileTeleSystems/RecTools/pull/120))
- Torch datasets to support warm recommendations ([#122](https://github.com/MobileTeleSystems/RecTools/pull/122))
- [Breaking] Replaced `include_warm` parameter in `Dataset.get_user_item_matrix` to pair `include_warm_users` and `include_warm_items` ([#122](https://github.com/MobileTeleSystems/RecTools/pull/122))
- [Breaking] Renamed torch datasets and `dataset_type` to `train_dataset_type` param in `DSSMModel` ([#122](https://github.com/MobileTeleSystems/RecTools/pull/122))
- [Breaking] Updated minimum versions of `numpy`, `scipy`, `pandas`, `typeguard` ([#126](https://github.com/MobileTeleSystems/RecTools/pull/126))
- [Breaking] Set restriction `scipy < 1.13` ([#126](https://github.com/MobileTeleSystems/RecTools/pull/126))

### Removed
- [Breaking] `return_external_ids` parameter in `recommend` and `recommend_to_items` model methods ([#77](https://github.com/MobileTeleSystems/RecTools/pull/77))
- [Breaking] Python `3.7` support ([#126](https://github.com/MobileTeleSystems/RecTools/pull/126))


## [0.5.0] - 22.03.2024

### Added
- `VisualApp` and `ItemToItemVisualApp` widgets for visual comparison of recommendations ([#80](https://github.com/MobileTeleSystems/RecTools/pull/80), [#82](https://github.com/MobileTeleSystems/RecTools/pull/82), [#85](https://github.com/MobileTeleSystems/RecTools/pull/85),  [#115](https://github.com/MobileTeleSystems/RecTools/pull/115))
- Methods for conversion `Interactions` to raw form and for getting raw interactions from `Dataset` ([#69](https://github.com/MobileTeleSystems/RecTools/pull/69))
- `AvgRecPopularity (Average Recommendation Popularity)` to `metrics` ([#81](https://github.com/MobileTeleSystems/RecTools/pull/81))
- Added `normalized` parameter to `AvgRecPopularity` metric ([#89](https://github.com/MobileTeleSystems/RecTools/pull/89))
- Added `EASE` model ([#107](https://github.com/MobileTeleSystems/RecTools/pull/107))

### Changed
- Loosened `pandas`, `torch` and `torch-light` versions for `python >= 3.8` ([#58](https://github.com/MobileTeleSystems/RecTools/pull/58))

### Fixed
- Bug in `Interactions.from_raw` method ([#58](https://github.com/MobileTeleSystems/RecTools/pull/58))
- Mistakes in formulas for Serendipity and MIUF in docstrings ([#115](https://github.com/MobileTeleSystems/RecTools/pull/115))
- Examples reproducibility on Google Colab ([#115](https://github.com/MobileTeleSystems/RecTools/pull/115))


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
