# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Uncommited

### Added 
- `get_val_mask_func_kwargs` and  `get_trainer_func_kwargs` kwargs for corresponding functions 


## [0.13.0] - 10.04.2025

### Added 
- `TransformerNegativeSamplerBase` and `CatalogUniformSampler` classes, `negative_sampler_type` and `negative_sampler_kwargs` parameters to transformer-based models ([#275](https://github.com/MobileTeleSystems/RecTools/pull/275))
- `SimilarityModuleBase`, `DistanceSimilarityModule`, similarity module to `TransformerTorchBackbone` parameters to transformer-based models `similarity_module_type`, `similarity_module_kwargs` ([#272](https://github.com/MobileTeleSystems/RecTools/pull/272))
- `out_dim` property to `IdEmbeddingsItemNet`, `CatFeaturesItemNet` and `SumOfEmbeddingsConstructor` ([#276](https://github.com/MobileTeleSystems/RecTools/pull/276))
- `TransformerBackboneBase`, `backbone_type` and `backbone_kwargs` parameters to transformer-based models ([#277](https://github.com/MobileTeleSystems/RecTools/pull/277))
- `sampled_softmax` loss option for transformer models ([#274](https://github.com/MobileTeleSystems/RecTools/pull/274))


## [0.12.0] - 24.02.2025

### Added
- `CatalogCoverage` metric ([#266](https://github.com/MobileTeleSystems/RecTools/pull/266), [#267](https://github.com/MobileTeleSystems/RecTools/pull/267))
- `divide_by_achievable` argument to `NDCG` metric ([#266](https://github.com/MobileTeleSystems/RecTools/pull/266))

### Changed
- Interactions extra columns are not dropped in `Dataset.filter_interactions` method [#267](https://github.com/MobileTeleSystems/RecTools/pull/267)

## [0.11.0] - 17.02.2025

### Added
- `SASRecModel` and `BERT4RecModel` - models based on transformer architecture ([#220](https://github.com/MobileTeleSystems/RecTools/pull/220))
- Transfomers extended theory & practice tutorial, advanced training guide and customization guide ([#220](https://github.com/MobileTeleSystems/RecTools/pull/220))
- `use_gpu` for PureSVD ([#229](https://github.com/MobileTeleSystems/RecTools/pull/229))
- `from_params` method for models and `model_from_params` function ([#252](https://github.com/MobileTeleSystems/RecTools/pull/252))
- `TorchRanker` ranker which calculates scores using torch. Supports GPU. [#251](https://github.com/MobileTeleSystems/RecTools/pull/251)
- `Ranker` ranker protocol which unify rankers call. [#251](https://github.com/MobileTeleSystems/RecTools/pull/251)

### Changed

- `ImplicitRanker` `rank` method compatible with `Ranker` protocol. `use_gpu` and `num_threads` params moved from `rank` method to `__init__`. [#251](https://github.com/MobileTeleSystems/RecTools/pull/251)

## [0.10.0] - 16.01.2025

### Added
- `ImplicitBPRWrapperModel` model with algorithm description in extended baselines tutorial ([#232](https://github.com/MobileTeleSystems/RecTools/pull/232), [#239](https://github.com/MobileTeleSystems/RecTools/pull/239))
- All vector models and `EASEModel` support for enabling ranking on GPU and selecting number of threads for CPU ranking. Added `recommend_n_threads` and `recommend_use_gpu_ranking` parameters to `EASEModel`, `ImplicitALSWrapperModel`, `ImplicitBPRWrapperModel`, `PureSVDModel` and `DSSMModel`. Added `recommend_use_gpu_ranking` to `LightFMWrapperModel`. GPU and CPU ranking may provide different ordering of items with identical scores in recommendation table, so this could change ordering items in recommendations since GPU ranking is now used as a default one. ([#218](https://github.com/MobileTeleSystems/RecTools/pull/218))

## [0.9.0] - 11.12.2024

### Added
- `from_config`, `get_config` and `get_params` methods to all models except neural-net-based ([#170](https://github.com/MobileTeleSystems/RecTools/pull/170))
- `fit_partial` implementation for `ImplicitALSWrapperModel` and `LightFMWrapperModel` ([#203](https://github.com/MobileTeleSystems/RecTools/pull/203), [#210](https://github.com/MobileTeleSystems/RecTools/pull/210), [#223](https://github.com/MobileTeleSystems/RecTools/pull/223))
- `save` and `load` methods to all of the models ([#206](https://github.com/MobileTeleSystems/RecTools/pull/206))
- Model configs example ([#207](https://github.com/MobileTeleSystems/RecTools/pull/207),[#219](https://github.com/MobileTeleSystems/RecTools/pull/219))
- `use_gpu` argument to `ImplicitRanker.rank` method ([#201](https://github.com/MobileTeleSystems/RecTools/pull/201))
- `keep_extra_cols` argument to `Dataset.construct` and `Interactions.from_raw` methods. `include_extra_cols` argument to `Dataset.get_raw_interactions` and `Interactions.to_external` methods ([#208](https://github.com/MobileTeleSystems/RecTools/pull/208))
- dtype adjustment to `recommend`, `recommend_to_items` methods of `ModelBase` ([#211](https://github.com/MobileTeleSystems/RecTools/pull/211))
- `load_model` function ([#213](https://github.com/MobileTeleSystems/RecTools/pull/213))
- `model_from_config` function ([#214](https://github.com/MobileTeleSystems/RecTools/pull/214))
- `get_cat_features` method to `SparseFeatures` ([#221](https://github.com/MobileTeleSystems/RecTools/pull/221))
- LightFM Python 3.12+ support ([#224](https://github.com/MobileTeleSystems/RecTools/pull/224))

### Fixed
- Implicit ALS matrix zero assignment size ([#228](https://github.com/MobileTeleSystems/RecTools/pull/228))

### Removed
- Python 3.8 support ([#222](https://github.com/MobileTeleSystems/RecTools/pull/222))


## [0.8.0] - 28.08.2024

### Added
- `Debias` mechanism for classification, ranking and auc metrics. New parameter `is_debiased` to `calc_from_confusion_df`, `calc_per_user_from_confusion_df` methods of classification metrics, `calc_from_fitted`, `calc_per_user_from_fitted` methods of auc and rankning (`MAP`) metrics, `calc_from_merged`, `calc_per_user_from_merged` methods of ranking (`NDCG`, `MRR`) metrics. ([#152](https://github.com/MobileTeleSystems/RecTools/pull/152))
- `nbformat >= 4.2.0` dependency to `[visuals]` extra ([#169](https://github.com/MobileTeleSystems/RecTools/pull/169))
- `filter_interactions` method of `Dataset` ([#177](https://github.com/MobileTeleSystems/RecTools/pull/177))
- `on_unsupported_targets` parameter to  `recommend` and `recommend_to_items` model methods ([#177](https://github.com/MobileTeleSystems/RecTools/pull/177))
- Use nmslib-metabrainz for Python 3.11 and upper ([#180](https://github.com/MobielTeleSystems/RecTools/pull/180))

### Fixed
- `display()` method in `MetricsApp` ([#169](https://github.com/MobileTeleSystems/RecTools/pull/169))
- `IntraListDiversity` metric computation in `cross_validate` ([#177](https://github.com/MobileTeleSystems/RecTools/pull/177))
- Allow warp-kos loss for LightFMWrapperModel ([#175](https://github.com/MobileTeleSystems/RecTools/pull/175))

### Removed
- [Breaking] `assume_external_ids` parameter in `recommend` and `recommend_to_items` model methods ([#177](https://github.com/MobileTeleSystems/RecTools/pull/177))

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
