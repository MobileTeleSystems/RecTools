# RecTools

[![Python versions](https://img.shields.io/pypi/pyversions/rectools.svg)](https://pypi.org/project/rectools)
[![PyPI](https://img.shields.io/pypi/v/rectools.svg)](https://pypi.org/project/rectools)
[![Docs](https://img.shields.io/github/actions/workflow/status/MobileTeleSystems/RecTools/publish.yml?label=docs)](https://rectools.readthedocs.io)

[![License](https://img.shields.io/github/license/MobileTeleSystems/RecTools.svg)](https://github.com/MobileTeleSystems/RecTools/blob/main/LICENSE)
[![Coverage](https://img.shields.io/codecov/c/github/MobileTeleSystems/RecTools.svg)](https://app.codecov.io/gh/MobileTeleSystems/RecTools)
[![Tests](https://img.shields.io/github/actions/workflow/status/MobileTeleSystems/RecTools/test.yml?branch=main&label=tests)](https://github.com/MobileTeleSystems/RecTools/actions/workflows/test.yml?query=branch%3Amain++)

[![Contributors](https://img.shields.io/github/contributors/MobileTeleSystems/RecTools.svg)](https://github.com/MobileTeleSystems/RecTools/graphs/contributors)
[![Downloads](https://static.pepy.tech/badge/rectools)](https://pepy.tech/project/rectools)
[![Telegram](https://img.shields.io/badge/channel-telegram-blue)](https://t.me/RecTools_Support)

<p align="center">
  <a href="https://rectools.readthedocs.io/en/stable/">Documentation</a> |
  <a href="https://github.com/MobileTeleSystems/RecTools/tree/main/examples">Examples</a> |
    <a href="https://github.com/MobileTeleSystems/RecTools/tree/main/examples/tutorials">Tutorials</a> |
  <a href="https://github.com/MobileTeleSystems/RecTools/blob/main/CONTRIBUTING.rst">Contributing</a> |
  <a href="https://github.com/MobileTeleSystems/RecTools/releases">Releases</a> |
  <a href="https://github.com/orgs/MobileTeleSystems/projects/1">Developers Board</a>
</p>

RecTools is an easy-to-use Python library which makes the process of building recommender systems easier and
faster than ever before.

## ✨ Highlights: HSTU model released! ✨

**HSTU arhictecture from ["Actions speak louder then words..."](https://arxiv.org/abs/2402.17152) is now available in RecTools as `HSTUModel`:**
- Fully compatible with our  `fit` / `recommend` paradigm and require NO special data processing
- Supports context-aware recommendations in case Relative Time Bias is enabled
- Supports all loss options, item embedding options, category features utilization and other common modular functionality of RecTools transformer models
- In [HSTU tutorial](examples/tutorials/transformers_HSTU_tutorial.ipynb) we show that original metrics reported for HSTU on public Movielens datasets may actually be **underestimated**
- Configurable, customizable, callback-friendly, checkpoints-included, logs-out-of-the-box, custom-validation-ready, multi-gpu-compatible! See  [Transformers Advanced Training User Guide](examples/tutorials/transformers_advanced_training_guide.ipynb) and [Transformers Customization Guide](examples/tutorials/transformers_customization_guide.ipynb)


## ✨ Highlights: RecTools framework at ACM RecSys'25 ✨

**RecTools implementations are featured in ACM RecSys'25: ["eSASRec: Enhancing Transformer-based Recommendations in a Modular Fashion"](https://www.arxiv.org/abs/2508.06450):**
- The article presents a systematic benchmark of Transformer modifications using RecTools models. It offers a detailed evaluation of training objectives, Transformer architectures, loss functions, and negative sampling strategies in realistic, production-like settings
- We introduce a new SOTA baseline, **eSASRec**, which combines SASRec’s training objective with LiGR Transformer layers and Sampled Softmax loss, forming a simple yet powerful recipe
- **eSASRec** shows 23% boost over SOTA models, such as ActionPiece, on academic benchmarks
- [LiGR](https://arxiv.org/pdf/2502.03417) Transformer layers used in **eSASRec** are now in RecTools

Plase note that we always compare the quality of our implementations to academic papers results. [Public benchmarks for transformer models SASRec and BERT4Rec](https://github.com/blondered/bert4rec_repro?tab=readme-ov-file#rectools-transformers-benchmark-results) show that RecTools implementations achieve highest scores on multiple datasets compared to other published results.


## Get started

Prepare data with

```shell
wget https://files.grouplens.org/datasets/movielens/ml-1m.zip
unzip ml-1m.zip
```

```python
import pandas as pd
    
from rectools import Columns
from rectools.dataset import Dataset
from rectools.models import SASRecModel

# Read the data
ratings = pd.read_csv(
    "ml-1m/ratings.dat", 
    sep="::",
    engine="python",  # Because of 2-chars separators
    header=None,
    names=[Columns.User, Columns.Item, Columns.Weight, Columns.Datetime],
)
    
# Create dataset
dataset = Dataset.construct(ratings)
    
# Fit model
model = SASRecModel(n_factors=64, epochs=100, loss="sampled_softmax")
model.fit(dataset)

# Make recommendations
recos = model.recommend(
    users=ratings[Columns.User].unique(),
    dataset=dataset,
    k=10,
    filter_viewed=True,
)
```

## Installation

RecTools is on PyPI, so you can use `pip` to install it.
```
pip install rectools
```
The default version doesn't contain all the dependencies, because some of them are needed only for specific functionality. Available user extensions are the following:

- `lightfm`: adds wrapper for LightFM model,
- `torch`: adds models based on neural nets,
- `visuals`: adds visualization tools,
- `nmslib`: adds fast ANN recommenders.

Install extension:
```
pip install rectools[extension-name]
```

Install all extensions:
```
pip install rectools[all]
```


## Recommender Models
The table below lists recommender models that are available in RecTools.  

| Model               | Type | Description (🎏 for user/item features, 🔆 for warm inference, ❄️ for cold inference support)                                                                               | Tutorials & Benchmarks |
|---------------------|----|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------|
| HSTU                | Neural Network | `rectools.models.HSTUModel` - Sequential model with unidirectional pointwise aggregated attention mechanism, incorporating relative attention bias from positional and temporal information, introduced in ["Actions speak louder then words..."](https://arxiv.org/pdf/2402.17152), combined with "Shifted Sequence" training objective as in original public benchmarks<br>🎏                                    | 📓 [HSTU Theory & Practice](examples/tutorials/transformers_HSTU_tutorial.ipynb) <br>  📕 [Transformers Theory & Practice](examples/tutorials/transformers_tutorial.ipynb)<br>  📗 [Advanced training guide](examples/tutorials/transformers_advanced_training_guide.ipynb) <br> 🚀 [Top performance on public datasets](examples/tutorials/transformers_HSTU_tutorial.ipynb)
| SASRec              | Neural Network | `rectools.models.SASRecModel` - Transformer-based sequential model with unidirectional attention mechanism and "Shifted Sequence" training objective. <br> For eSASRec variant specify `rectools.models.nn.transformers.ligr.LiGRLayers` for `transformer_layers_type` and `sampled_softmax` for `loss` <br>🎏                 | 📕 [Transformers Theory & Practice](examples/tutorials/transformers_tutorial.ipynb)<br>  📗 [Advanced training guide](examples/tutorials/transformers_advanced_training_guide.ipynb) <br>  📘 [Customization guide](examples/tutorials/transformers_customization_guide.ipynb) <br> 🚀 [Top performance on public benchmarks](https://github.com/blondered/bert4rec_repro?tab=readme-ov-file#rectools-transformers-benchmark-results) |
| BERT4Rec            | Neural Network | `rectools.models.BERT4RecModel` - Transformer-based sequential model with bidirectional attention mechanism and "MLM" (masked item) training objective <br>🎏               | 📕 [Transformers Theory & Practice](examples/tutorials/transformers_tutorial.ipynb)<br>  📗 [Advanced training guide](examples/tutorials/transformers_advanced_training_guide.ipynb) <br>  📘 [Customization guide](examples/tutorials/transformers_customization_guide.ipynb) <br> 🚀 [Top performance on public benchmarks](https://github.com/blondered/bert4rec_repro?tab=readme-ov-file#rectools-transformers-benchmark-results) |
| [implicit](https://github.com/benfred/implicit) ALS Wrapper | Matrix Factorization | `rectools.models.ImplicitALSWrapperModel` - Alternating Least Squares Matrix Factorizattion algorithm for implicit feedback. <br>🎏                                         | 📙 [Theory & Practice](https://rectools.readthedocs.io/en/latest/examples/tutorials/baselines_extended_tutorial.html#Implicit-ALS)<br> 🚀 [50% boost to metrics with user & item features](examples/5_benchmark_iALS_with_features.ipynb) |
| [implicit](https://github.com/benfred/implicit) BPR-MF Wrapper | Matrix Factorization | `rectools.models.ImplicitBPRWrapperModel` - Bayesian Personalized Ranking Matrix Factorization algorithm.                                                                   | 📙 [Theory & Practice](https://rectools.readthedocs.io/en/latest/examples/tutorials/baselines_extended_tutorial.html#Bayesian-Personalized-Ranking-Matrix-Factorization-(BPR-MF)) |
| [implicit](https://github.com/benfred/implicit) ItemKNN Wrapper | Nearest Neighbours | `rectools.models.ImplicitItemKNNWrapperModel` - Algorithm that calculates item-item similarity matrix using distances between item vectors in user-item interactions matrix | 📙 [Theory & Practice](https://rectools.readthedocs.io/en/latest/examples/tutorials/baselines_extended_tutorial.html#ItemKNN) |
| [LightFM](https://github.com/lyst/lightfm) Wrapper | Matrix Factorization | `rectools.models.LightFMWrapperModel` - Hybrid matrix factorization algorithm which utilises user and item features and supports a variety of losses.<br>🎏 🔆 ❄️           | 📙 [Theory & Practice](https://rectools.readthedocs.io/en/latest/examples/tutorials/baselines_extended_tutorial.html#LightFM)<br>🚀 [10-25 times faster inference with RecTools](examples/6_benchmark_lightfm_inference.ipynb)|
| EASE                | Linear Autoencoder | `rectools.models.EASEModel` - Embarassingly Shallow Autoencoders implementation that explicitly calculates dense item-item similarity matrix                                | 📙 [Theory & Practice](https://rectools.readthedocs.io/en/latest/examples/tutorials/baselines_extended_tutorial.html#EASE) |
| PureSVD             | Matrix Factorization | `rectools.models.PureSVDModel` - Truncated Singular Value Decomposition of user-item interactions matrix                                                                    | 📙 [Theory & Practice](https://rectools.readthedocs.io/en/latest/examples/tutorials/baselines_extended_tutorial.html#PureSVD) |
| DSSM                | Neural Network | `rectools.models.DSSMModel` - Two-tower Neural model that learns user and item embeddings utilising their explicit features and learning on triplet loss.<br>🎏 🔆          | - |
| Popular             | Heuristic | `rectools.models.PopularModel` - Classic baseline which computes popularity of items and also accepts params like time window and type of popularity computation.<br>❄️     | - |
| Popular in Category | Heuristic | `rectools.models.PopularInCategoryModel` - Model that computes poularity within category and applies mixing strategy to increase Diversity.<br>❄️                           | - |
| Random              |  Heuristic | `rectools.models.RandomModel` - Simple random algorithm useful to benchmark Novelty, Coverage, etc.<br>❄️                                                                   | - |

- All of the models follow the same interface. **No exceptions**
- No need for manual creation of sparse matrixes, torch dataloaders or mapping ids. Preparing data for models is as simple as `dataset = Dataset.construct(interactions_df)`
- Fitting any model is as simple as `model.fit(dataset)`
- For getting recommendations `filter_viewed` and `items_to_recommend` options are available
- For item-to-item recommendations use `recommend_to_items` method
- For feeding user/item features to model just specify dataframes when constructing `Dataset`. [Check our example](examples/4_dataset_with_features.ipynb)
- For warm / cold inference just provide all required ids in `users` or `target_items` parameters of `recommend` or `recommend_to_items` methods and make sure you have features in the dataset for warm users/items. **Nothing else is needed, everything works out of the box.**
- Our models can be initialized from configs and have useful methods like `get_config`, `get_params`, `save`, `load`. Common functions `model_from_config`, `model_from_params` and `load_model` are available. [Check our example](examples/9_model_configs_and_saving.ipynb)


## Extended validation tools

### `calc_metrics` for classification, ranking, "beyond-accuracy", DQ, popularity bias and between-model metrics


[User guide](https://github.com/MobileTeleSystems/RecTools/blob/main/examples/3_metrics.ipynb) | [Documentation](https://rectools.readthedocs.io/en/stable/features.html#metrics)


### `DebiasConfig` for debiased metrics calculation

[User guide](https://github.com/MobileTeleSystems/RecTools/blob/main/examples/8_debiased_metrics.ipynb) | [Documentation](https://rectools.readthedocs.io/en/stable/api/rectools.metrics.debias.DebiasConfig.html)

### `cross_validate` for model metrics comparison


[User guide](https://github.com/MobileTeleSystems/RecTools/blob/main/examples/2_cross_validation.ipynb) 


### `VisualApp` for model recommendations comparison

<img src="https://recsysart.ru/images/visual_app.gif" width=500>


[Example](https://github.com/MobileTeleSystems/RecTools/blob/main/examples/7_visualization.ipynb) | [Demo](https://recsysart.ru/voila/) | [Documentation](https://rectools.readthedocs.io/en/stable/api/rectools.visuals.visual_app.VisualApp.html)



### `MetricsApp` for metrics trade-off analysis


<img src="https://recsysart.ru/images/metrics_app.gif" width=600>

[Example](https://github.com/MobileTeleSystems/RecTools/blob/main/examples/2_cross_validation.ipynb) |
[Documentation](https://rectools.readthedocs.io/en/stable/api/rectools.visuals.metrics_app.MetricsApp.html)


## Contribution
[Contributing guide](CONTRIBUTING.rst)

To install all requirements
- you must have `python3` and `poetry` installed
- make sure you have no active virtual environments (deactivate conda `base` if applicable)
- run
```
make install
```


For autoformatting run 
```
make format
```

For linters check run 
```
make lint
```

For tests run 
```
make test
```

For coverage run 
```
make coverage
```

To remove virtual environment run
```
make clean
```

## RecTools Team

- [Emiliy Feldman](https://github.com/feldlime) [Maintainer]
- [Daria Tikhonovich](https://github.com/blondered) [Maintainer]
- [Andrey Semenov](https://github.com/In48semenov)
- [Mike Sokolov](https://github.com/mikesokolovv)
- [Maya Spirina](https://github.com/spirinamayya)
- [Grigoriy Gusarov](https://github.com/Gooogr)
- [Aki Ariga](https://github.com/chezou)
- [Nikolay Undalov](https://github.com/nsundalov)
- [Aleksey Kuzin](https://github.com/teodor-r)

Previous contributors: [Ildar Safilo](https://github.com/irsafilo) [ex-Maintainer], [Daniil Potapov](https://github.com/sharthZ23) [ex-Maintainer], [Alexander Butenko](https://github.com/iomallach), [Igor Belkov](https://github.com/OzmundSedler), [Artem Senin](https://github.com/artemseninhse), [Mikhail Khasykov](https://github.com/mkhasykov), [Julia Karamnova](https://github.com/JuliaKup), [Maxim Lukin](https://github.com/groundmax), [Yuri Ulianov](https://github.com/yukeeul), [Egor Kratkov](https://github.com/jegorus), [Azat Sibagatulin](https://github.com/azatnv), [Vadim Vetrov](https://github.com/Waujito)

