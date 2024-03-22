# RecTools

[![Python versions](https://img.shields.io/pypi/pyversions/rectools.svg)](https://pypi.org/project/rectools)
[![PyPI](https://img.shields.io/pypi/v/rectools.svg)](https://pypi.org/project/rectools)
[![Docs](https://img.shields.io/github/actions/workflow/status/MobileTeleSystems/RecTools/publish.yml?label=docs)](https://rectools.readthedocs.io)

[![License](https://img.shields.io/github/license/MobileTeleSystems/RecTools.svg)](https://github.com/MobileTeleSystems/RecTools/blob/main/LICENSE)
[![Coverage](https://img.shields.io/codecov/c/github/MobileTeleSystems/RecTools.svg)](https://app.codecov.io/gh/MobileTeleSystems/RecTools)
[![Tests](https://img.shields.io/github/actions/workflow/status/MobileTeleSystems/RecTools/test.yml?branch=main&label=tests)](https://github.com/MobileTeleSystems/RecTools/actions/workflows/test.yml?query=branch%3Amain++)

[![Contributors](https://img.shields.io/github/contributors/MobileTeleSystems/RecTools.svg)](https://github.com/MobileTeleSystems/RecTools/graphs/contributors)
[![Telegram](https://img.shields.io/badge/channel-telegram-blue)](https://t.me/RecTools_Support)

RecTools is an easy-to-use Python library which makes the process of building recommendation systems easier, 
faster and more structured than ever before.
It includes built-in toolkits for data processing and metrics calculation, 
a variety of recommender models, some wrappers for already existing implementations of popular algorithms 
and model selection framework.
The aim is to collect ready-to-use solutions and best practices in one place to make processes 
of creating your first MVP and deploying model to production as fast and easy as possible.

For more details, see the [Documentation](https://rectools.readthedocs.io/) 
and [Tutorials](https://github.com/MobileTeleSystems/RecTools/tree/main/examples).

## Get started

Prepare data with

```shell
wget https://files.grouplens.org/datasets/movielens/ml-1m.zip
unzip ml-1m.zip
```

```python
import pandas as pd
from implicit.nearest_neighbours import TFIDFRecommender
    
from rectools import Columns
from rectools.dataset import Dataset
from rectools.models import ImplicitItemKNNWrapperModel

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
model = ImplicitItemKNNWrapperModel(TFIDFRecommender(K=10))
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
The default version doesn't contain all the dependencies, because some of them are needed only for specific models. Available user extensions are the following:

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

**Important:** If you're using `poetry` and you want to add `rectools` to your project, then you should either install `rectools` without `lightfm` extras or use `poetry==1.4.0` and add to your `poetry.toml` file the next lines:
```
[experimental]
new-installer = false
```

## Recommender Models
The table below lists recommender models that are available in RecTools. 

| Model | Type | Description | Extra features |
|----|----|-----------|--------|
| [implicit](https://github.com/benfred/implicit) ALS Wrapper | Matrix Factorization | `rectools.models.ImplicitALSWrapperModel` - Alternating Least Squares Matrix Factorizattion algorithm for implicit feedback | Support for user/item features! [Check our boost to metrics](examples/5_benchmark_iALS_with_features.ipynb) |
| [implicit](https://github.com/benfred/implicit) ItemKNN Wrapper | Collaborative Filtering | `rectools.models.ImplicitItemKNNWrapperModel` - Algorithm that calculates item-item similarity matrix using distances between item vectors in user-item interactions matrix | - |
| [LightFM](https://github.com/lyst/lightfm) Wrapper | Matrix Factorization | `rectools.models.LightFMWrapperModel` - Hybrid matrix factorization algorithm which utilises user and item features and supports a variety of losses | 10-25 times faster inference! [Check our boost to inference](examples/6_benchmark_lightfm_inference.ipynb)|
| EASE | Collaborative Filtering | `rectools.models.EASEModel` - Embarassingly Shallow Autoencoders implementation that explicitly calculates dense item-item similarity matrix | - |
| PureSVD | Matrix Factorization | `rectools.models.PureSVDModel` - Truncated Singular Value Decomposition of user-item interactions matrix | - |
| DSSM | Neural Network | `rectools.models.DSSMModel` - Two-tower Neural model that learns user and item embeddings utilising their explicit features and learning on triplet loss | - |
| Popular | Heuristic | `rectools.models.PopularModel` - Classic baseline which computes popularity of items | Hyperparams (time window, pop computation) |
| Popular in Category | Heuristic |  `rectools.models.PopularInCategoryModel` - Model that computes poularity within category and applies mixing strategy to increase Diversity | Hyperparams (time window, pop computation, mixing/ratio strategy) |
| Random |  Heuristic | `rectools.models.RandomModel` - Simple random algorithm useful to benchmark Novelty, Coverage, etc.  | - |

- All of the models follow the same interface. **No exceptions**
- No need for manual creation of sparse matrixes or mapping ids. Preparing data for models is as simple as `dataset = Dataset.construct(interactions_df)`
- Fitting any model is as simple as `model.fit(dataset)`
- For getting recommendations `filter_viewed` and `items_to_recommend` options are available
- For item-to-item recommendations use `recommend_to_items` method
- For feeding user/item features to model just specify dataframes when constructing `Dataset`. [Check our tutorial](examples/4_dataset_with_features.ipynb)


## Contribution
[Contributing guide](CONTRIBUTING.rst)

To install all requirements
- you must have `python3` and `poetry==1.4.0` installed
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
- [Alexander Butenko](https://github.com/iomallach)
- [Andrey Semenov](https://github.com/In48semenov)
- [Mike Sokolov](https://github.com/mikesokolovv)

Previous contributors: [Ildar Safilo](https://github.com/irsafilo) [ex-Maintainer], [Daniil Potapov](https://github.com/sharthZ23) [ex-Maintainer], [Igor Belkov](https://github.com/OzmundSedler), [Artem Senin](https://github.com/artemseninhse), [Mikhail Khasykov](https://github.com/mkhasykov), [Julia Karamnova](https://github.com/JuliaKup) 

