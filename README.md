# RecTools

[![Python versions](https://img.shields.io/pypi/pyversions/rectools.svg)](https://pypi.org/project/rectools)
[![PyPI](https://img.shields.io/pypi/v/rectools.svg)](https://pypi.org/project/rectools)
[![Docs](https://img.shields.io/github/workflow/status/MobileTeleSystems/RecTools/Publish?label=docs)](https://rectools.readthedocs.io)

[![License](https://img.shields.io/github/license/MobileTeleSystems/RecTools.svg)](https://github.com/MobileTeleSystems/RecTools/blob/main/LICENSE)
[![Coverage](https://img.shields.io/codecov/c/github/MobileTeleSystems/RecTools.svg)](https://app.codecov.io/gh/MobileTeleSystems/RecTools)
[![Tests](https://img.shields.io/github/workflow/status/MobileTeleSystems/RecTools/Test/main?label=tests)](https://github.com/MobileTeleSystems/RecTools/actions/workflows/test.yml?query=branch%3Amain++)

[![Contributors](https://img.shields.io/github/contributors/MobileTeleSystems/RecTools.svg)](https://github.com/MobileTeleSystems/RecTools/graphs/contributors)
[![Telegram](https://img.shields.io/badge/channel-telegram-blue)](https://t.me/RecTools_Support)

RecTools is an easy-to-use Python library which makes the process of building recommendation systems easier, 
faster and more structured than ever before.
It includes built-in in toolkits for data processing and metrics calculation, 
a variety of recommender models, some wrappers for already existing implementations of popular algorithms 
and model selection framework.
The aim is collecting ready-to-use solutions and best practices in one place to make processes 
of creating your first MVP and deploying model to production as fast and easy as possible.

RecTools allows to work with dense and sparse features easily.
There are a lot of useful features such as basic model which based on random suggestions or popularity, and more advanced, e.g. LightFM.
Also it contains a wide variety of metrics to choose from to better suit recommender system to your needs.

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


## Contribution

To install all requirements run
```
make install
```
You must have `python3` and `poetry` installed.

For autoformatting run 
```
make autoformat
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

## RecTools.Team

- [Emiliy Feldman](https://github.com/feldlime)
- [Ildar Safilo](https://github.com/irsafilo)
- [Daniil Potapov](https://github.com/sharthZ23) 
- [Igor Belkov](https://github.com/OzmundSedler)
- [Artem Senin](https://github.com/artemseninhse)
- [Alexander Butenko](https://github.com/iomallach)
- [Mikhail Khasykov](https://github.com/mkhasykov)
- [Daria Tikhonovich](https://github.com/blondered)
