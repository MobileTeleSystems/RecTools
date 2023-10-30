.. highlight:: console
.. highlight:: python

Welcome to RecTools's documentation!
====================================

RecTools is an easy-to-use Python library which makes the process of building recommendation systems easier,
faster and more structured than ever before. The aim is to collect ready-to-use solutions and best practices in one place to make processes
of creating your first MVP and deploying model to production as fast and easy as possible.
The package also includes useful tools, such as ANN indexes for vector models and fast metric calculation.

Quick Start
-----------

Download data.

.. code-block:: bash

    $ wget https://files.grouplens.org/datasets/movielens/ml-1m.zip
    $ unzip ml-1m.zip

Train model and infer recommendations.

.. code-block:: python

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

Installation
------------
PyPI
~~~~
Install from PyPi using pip

.. code-block:: bash

    $ pip install rectools

RecTools is compatible with all operating systems and with Python 3.7+.
The default version doesn't contain all the dependencies. Optional dependencies are the following:

lightfm: adds wrapper for LightFM model,
torch: adds models based on neural nets,
nmslib: adds fast ANN recommenders.
all: all extra dependencies

Install RecTools with selected dependencies:

.. code-block:: bash

    $ pip install rectools[lightfm,torch]

Why RecTools?
-------------
The one, the only and the best.

RecTools provides unified interface for most commonly used recommender models. They include Implicit ALS, Implicit KNN,
LightFM, SVD and DSSM. Recommendations based on popularity and random are also possible.
For model validation, RecTools contains implementation of time split methodology and numerous metrics
to evaluate model's performance. As well as basic ones they also include Diversity, Novelty and Serendipity.
The package also provides tools that allow to evaluate metrics as easy and as fast as possible.

.. toctree::
   :hidden:
   :caption: Table of Contents
   :titlesonly:
   :maxdepth: 2

   features
   api
   tutorials
   benchmarks
   faq
   support
