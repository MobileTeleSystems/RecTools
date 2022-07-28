FAQ
===

.. currentmodule:: rectools

1. What kind of features should I use: Dense or Sparse?
    It depends. In most cases you're better off using `SparseFeatures` because they are better suited
    to categorical features. Even if you have a feature with real numerical values you're often better off
    if you binarize or discretize it. But there are exceptions to this rule, e.g. ALS features.

2. How do I calculate several metrics at once?
    Use function `calc_metrics`. It allows to calculate a batch of metrics more efficiently.
    It's similar to `reports` from `sklearn`.

3. What is the benefit of model wrappers?
    They all have the same set of parameters allowing for easier usage.
    They also provide extension of existing functionality, such as allowing to filters to eliminate items
    that has already been seen, whitelist, features in ALS, I2I. Wrappers have unified interface of output
    that is easy to use as input to calculate metrics. They also allowed to speed up performance of some models.

4. What is the benefit of using `Dataset`?
    It's an easy-to-use wrapping of interactions, features and mapping between item and user ids in feature sets and
    those in interaction matrix.

5. Why do I need to pass `Dataset` object as an argument to method `recommend`?
    It conceals mapping between internal and external user and item ids. Additionally it allows to filter out items
    that users have already seen. Some models, such as `LightFM` or `DSSM`, require to pass features.

6. Should the same `Dataset` object be used for fitting of a model and for inference of recommendations?
    It almost always has to be exactly the same `Dataset` object.

    One of possible exceptions is if during the fitting stage you use both viewing and purchase of an item
    as a positive event but you want exempt an item from being recommended only if it was purchased.
    In this case you should pass all interactions to train a model and only purchases to infer recommendations.

    Another exception is if a model requires to pass features to infer recommendations and values of those features
    have changed.
