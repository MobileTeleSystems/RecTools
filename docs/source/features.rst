Components
==========

.. currentmodule:: rectools

Basic Concepts
--------------

Columns
~~~~~~~
Names of columns are fixed. They are `user_id`, `item_id`, `weight` (numerical value of interaction's importance),
`datetime` (date and time of interaction), `rank` (rank of recommendation according to score)
and `score` (numeric value estimating how good recommendation it is).
Column names are fixed in order to not constantly require mapping of columns in data and their actual meaning.
So you'll need to rename your columns.

.. currentmodule:: rectools.columns

.. moduleautosummary::
   :toctree: api/
   :template: custom-module-template.rst
   :recursive:

   rectools.columns

Identifiers
~~~~~~~~~~~
Mappings of external identifiers of users or items to internal ones.
Recommendation systems always require to have a mapping between external item ids in data sources
and internal ids in interaction matrix. Managing such mapping requires a lot of diligence. RecTools does it for you.
Every user and item must have a unique id.
External ids may be any unique hashable values, internal - always integers from ``0`` to ``n_objects-1``.

Interactions
~~~~~~~~~~~~
This table stores history of interactions between users and items. It carries the most importance.
Interactions table might also contain column describing importance of an interaction. Also timestamp of interaction.
If no such column is provided, all interactions are assumed to be of equal importance.

User Features
~~~~~~~~~~~~~
This table stores data about users.
It might include age, gender or any other features which may prove to be important for a recommender model.

Item Features
~~~~~~~~~~~~~
This table stores data about items.
It might include category, price or any other features which may prove to be important for a recommender model.

All of the above concepts are combined in `Dataset`.
`Dataset` is used to build recommendation models and infer recommendations.

.. include:: dataset.rst


Hot, warm, cold
~~~~~~~~~~~~~~~
There is a concept of a temperature we're using for users and items:

* **hot** - the ones that are present in interactions used for training (they may or may not have features);
* **warm** - the ones that are not in interactions, but have some features;
* **cold** - the ones we don't know anything about (they are not in interactions and don't have any features).

All the models are able to generate recommendations for the *hot* users (items).  
But as for warm and cold ones, there may be all possible combinations (neither of them, only cold, only warm, both).  
The important thing is that if model is able to recommend for cold users (items), but not for warm ones (see table below), 
it is still able to recommend for warm ones, but they will be considered as cold (no personalisation should be expected).

.. include:: models.rst

What are you waiting for? Train and apply them!

Recommendation Table
~~~~~~~~~~~~~~~~~~~~
Recommendation table contains recommendations for each user.
It has a fixed set of columns, though they are different for i2i and u2i recommendations.
Recommendation table can also be used for calculation of metrics.


.. include:: metrics.rst

Oops, yeah, can't forget about them.


.. include:: model_selection.rst


.. include:: tools.rst


.. include:: visuals.rst
