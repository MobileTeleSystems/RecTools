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


Recommendation Table
~~~~~~~~~~~~~~~~~~~~
Recommendation table contains recommendations for each user.
It has a fixed set of columns, though they are different for i2i and u2i recommendations.
Recommendation table can also be used for calculation of metrics.


.. include:: models.rst

What are you waiting for? Train and apply them!


.. include:: metrics.rst

Oops, yeah, can't forget about them.


.. include:: model_selection.rst


.. include:: tools.rst


.. include:: visuals.rst
