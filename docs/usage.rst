Usage: modeling
===============

Instantiation
-------------

If you followed the `Quick start <evaluation>`__, you hopefully already
have a sense of how **pylift** is structured: the package is class-based
and so the entire modeling process takes place within instantiation of a
``TransformedOutcome`` class. This method in particular implements the
Transformed Outcome method, as described in `Introduction to
uplift <introduction>`__.

In particular, the ``TransformedOutcome`` class inherits from a
``BaseProxyMethod`` class, and only adds to said class a
``_transform_func`` and an ``_untransform_func`` which perform the
transformation to obtain :math:`Y^{*}` (the transformed outcome) from
:math:`Y` and :math:`W` (1 or 0 indicating the presence of a treatment) and
vice versa, respectively. Custom transformation methods are therefore
possible by explicitly providing the ``transform_func`` and
``untransform_func`` to ``BaseProxyMethod``.

Instantiation is accomplished as follows:

::

   up = TransformedOutcome(df, col_treatment='Treatment', col_outcome='Converted')

A number of custom parameters can be passed, which are all documented in
the docstring. Of particular note may be the ``stratify`` keyword
argument (whose argument is directly passed to
``sklearn.model_selection.train_test_split``).

The instantiation step accomplishes several things:

1. Define the transform function and transform the outcome (this is
   added to the dataframe you pass in, by default, as a new column,
   ``TransformedOutcome``).
2. Split the data using ``train_test_split``.
3. Set a random state (we like determinism!). This random state is used
   wherever possible.
4. Define an ``untransform`` function and use this to define a scoring
   function for hyperparameter tuning. The scoring function is saved
   within ``up.randomized_search_params`` and ``up.grid_search_params``,
   which are dictionaries that are used by default whenever
   ``up.randomized_search()`` or ``up.grid_search()`` are called.
5. Define some default hyperparameters.

Fit and hyperparameter tunings: passing custom parameters
---------------------------------------------------------

Anything that can be taken by ``RandomizedSearchCV()``,
``GridSearchCV()``, or ``Regressor()`` can be similarly passed to
``up.randomized_search``, ``up.grid_search``, or ``up.fit``,
respectively.

::

   up.fit(max_depth=2, nthread=-1)

``XGBRegressor`` is the default regressor, but a different ``Regressor``
object can also be used. To do this, pass the object to the keyword
argument ``sklearn_model`` during ``TransformedOutcome`` instantiation.

::

   up = TransformedOutcome(df, col_treatment='Test', col_outcome='Converted', sklearn_model=RandomForestRegressor)

   grid_search_params = {
       'estimator': RandomForestRegressor(),
       'param_grid': {'min_samples_split': [2,3,5,10,30,100,300,1000,3000,10000]},
       'verbose': True,
       'n_jobs': 35,
   }
   up.grid_search(**grid_search_params)

We tend to prefer ``xgboost``, however, as it tends to give favorable
results quickly, while also allowing the option for a custom objective
function. This extensibility allows for the possibility of an objective
function that takes into account :math:`P(W=1)` within each leaf, though we
because we have had mixed results with this approach, we have left the
package defaults as is.

Regardless of what regressor you use, the ``RandomizedSearchCV`` default
params are contained in ``up.randomized_search_params``, and the
``GridSearchCV`` params are located in ``up.grid_search_params``. These
can be manually replaced, but doing so will remove the scoring
functions, so it is highly recommended that any alterations to these
class attributes be done as an update, or that alterations be simply
passed as arguments to ``randomized_search`` or ``grid_search``, as
shown above.

Accessing sklearn objects
-------------------------

The class objects produced by the sklearn classes,
``RandomizedSearchCV``, ``GridSearchCV``, ``XGBRegressor``, etc. are
preserved in the ``TransformedOutcome`` class as class attributes.

``up.randomized_search`` -> ``up.rand_search_``

``up.grid_search`` -> ``up.grid_search_``

``up.fit`` -> ``up.model``

``up.fit(productionize=True)`` -> ``up.model_final``
