Raw data
========

Raw data and wrapped class objects for the ``TransformedOutcome`` method are stored as class attributes. The wrapped class objects are described in the `Usage: modeling <usage>`__ section.

Everything else, from processed data to the transformation functions can be accessed as listed below:

::

   up.randomized_search_params # Parameters that are used in `up.randomized_search()`
   up.grid_search_params       # Parameters that are used in `up.grid_search()`


   up.transform                # Outcome transform function.
   up.untransform              # Reverse of outcome transform function.

   # Data (`y` in any of these can be replaced with `tc` for treatment or `x`).
   up.transformed_y_train_pred  # The predicted uplift.
   up.transformed_y_train  # The transformed outcome.
   up.y_train
   up.y_test
   up.y                    # All the `y` data.
   up.df
   up.df_train
   up.df_test

   # Once a model has been created...
   up.model
   up.model_final
   up.Q_cgains # 'aqini' or 'qini' can be used in place of 'cgains'
   up.q1_cgains
   up.q2_cgains

Evaluation curve information
----------------------------

The raw data for all evaluation curves can be accessed within any ``UpliftEval`` object (``upev`` below):

::

   upev.PLOTTYPE_x  # percentile
   upev.PLOTTYPE_y

where the phrase ``PLOTTYPE`` can be replaced with any of the following: ``qini``, ``aqini``, ``cgains``, ``cuplift``, ``balance``, ``uplift``.  Because ``up.test_results_`` and ``up.train_results_`` are ``UpliftEval`` class objects, they can also be similarly accessed as shown above.

The theoretical maximum curves can also be extracted:

::

   # Overfitting theoretical maximal qini curve.
   upev.qini_max_x  # percentile
   upev.qini_max_y

   # "Practical" max curve.
   upev.qini_pmax_x
   upev.qini_pmax_y

   # No sleeping dogs curve.
   upev.qini_nosdmax_x
   upev.qini_nosdmax_y

``up.train_results_`` can be used to plot the qini performance on the training data, as follows: ``up.train_results_.plot_qini()``.
