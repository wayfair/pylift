.. role:: raw-latex(raw)
   :format: latex
..

Usage: evaluation
=================

Within the ``up`` object
------------------------

All curves can be plotted using

::

   up.plot(plot_type='qini')

Where ``plot_type`` can be any of the following values. In the formulaic representations

-  ``qini``: typical Qini curve (see Radcliffe 2007), except we normalize by the total number of people in treatment. The typical definition is

   .. math:: n_{t,1} - n_{c,1} N_t/N_c.

-  ``aqini``: adjusted Qini curve, calculated as

   .. math:: n_{t,1}/N_t - n_{c,1} n_t/(n_c N_t).

-  ``cuplift``: cumulative uplift curve, calculated as

   .. math:: n_{t,1}/n_t - n_{c,1}/n_c.

-  ``uplift``: typical uplift curve, calculated the same as cuplift but only returning the average value within the bin, rather than cumulatively.
-  ``cgains``: cumulative gains curve (see Gutierrez, Gerardy 2016), defined as

   .. math:: ((n_{t,1}/n_t - n_{c,1}/n_c)\phi.

-  ``balance``: ratio of treatment group size to total group size within each bin,

   .. math:: n_t/(n_c + n_t).

Above, :math:`\phi` corresponds to the fraction of individuals targeted – the x-axis of these curves. :math:`n` and :math:`N` correspond to counts up to :math:`phi` (except for the uplift curve, which is only within the bin at the :math:`phi` position) or within the entire group, respectively. The subscript :math:`t` indicates the treatment group, and :math:`c`, the control. The subscript :math:`1` indicates the subset of the count for which individuals had a positive outcome.

A number of scores are stored in both the ``test_results_`` and ``train_results_`` objects, containing scores calculated over the test set and train set, respectively. Namely, there are three important scores: \* ``Q``: unnormalized area between the qini curve and the random selection line. \* ``q1``: ``Q``, normalized by the theoretical maximum value of ``Q``. \* ``q2``: ``Q``, normalized by the practical maximum value of ``Q``.

Each of these can be accesses as attributes of ``test_results_`` or ``train_results_``. Either ``_qini``, ``_aqini``, or ``_cgains`` can be appended to obtain the same calculation for the qini curve, adjusted qini curve, or the cumulative gains curve, respectively. The score most unaffected by anomalous treatment/control ordering, without any bias to treatment or control (i.e. if you’re looking at lift between two equally viable treatments) is the ``q1_cgains`` score, but if you are looking at a simple treatment vs. control situation, ``q1_aqini`` is preferred.  Because this only really has meaning over an independent holdout [test] set, the most valuable value to access, then, would likely be ``up.test_results_.q1_aqini``.

::

   up.test_results_.q1_aqini # Over training set.

Maximal curves can also be toggled by passing flags into ``up.plot()``.
\* ``show_theoretical_max`` \* ``show_practical_max`` \*
``show_no_dogs`` \* ``show_random_selection``

Each of these curves satisfies shows the maximally attainable curve given different assumptions about the underlying data. The ``show_theoretical_max`` curve corresponds to a sorting in which we assume that an individual is persuadable (uplift = 1) if and only if they respond in the treatment group (and the same reasoning applies to the control group, for sleeping dogs). The ``show_practical_max`` curve assumes that all individuals that have a positive outcome in the treatment group must also have a counterpart (relative to the proportion of individuals in the treatment and control group) in the control group that did not respond. This is a more conservative, realistic curve. The former can only be attained through overfitting, while the latter can only be attained under very generous circumstances. Within the package, we also calculate the ``show_no_dogs`` curve, which simply precludes the possibility of negative effects.

The random selection line is shown by default, but the option to toggle it off is included in case you’d like to plot multiple plots on top of each other.

The below code plots the practical max over the aqini curve of a model contained in the TransformedOutcome object ``up``, then overlays the aqini curve of a second model contained in ``up1``, also changing the line color.

::

   ax = up.plot(show_practical_max=True, show_random_selection=False, label='Model 1')
   up1.plot(ax=ax, label='Model 2', color=[0.7,0,0])```

Error bars
~~~~~~~~~~

It is often useful to obtain error bars on your qini curves. We’ve implemented two ways to do this: 1. ``up.shuffle_fit()``: Seeds the ``train_test_split``, fit the model over the new training data, and evaluate on the new test data. Average these curves. 1.  ``up.noise_fit()``: Randomly shuffle the labels independently of the features and fit a model. This can help distinguish your evaluation curves from noise.

::

   up.shuffle_fit()
   up.plot(plot_type='aqini', show_shuffle_fits=True)

Adjustments can also be made to the aesthetics of these curves by passing in dictionaries that pass down to plot elements. For example, ``shuffle_band_kwargs`` is a dictionary of kwargs that modifies the ``fill_between`` shaded error bar region.

With ``UpliftEval``
-------------------

The ``UpliftEval`` class can also independently be used to apply the above evaluation visualizations and calculations. Note that the ``up`` object uses ``UpliftEval`` to generate the plots, so the ``UpliftEval`` class object for the train set and test set can be obtained in ``up.train_results_`` and ``up.test_results_``, respectively.

::

   from pylift.eval import UpliftEval
   upev = UpliftEval(treatment, outcome, predictions)
   upev.plot(plot_type='aqini')

It generally functions the same as the ``up.plot()`` function, except error bars cannot be obtained. Note that ``UpliftEval`` could still be used, however, to manually generate the curves that can be aggregated to make error bars.
