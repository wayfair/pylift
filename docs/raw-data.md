# Raw data

Raw data for the `TransformedOutcome` method are stored as class attributes:
```
up.randomized_search_params
up.grid_search_params
up.transform                # Outcome transform function.
up.untransform              # Reverse of outcome transform function.

# Data (`y` in any of these can be replaced with `tc` for treatment or `x`).
up.transformed_y_train  # The predicted uplift.
up.y_train
up.y_test
up.y                    # All the `y` data.
up.df
up.df_train
up.df_test

# Once a model has been created...
up.model
up.model_final
up.frost_score_test (or train)
```

## Qini information

`up.test_results_` and `up.train_results_` are `UpliftEval` class
objects, and consequently contain all data about your qini curves, which can be
accessed as follows.

```
up.test_results_.qini_x  # percentile
up.test_results_.qini_y
# Best theoretical qini curve.
up.test_results_.qini_max_x  # percentile
up.test_results_.qini_max_y
# Uplift curve.
up.test_results_.uplift_x  # percentile
up.test_results_.uplift_y
```

`up.train_results_` can be used to plot the qini performance on the training
data, as follows: `up.train_results_.plot_qini()`.

