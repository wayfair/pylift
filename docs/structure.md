# Getting used to the structure

Because `pylift` is created as a wrapper around `sklearn`-style objects, it's generally possible to do anything you can do with `sklearn` within `pylift` as well.

## Passing custom parameters

Anything that can be taken by `RandomizedSearchCV()`, `GridSearchCV()`, or
`Regressor()` can be similarly passed to `up.randomized_search`,
`up.grid_search`, or `up.fit`, respectively.

```
up.fit(max_depth=2, nthread=-1)
```

`XGBRegressor` is the default regressor, but a different `Regressor` object can
also be used. To do this, pass the object to the keyword argument
`sklearn_model` during `TransformedOutcome` instantiation.

```
up = TransformedOutcome(df, col_treatment='Test', col_outcome='Converted', sklearn_model=RandomForestRegressor)

grid_search_params = {
    'estimator': RandomForestRegressor(),
    'param_grid': {'min_samples_split': [2,3,5,10,30,100,300,1000,3000,10000]},
    'verbose': True,
    'n_jobs': 35,
}
up.grid_search(**grid_search_params)
```

Regardless of what regressor you use, the `RandomizedSearchCV` default params
are contained in `up.randomized_search_params`, and the `GridSearchCV` params
are located in `up.grid_search_params`. These can be manually replaced, but
doing so will remove the scoring functions, so it is highly recommended that
any alterations to these class attributes be done as an update, or that
alterations be simply passed as arguments to `randomized_search` or
`grid_search`, as shown above.


## Accessing sklearn objects

The class objects produced by the sklearn classes, `RandomizedSearchCV`,
`GridSearchCV`, `XGBRegressor`, etc. are preserved in the `TransformedOutcome`
class as class attributes.

`up.randomized_search` -> `up.rand_search_`

`up.grid_search` -> `up.grid_search_`

`up.fit` -> `up.model`

`up.fit(productionize=True)` -> `up.model_final`

~
