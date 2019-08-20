# Quick start
To start, you simply need a `pandas.DataFrame` with a treatment column of 0s
and 1s (0 for control, 1 for test) and a outcome column of 0s and 1s.
Implementation can be as simple as follows:

```
from pylift import TransformedOutcome
up = TransformedOutcome(df1, col_treatment='Treatment', col_outcome='Converted')

up.randomized_search()
up.fit(**up.rand_search_.best_params_)

up.plot(plot_type='aqini', show_theoretical_max=True)
print(up.test_results_.Q_aqini)
```

`up.fit()` can also be passed a flag `productionize=True`, which when `True`
will create a productionizable model trained over the entire data set, stored
in `self.model_final` (though it is contentious whether it's safe to use a
model that has not been evaluated in production -- if you have enough data, it
may be prudent not to). This can then be pickled with
`self.model_final.to_pickle(PATH)`, as usually done with `sklearn`-style
models.

A fully-fledged example of **pylift** in action can be found in
`pylift/examples/simulated_data/sample.ipynb`.
