import copy
import functools
import numpy as np
import pandas as pd
import warnings

from scipy.stats import uniform
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split
from xgboost import XGBRegressor

from ..eval import _plot_defaults, get_scores, UpliftEval
from ..explore.base import _add_bins, _NWOE, _NIV, _NIV_bootstrap, _plot_NWOE_bins, _plot_NIV_bs

class BaseProxyMethod:
    """Provide common functionalities for all label transformation methods.

    Requires an input function `transform_func` that transforms `treatment` and
    `outcome` into a single `transformed_outcome`. This is typically the TOT
    transformation, but can be whatever you want.

    Also complete a number of tasks that enable use of the proxy method: save
    dataframe and important dataframe column names to class object, calculate
    the transformed outcome, create an `untransform` method that undoes
    `transform`.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the features, test/control flag, and outcome.
    transform_func : function
        Function that takes two keyword arguments, `treatment` and
        `outcome`, and outputs a transformed outcome.
    untransform_func : function
        Function that inverts `transform_func`.
    col_treatment : string, optional
        Name of the treatment column. Depends on input dataframe.
    col_outcome : string, optional
        Name of the original outcome column. Depends on input dataframe.
    col_transformed_outcome : string, optional
        Name of the new, transformed outcome. Can be whatever you want.
    col_policy : string or float, optional
        Name of the column that indicates treatment policy (probability of
        treatment). If a float is given, the treatment policy is assumed to be
        even across all rows. If not given, it is assumed that application of
        treatment was randomly assigned with the same probability across the
        entire population.
    continuous_outcome : "infer" or Bool, optional
        Flag that indicates whether or not the Outcome column is continuous.
        Inferred by default.
    random_state : int
        Random seed for deterministic behavior.
    test_size : float
        test_size parameter for skleran.metrics.train_test_split.
    stratify : column name or anything that can be passed to parameter of same
    name in train_test_split, optional
        If not None, stratify is used as input into train_test_split.
    sklearn_model : scikit-learn regressor
        Model used for grid searching and fitting.

    """

    def _score(self, y_true, y_pred, method, plot_type, score_name):
        """ scoring function to be passed to make_scorer.

        """
        treatment_true, outcome_true, p = self.untransform(y_true)
        scores = get_scores(treatment_true, outcome_true, y_pred, p, scoring_range=(0,self.scoring_cutoff[method]), plot_type=plot_type)
        return scores[score_name]

    def __init__(self, df, transform_func, untransform_func, col_treatment='Treatment', col_outcome='Outcome', col_transformed_outcome='TransformedOutcome', col_policy=None, continuous_outcome='infer', random_state=2701, test_size=0.2, stratify=None, scoring_cutoff=1, scoring_method='aqini', sklearn_model=XGBRegressor):

        # ACCOUNT FOR MODULARITY.
        # To allow for modular behavior, different types can be allowed for a
        # number of the above keyword arguments. Below, the snippets of code
        # that account for this modularity are labeled as AFMB(#), which stands
        # for Account for Modular Behavior.

        # Explicitly define some flags that infer the state specified by the
        # modular inputs. I.e. get the `type` and interpret what it means.
        train_test_split_specified = type(df) == tuple
        self.individual_policy_given = type(col_policy) == str

        # AFMB(1): Is train test split specified?
        # If train test split is specified, combine and create a single dataframe.
        if train_test_split_specified:
            df_train = df[0]
            df_test = df[1]
            df = pd.concat([df_train, df_test], keys=[0,1])

        # AFMB(2): Is a policy for each individual? If so, is it a row-level
        # specification?
        # Deal with possible inputs and raise exception if not acceptable.
        if not col_policy:
            treatment = df[col_treatment]
            self.p = len(treatment[treatment==1])/len(treatment)
        elif self.individual_policy_given:
            self.p = df[col_policy]
        elif type(col_policy) == float:  # In the case that it is a numerical value.
            self.p = col_policy
        else:
            raise Exception('col_policy must be a str (column name), float (probability of treatment), or None.')

        # AFMB(3): Is `df[col_outcome]` a continuous variable?
        # By default, this is inferred from the number of distinct values of
        # the Outcome column.
        if continuous_outcome == 'infer':
            num_outcomes = df[col_outcome].nunique()
            continuous_outcome = num_outcomes > 2  # Assign True or False depending on the number of outcomes.

        # AFMB(2) and (3) have an interaction problem in the scenario where
        # both a continuous variable is given for outcome and the policy is
        # specified by row. If this happens, raise exception.
        if continuous_outcome and self.individual_policy_given:
            raise Exception('Cannot give continuous outcome and row-level treatment policy (col_policy).')


        # BEGIN UNMODULAR BEHAVIOR.

        # Save some inputs as class attributes.
        self.random_state = random_state
        self.transform = transform_func
        self.untransform = untransform_func

        # Perform outcome transformation.
        df[col_transformed_outcome] = self.transform(treatment=df[col_treatment], outcome=df[col_outcome], p=self.p)
        # Train test split (or recover the original train test split).
        if train_test_split_specified:
            df_train = df.xs(0)
            df_test = df.xs(1)
        else:
            df_train, df_test = train_test_split(df, test_size=test_size, random_state=self.random_state, stratify=stratify)

        # Save data to class attributes.
        # Full dataframes.
        self.df = df
        self.df_train = df_train
        self.df_test = df_test
        # Column names.
        self.col_treatment = col_treatment
        self.col_outcome = col_outcome
        self.col_transformed_outcome = col_transformed_outcome
        # Transformed outcome.
        self.transformed_y_train = df_train[col_transformed_outcome]
        self.transformed_y_test = df_test[col_transformed_outcome]
        self.transformed_y = df[col_transformed_outcome]
        # Untransformed outcome.
        self.y_train = df_train[col_outcome]
        self.y_test = df_test[col_outcome]
        self.y = df[col_outcome]
        # Features.
        feature_columns = [column for column in df.columns if column not in [col_treatment, col_outcome, col_transformed_outcome, col_policy]]
        self.x_train = df_train[feature_columns]
        self.x_test = df_test[feature_columns]
        self.x = df[feature_columns]
        # Treatment.
        self.tc_train = df_train[col_treatment]
        self.tc_test = df_test[col_treatment]
        self.tc = df[col_treatment]
        # Policy.
        if self.individual_policy_given:
            self.p_train = df_train[col_policy]
            self.p_test = df_test[col_policy]
        else:
            self.p_train = self.p
            self.p_test = self.p
        self.sklearn_model = sklearn_model

        # For backwards compatibility, create plot functions using the plot_type in the function name.
        plot_types = ['qini', 'aqini', 'cgains', 'cuplift', 'uplift', 'balance']
        for plot_type in plot_types:
            new_func = functools.partial(self.plot, plot_type=plot_type)
            setattr(self, 'plot_'+plot_type, new_func)

        # Define scoring cutoff for scoring functions.
        all_scoring_methods = ['qini', 'cgains', 'aqini', 'max_qini', 'max_cgains', 'max_aqini']
        if isinstance(scoring_cutoff, (int, float)):
            self.scoring_cutoff = { method: scoring_cutoff for method in all_scoring_methods }
        elif type(scoring_cutoff) == dict:
            self.scoring_cutoff = scoring_cutoff
        # Define scoring functions.
        for method in all_scoring_methods:
            score_func_name = '_'+method+'_score'
            # Deal with `max` functions differently.
            if method[:3]=='max':
                score_name = method
                plot_type = method[4:]
            else:
                score_name = 'q1_'+method
                plot_type = method
            # Customize `_score` function with `method`.
            new_func = functools.partial(self._score, method=method, plot_type=plot_type, score_name=score_name)
            new_func.__name__ = score_func_name
            setattr(self, score_func_name, new_func)

        # Put scoring functions in form accepted by sklearn functions.
        if type(scoring_method)==str:
            score_func_name = '_'+scoring_method+'_score'
            scoring_arg = make_scorer(getattr(self, score_func_name))
        elif type(scoring_method)==list:
            scoring_arg = {}
            for method in scoring_method:
                if method in all_scoring_methods:
                    score_func_name = '_'+method+'_score'
                    scoring_arg.update({method: make_scorer(getattr(self, score_func_name))})
                else:
                    scoring_arg.update({method: method})  # Allow RandomizedSearchCV built-in scoring methods.

        # Pass scoring function as default arguments in hyperparameter search functions.
        default_params = {
            'verbose': 3,
            'scoring': scoring_arg,
            'refit': False,
        }
        # Default XGBRegressor params.
        if self.sklearn_model == XGBRegressor:
            min_colsamp = np.amax([1/len(self.x_train.columns), 0.3])
            self.randomized_search_params = {
                'estimator': self.sklearn_model(nthread=1),
                'param_distributions': {
                    'n_estimators': range(10,500),
                    'max_depth': list(range(2, 21, 1)),
                    'min_child_weight': list(range(1,500,1)),
                    'gamma': uniform(0, 10),
                    'subsample': uniform(0.3, 0.7),
                    'colsample_bytree': uniform(min_colsamp, 1-min_colsamp),
                },
                'n_iter': 200,
                **default_params
            }
            self.grid_search_params = {
                'estimator': self.sklearn_model(),
                'param_grid': {'min_child_weight': list(range(1,200,1))},
                **default_params
            }
        # Default parameters for other models.
        else:
            self.randomized_search_params = {
                'estimator': self.sklearn_model(),
                **default_params
                }
            self.grid_search_params = {
                'estimator': self.sklearn_model(),
                **default_params
            }



    def NWOE(self, feats_to_use=None, n_bins=10):
        """Net weight of evidence.

        Parameters
        ----------
        feats_to_use : list, optional
            A list of features to use. If no list is specified, all features are used.
        n_bins : int, optional
            Number of bins to use.

        Returns
        -------
        ax : matplotlib.axes._subplots.AxesSubplot
            The matplotlib axes handle.

        """
        feats = feats_to_use if feats_to_use else self.x_train.columns
        df_with_bins = _add_bins(self.df_train, feats, n_bins=n_bins)
        self.NWOE_dict = _NWOE(df_with_bins, feats, col_treatment=self.col_treatment, col_outcome=self.col_outcome)
        ax = _plot_NWOE_bins(self.NWOE_dict, feats)
        return ax

    def NIV(self, feats_to_use=None, n_bins=10, n_iter=3):
        """Net information value, calculated for each feature averaged over `n_iter` bootstrappings of `df`.

        Parameters
        ----------
        feats_to_use : list, optional
            A list of features to use. If no list is specified, all features are used.
        n_bins : int, optional
            Number of bins to use.
        n_iter : int, optional
            Number of iterations.

        Returns
        -------
        ax : matplotlib.axes._subplots.AxesSubplot
            The matplotlib axes handle.

        """
        feats = feats_to_use if feats_to_use else self.x_train.columns
        df_with_bins = _add_bins(self.df_train, feats, n_bins=n_bins)
        # self.NIV_dict = _NIV(df_with_bins, feats, col_treatment=self.col_treatment, col_outcome=self.col_outcome)
        means_dict, low_perc_dict, high_perc_dict = _NIV_bootstrap(self.df_train, feats, n_bins=n_bins, perc=[20,80], n_iter=n_iter, frac=0.5, col_treatment=self.col_treatment, col_outcome=self.col_outcome)
        self.NIV_dict = means_dict
        ax = _plot_NIV_bs(means_dict, low_perc_dict, high_perc_dict, feats)
        return ax

    def randomized_search(self, **kwargs):
        """Randomized search using sklearn.model_selection.RandomizedSearchCV.

        Any parameters typically associated with RandomizedSearchCV (see
        sklearn documentation) can be passed as keyword arguments to this
        function.

        The final dictionary used for the randomized search is saved to
        `self.randomized_search_params`. This is updated with any parameters
        that are passed.

        Examples
        --------
        # Passing kwargs.
        self.randomized_search(param_distributions={'max_depth':[2,3,5,10]}, refit=True)

        """
        self.randomized_search_params.update(kwargs)
        self.rand_search_ = RandomizedSearchCV(**self.randomized_search_params)
        self.rand_search_.fit(self.x_train, self.transformed_y_train)
        return self.rand_search_

    def grid_search(self, **kwargs):
        """Grid search using sklearn.model_selection.GridSearchCV.

        Any parameters typically associated with GridSearchCV (see
        sklearn documentation) can be passed as keyword arguments to this
        function.

        The final dictionary used for the grid search is saved to
        `self.grid_search_params`. This is updated with any parameters that are
        passed.

        Examples
        --------
        # Passing kwargs.
        self.grid_search(param_grid={'max_depth':[2,3,5,10]}, refit=True)

        """
        self.grid_search_params.update(kwargs)
        self.grid_search_ = GridSearchCV(**self.grid_search_params)
        self.grid_search_.fit(self.x_train, self.transformed_y_train)
        return self.grid_search_


    def fit(self, productionize=False, **kwargs):
        """A fit wrapper around any sklearn Regressor.

        Any parameters typically associated with the model can be passed
        as keyword arguments to this function.

        The sklearn model object is saved to `self.model`, or if
        `productionize=True`, `self.model_final`.

        Parameters
        ----------
        productionize: boolean, optional
            If False, fits the model over the train set only. Otherwise, fits
            to all data available.

        """
        if productionize:
            # Just calculate the full new model.
            self.model_final = self.sklearn_model(**kwargs)
            self.model_final.fit(self.x, self.transformed_y)
        else:
            self.model = self.sklearn_model(**kwargs)
            self.model.fit(self.x_train, self.transformed_y_train)

            # Calculate evaluation metrics on test set.
            self.transformed_y_test_pred = self.model.predict(self.x_test)
            self.test_results_ = UpliftEval(self.tc_test, self.y_test, self.transformed_y_test_pred, p=self.p_test)

            # Calculate evaluation metrics for train set.
            self.transformed_y_train_pred = self.model.predict(self.x_train)
            self.train_results_ = UpliftEval(self.tc_train, self.y_train, self.transformed_y_train_pred, p=self.p_train)

    def noise_fit(self, iterations=10, n_bins=10, **kwargs):
        """Shuffle predictions to get a sense of the range of possible curves you
        might expect from fitting to noise.

        Parameters
        ----------
        iterations : int, optional
            Number of times to shuffle the data and retrain.
        n_bins : int, optional
            Number of bins to use when calculating the qini curves.

        """
        noise_fits = []
        shuffled_predictions = copy.deepcopy(self.transformed_y_test_pred)
        for i in range(iterations):
            np.random.shuffle(shuffled_predictions)
            upev = UpliftEval(self.tc_test, self.y_test, shuffled_predictions, n_bins=n_bins, p=self.p_test)
            noise_fits.append(upev)
        self.noise_fits = noise_fits

    def shuffle_fit(self, iterations=10, n_bins=20, params=None, transform_train=None, clear=False, plot_type='cgains', stratify=None, starting_seed=0, **kwargs):
        """Try the train-test split `iterations` times, and fit a model using `params`.

        Parameters
        ----------
        iterations : int
            Number of shuffle-fit sequences to run.
        n_bins : int
            Number of bins for the resulting curve to have.
        params : dict
            Dictionary of parameters to pass to each fit. If not given, will
            default to `self.rand_search_.best_params_`.
        transform_train : func, optional
            A function that will be applied to the training data only. Extended
            functionality that may be useful if the distribution of the
            y-variable is heavy-tailed, and a transformation would produce a
            better model, but you still want to evaluate on the untransformed
            data.
        clear : boolean, optional
            Data for the shuffle fits is saved in `self.shuffle_fit_`. If clear
            is True, this data is rewritten with each shuffle_fit iteration.
        plot_type : string, optional
            Type of plot to show. Can be `aqini`, `qini`, `cgains`.
        stratify : anything that can be passed to parameter of same name in train_test_split, optional
            If not None, stratify is used as input into train_test_split.
        starting_seed : the random seed used for the first iteration of train_test_split. All subsequent iterations increment from this value.

        """
        bootstrap_fits = []

        if not params:
            if hasattr(self, 'rand_search_'):
                if hasattr(self.rand_search_, 'best_params_'):
                    params = self.rand_search_.best_params_
                else:
                    print('ERROR: You need to pass parameters to shuffle fit.')
            else:
                print('ERROR: You need to pass parameters to shuffle fit.')

        ups = {}
        for seed in range(starting_seed, starting_seed+iterations):
            df_train, df_test = train_test_split(self.df, test_size=0.2, random_state=seed, stratify=stratify)
            if transform_train:
                df_train[self.col_outcome] = transform_train(df_train[self.col_outcome])
            uptmp = self.__class__((df_train, df_test), col_treatment=self.col_treatment, col_outcome=self.col_outcome, random_state=seed)
            uptmp.fit(**params, **kwargs)
            uptmp.test_results_.calc(plot_type=plot_type, n_bins=n_bins)
            ups[seed] = uptmp
            print('Seed', seed, 'finished.')

        # Save bootstrap fits to class.
        if (hasattr(self, 'shuffle_fit_')) and (not clear):
            self.shuffle_fit_.update(ups)
        else:
            self.shuffle_fit_ = ups
        return ups

    def plot(self, plot_type='cgains', ax=None, n_bins=None, show_noise_fits=False, noise_lines_kwargs={}, noise_band_kwargs={}, show_shuffle_fits=False, shuffle_lines_kwargs={}, shuffle_band_kwargs={}, shuffle_avg_line_kwargs={}, *args, **kwargs):
        """ Function to plot all curves.

        args and kwargs are passed to the default
        plot function, inherited from the UpliftEval class.

        Parameters
        ----------
        plot_type : string, optional
            Either 'qini', 'aqini', 'uplift', 'cuplift', or 'balance'.
            'aqini' refers to an adjusted qini plot, 'cuplift' gives a
            cumulative uplift plot. 'balance' gives the test-control balance
            for each of the bins. All others are self-explanatory.
        ax : matplotlib.Axes
            Pass axes to allow for overlaying on top of existing plots.
        n_bins : int, optional
            Number of bins to use for the main plot. This has no bearing on the
            shuffle or shuffle plots, which have to be calculated through
            their respective methods.
        show_noise_fits : bool, optional
            Toggle the display of fits to random noise.
        noise_lines_kwargs : dict, optional
            Kwargs to be passed to the lines that display the different curves
            for each noise fit iteration.
        noise_band_kwargs : dict, optional
            Kwargs to be passed to the colored band that displays the standard
            deviation of the noise fit iterations.
        show_shuffle_fits : bool, optional
            Toggle the display of fits with different train test split seeds.
        shuffle_lines_kwargs : dict, optional
            Kwargs to be passed to the lines that display the different curves
            for each shuffle fit iteration.
        shuffle_band_kwargs : dict, optional
            Kwargs to be passed to the colored band that displays the standard
            deviation of the shuffleped fit iterations.
        shuffle_avg_line_kwargs : dict, optional
            Kwargs to be passed to the average line that displays the average
            value of the shuffleped fit iterations.

        """

        # Define default parameters and update with the kwarg dictionaries.
        noise_lines_dict = {'alpha': 0.1}
        noise_lines_dict.update(noise_lines_kwargs)
        noise_band_dict = {'color':[0,0,0], 'alpha':0.3}
        noise_band_dict.update(noise_band_kwargs)
        shuffle_lines_dict = {'alpha':0.1}
        shuffle_lines_dict.update(shuffle_lines_kwargs)
        shuffle_band_dict = {'color':[0,0,0.4], 'alpha':0.3}
        shuffle_band_dict.update(shuffle_band_kwargs)
        shuffle_avg_line_dict = {'color': [0,0,0]}
        shuffle_avg_line_dict.update(shuffle_avg_line_kwargs)

        # If no axes are given, generate axes.
        if not ax:
            fig, ax = _plot_defaults()

        # If we are showing shuffle fits, an average line is shown, so don't plot the default.
        if not show_shuffle_fits:
            self_n_bins = getattr(self.test_results_, plot_type+'_n_bins')
            self_plot_func = getattr(self.test_results_, 'plot_'+plot_type)

            # Recalculate the values if n_bins is not the same.
            if n_bins and (n_bins!=self_n_bins):
                self.test_results_.calc(plot_type=plot_type, n_bins=n_bins)

            ax = self_plot_func(ax=ax, *args, **kwargs)

        # For qini-style curves, remove the random selection line in shuffle & noise fits.
        if show_noise_fits or show_shuffle_fits:
            extra_kwargs = {}
            if plot_type in ['qini', 'aqini', 'cgains']:
                extra_kwargs = {'show_random_selection': False}

        # Plot noise fits if flagged.
        if show_noise_fits and (hasattr(self,'noise_fits')):
            noise_y_values = []

            # Remove random selection line for qini and aqini curves.
            for upev in self.noise_fits:
                getattr(upev, 'plot_'+plot_type)(ax=ax, **noise_lines_dict, **extra_kwargs)
                noise_y_values.append(getattr(upev, plot_type+'_y'))
            noise_means = np.array(list(map(np.mean, list(zip(*noise_y_values)))))
            noise_std = np.array(list(map(np.std, list(zip(*noise_y_values)))))
            ax.fill_between(getattr(self.noise_fits[0], plot_type+'_x'), noise_means-noise_std, noise_means+noise_std, **noise_band_dict)

        # Plot shuffled fits if flagged.
        shuffle_y_values = []
        if show_shuffle_fits and hasattr(self, 'shuffle_fit_'):
            for key, up in self.shuffle_fit_.items():
                up.plot(plot_type=plot_type, ax=ax, **shuffle_lines_dict, **extra_kwargs)
                shuffle_y_values.append(getattr(up.test_results_, plot_type+'_y'))
            bs_means = np.array(list(map(np.mean, list(zip(*shuffle_y_values)))))
            bs_std = np.array(list(map(np.std, list(zip(*shuffle_y_values)))))
            x = getattr(self.shuffle_fit_[0].test_results_, plot_type+'_x')
            ax.fill_between(x, bs_means-bs_std, bs_means+bs_std, **shuffle_band_dict)
            ax.plot(x, bs_means, **shuffle_avg_line_dict)
            ax.plot([x[0], x[-1]], [bs_means[0], bs_means[-1]], '--', color=[0.6,0.6,0.6])

        return ax

