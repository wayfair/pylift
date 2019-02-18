from .base import BaseProxyMethod
from ..eval import get_scores, UpliftEval
import functools
from scipy.stats import uniform
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from xgboost import XGBClassifier, XGBRegressor

import numpy as np
import pickle

TOL = 1e-37
EPS = 1e-20

def custom_objective(dtrain, preds):
    # Error.
    err = preds - dtrain
    treatment, outcome, p = TransformedOutcome._untransform_func(dtrain)
    # Correct for policy.
    nt_scaled = np.sum(0.5*treatment/p)
    nc_scaled = np.sum(0.5*(1-treatment)/(1-p))
    p_scaled = nt_scaled/(nt_scaled + nc_scaled)

    w_i = 0.25*(treatment/(p_scaled*p) + (1-treatment)/((1-p_scaled)*(1-p)))
    grad = w_i*err
    hess = np.ones(dtrain.shape)
    return grad, hess

class FlaggedFloat(float):
    """Float subclass that retains a Treatment flag property.
    """
    def save(self, outcome, treatment, p):
        self.outcome = outcome
        self.treatment = treatment
        self.p = p

class TransformedOutcome(BaseProxyMethod):
    """Implement Transformed Outcome [Trees] method.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the features, test/control flag, and outcome.
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
    stratify : string or same format as parameter of same name in
    train_test_split, optional
        If not None, stratify is used as input into train_test_split.
    scoring_method : string or list, optional
        Either `qini`, `aqini`, `cgains` or `max_` prepended to any of the
        previous values. Any strings available to the parameter `scoring` in
        `sklearn.model_selection.RandomizedSearchCV` can also be passed.
    scoring_cutoff : float or dict, optional
        The fraction of observations used to score qini for hyperparam
        searches. E.g. if 0.4, the 40% of observations with the highest
        predicted uplift are used to determine the frost score in the
        randomized search scoring function. If a list of scoring_methods is
        passed, a dictionary can also be passed here, where the keys are the
        scoring_method strings and the values are the scoring cutoff for those
        specific methods.
    sklearn_model : sklearn regressor class, optional
        Sklearn model object to for all successive operations (don't pass any
        parameters).
    """

    @staticmethod
    def _transform_func(treatment, outcome, p):
        """Function that executes the Transformed Outcome.

        Parameters
        ----------
        treatment : array-like
            Array of 1s and 0s indicating treatment.
        outcome : array-like
            Array of 1s and 0s indicating outcome.
        p : float or np.array
            Probability of observing a treatment=1 flag, used for the
            transformation.

        Returns
        -------
        y : np.array
            Transformed label.

        """
        treatment = list(treatment)
        outcome = list(outcome)
        if np.size(p) == 1:  # If numerical.
            p = [p for i in range(len(treatment))]
        else:  # If array-like.
            p = list(p)

        # Change nonzero outcomes according to treatment/control split.
        y = [FlaggedFloat(o*((t-pr)/(pr*(1-pr)))) for o, t, pr in zip(outcome, treatment, p)]
        for yi, oi, ti, pi in zip(y, outcome, treatment, p):
            yi.save(oi, ti, pi)
        y = np.array(y, dtype=np.dtype(FlaggedFloat))
        return y

    @staticmethod
    def _untransform_func(ys):
        """Function that recovers original data from Transformed Outcome.

        Parameters
        ----------
        ys : numpy.array
            Transformed label.
        p : float or np.array, optional
            Probability of observing a treatment=1 flag, used to reverse the
            transformation.

        Returns
        -------
        treatment : np.array
            Array of 1s and 0s indicating treatment.
        outcome : np.array
            Array of 1s and 0s indicating outcome.

        """
        treatment = np.array([i.treatment for i in ys])
        p = np.array([i.p for i in ys])
        outcome = np.array([i.outcome for i in ys])

        return treatment, outcome, p

    def __init__(self, df, col_treatment='Treatment', col_outcome='Outcome', col_transformed_outcome='TransformedOutcome', col_policy=None, continuous_outcome='infer', random_state=2701, test_size=0.2, stratify=None, scoring_cutoff=1, sklearn_model=XGBRegressor, scoring_method='cgains'):

        super().__init__(df, transform_func=self._transform_func, untransform_func=self._untransform_func, col_treatment=col_treatment, col_outcome=col_outcome, col_transformed_outcome=col_transformed_outcome, col_policy=col_policy, continuous_outcome=continuous_outcome, random_state=random_state, test_size=test_size, stratify=stratify, scoring_cutoff=scoring_cutoff, scoring_method=scoring_method, sklearn_model=sklearn_model)
