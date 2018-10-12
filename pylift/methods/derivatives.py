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
        treatment = np.array(treatment)
        outcome = np.array(outcome).astype(float)
        # Make sure outcome=0 maps to EPS, to preserve sign.
        outcome[outcome==0] = EPS
        # Change nonzero outcomes according to treatment/control split.
        y = outcome*((treatment-p)/(p*(1-p)))
        return y

    @staticmethod
    def _untransform_func(ys, p=None):
        """Function that recovers original data from Transformed Outcome.

        Parameters
        ----------
        ys : array-like
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
        ys = np.array(ys)

        treatment = np.zeros(ys.shape)
        outcome = np.zeros(ys.shape)
        nonzeros = (abs(ys)!=EPS)

        # Get the treatment label (positive or negative).
        t1 = (ys > 0)
        t0 = (ys < 0)
        treatment[t1] = 1  # All other entrise are by default 0.

        # Get the policy back, if not given. p ranges between 0 and 1. Hack here: as long as it's bigger than EPS and less than 1-EPS, we can recover it.
        if not p:
            t1o1 = t1 & (ys>=1)
            t1o0 = t1 & (ys<=1)
            t0o1 = t0 & (ys<=-1)
            t0o0 = t0 & (ys>-1)
            p = np.zeros(ys.shape)
            p[t1o1] = 1/ys[t1o1]
            p[t1o0] = EPS/ys[t1o0]
            p[t0o1] = 1+1/ys[t0o1]
            p[t0o0] = 1+EPS/ys[t0o0]
            outcome[t1] = ys[t1]*p[t1]
            outcome[t0] = -ys[t0]*(1-p[t0])
        else:
            outcome[t1] = ys[t1]*p
            outcome[t0] = -ys[t0]*(1-p)
            p = np.ones(ys.shape)*p
        outcome[np.abs(outcome) - EPS < TOL] = 0

        return treatment, outcome, p

    def __init__(self, df, col_treatment='Treatment', col_outcome='Outcome', col_transformed_outcome='TransformedOutcome', col_policy=None, continuous_outcome="infer", random_state=2701, test_size=0.2, stratify=None, scoring_cutoff=1, sklearn_model=XGBRegressor, scoring_method='cgains'):

        super().__init__(df, transform_func=self._transform_func, untransform_func=self._untransform_func, col_treatment=col_treatment, col_outcome=col_outcome, col_transformed_outcome=col_transformed_outcome, col_policy=col_policy, continuous_outcome=continuous_outcome, random_state=random_state, test_size=test_size, stratify=stratify, scoring_cutoff=scoring_cutoff, scoring_method=scoring_method, sklearn_model=sklearn_model)
