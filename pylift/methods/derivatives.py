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

EPS = 1e-37

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
        Either qini, aqini, cgains or max_ prepended to any of the previous
        values. Any strings available to the parameter `scoring` in
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
    def _transform_func(treatment, outcome, p=None):
        """Function that executes the Transformed Outcome.

        Parameters
        ----------
        treatment : array-like
            Array of 1s and 0s indicating treatment.
        outcome : array-like
            Array of 1s and 0s indicating outcome.

        Returns
        -------
        y : np.array
            Transformed label.

        """
        treatment = np.array(treatment)
        outcome = np.array(outcome).astype(float)
        # Make sure outcome=0 maps to EPS, to preserve sign.
        outcome[outcome==0] = EPS
        y = (treatment*2 - 1)*outcome
        # Change nonzero outcomes (currently 1 or -1) according to test/control split.
        if not p:
            p = len(treatment[treatment==1])/len(treatment)
        ones = abs(y)>EPS
        y[ones] = ((treatment[ones]-p)/(p*(1-p)))*outcome[ones] # Change the 1s.
        return y

    @staticmethod
    def _untransform_func(ys, p=None):
        """Function that recovers original data from Transformed Outcome.

        Parameters
        ----------
        ys : array-like
            Transformed label.

        Returns
        -------
        treatment : np.array
            Array of 1s and 0s indicating treatment.
        outcome : np.array
            Array of 1s and 0s indicating outcome.

        """
        ys = np.array(ys)
        if not p:
            p = len(ys[ys>0])/len(ys)
        treatment = np.zeros(ys.shape)
        outcome = np.zeros(ys.shape)
        nonzeros = (abs(ys)!=EPS)
        # Get treatment back (binary depending on sign of ys).
        treatment[ys==-EPS] = 0
        treatment[ys==EPS] = 1
        treat = (np.sign(ys)+1)/2 # One or zero for sign.
        treatment[nonzeros] = treat[nonzeros]
        # Get outcome back (EPS, or transformation of pure ys value).
        outcome[abs(ys)==EPS] = 0
        outcome[nonzeros] = ys[nonzeros]*(p*(1-p))/(treatment[nonzeros]-p)
        return treatment, outcome

    def __init__(self, df, col_treatment='Treatment', col_outcome='Outcome', col_transformed_outcome='TransformedOutcome', random_state=2701, test_size=0.2, stratify=None, scoring_cutoff=1, sklearn_model=XGBRegressor, scoring_method='cgains'):

        super().__init__(df, transform_func=self._transform_func, untransform_func=self._untransform_func, col_treatment=col_treatment, col_outcome=col_outcome, col_transformed_outcome=col_transformed_outcome, random_state=random_state, test_size=test_size, stratify=stratify, scoring_cutoff=scoring_cutoff, scoring_method=scoring_method, sklearn_model=sklearn_model)
