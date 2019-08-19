import numpy as np
import pandas as pd
from scipy.stats import norm

def dgp(N=1000, n_features=3, beta=None, error_std=0.5, tau=3, tau_std=1, discrete_outcome=False, seed=None, feature_effect=0.5):
    """
    dgp(N=1000, n_features=3, beta=[1,-2,3,-0.8], error_std=0.5, tau=3, discrete_outcome=False)

    Generates random data with a ground truth data generating process.

    Draws random values for features from [0, 1), errors from a 0-centered
    distribution with std `error_std`, and creates an outcome y.

    Parameters
    ----------
    N : int, optional
        Number of observations.
    n_features : int, optional
        Number of features.
    beta : np.array, optional
        Array of beta coefficients to multiply by X to get y.
    error_std : float, optional
        Standard deviation (scale) of distribution from which errors are drawn.
    tau : float, optional
        Effect of treatment.
    tau_std : float, optional
        When not None, draws tau from a normal distribution centered around tau
        with standard deviation tau_std rather than just using a constant value
        of tau.
    discrete_outcome : boolean, optional
        If True, outcomes are 0 or 1; otherwise continuous.
    seed : int, optional
        Random seed fed to np.random.seed to allow for deterministic behavior.

    Output
    ------
    df : pd.DataFrame
        A DataFrame containing the generated data.

    """

    np.random.seed(seed=seed)

    if(beta and (n_features != len(beta) -1)):
        raise ValueError('If custom beta supplied, len(beta) must be equal to n_features+1')

    # Effect of features on outcome.
    if beta is None:
        beta = np.random.random(n_features+1)

    # Define features, error, and random treatment assignment.
    X = np.random.random(size=(N, n_features))
    error = np.random.normal(size=(N), loc=0, scale=error_std)
    treatment = np.random.binomial(1, .5, size=(N))

    # Treatment heterogeneity.
    tau_vec = np.random.normal(loc=tau, scale=tau_std, size=N) + np.dot(X, beta[1:])*feature_effect

    # Calculate outcome.
    y = beta[0] + np.dot(X, beta[1:]) + error + treatment*tau_vec

    if discrete_outcome:
        y = y > 0

    names = list(range(n_features))
    names.extend(['Treatment', 'Outcome'])

    df = pd.DataFrame(np.concatenate((X, treatment.reshape(-1,1), y.reshape(-1,1)), axis=1), columns=names)
    return df



def sim_pte(N=1000, n_features=20, beta=None, rho=0, sigma=np.sqrt(2), beta_den=4, discrete_outcome=False, seed=None):
    """
    sim_pte(N=1000, p=20, rho=0, sigma=np.sqrt(2), beta_den=4)

    Numerical simulation for treatment effect heterogeneity estimation as described in Tian et al. (2012)
    Translated from the R uplift package (Leo Guelman <leo.guelman@gmail.com>).

    Parameters
    ----------
    N : int, optional
        Number of observations.
    n_features : int, optional
        Number of features.
    beta : np.array, optional
        Array of beta coefficients to multiply by X to get y.
    rho : covariance matrix between predictors.
    sigma : multiplier of error term.
    beta_den : size of main effects relative to interaction effects.
    discrete_outcome : boolean, optional
        If True, outcomes are 0 or 1; otherwise continuous.
    seed : int, optional
        Random seed fed to np.random.seed to allow for deterministic behavior.

    Output
    ------
    A data frame including the response variable (Y), the treatment (treat=1)
    and control (treat=-1) assignment, the predictor variables (X) and the "true"
    treatment effect score (ts).

    """
    p = n_features

    # Check arguments
    if N < 2:
        raise ValueError("uplift: The number of observations must be greater than 2")

    if p < 4:
        raise ValueError("uplift: The number predictors must be equal or greater than 4")

    if rho < 0 | rho > 1:
        raise ValueError("uplift: rho must be between 0 and 1")

    if sigma < 0:
        raise ValueError("uplift: beta.den must be equal or greater than 0")

    if beta_den <= 0:
        raise ValueError("uplift: beta.den must be greater than 0")

    # Main Effects
    if beta is None:
        beta = np.zeros(p)
        for j in range(p):
            beta[j] = (-1)**j * (j >= 3 & j <= 10) / beta_den

    ### Generate x ~ N~p(0, rho)
    mean = np.zeros(p)
    cov = np.identity(n=p)
    x = np.random.multivariate_normal(mean, cov, N)

    ### Random error from N~1(0,1)
    eps = np.random.normal(loc=0, scale=1, size=N)[:,np.newaxis]

    ### Treatment generated with equal probability at random
    treat = np.random.binomial(n=1, p=0.5, size=N)


    ### Interaction effects
    gamma = np.array([0.5, -0.5, 0.5, -0.5] + [0]*(p-4))

    ### Response variable
    y = np.matmul(x, beta[:,np.newaxis]) + (x * treat[:, np.newaxis]) @ gamma[:, np.newaxis] + sigma * eps

    ### "True" score
    ts = norm.pdf(np.matmul(x, (beta + gamma)[:, np.newaxis]) / sigma) - norm.pdf(np.matmul(x, (beta - gamma)[:, np.newaxis]) / sigma)

    if discrete_outcome:
        y = (y > 0).astype(int)

    ### Returned value
    out = pd.DataFrame({'y':y[:,0], 'treat':treat, 'ts':ts[:,0]})
    dfx = pd.DataFrame(x)
    dfx.columns = ['x' + str(colname) for colname in list(dfx.columns)]

    out = pd.concat([out, dfx], axis=1)

    return out

