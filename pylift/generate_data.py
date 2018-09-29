import numpy as np
import pandas as pd


def dgp(N=1000, n_features=3, beta=[-3,-8,13,-8], error_std=0.5, tau=3, tau_std=1, discrete_outcome=False, seed=2701, feature_effect=0.5):
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

    # Define features, error, and random treatment assignment.
    X = np.random.random(size=(N, n_features))
    error = np.random.normal(size=(N), loc=0, scale=error_std)
    treatment = np.random.binomial(1, .5, size=(N))

    # Effect of features on outcome.
    if beta is None:
        beta = np.random.random(n_features+1)

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
