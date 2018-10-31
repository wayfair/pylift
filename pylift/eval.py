import copy
import functools
import numpy as np
import matplotlib.pyplot as plt, matplotlib as mpl
from .style import _plot_defaults

EPS = 1e-20

_ = lambda x: x

def _ensure_array(arraylike):
    """ Make sure inputs are np.array.

    A number of functions exploit np.array's element-level operations, so we
    add this snippet of code to ensure an object is a numpy array.

    Parameters
    ----------
    arraylike : array-like
        Array-like object to be processed.

    Returns
    -------
    arraylike : np.array
        Numpy array with elements from `arraylike`.

    """
    if type(arraylike)!=np.ndarray:
        arraylike = np.array(arraylike)
    return arraylike

def _get_counts(treatment, outcome, p):
    """Extract (treatment,outcome) pair counts from treatment and outcome vectors.

    Calculate counts of outcome/treatment combinations. Variables are named
    nt#o#, where the # corresponds to a binary value for test and outcome,
    respectively.

    Parameters
    ----------
    treatment : np.array
        Vector of 1s and 0s indicating exposure to a treatment.
    outcome : np.array
        Vector of 1s and 0s indicating successful outcome.
    p : np.array
        Vector of values 0 < p < 1 indicating probability of treatment.

    Returns
    -------
    Nt1o1 : int
        Number of entries where (treatment, outcome) == (1,1).
    Nt0o1 : int
        Number of entries where (treatment, outcome) == (0,1).
    Nt1o0 : int
        Number of entries where (treatment, outcome) == (1,0).
    Nt0o0 : int
        Number of entries where (treatment, outcome) == (0,0).

    """
    Nt1o1 = 0.5*np.sum(1/p[(treatment == 1) & (outcome > 0)])
    Nt0o1 = 0.5*np.sum(1/(1-p[(treatment == 0) & (outcome > 0)]))
    Nt1o0 = 0.5*np.sum(1/p[(treatment == 1) & (outcome == 0)])
    Nt0o0 = 0.5*np.sum(1/(1-p[(treatment == 0) & (outcome == 0)]))
    return Nt1o1, Nt0o1, Nt1o0, Nt0o0

def _get_tc_counts(Nt1o1, Nt0o1, Nt1o0, Nt0o0):
    """Get treatment and control group sizes from `_get_counts` output.

    Parameters
    ----------
    Nt1o1 : int
        Number of entries where (treatment, outcome) == (1,1).
    Nt0o1 : int
        Number of entries where (treatment, outcome) == (0,1).
    Nt1o0 : int
        Number of entries where (treatment, outcome) == (1,0).
    Nt0o0 : int
        Number of entries where (treatment, outcome) == (0,0).

    Returns
    -------
    Nt1 : int
        Size of treatment group.
    Nt0 : int
        Size of control group.
    N : int
        Size of full group.

    """
    Nt1 = Nt1o0 + Nt1o1
    Nt0 = Nt0o0 + Nt0o1
    N = Nt0 + Nt1
    return Nt1, Nt0, N

def _get_no_sure_thing_counts(Nt1o1, Nt0o1, Nt1o0, Nt0o0, func=sum):
    """Calculate counts of causal behavior under assumption: no "sure things".

    Determining the number of persuadables, sure things, lost causes, and
    sleeping dogs is impossible, but we can do so if we make assumptions and
    follow these assumptions to their logical end. We can then calculate
    maximal qini curves.

    In this case, we assume there are no "sure things" -- in other words, all
    those in the treatment group that have a positive outcome are "persuadables".

    Parameters
    ----------
    Nt1o1 : int
        Number of entries where (treatment, outcome) == (1,1).
    Nt0o1 : int
        Number of entries where (treatment, outcome) == (0,1).
    Nt1o0 : int
        Number of entries where (treatment, outcome) == (1,0).
    Nt0o0 : int
        Number of entries where (treatment, outcome) == (0,0).
    func : function
        Function to process tuples before output is returned. `sum` is typical,
        but an identity function can be passed if you want to keep the tuples.

    Returns
    -------
    persuadables : func(tuple)
        Number of persuadables in control and treatment, respectively, under the given assumption.
    dogs : func(tuple)
        Number of sleeping dogs, in control and treatment, respectively, under the given assumption.
    sure_things : func(tuple)
        Number of sure things, in control and treatment, respectively, under the given assumption.
    lost_causes : func(tuple)
        Number of lost causes, in control and treatment, respectively, under the given assumption.

    """
    Nt1, Nt0, N = _get_tc_counts(Nt1o1, Nt0o1, Nt1o0, Nt0o0)
    sure_things = (0, 0)
    persuadables = (Nt1o1*Nt0/Nt1, Nt1o1)  # Add the persuadables in control.
    dogs = (Nt0o1, Nt0o1*Nt1/Nt0)
    lost_causes = (Nt0-dogs[0]-sure_things[0]-persuadables[0], \
        Nt1-dogs[1]-sure_things[1]-persuadables[1])
    return func(persuadables), func(dogs), func(sure_things), func(lost_causes)

def _get_no_sleeping_dog_counts(Nt1o1, Nt0o1, Nt1o0, Nt0o0, func=sum):
    """Calculate counts of causal behavior under assumption: no "sleeping dogs".

    Determining the number of persuadables, sure things, lost causes, and
    sleeping dogs is impossible, but we can do so if we make assumptions and
    follow these assumptions to their logical end. We can then calculate
    maximal qini curves.

    In this case, we assume there are no "sleeping dogs" -- in other words, all
    those in the control group that have a positive outcome are "sure things".

    Parameters
    ----------
    Nt1o1 : int
        Number of entries where (treatment, outcome) == (1,1).
    Nt0o1 : int
        Number of entries where (treatment, outcome) == (0,1).
    Nt1o0 : int
        Number of entries where (treatment, outcome) == (1,0).
    Nt0o0 : int
        Number of entries where (treatment, outcome) == (0,0).
    func : function
        Function to process tuples before output is returned. `sum` is typical,
        but an identity function can be passed if you want to keep the tuples.

    Returns
    -------
    persuadables : func(tuple)
        Number of persuadables in control and treatment, respectively, under the given assumption.
    dogs : func(tuple)
        Number of sleeping dogs, in control and treatment, respectively, under the given assumption.
    sure_things : func(tuple)
        Number of sure things, in control and treatment, respectively, under the given assumption.
    lost_causes : func(tuple)
        Number of lost causes, in control and treatment, respectively, under the given assumption.

    """
    Nt1, Nt0, N = _get_tc_counts(Nt1o1, Nt0o1, Nt1o0, Nt0o0)
    dogs = (0,0)
    sure_things = (Nt0o1, Nt0o1*Nt1/Nt0)
    lost_causes = (Nt1o0*Nt0/Nt1, Nt1o0)
    persuadables = (Nt0-dogs[0]-sure_things[0]-lost_causes[0], \
        Nt1-dogs[1]-sure_things[1]-lost_causes[1])
    return func(persuadables), func(dogs), func(sure_things), func(lost_causes)

def _get_overfit_counts(Nt1o1, Nt0o1, Nt1o0, Nt0o0, func=sum):
    """Calculate counts of causal behavior under assumption: all treated are
    persuadables, and these are the only persuadables.

    Determining the number of persuadables, sure things, lost causes, and
    sleeping dogs is impossible, but we can do so if we make assumptions and
    follow these assumptions to their logical end. We can then calculate
    maximal qini curves.

    In this case, we assume that all those with a positive outcome in the
    treatment group are persuadables, and that these are the only persuadables.
    In almost all circumstances, this combination of counts are impossible, but
    they do provide the points for the absolute maximal qini curve possible.

    Parameters
    ----------
    Nt1o1 : int
        Number of entries where (treatment, outcome) == (1,1).
    Nt0o1 : int
        Number of entries where (treatment, outcome) == (0,1).
    Nt1o0 : int
        Number of entries where (treatment, outcome) == (1,0).
    Nt0o0 : int
        Number of entries where (treatment, outcome) == (0,0).
    func : function
        Function to process tuples before output is returned. `sum` is typical,
        but an identity function can be passed if you want to keep the tuples.

    Returns
    -------
    persuadables : func(tuple)
        Number of persuadables in control and treatment, respectively, under the given assumption.
    dogs : func(tuple)
        Number of sleeping dogs, in control and treatment, respectively, under the given assumption.
    sure_things : func(tuple)
        Number of sure things, in control and treatment, respectively, under the given assumption.
    lost_causes : func(tuple)
        Number of lost causes, in control and treatment, respectively, under the given assumption.

    """
    persuadables = (0, Nt1o1)
    dogs = (Nt0o1, 0)
    sure_things = (0,0)
    lost_causes = (Nt0o0, Nt1o0)
    return func(persuadables), func(dogs), func(sure_things), func(lost_causes)

def _maximal_qini_curve(func, Nt1o1, Nt0o1, Nt1o0, Nt0o0):
    """Use counts of (treatment,outcome) pairs to determine the optimal Qini
    curve.

    Calculates the maximal Qini-style curves using the number of entries in the
    different (treatment,outcome) pairs and the _get_* function, `func`.

    Parameters
    ----------
    func: function
        Function that returns persuadables, etc. counts.
    Nt1o1 : int
        Number of entries where (treatment, outcome) == (1,1).
    Nt0o1 : int
        Number of entries where (treatment, outcome) == (0,1).
    Nt1o0 : int
        Number of entries where (treatment, outcome) == (1,0).
    Nt0o0 : int
        Number of entries where (treatment, outcome) == (0,0).

    Returns
    -------
    x : list
        x-values of the maximal Qini curve.
    y : list
        y-values of the maximal Qini curve.

    """
    Nt1, Nt0, N = _get_tc_counts(Nt1o1, Nt0o1, Nt1o0, Nt0o0)
    persuadables, dogs, sure_things, lost_causes = func(Nt1o1, Nt0o1, Nt1o0, Nt0o0)
    # For the overfit case, this is simply the entire treated group.
    if func == _get_overfit_counts:
        slope = 2
    else:
        slope = 1
    x = [0, persuadables/N, 1-dogs/N, 1]
    # Deal with edge case where number of persuadables is greater than sleeping
    # dogs (common if this is not a treatment/control experiment, but an
    # experiment between two treatments).
    if x[1]>x[2]:
        new_val = (x[1]+x[2])/2
        x[1] = new_val
        x[2] = new_val
    y = [0, x[1]*slope, x[1]*slope, (Nt1o1/Nt1-Nt0o1/Nt0)]

    return x, y

def _maximal_uplift_curve(func, Nt1o1, Nt0o1, Nt1o0, Nt0o0):
    """Use counts of (treatment,outcome) pairs to determine the optimal uplift
    curve.

    Calculates the maximal uplift curves using the number of entries in the
    different (treatment,outcome) pairs and the _get_* function, `func`.

    Parameters
    ----------
    func: function
        Function that returns persuadables, etc. counts.
    Nt1o1 : int
        Number of entries where (treatment, outcome) == (1,1).
    Nt0o1 : int
        Number of entries where (treatment, outcome) == (0,1).
    Nt1o0 : int
        Number of entries where (treatment, outcome) == (1,0).
    Nt0o0 : int
        Number of entries where (treatment, outcome) == (0,0).

    Returns
    -------
    x : list
        x-values of the maximal uplift curve.
    y : list
        y-values of the maximal uplift curve.

    """
    Nt1, Nt0, N = _get_tc_counts(Nt1o1, Nt0o1, Nt1o0, Nt0o0)
    persuadables, dogs, sure_things, lost_causes = func(Nt1o1, Nt0o1, Nt1o0, Nt0o0)
    y = [1, 1, 0, 0, -1, -1]
    x = [0, persuadables/N, persuadables/N, 1-dogs/N, 1-dogs/N, 1]
    return x, y

def _maximal_cuplift_curve(func, Nt1o1, Nt0o1, Nt1o0, Nt0o0):
    """Use counts of (treatment,outcome) pairs to determine the optimal
    cumulative uplift curve.

    Calculates the maximal cumulative uplift curves using the number of entries
    in the different (treatment,outcome) pairs and the _get_* function, `func`.

    Parameters
    ----------
    func: function
        Function that returns persuadables, etc. counts.
    Nt1o1 : int
        Number of entries where (treatment, outcome) == (1,1).
    Nt0o1 : int
        Number of entries where (treatment, outcome) == (0,1).
    Nt1o0 : int
        Number of entries where (treatment, outcome) == (1,0).
    Nt0o0 : int
        Number of entries where (treatment, outcome) == (0,0).

    Returns
    -------
    x : list
        x-values of the maximal cumulative uplift curve.
    y : list
        y-values of the maximal cumulative uplift curve.

    """
    Nt1, Nt0, N = _get_tc_counts(Nt1o1, Nt0o1, Nt1o0, Nt0o0)
    persuadables, dogs, sure_things, lost_causes = func(Nt1o1, Nt0o1, Nt1o0, Nt0o0, func=lambda x: x)
    kink_y = (persuadables[1] + sure_things[1])/(persuadables[1] + sure_things[1] + lost_causes[1]) \
        - (sure_things[0])/(persuadables[0] + sure_things[0] + lost_causes[0])
    y = [1, 1, kink_y, (Nt1o1/Nt1-Nt0o1/Nt0)]
    x = [0, sum(persuadables)/N, 1-sum(dogs)/N, 1]
    return x, y

def get_scores(treatment, outcome, prediction, p, scoring_range=(0,1), plot_type='all'):
    """Calculate AUC scoring metrics.

    Parameters
    ----------
    treatment : array-like
    outcome : array-like
    prediction : array-like
    p : array-like
        Treatment policy (probability of treatment for each row).
    scoring_range : 2-tuple
        Fractional range over which frost score is calculated. First element
        must be less than second, and both must be less than 1.

    Returns
    -------
    scores : dict
        A dictionary containing the following values. Each is also appended
        with `_cgains` and `_aqini` for the corresponding values for the
        cumulative gains curve and adjusted qini curve, respectively.

        q1: Traditional Q score normalized by the theoretical
        maximal qini. Note the theoretical max here goes up with a slope of 2.

        q2: Traditional Q score normalized by the practical maximal qini. This
        curve increases with a slope of 1.

        Q: Area between qini curve and random selection line. This is named
        after the notation in Radcliffe & Surry 2011, but note that they
        normalize their curves differently.

        Q_max: Maximal possible qini score, which is used for normalization
        of qini to get frost score. Only obtainable by overfitting.

        Q_practical_max: Practical maximal qini score, if you are not
        overfitting. This assumes that all (outcome, treatment) = (1,1) were
        persuadables, but that there are also an equal number of persuadables
        in the control group. This is the best possible scenario, but likely
        assumes too few "sure things".

        overall_lift: The lift expected from random application of treatment.

    """
    treatment = _ensure_array(treatment)
    outcome = _ensure_array(outcome)
    prediction = _ensure_array(prediction)
    p = _ensure_array(p)

    Nt1o1, Nt0o1, Nt1o0, Nt0o0 = _get_counts(treatment, outcome, p)
    Nt1, Nt0, N = _get_tc_counts(Nt1o1, Nt0o1, Nt1o0, Nt0o0)

    def riemann(x, y):
        avgy = [(a+b)/2 for (a,b) in zip(y[:-1], y[1:])]
        dx = [b-a for (a,b) in zip(x[:-1], x[1:])]
        return sum([a*b for (a,b) in zip(dx, avgy)])

    qini_riemann = riemann(*_maximal_qini_curve(_get_overfit_counts, Nt1o1, Nt0o1, Nt1o0, Nt0o0))
    practical_qini_riemann = riemann(*_maximal_qini_curve(_get_no_sure_thing_counts, Nt1o1, Nt0o1, Nt1o0, Nt0o0))

    overall_lift = (Nt1o1/Nt1-Nt0o1/Nt0)
    qini_max = qini_riemann - 0.5*overall_lift
    practical_qini_max = practical_qini_riemann - 0.5*overall_lift

    # The predicted Qini curve.
    # First we need to reorder the y values and y_pred based on this reordering
    # We calculate TOT roughly here so we have a way of distinguishing those that (ordered, treated) and those that (ordered, untreated).
    y = (2*treatment - 1)*outcome

    def sortbyprediction(vec):
        list2 = list(zip(prediction,vec))
        # Sort by prediction.
        list2.sort(key=lambda tup: tup[0], reverse=True) # included the tup[0] because otherwise we run into problems when there are only a few predicted values -- it orders by index i instead -- not what we want!
        # Extract `y`, sorted by prediction.
        _, vec_ordered = zip(*list2)
        return vec_ordered

    y_ordered = sortbyprediction(y)
    tr_ordered = sortbyprediction(treatment)
    p_ordered = sortbyprediction(p)

    def auc(method='qini'):
        # Calculate the area.
        uplift_last = 0
        nt1o1 = 0
        nt0o1 = 0
        nt1 = EPS
        nt0 = EPS
        pred_riemann = 0
        uplifts = []
        for i in range(round(scoring_range[0]*len(treatment)), round(scoring_range[1]*len(treatment))):
            if y_ordered[i] > 0:
                nt1o1 += 0.5*(1/p_ordered[i])
            elif y_ordered[i] < 0:
                nt0o1 += 0.5*(1/(1-p_ordered[i]))

            if tr_ordered[i] == 1:
                nt1 += 0.5*(1/p_ordered[i])
            else:
                nt0 += 0.5*(1/(1-p_ordered[i]))

            if method=='qini':
                uplift_next = nt1o1/Nt1-nt0o1/Nt0
            elif method=='cgains':
                uplift_next = (nt1o1/nt1-nt0o1/nt0)*(nt1+nt0)/N
            elif method=='aqini':
                uplift_next = nt1o1/Nt1-nt0o1*nt1/(nt0*Nt1 + EPS)

            uplifts.append(uplift_next)
            # each point corresponds to an x delta of 1/N
            pred_riemann += 1/2*(uplift_next+uplift_last)/N
            uplift_last = uplift_next

        AUC = pred_riemann - 0.5*overall_lift*(scoring_range[1]**2 - scoring_range[0]**2)
        maxgain = np.amax(uplifts)
        return AUC, maxgain

    # Dictionary to store all scores.
    scores = {}
    # Raw max scores.
    scores['Q_max'] = qini_max
    scores['overall_lift'] = overall_lift
    scores['Q_practical_max'] = practical_qini_max
    if (plot_type=='qini') or (plot_type=='all'):
        # Qini curve scores.
        scores['Q_qini'], scores['max_qini'] = auc(method='qini')
        scores['q1_qini'] = scores['Q_qini']/scores['Q_max']
        scores['q2_qini'] = scores['Q_qini']/scores['Q_practical_max']
    if (plot_type=='cgains') or (plot_type=='all'):
        # Scores for cumulative gains curve.
        scores['Q_cgains'], scores['max_cgains'] = auc(method='cgains')
        scores['q1_cgains'] = scores['Q_cgains']/scores['Q_max']
        scores['q2_cgains'] = scores['Q_cgains']/scores['Q_practical_max']
    if (plot_type=='aqini') or (plot_type=='all'):
        # Scores for adjusted qini curve.
        scores['Q_aqini'], scores['max_aqini'] = auc(method='aqini')
        scores['q1_aqini'] = scores['Q_aqini']/scores['Q_max']
        scores['q2_aqini'] = scores['Q_aqini']/scores['Q_practical_max']

    return scores

class UpliftEval:
    """Calculate qini and uplift curves given some input data.

    Requires three input vectors: `treatment`, `outcome`, `prediction`. Generates
    based qini and uplift curves where populations are ranked by `prediction`, and
    uplift is calculated over `treatment` and `outcome`.

    This class can be used independently of the methods in this package, i.e.
    if you want to evaluate the performance of an externally generated model.

    NOTE: the maximal Qini curves do not currently work with continuous outcomes.

    Parameters
    ----------
    treatment : array-like
        Array of 1s and 0s indicating whether a treatment was served.
    outcome : array-like
        Arrays of nonzero values and zeros indicating whether a response occurred.
    prediction : array-like
        Predicted value used to rank.
    p : float, None, or array-like
        The treatment policy, P(treatment==1). Can be a float if uniform across
        all individuals; or an array if individual-dependent.

    """

    def __init__(self, treatment, outcome, prediction, p="infer", n_bins=20):

        # Cast input arrays as np.array.
        self.treatment = _ensure_array(treatment)
        self.outcome = _ensure_array(outcome)
        self.prediction = _ensure_array(prediction)

        # Deal with `p`, in case float or None.
        if type(p) == str:
            if p == "infer":
                self.p = np.ones(self.prediction.shape)*len(self.treatment[self.treatment==1])/len(self.treatment)
        elif type(p) == float:
            self.p = np.ones(self.prediction.shape)*p
        else:  # If array.
            self.p = _ensure_array(p)

        plot_types = ['qini', 'aqini', 'cgains', 'cuplift', 'uplift', 'balance']

        # Counts.
        Nt1o1, Nt0o1, Nt1o0, Nt0o0 = _get_counts(self.treatment, self.outcome, self.p)
        self.N_treat = Nt1o0 + Nt1o1
        self.N_contr = Nt0o0 + Nt0o1
        self.N = self.N_treat + self.N_contr

        # Calculate maximal curves.
        count_functions = [_get_overfit_counts, _get_no_sure_thing_counts, _get_no_sleeping_dog_counts]
        count_names = ['max', 'pmax', 'nosdmax']
        curve_functions = [_maximal_qini_curve, _maximal_uplift_curve, _maximal_cuplift_curve]
        curve_names = ['qini', 'uplift', 'cuplift']
        for count_function, count_name in zip(count_functions, count_names):
            for curve_function, curve_name in zip(curve_functions, curve_names):
                x, y = curve_function(count_function, Nt1o1, Nt0o1, Nt1o0, Nt0o0)
                setattr(self, '{}_{}_x'.format(curve_name, count_name), x)
                setattr(self, '{}_{}_y'.format(curve_name, count_name), y)

        # Calculate curve (qini, uplift, cumulative uplift).
        for plot_type in plot_types:
            self.calc(plot_type=plot_type, n_bins=n_bins)

        # Calculate Q, q1, q2 scores and other metrics.
        scores = get_scores(self.treatment, self.outcome, self.prediction, self.p)
        for key in scores:
            setattr(self, key, scores[key])

        # For backwards compatibility, create a bunch of plotting functions using the plot_type in the function name.
        for plot_type in plot_types:
            new_func = functools.partial(self.plot, plot_type=plot_type)
            setattr(self, 'plot_'+plot_type, new_func)

    def calc(self, plot_type, n_bins=20):
        """Calculate the different curve types.

        Parameters
        ----------
        plot_type: string
            Type of curve to calculate. Options: qini, aqini, cgains, cuplift, balance, uplift.
        n_bins : int, optional
            Number of bins to use.

        Returns
        -------
        percentile: list
            The percentile of the population, calculated from the bins.
        qini_y: list
            The qini value for each of the percentile points.
        """
        setattr(self, plot_type+'_n_bins', n_bins)

        # Create bins.
        bin_range = np.linspace(0, len(self.treatment), n_bins+1).astype(int)

        # Define whether the curve uses all data up to the percentile, or the data within that percentile.
        def noncumulative_subset_func(i):
            return np.isin(list(range(len(self.treatment))), prob_index[bin_range[i]:bin_range[i+1]])
        def cumulative_subset_func(i):
            return np.isin(list(range(len(self.treatment))), prob_index[:bin_range[i+1]])

        subsetting_functions = {
            'qini': cumulative_subset_func,
            'aqini': cumulative_subset_func,
            'cgains': cumulative_subset_func,
            'cuplift': cumulative_subset_func,
            'balance': noncumulative_subset_func,
            'uplift': noncumulative_subset_func,
        }

        # Define the function that is calculated within the above bins.
        y_calculating_functions = {
            'qini': lambda nt1o1, nt0o1, nt1, nt0: nt1o1/self.N_treat - nt0o1/self.N_contr,
            'aqini': lambda nt1o1, nt0o1, nt1, nt0: nt1o1/self.N_treat - nt0o1*nt1/(nt0*self.N_treat + EPS),
            'cgains': lambda nt1o1, nt0o1, nt1, nt0: (nt1o1/(nt1+EPS)- nt0o1/(nt0+EPS))*(nt1+nt0)/self.N,
            'cuplift': lambda nt1o1, nt0o1, nt1, nt0: nt1o1/(nt1+EPS) - nt0o1/(nt0+EPS),
            'uplift': lambda nt1o1, nt0o1, nt1, nt0: nt1o1/(nt1+EPS) - nt0o1/(nt0+EPS),
            'balance': lambda nt1o1, nt0o1, nt1, nt0: nt1/(nt0+nt1+EPS)
        }

        # Initialize output lists.
        x = []
        y = []

        # Sort `self.prediction`, descending, then get the indices in the test set that
        # these correspond to.
        prob_index = np.flip(np.argsort(self.prediction), 0)

        # Calculate qini curve points for each bin.
        for i in range(n_bins):
            current_subset = subsetting_functions[plot_type](i)
            # Get the values of outcome in this subset for test and control.
            treated_subset = (self.treatment==1) & current_subset
            resp_treated = self.outcome[treated_subset]
            untreated_subset = (self.treatment==0) & current_subset
            resp_untreated = self.outcome[untreated_subset]
            # Get the policy for each of these as well.
            p_treated = self.p[treated_subset]
            p_untreated = self.p[untreated_subset]

            # Count the number of correct values (i.e. y==1) within each of these
            # sections as a fraction of total ads shown.
            nt1o1 = np.sum(resp_treated*0.5/p_treated)
            nt0o1 = np.sum(resp_untreated*0.5/(1-p_untreated))
            nt1 = np.sum(0.5/p_treated)
            nt0 = np.sum(0.5/(1-p_untreated))
            y.append(y_calculating_functions[plot_type](nt1o1, nt0o1, nt1, nt0))

            x.append(nt1+nt0)

        # For non-cumulative functions, we need to do a cumulative sum of the x
        # values, because the sums in the loop only captured the counts within
        # the non-cumulative bins.
        if plot_type in ['balance', 'uplift']:
            x = np.cumsum(x)

        # Rescale x so it's between 0 and 1.
        percentile = x/np.amax(x)

        if plot_type not in ['balance', 'uplift', 'cuplift']:
            percentile = np.insert(percentile, 0, 0)
            y.insert(0,0)

        setattr(self, plot_type+'_x', percentile)
        setattr(self, plot_type+'_y', y)

        return percentile, y

    def plot(self, plot_type='cgains', ax=None, show_theoretical_max=False, show_practical_max=False, show_random_selection=True, show_no_dogs=False, **kwargs):
        """Plots the different kinds of percentage-targeted curves.

        Parameters
        ----------
        plot_type : string, optional
            Either 'qini', 'aqini', 'uplift', 'cuplift', or 'balance'.
            'aqini' refers to an adjusted qini plot, 'cuplift' gives a
            cumulative uplift plot. 'balance' gives the test-control balance
            for each of the bins. All others are self-explanatory.
        ax: matplotlib.axes._subplots.AxesSubplot, optional
            A matplotlib axis referencing where to plot.
        show_theoretical_max: boolean, optional
            Toggle theoretical maximal qini curve, if overfitting to
            treatment/control. Only works for Qini-style curves.
        show_practical_max : boolean, optional
            Toggle theoretical maximal qini curve, if not overfitting to
            treatment/control. Only works for Qini-style curves.
        show_no_dogs : boolean, optional
            Toggle theoretical maximal qini curve, if you believe there are no
            sleeping dogs. Only works for Qini-style curves.
        show_random_selection: boolean, optional
            Toggle straight line indicating a random ordering. Only works for
            Qini-style curves.

        """
        if not ax:
            fig, ax = _plot_defaults()

        titles = {
            'qini': 'Qini curve',
            'aqini': 'Adjusted Qini curve',
            'cgains': 'Cumulative gain chart',
            'cuplift': 'Cumulative uplift curve',
            'uplift': 'Uplift curve',
            'balance': 'Treatment balance curve',
        }

        ylabels = {
            'qini': 'Uplift gain',
            'aqini': 'Uplift gain',
            'cgains': 'Uplift gain',
            'cuplift': 'Cumulative lift',
            'uplift': 'Lift',
            'balance': 'Treatment size / (treatment size + control size)',
        }

        ax.plot(getattr(self, plot_type+'_x'), getattr(self, plot_type+'_y'), '.-', **kwargs)

        if (plot_type=='aqini') or (plot_type=='cgains') or (plot_type=='qini'):
            max_plot_type='qini'
            # Plot random selection line.
            if show_random_selection:
                ax.plot([0,1], [0, getattr(self, plot_type+'_y')[-1]], '--', color=[0.6, 0.6, 0.6], label='Random selection')
        else:
            max_plot_type=plot_type

        if plot_type!='balance':
            # Show theoretical maxes.
            if show_theoretical_max:
                ax.plot(getattr(self, max_plot_type+'_max_x'), \
                    getattr(self, max_plot_type+'_max_y'), 'g--', label='Theoretical max')
            if show_practical_max:
                ax.plot(getattr(self, max_plot_type+'_pmax_x'), \
                    getattr(self, max_plot_type+'_pmax_y'), '--', color=[0.7,0.3,0.3], label='Practical max')
            if show_no_dogs:
                ax.plot(getattr(self, max_plot_type+'_nosdmax_x'), \
                    getattr(self, max_plot_type+'_nosdmax_y'), '--', color=[0.5,0.3,0.7], label='No sleeping dogs')

        ax.set_xlabel('Fraction of data')
        ax.set_ylabel(ylabels[plot_type])
        ax.set_title(titles[plot_type])

        ax.legend(frameon=False)
        return ax

