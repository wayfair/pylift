# Introduction to uplift

In traditional binary classification, we attempt to predict a binary outcome
\(y\) based on some set of features `X`. Consequently, our success criterion is
usually something that measures how well we can predict `y` based on values of
`X` (e.g. accuracy, precision, recall). In uplift modeling, however, the story
is quite different, because while we are still trying to predict the outcome
`y`, we want to predict this outcome *conditioned* on a treatment.  For
example, in traditional classification, we might want to know if someone will
click an ad (`y`), period, while in uplift modeling, we would want to know if
someone will click an ad based on what kind of ad we serve them (treatment) (`y
| treatment`).

The practical ramifications of this constraint can be readily understood in the
context of decision trees. In decision trees, splits are chosen to minimize
impurity -- i.e. to maximize homogeneity of nodes. In the previously mentioned
advertising example, this would mean finding splits that separate those who
would visit our website from those who wouldn't. In uplift modeling, the split
needs to be chosen to take into effect the treatment, and so roughly speaking,
we want to sort out those who would visit our website only when shown an ad
(serving an ad is useful) from those who would visit our website regardless
(serving an ad is useless).

An obvious way to do this is to change the splitting criterion, but because
many of `sklearn`'s built-in implementations do not allow for a custom split
criterion/optimization function, this is not completely trivial. While we could
recreate each method to allow for this customized uplift criterion (and others
have done this), we have discovered that `sklearn` is blazingly fast, and in
making our own homebrew algorithms, we sacrifice some of the benefits
(efficient hyperparameter tuning) that come with speed (the other algorithms we
have encountered are, regrettably, slow). Thus, we seek alternate methods that
allow us to leverage `sklearn` (and other) modules. In particular, we implement
methods here that simply transform the data, but in doing so, code the
Test/Control information in the transformed outcome.

Once the model has been made, evaluation is traditionally accomplished using
what is known as a Qini curve. The Qini curve is defined as
$$\text{Qini curve}(\phi) = \frac{n_{y=1,A}(\phi)}{n_A} - \frac{n_{y=1,B}(\phi)}{n_B}$$

where $`\phi`$ is the fraction of population treated (in either A or B) ordered
by predicted uplift (from highest to lowest). "Random chance" is therefore a
model that cannot distinguish positive and negative uplift and results in a
straight line from (0,0) to $$(1, \frac{n_{y=1,A}}{n_A} -
\frac{n_{y=1,B}}{n_B}).$$ The value `Q` is then the area between the model's
Qini curve and the random chance Qini curve. `Q` has been used throughout the
literature as a way of measuring how good a model is at separating positive and
negative uplift values. However, a problem with this curve that its absolute
value is dependent on how people generally respond to your treatment.
Consequently, it is not particularly useful in understanding how much of the
*potential* uplift you have captured with your model. To this end, we generally
normalize `Q` in a two different ways:

* `q1`: `Q` normalized by the theoretical maximal area.
* `q2`: `Q` normalized by the practical maximal area.

The theoretical maximal curve corresponds to a sorting in which we assume that
an individual is persuadable (uplift = 1) if and only if they respond in the
treatment group (and the same reasoning applies to the control group, for
sleeping dogs). The practical max curve corresponds to a similar sorting, for
which we also assume that all individuals that have a positive outcome in the
treatment group must also have a counterpart (relative to the proportion of
individuals in the treatment and control group) in the control group that did
not respond. This is a more conservative, realistic curve. The former can only
be attained through overfitting, while the latter can only be attained under
very generous circumstances. Within the package, we also calculate the "no
sleeping dogs" curve, which simply precludes the possibility of negative
effects.

To evaluate `Q`, we predict the uplift for each row in our dataset.
We then order the dataset from highest uplift to lowest
uplift and evaluate the Qini curve as a function of the population targeted. The area
between this curve and the x-axis can be approximated by a Riemann sum on the
$`N`$ data points:
$$\text{Qini Curve Area} = \sum_{i=0}^{N-1} \frac{1}{2}\left(\text{Qini curve}(\phi_{i+1})+\text{Qini curve}(\phi_{i})\right)\left(\phi_{i+1} - \phi_{i}\right)$$
where $$\phi_{i} = i/N,$$ and so
$$\text{Qini Curve Area} = \sum_{i=0}^{N-1} \frac{1}{2}\left(\frac{n_{y=1,A}(\phi_{i+1})-n_{y=1,A}(\phi_{i})}{n_A} - \frac{n_{y=1,B}(\phi_{i+1})-n_{y=1,B}(\phi_i)}{n_B}\right)\frac{1}{N}$$
We then need to subtract off the randomized curve area which is given by:
$$\text{Randomized Qini Area} = \frac{1}{2}\left(\frac{n_{y=1,A}}{n_A} - \frac{n_{y=1,B}}{n_B}\right)$$
and so the Qini coefficient is:
$$Q = \text{Qini Curve Area} - \text{Randomized Qini Area}$$
