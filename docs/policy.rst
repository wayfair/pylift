Usage: custom targeting policy
==============================

Often a randomized experiment is not economical, and so the only data available is data collected according to some *targeting policy*. In other words, you vary probability of treatment :math:`P(W=1)`, often according either to some business logic or a model output. As long as all individuals are targeted with some non-zero probability, it is still possible to train an unbiased model using this data by simply weighting calculations on each individual according to :math:`P(W=1)`.

In its current state, pylift supports this kind of correction to an extent. We have added the ability to correct the Qini-style evaluation curves according to a treatment policy (simply add an argument ``p``, defined as :math:`P(W=1)`. We’ve also adjusted the transformation to allow the policy information to be encoded in the transformation (``pylift.methods.derivatives.TransformedOutcome._transform_func``). By specifying a column string that in the keyword argument ``col_policy`` that specifies the row-level probability of treatment, this encoding is automatically created.

It is therefore possible to write a custom objective function that recovers ``treatment``, ``outcome``, and ``policy`` information from the transformed outcome (using the ``TransformedOutcome._untransform_func`` function), then adapting the objective function (if possible – this works with ``xgboost``, but not ``sklearn``) accordingly. However, we have not yet explicitly implemented this custom objective function.
