.. pylift documentation master file, created by
   sphinx-quickstart on Tue Aug 19 09:15:00 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pylift's documentation!
==================================

**pylift** is an uplift library that provides, primarily, (1) fast uplift modeling implementations and (2) evaluation tools. While other packages and more exact methods exist to model uplift, **pylift** is designed to be quick, flexible, and effective. **pylift** heavily leverages the optimizations of other packages -- namely, `xgboost`, `sklearn`, `pandas`, `matplotlib`, `numpy`, and `scipy`.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   introduction
   quick-start
   usage
   evaluation
   policy
   raw-data
   contributing

**pylift** has two main features:

#. A `TransformedOutcome` class (inheriting a more general `BaseProxyMethod` class) that allows for full end-to-end uplift modeling.
#. An `UpliftEval` class that allows for evaluation of any model prediction. This class is used within the `TransformedOutcome` class, but can be called independently to evaluate the performance of, for example, scores from  a modeling approach external to **pylift**.

The `TransformedOutcome` class (and so, the `BaseProxyMethod` class) simply wraps `sklearn` classes and functions. Therefore, it's generally possible to do anything you can do with `sklearn` within `pylift` as well. Advanced usage of **pylift**, therefore, should feel familiar to those well-versed in `sklearn`.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
