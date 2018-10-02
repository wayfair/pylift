# pylift

[![Documentation Status](https://readthedocs.org/projects/pylift/badge/?version=latest)](https://pylift.readthedocs.io/en/latest/?badge=latest)

**pylift** is an uplift library that provides, primarily, (1) fast uplift modeling implementations and (2) evaluation tools. While other packages and more exact methods exist to model uplift, **pylift** is designed to be quick,         flexible, and effective. **pylift** heavily leverages the optimizations of other packages -- namely, `xgboost`,       `sklearn`, `pandas`, `matplotlib`, `numpy`, and `scipy`. The primary method currently implemented is the Transformed Outcome proxy method (Athey 2015).

# Reference
Athey, S., & Imbens, G. W. (2015). Machine learning methods for estimating heterogeneous causal effects. stat, 1050(5).
