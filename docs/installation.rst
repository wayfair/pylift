Installation
============

**pylift** has only been tested on Python **3.6** and **3.7**. It currently requires the following package versions:

::

   matplotlib >= 2.1.0
   numpy >= 1.13.3
   scikit-learn >= 0.19.1
   scipy >= 1.0.0
   xgboost >= 0.6a2

A ``requirements.txt`` file is included in the parent directory of the github repo that contains these lower-limit package versions, as these are the versions we have most extensively tested pylift on, but newer versions generally appear to work.

The package can be built from source (for the latest version) or simply sourced from pypi. To install from source, clone the repo and install, using the following commands:

::

   git clone https://github.com/wayfair/pylift
   cd pylift
   pip install .

To upgrade, ``git pull origin master`` in the repo folder, and then run ``pip install --upgrade --no-cache-dir .``.

Alternatively, install from pypi by simply running ``pip install pylift``.
