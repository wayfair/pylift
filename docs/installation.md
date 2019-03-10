# Installation

**pylift** has only been tested on Python **3.6** and **3.7**. It currently requires the following package versions:

```
matplotlib >= 2.1.0
numpy >= 1.13.3
scikit-learn >= 0.19.1
scipy >= 1.0.0
xgboost >= 0.6a2
```
A `requirements.txt` file is included in the parent directory of the github repo that contains these lower-limit package versions, as these are the versions we have most extensively tested pylift on, but newer versions generally appear to work.

At the moment, the package must be built from source. This means cloning the repo and installing, using the following commands:

```
git clone https://github.com/wayfair/pylift
cd pylift
pip install .
```

To upgrade, `git pull origin master` in the repo folder, and
then run `pip install --upgrade --no-cache-dir .`.

