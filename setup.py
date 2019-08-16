from setuptools import find_packages, setup

setup(name='pylift',
      version='0.0.1',
      description='Python implementation of uplift modeling.',
      author='Robert Yi, Will Frost',
      author_email='robert@ryi.me',
      url='https://github.com/rsyi/pylift',
      install_requires=[
            'numpy',
            'matplotlib',
            'scikit-learn',
            'scipy',
            'seaborn',
            'xgboost'
          ],
      packages=find_packages(),
      zip_safe=False)
