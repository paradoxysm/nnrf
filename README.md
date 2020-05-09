## Neural Network with Random Forest Structure

[![Travis](https://flat.badgen.net/travis/paradoxysm/nnrf?label=build)](https://travis-ci.com/paradoxysm/nnrf)
[![Codecov](https://flat.badgen.net/codecov/c/github/paradoxysm/nnrf?label=coverage)](https://codecov.io/gh/paradoxysm/nnrf)
[![GitHub](https://flat.badgen.net/github/license/paradoxysm/nnrf)](https://github.com/paradoxysm/nnrf/blob/master/LICENSE)

NNRF is a neural network with random forest structure as detailed by [Wang, S. et. al.](https://pdfs.semanticscholar.org/c0b1/2e04be429e70c0303215a3df21f5c5843052.pdf). This package implements the decision tree structured neural network (NNDT) and the accompanying NNRF, both in sklearn API style.

The NNDT is structured like a binary decision tree with each node simulating a split. Furthermore, each node is a visible-hidden hybrid, taking in input from it ancestor node along with *r* features from the data. By training through backpropagation, NNDTs are able to model more powerful splits as well as tune all splits from leaf to root, resulting in better performance compared to traditional decision trees.

The NNRF creates an ensemble of NNDTs that are each trained with bootstrapped data and features, resulting in a powerful model that generalizes well.

More details regarding nnrf can be found in the documentation [here](https://github.com/paradoxysm/nnrf/tree/master/doc).

## Installation

Once you have a suitable python environment setup, nnrf can be easily installed using `pip`:
```
pip install nnrf
```
> nnrf is tested and supported on Python 3.4+ up to Python 3.7. Usage on other versions of Python is not guaranteed to work as intended.

## Using NNRF and NNDT

NNRF and NNDT can be used to classify data very easily. Furthermore, they extend sklearn's BaseEstimator and generally follow sklearn's API.

```python
from nnrf import NNRF

# Create and fit an NNRF model with 50 NNDTs with depth of 5
nnrf = NNRF(n=50, d=5).fit(X_train, Y_train)

# Predict some data
predictions = nnrf.predict(X_test)
```

For full details on usage, see the [documentation](https://github.com/paradoxysm/nnrf/tree/master/doc).

## Changelog

See the [changelog](https://github.com/paradoxysm/nnrf/blob/master/CHANGES.md) for a history of notable changes to nnrf.

## Development

[![Code Climate maintainability](https://img.shields.io/codeclimate/maintainability-percentage/paradoxysm/nnrf?style=flat-square)](https://codeclimate.com/github/paradoxysm/nnrf/maintainability)

nnrf is mostly complete, however could use some more rigorous testing.
Finally, nnrf doesn't faithfully implement sklearn's API all too well - it was mean to follow the API in spirit but wasn't intended to be integrated (as of yet). Any assistance on this is more than welcome!

## Help and Support

### Documentation

Documentation for nnrf can be found [here](https://github.com/paradoxysm/nnrf/tree/master/doc).

### Issues and Questions

Issues and Questions should be posed to the issue tracker [here](https://github.com/paradoxysm/nnrf/issues).
