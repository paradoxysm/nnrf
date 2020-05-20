## Documentation

All documentation for `nnrf` is located here!

[**pydoc**](https://github.com/paradoxysm/nnrf/tree/master/doc/pydoc) : Documentation regarding python classes and functions.

[**guides**](https://github.com/paradoxysm/nnrf/tree/master/doc/guides) : Guides on using `nnrf`.

## Overview

Decision Trees and Random Forests are simple and powerful models for classification. However, they are restricted to making linear splits at each node in a tree and, critically, cannot generalize their loss metrics. Here, a neural network overcomes these limitations. By structuring a neural network as a decision tree, a model can be created with properties of both decision trees and neural networks.

`nnrf` implements models to work with such neural networks with decision tree structure (NNDTs). This includes NNDTs, and a random forest implementation (NNRF). `nnrf` also includes basic neural networks and a Dynamic Ensemble Selection (DES-KNN) model, which can be used to stack on top of an NNRF, enabling more powerful decision making over the basic random forest implementation.

For more details regarding NNDTs and NNRFs, see the original paper by [Wang, S. et. al.](https://pdfs.semanticscholar.org/c0b1/2e04be429e70c0303215a3df21f5c5843052.pdf).

This package is implemented similar to the `sklearn` API style. It is not fully faithful as it includes modularity regarding activation, regularization, loss, and optimizer functions.

### Neural Networks

`nnrf` implements simple neural networks as a quick utility accompaniment. Read more [here](https://github.com/paradoxysm/nnrf/blob/master/doc/nn.md).

### Neural Networks with Decision Tree Structure

NNDTs are neural networks structured as a binary decision tree. They modify conventional neural networks by hybridizing hidden layers into input-hidden layers and by restricting activation paths. Read more [here](https://github.com/paradoxysm/nnrf/blob/master/doc/nndt.md).

### Random Forest of NNDTs

NNRFs are ensembles of NNDTs. Read more [here](https://github.com/paradoxysm/nnrf/blob/master/doc/nnrf.md).

### Dynamic Ensemble Selection

Dynamic Ensemble Selection (DES) is a class of ensembles that select a subset of the ensemble classifiers at inference time (dynamically). This strategy rests on the observation that some members of the ensemble perform better than other members for a given region of data. `nnrf` implements a simplistic k-Nearest Neighbors version of DES. Read more [here](https://github.com/paradoxysm/nnrf/blob/master/doc/des.md).
