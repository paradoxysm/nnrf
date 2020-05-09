## Documentation

All documentation for nnrf is located here!

[**pydoc**](https://github.com/paradoxysm/nnrf/tree/master/doc/pydoc) : Documentation regarding python classes and functions.

[**guides**](https://github.com/paradoxysm/nnrf/tree/master/doc/guides) : Guides on using SleepCluster for cluster analysis and classification.

## NNRF Overview

NNRF is a neural network with random forest structure as detailed by [Wang, S. et. al.](https://pdfs.semanticscholar.org/c0b1/2e04be429e70c0303215a3df21f5c5843052.pdf). This package implements the decision tree structured neural network (NNDT) and the accompanying NNRF, both in sklearn API style.

The NNDT is structured like a binary decision tree with each node simulating a split. Furthermore, each node is a visible-hidden hybrid, taking in input from it ancestor node along with *r* features from the data. By training through backpropagation, NNDTs are able to model more powerful splits as well as tune all splits from leaf to root, resulting in better performance compared to traditional decision trees.

The NNRF creates an ensemble of NNDTs that are each trained with bootstrapped data and features, resulting in a powerful model that generalizes well.
