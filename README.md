# WeightedLinearBandits

# Description

Code associated with the NeurIPS19 paper *Weighted Linear Bandits in Non-Stationary Environments* available [here](https://arxiv.org/abs/1909.09146). The article is a joint work with Claire Vernade and Olivier Cappé.

This package implements the `D-LinUCB` algorithm presented in the paper. This algorithm builds an estimate of the unkwown vector of the linear model based on weighted least-squares rather than least-squares. By doing so, the new estimate is more robust to non-stationarity.

The empirical performance of the algorithm is reported in two simulated experiments. Those experiments can be easily reproduced by cloning the package, and running the Jupyter notebook in the [Experiment](Experiments/) folder.

