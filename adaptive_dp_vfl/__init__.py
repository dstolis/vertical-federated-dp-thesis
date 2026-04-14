"""Adaptive Differentially Private Vertical Federated Learning on Flower.

This package implements a two-party VFL system in which each party holds a
disjoint subset of the features for a shared set of entities. Bottom models
live on Flower ``ClientApp``s; a top model and a custom ``Strategy`` live on
the Flower ``ServerApp``. Per-sample clipping and calibrated Gaussian noise
are applied to intermediate representations before they leave each client;
the server runs an adaptive tuner that adjusts the learning rate, local
epochs, and clipping threshold between rounds.
"""

__version__ = "0.2.0"
