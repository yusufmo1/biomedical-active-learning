"""
Biomedical Active Learning Package

A comprehensive framework for active learning on biomedical datasets,
demonstrating superior performance with minimal labeled data.

Key Features:
- Blood-Brain Barrier Penetration prediction with molecular features
- Breast Cancer classification with clinical features
- Random Forest and Query-by-Committee active learning strategies
- Comprehensive evaluation and visualization tools
- Parallel processing support for large-scale experiments

Example Usage:
    >>> from src.data import DatasetLoader
    >>> from src.active_learning import ALExperiment
    >>> 
    >>> # Load data
    >>> loader = DatasetLoader()
    >>> bbb_df, bc_df = loader.load_both_datasets()
    >>> 
    >>> # Run active learning experiment
    >>> config = {'max_queries': -1, 'batch_size': 10, 'n_runs': 10}
    >>> experiment = ALExperiment(config)
    >>> results = experiment.run_experiment(X_train, y_train, X_test, y_test)

Authors: Yusuf Mohammed
License: MIT
"""

__version__ = "0.1.0"
__author__ = "Yusuf Mohammed"
__email__ = "your.email@example.com"

# Import main modules
from . import data
from . import features
from . import active_learning
from . import evaluation
from . import dimensionality
from . import utils

__all__ = [
    "data",
    "features", 
    "active_learning",
    "evaluation",
    "dimensionality",
    "utils"
]