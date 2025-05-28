"""
Active learning sampling strategies.

Extracted from the original notebook's sampling functions.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Union, List
from scipy.stats import entropy


class SamplingStrategy(ABC):
    """
    Abstract base class for sampling strategies.
    """
    
    @abstractmethod
    def select_sample(self, learner, X_pool: np.ndarray) -> int:
        """
        Select the next sample to query.
        
        Parameters:
        -----------
        learner : object
            Trained model or committee
        X_pool : np.ndarray
            Pool of unlabeled samples
            
        Returns:
        --------
        int
            Index of the selected sample
        """
        pass


class LeastConfidenceSampling(SamplingStrategy):
    """
    Least confidence sampling strategy.
    """
    
    def select_sample(self, learner, X_pool: np.ndarray) -> int:
        """
        Select the sample with highest uncertainty (least confidence).
        
        Parameters:
        -----------
        learner : sklearn classifier
            Trained classifier with predict_proba method
        X_pool : np.ndarray
            Pool of unlabeled samples
            
        Returns:
        --------
        int
            Index of the most uncertain sample
        """
        probs = learner.predict_proba(X_pool)
        uncertainties = 1 - np.max(probs, axis=1)
        return np.argmax(uncertainties)


class QBCVoteEntropySampling(SamplingStrategy):
    """
    Query-by-Committee vote entropy sampling strategy.
    """
    
    def select_sample(self, committee: List, X_pool: np.ndarray) -> int:
        """
        Select the sample with highest vote entropy among committee members.
        
        Parameters:
        -----------
        committee : List
            List of trained classifiers
        X_pool : np.ndarray
            Pool of unlabeled samples
            
        Returns:
        --------
        int
            Index of the sample with highest disagreement
        """
        votes = np.array([clf.predict(X_pool) for clf in committee])
        n_classifiers = votes.shape[0]
        n_samples = votes.shape[1]
        vote_entropies = np.zeros(n_samples)
        
        for i in range(n_samples):
            sample_votes = votes[:, i]
            _, counts = np.unique(sample_votes, return_counts=True)
            proportions = counts / n_classifiers
            vote_entropies[i] = entropy(proportions)
            
        return np.argmax(vote_entropies)


class RandomSampling(SamplingStrategy):
    """
    Random sampling strategy (baseline).
    """
    
    def select_sample(self, learner, X_pool: np.ndarray) -> int:
        """
        Randomly select a sample.
        
        Parameters:
        -----------
        learner : object
            Learner (not used in random sampling)
        X_pool : np.ndarray
            Pool of unlabeled samples
            
        Returns:
        --------
        int
            Randomly selected index
        """
        return np.random.choice(range(len(X_pool)))


# Functional interface for backward compatibility
def least_confidence_sampling(learner, X_pool: np.ndarray) -> int:
    """
    Selects the sample with the highest uncertainty from X_pool.
    
    Parameters:
    -----------
    learner : sklearn classifier
        Trained classifier with predict_proba method
    X_pool : np.ndarray
        Pool of unlabeled samples
        
    Returns:
    --------
    int
        Index of the most uncertain sample
    """
    strategy = LeastConfidenceSampling()
    return strategy.select_sample(learner, X_pool)


def qbc_vote_entropy_sampling(committee: List, X_pool: np.ndarray) -> int:
    """
    Computes vote entropy for each sample and returns the index with highest entropy.
    
    Parameters:
    -----------
    committee : List
        List of trained classifiers
    X_pool : np.ndarray
        Pool of unlabeled samples
        
    Returns:
    --------
    int
        Index of the sample with highest disagreement
    """
    strategy = QBCVoteEntropySampling()
    return strategy.select_sample(committee, X_pool)


def random_sample(learner, X_pool: np.ndarray) -> int:
    """
    Randomly selects a sample from X_pool.
    
    Parameters:
    -----------
    learner : object
        Learner (not used in random sampling)
    X_pool : np.ndarray
        Pool of unlabeled samples
        
    Returns:
    --------
    int
        Randomly selected index
    """
    strategy = RandomSampling()
    return strategy.select_sample(learner, X_pool)