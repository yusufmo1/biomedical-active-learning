"""
Active learning models and factories.

Extracted from the original notebook's model factory functions.
"""

import copy
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from scipy.stats import mode

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


class ActiveLearner(ABC):
    """
    Abstract base class for active learners.
    """
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the learner to labeled data.
        
        Parameters:
        -----------
        X : np.ndarray
            Training features
        y : np.ndarray
            Training labels
        """
        pass
        
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Parameters:
        -----------
        X : np.ndarray
            Features to predict
            
        Returns:
        --------
        np.ndarray
            Predictions
        """
        pass
        
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Parameters:
        -----------
        X : np.ndarray
            Features to predict
            
        Returns:
        --------
        np.ndarray
            Prediction probabilities
        """
        pass


class RandomForestAL(ActiveLearner):
    """
    Random Forest active learner.
    """
    
    def __init__(self, **params):
        """
        Initialize Random Forest learner.
        
        Parameters:
        -----------
        **params : dict
            Parameters for RandomForestClassifier
        """
        default_params = {'n_estimators': 100, 'random_state': 42}
        default_params.update(params)
        self.model = RandomForestClassifier(**default_params)
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the Random Forest model."""
        return self.model.fit(X, y)
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        return self.model.predict_proba(X)


class QueryByCommitteeAL(ActiveLearner):
    """
    Query-by-Committee active learner.
    """
    
    def __init__(self, committee: Optional[List] = None, **params):
        """
        Initialize QBC learner.
        
        Parameters:
        -----------
        committee : List, optional
            List of base learners. If None, default committee is created
        **params : dict
            Parameters for base learners
        """
        if committee is None:
            self.committee = self._create_default_committee(**params)
        else:
            self.committee = committee
            
    def _create_default_committee(self, **params) -> List:
        """
        Create default committee of diverse learners.
        
        Returns:
        --------
        List
            List of base learners
        """
        run_iter = params.get('run_iter', 0)
        base_random_state = params.get('random_state', 42)
        
        committee = []
        
        # Random Forest
        rf_params = {'n_estimators': 100, 'random_state': base_random_state + run_iter}
        rf_params.update(params.get('rf', {}))
        committee.append(RandomForestClassifier(**rf_params))
        
        # Extra Trees
        et_params = {'n_estimators': 100, 'random_state': base_random_state + run_iter}
        et_params.update(params.get('extratrees', {}))
        committee.append(ExtraTreesClassifier(**et_params))
        
        # Gradient Boosting
        gb_params = {'n_estimators': 100, 'random_state': base_random_state + run_iter}
        gb_params.update(params.get('gb', {}))
        committee.append(GradientBoostingClassifier(**gb_params))
        
        # Logistic Regression
        lr_params = {'solver': 'lbfgs', 'max_iter': 1000, 'random_state': base_random_state + run_iter}
        lr_params.update(params.get('lr', {}))
        committee.append(LogisticRegression(**lr_params))
        
        # K-Nearest Neighbors
        knn_params = {'n_neighbors': 5}
        knn_params.update(params.get('knn', {}))
        committee.append(KNeighborsClassifier(**knn_params))
        
        return committee
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit all committee members."""
        for clf in self.committee:
            clf.fit(X, y)
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using majority vote."""
        predictions = np.array([clf.predict(X) for clf in self.committee])
        maj_vote = mode(predictions, axis=0)[0].flatten()
        return maj_vote
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities by averaging committee members."""
        probas_list = []
        for clf in self.committee:
            if hasattr(clf, "predict_proba"):
                probas_list.append(clf.predict_proba(X))
        
        if probas_list:
            return np.mean(probas_list, axis=0)
        else:
            raise ValueError("No committee members support predict_proba")


# Factory functions for backward compatibility
def rf_factory(params: Optional[Dict] = None) -> RandomForestClassifier:
    """
    Creates and returns a RandomForestClassifier using parameters from params.
    
    Parameters:
    -----------
    params : dict, optional
        Parameters for RandomForestClassifier
        
    Returns:
    --------
    RandomForestClassifier
        Configured Random Forest classifier
    """
    if params is None:
        params = {}
    default_params = {'n_estimators': 100, 'random_state': 42}
    default_params.update(params)
    return RandomForestClassifier(**default_params)


def base_learner_factory(qbc_params: Optional[Dict] = None, run_iter: int = 0) -> List:
    """
    Returns a list of 5 new base learners for the Query-By-Committee (QBC) ensemble.
    The run_iter parameter is used to set the random_state as 42 + run_iter.
    
    Parameters:
    -----------
    qbc_params : dict, optional
        Parameters for base learners
    run_iter : int
        Run iteration for random state variation
        
    Returns:
    --------
    List
        List of base learners
    """
    if qbc_params is None:
        qbc_params = {}
        
    learners = []
    
    # RF learner
    rf_params = {'n_estimators': 100, 'random_state': 42 + run_iter}
    rf_params.update(qbc_params.get('rf', {}))
    learners.append(RandomForestClassifier(**rf_params))
    
    # ExtraTrees learner
    et_params = {'n_estimators': 100, 'random_state': 42 + run_iter}
    et_params.update(qbc_params.get('extratrees', {}))
    learners.append(ExtraTreesClassifier(**et_params))
    
    # GradientBoosting learner
    gb_params = {'n_estimators': 100, 'random_state': 42 + run_iter}
    gb_params.update(qbc_params.get('gb', {}))
    learners.append(GradientBoostingClassifier(**gb_params))
    
    # LogisticRegression learner
    lr_params = {'solver': 'lbfgs', 'max_iter': 1000, 'random_state': 42 + run_iter}
    lr_params.update(qbc_params.get('lr', {}))
    learners.append(LogisticRegression(**lr_params))
    
    # KNeighbors learner
    knn_params = {'n_neighbors': 5}
    knn_params.update(qbc_params.get('knn', {}))
    learners.append(KNeighborsClassifier(**knn_params))
    
    return learners


def qbc_predict(committee: List, X: np.ndarray) -> np.ndarray:
    """
    Aggregates predictions from the committee by majority vote.
    
    Parameters:
    -----------
    committee : List
        List of trained classifiers
    X : np.ndarray
        Features to predict
        
    Returns:
    --------
    np.ndarray
        Majority vote predictions
    """
    from joblib import Parallel, delayed
    predictions = Parallel(n_jobs=len(committee))(
        delayed(clf.predict)(X) for clf in committee
    )
    votes = np.array(predictions)
    maj_vote = mode(votes, axis=0)[0].flatten()
    return maj_vote