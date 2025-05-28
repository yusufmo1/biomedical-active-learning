"""
Evaluation metrics for biomedical active learning.

Extracted from the original notebook's evaluation functions.
"""

import numpy as np
from typing import Dict, List, Tuple
from scipy.stats import mode

from sklearn.metrics import (
    matthews_corrcoef, f1_score, roc_auc_score, roc_curve,
    confusion_matrix, classification_report, accuracy_score
)


def evaluate_model(model, X: np.ndarray, y: np.ndarray) -> Dict[str, any]:
    """
    Evaluate a single classifier on data (X, y).
    
    Parameters:
    -----------
    model : sklearn classifier
        Trained model with predict and predict_proba methods
    X : np.ndarray
        Features
    y : np.ndarray
        True labels
        
    Returns:
    --------
    Dict[str, any]
        Dictionary with keys: 'mcc', 'f1', 'roc_auc', and 'roc_curve'
    """
    y_pred = model.predict(X)
    mcc = matthews_corrcoef(y, y_pred)
    f1 = f1_score(y, y_pred, average='weighted')
    
    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(X)[:, 1]
        roc_auc = roc_auc_score(y, probas)
        fpr, tpr, thresholds = roc_curve(y, probas)
    else:
        roc_auc = np.nan
        fpr, tpr, thresholds = None, None, None
        
    return {
        "mcc": mcc, 
        "f1": f1, 
        "roc_auc": roc_auc, 
        "roc_curve": (fpr, tpr, thresholds),
        "accuracy": accuracy_score(y, y_pred),
        "confusion_matrix": confusion_matrix(y, y_pred)
    }


def evaluate_committee(committee: List, X: np.ndarray, y: np.ndarray) -> Dict[str, any]:
    """
    Evaluate a QBC committee on data (X, y).
    Uses majority vote for predictions and averages probabilities for ROC AUC.
    
    Parameters:
    -----------
    committee : List
        List of trained classifiers
    X : np.ndarray
        Features
    y : np.ndarray
        True labels
        
    Returns:
    --------
    Dict[str, any]
        Dictionary with evaluation metrics
    """
    y_pred = qbc_predict(committee, X)
    mcc = matthews_corrcoef(y, y_pred)
    f1 = f1_score(y, y_pred, average='weighted')
    
    probas_list = []
    for clf in committee:
        if hasattr(clf, "predict_proba"):
            probas_list.append(clf.predict_proba(X)[:, 1])
            
    if probas_list:
        avg_probas = np.mean(probas_list, axis=0)
        roc_auc = roc_auc_score(y, avg_probas)
        fpr, tpr, thresholds = roc_curve(y, avg_probas)
    else:
        roc_auc = np.nan
        fpr, tpr, thresholds = None, None, None
        
    return {
        "mcc": mcc, 
        "f1": f1, 
        "roc_auc": roc_auc, 
        "roc_curve": (fpr, tpr, thresholds),
        "accuracy": accuracy_score(y, y_pred),
        "confusion_matrix": confusion_matrix(y, y_pred)
    }


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


def qbc_predict_proba(committee: List, X: np.ndarray) -> np.ndarray:
    """
    Get probability predictions from a QBC committee by averaging member probabilities.
    
    Parameters:
    -----------
    committee : List
        List of trained classifiers
    X : np.ndarray
        Features to predict
        
    Returns:
    --------
    np.ndarray
        Averaged probability predictions
    """
    # Get predictions from all committee members
    all_probs = np.array([member.predict_proba(X)[:, 1] for member in committee])
    # Return mean probability across committee
    return np.mean(all_probs, axis=0)


def calculate_per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[int, float]:
    """
    Calculate per-class accuracy.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
        
    Returns:
    --------
    Dict[int, float]
        Per-class accuracy dictionary
    """
    classes = np.unique(y_true)
    per_class_acc = {}
    
    for cls in classes:
        mask = (y_true == cls)
        if np.sum(mask) > 0:
            per_class_acc[cls] = accuracy_score(y_true[mask], y_pred[mask])
        else:
            per_class_acc[cls] = 0.0
            
    return per_class_acc


class ModelEvaluator:
    """
    Class for comprehensive model evaluation.
    """
    
    def __init__(self):
        """Initialize model evaluator."""
        self.results = {}
        
    def evaluate_single_model(self, 
                             model, 
                             X_test: np.ndarray, 
                             y_test: np.ndarray,
                             model_name: str = "model") -> Dict[str, any]:
        """
        Evaluate a single model comprehensively.
        
        Parameters:
        -----------
        model : sklearn classifier
            Trained model
        X_test : np.ndarray
            Test features
        y_test : np.ndarray
            Test labels
        model_name : str
            Name for the model
            
        Returns:
        --------
        Dict[str, any]
            Comprehensive evaluation results
        """
        results = evaluate_model(model, X_test, y_test)
        
        # Add per-class metrics
        y_pred = model.predict(X_test)
        results['per_class_accuracy'] = calculate_per_class_accuracy(y_test, y_pred)
        results['classification_report'] = classification_report(y_test, y_pred, output_dict=True)
        
        self.results[model_name] = results
        return results
        
    def evaluate_committee(self, 
                          committee: List, 
                          X_test: np.ndarray, 
                          y_test: np.ndarray,
                          committee_name: str = "committee") -> Dict[str, any]:
        """
        Evaluate a committee of models.
        
        Parameters:
        -----------
        committee : List
            List of trained models
        X_test : np.ndarray
            Test features
        y_test : np.ndarray
            Test labels
        committee_name : str
            Name for the committee
            
        Returns:
        --------
        Dict[str, any]
            Comprehensive evaluation results
        """
        results = evaluate_committee(committee, X_test, y_test)
        
        # Add per-class metrics
        y_pred = qbc_predict(committee, X_test)
        results['per_class_accuracy'] = calculate_per_class_accuracy(y_test, y_pred)
        results['classification_report'] = classification_report(y_test, y_pred, output_dict=True)
        
        # Individual member performance
        member_results = []
        for i, member in enumerate(committee):
            member_result = evaluate_model(member, X_test, y_test)
            member_results.append(member_result)
            
        results['member_results'] = member_results
        results['member_mcc_std'] = np.std([r['mcc'] for r in member_results])
        results['member_f1_std'] = np.std([r['f1'] for r in member_results])
        
        self.results[committee_name] = results
        return results
        
    def compare_models(self, metrics: List[str] = ['mcc', 'f1', 'roc_auc']) -> Dict[str, Dict[str, float]]:
        """
        Compare models across specified metrics.
        
        Parameters:
        -----------
        metrics : List[str]
            List of metrics to compare
            
        Returns:
        --------
        Dict[str, Dict[str, float]]
            Comparison results
        """
        comparison = {}
        
        for metric in metrics:
            comparison[metric] = {}
            for model_name, results in self.results.items():
                if metric in results:
                    comparison[metric][model_name] = results[metric]
                    
        return comparison
        
    def get_best_model(self, metric: str = 'mcc') -> Tuple[str, float]:
        """
        Get the best performing model for a given metric.
        
        Parameters:
        -----------
        metric : str
            Metric to use for comparison
            
        Returns:
        --------
        Tuple[str, float]
            Best model name and its score
        """
        best_score = -np.inf
        best_model = None
        
        for model_name, results in self.results.items():
            if metric in results and results[metric] > best_score:
                best_score = results[metric]
                best_model = model_name
                
        return best_model, best_score