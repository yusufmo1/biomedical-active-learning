"""
Active learning experiment management.

Extracted from the original notebook's experiment execution functions.
"""

import copy
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Callable
from pathlib import Path
from tqdm import tqdm

from sklearn.model_selection import StratifiedShuffleSplit
from joblib import Parallel, delayed

from .strategies import least_confidence_sampling, qbc_vote_entropy_sampling
from .learners import rf_factory, base_learner_factory, qbc_predict
from ..evaluation.metrics import evaluate_model, evaluate_committee

logger = logging.getLogger(__name__)


class ALExperiment:
    """
    Class for managing active learning experiments.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize active learning experiment.
        
        Parameters:
        -----------
        config : Dict[str, Any]
            Experiment configuration dictionary
        """
        self.config = config
        self.results = {}
        
    def run_experiment(self, 
                      X_unlab: np.ndarray, 
                      y_unlab: np.ndarray,
                      X_holdout: np.ndarray, 
                      y_holdout: np.ndarray,
                      experiment_type: str = "rf_first5") -> Dict[str, Any]:
        """
        Run a complete active learning experiment.
        
        Parameters:
        -----------
        X_unlab : np.ndarray
            Unlabeled pool features
        y_unlab : np.ndarray
            Unlabeled pool labels
        X_holdout : np.ndarray
            Holdout test features
        y_holdout : np.ndarray
            Holdout test labels
        experiment_type : str
            Type of experiment to run
            
        Returns:
        --------
        Dict[str, Any]
            Experiment results
        """
        if experiment_type == "rf_first5":
            return self._run_rf_first5_experiment(X_unlab, y_unlab, X_holdout, y_holdout)
        elif experiment_type == "rf_stratified":
            return self._run_rf_stratified_experiment(X_unlab, y_unlab, X_holdout, y_holdout)
        elif experiment_type == "qbc_first5":
            return self._run_qbc_first5_experiment(X_unlab, y_unlab, X_holdout, y_holdout)
        elif experiment_type == "qbc_stratified":
            return self._run_qbc_stratified_experiment(X_unlab, y_unlab, X_holdout, y_holdout)
        else:
            raise ValueError(f"Unknown experiment type: {experiment_type}")
            
    def _run_rf_first5_experiment(self, 
                                 X_unlab: np.ndarray, 
                                 y_unlab: np.ndarray,
                                 X_holdout: np.ndarray, 
                                 y_holdout: np.ndarray) -> Dict[str, Any]:
        """Run RF active learning with first 5 samples."""
        initial_indices = list(range(5))
        ref_metrics = self._get_reference_metrics(X_unlab, y_unlab, X_holdout, y_holdout, "rf")
        
        rf_params = self.config.get('rf_params', {}).copy()
        rf_params['random_state'] = 42
        learner = rf_factory(rf_params)
        
        return self._execute_active_learning_experiment(
            learner, X_unlab, y_unlab, X_holdout, y_holdout,
            initial_indices, least_confidence_sampling, ref_metrics
        )
        
    def _run_rf_stratified_experiment(self, 
                                     X_unlab: np.ndarray, 
                                     y_unlab: np.ndarray,
                                     X_holdout: np.ndarray, 
                                     y_holdout: np.ndarray) -> Dict[str, Any]:
        """Run RF active learning with stratified 5 samples."""
        seed = self.config.get('stratified_seeds', [42])[0]
        sss = StratifiedShuffleSplit(n_splits=1, train_size=5, random_state=seed)
        for train_index, _ in sss.split(X_unlab, y_unlab):
            initial_indices = train_index.tolist()
            
        ref_metrics = self._get_reference_metrics(X_unlab, y_unlab, X_holdout, y_holdout, "rf")
        learner = rf_factory(self.config.get('rf_params', {}))
        
        return self._execute_active_learning_experiment(
            learner, X_unlab, y_unlab, X_holdout, y_holdout,
            initial_indices, least_confidence_sampling, ref_metrics
        )
        
    def _run_qbc_first5_experiment(self, 
                                  X_unlab: np.ndarray, 
                                  y_unlab: np.ndarray,
                                  X_holdout: np.ndarray, 
                                  y_holdout: np.ndarray) -> Dict[str, Any]:
        """Run QBC active learning with first 5 samples."""
        initial_indices = list(range(5))
        ref_metrics = self._get_reference_metrics(X_unlab, y_unlab, X_holdout, y_holdout, "qbc")
        
        return self._qbc_active_learning_experiment(
            X_unlab, y_unlab, X_holdout, y_holdout,
            initial_indices, ref_metrics
        )
        
    def _run_qbc_stratified_experiment(self, 
                                      X_unlab: np.ndarray, 
                                      y_unlab: np.ndarray,
                                      X_holdout: np.ndarray, 
                                      y_holdout: np.ndarray) -> Dict[str, Any]:
        """Run QBC active learning with stratified 5 samples."""
        seed = self.config.get('stratified_seeds', [42])[0]
        sss = StratifiedShuffleSplit(n_splits=1, train_size=5, random_state=seed)
        for train_index, _ in sss.split(X_unlab, y_unlab):
            initial_indices = train_index.tolist()
            
        ref_metrics = self._get_reference_metrics(X_unlab, y_unlab, X_holdout, y_holdout, "qbc")
        
        return self._qbc_active_learning_experiment(
            X_unlab, y_unlab, X_holdout, y_holdout,
            initial_indices, ref_metrics
        )
        
    def _get_reference_metrics(self, 
                              X_unlab: np.ndarray, 
                              y_unlab: np.ndarray,
                              X_holdout: np.ndarray, 
                              y_holdout: np.ndarray,
                              model_type: str) -> Dict[str, float]:
        """Get reference metrics from full model."""
        if model_type == "rf":
            model = rf_factory(self.config.get('rf_params', {}))
            model.fit(X_unlab, y_unlab)
            return evaluate_model(model, X_holdout, y_holdout)
        elif model_type == "qbc":
            committee = base_learner_factory(self.config.get('qbc_params', {}))
            for clf in committee:
                clf.fit(X_unlab, y_unlab)
            return evaluate_committee(committee, X_holdout, y_holdout)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
    def _execute_active_learning_experiment(self,
                                          learner,
                                          X_unlab: np.ndarray,
                                          y_unlab: np.ndarray,
                                          X_holdout: np.ndarray,
                                          y_holdout: np.ndarray,
                                          initial_indices: List[int],
                                          sample_fn: Callable,
                                          ref_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Execute an active learning experiment for a single-model (RF).
        """
        max_queries = self.config.get('max_queries', -1)
        stop_ratio = self.config.get('stop_ratio', 0.8)
        batch_size = self.config.get('batch_size', 1)
        
        stop_count = int(stop_ratio * len(X_unlab))
        labeled_idx = list(initial_indices)
        unlabeled_idx = list(set(range(len(X_unlab))) - set(labeled_idx))
        
        available_queries = stop_count - len(labeled_idx)
        pbar_total = available_queries if max_queries == -1 else min(max_queries, available_queries)
        
        mcc_list = []
        f1_list = []
        roc_auc_list = []
        dmcc_list = []
        iteration_list = []
        
        iter_idx = 0
        pbar = tqdm(total=pbar_total, desc="AL Iterations", ncols=80, leave=True)
        
        while ((max_queries == -1 or iter_idx < max_queries) and 
               (len(labeled_idx) < stop_count) and unlabeled_idx):
            
            learner.fit(X_unlab[labeled_idx], y_unlab[labeled_idx])
            eval_res = evaluate_model(learner, X_holdout, y_holdout)
            
            mcc_list.append(eval_res['mcc'])
            f1_list.append(eval_res['f1'])
            roc_auc_list.append(eval_res['roc_auc'])
            dmcc_list.append(eval_res['mcc'] - ref_metrics['mcc'])
            iteration_list.append(iter_idx)
            iter_idx += 1
            
            for _ in range(batch_size):
                if not unlabeled_idx or ((max_queries != -1 and iter_idx >= max_queries) or 
                                       (len(labeled_idx) >= stop_count)):
                    break
                x_pool = X_unlab[unlabeled_idx]
                q_rel_idx = sample_fn(learner, x_pool)
                q_abs_idx = unlabeled_idx[q_rel_idx]
                unlabeled_idx.remove(q_abs_idx)
                labeled_idx.append(q_abs_idx)
                pbar.update(1)
                
        pbar.close()
        
        # Final evaluation
        learner.fit(X_unlab[labeled_idx], y_unlab[labeled_idx])
        final_eval = evaluate_model(learner, X_holdout, y_holdout)
        mcc_list.append(final_eval['mcc'])
        f1_list.append(final_eval['f1'])
        roc_auc_list.append(final_eval['roc_auc'])
        dmcc_list.append(final_eval['mcc'] - ref_metrics['mcc'])
        iteration_list.append(iter_idx)
        
        return {
            'mcc_scores': mcc_list,
            'f1_scores': f1_list,
            'roc_auc_scores': roc_auc_list,
            'dmcc_improvements': dmcc_list,
            'n_iterations': len(iteration_list),
            'final_model': copy.deepcopy(learner)
        }
        
    def _qbc_active_learning_experiment(self,
                                       X_unlab: np.ndarray,
                                       y_unlab: np.ndarray,
                                       X_holdout: np.ndarray,
                                       y_holdout: np.ndarray,
                                       initial_indices: List[int],
                                       ref_metrics: Dict[str, float],
                                       run_iter: int = 0) -> Dict[str, Any]:
        """
        Run a QBC-based active learning experiment.
        """
        max_queries = self.config.get('max_queries', -1)
        stop_ratio = self.config.get('stop_ratio', 0.8)
        batch_size = self.config.get('batch_size', 1)
        
        stop_count = int(stop_ratio * len(X_unlab))
        labeled_idx = list(initial_indices)
        unlabeled_idx = list(set(range(len(X_unlab))) - set(labeled_idx))
        
        available_queries = stop_count - len(labeled_idx)
        pbar_total = available_queries if max_queries == -1 else min(max_queries, available_queries)
        
        mcc_list = []
        f1_list = []
        roc_auc_list = []
        dmcc_list = []
        iteration_list = []
        
        iter_idx = 0
        pbar = tqdm(total=pbar_total, desc="AL Iterations", ncols=80, leave=True)
        
        while ((max_queries == -1 or iter_idx < max_queries) and 
               (len(labeled_idx) < stop_count) and unlabeled_idx):
            
            committee = base_learner_factory(self.config.get('qbc_params', {}), run_iter=run_iter)
            for clf in committee:
                clf.fit(X_unlab[labeled_idx], y_unlab[labeled_idx])
                
            eval_res = evaluate_committee(committee, X_holdout, y_holdout)
            mcc_list.append(eval_res['mcc'])
            f1_list.append(eval_res['f1'])
            roc_auc_list.append(eval_res['roc_auc'])
            dmcc_list.append(eval_res['mcc'] - ref_metrics['mcc'])
            iteration_list.append(iter_idx)
            iter_idx += 1
            
            for _ in range(batch_size):
                if not unlabeled_idx or ((max_queries != -1 and iter_idx >= max_queries) or 
                                       (len(labeled_idx) >= stop_count)):
                    break
                x_pool = X_unlab[unlabeled_idx]
                q_rel_idx = qbc_vote_entropy_sampling(committee, x_pool)
                q_abs_idx = unlabeled_idx[q_rel_idx]
                unlabeled_idx.remove(q_abs_idx)
                labeled_idx.append(q_abs_idx)
                pbar.update(1)
                
        pbar.close()
        
        # Final evaluation
        committee = base_learner_factory(self.config.get('qbc_params', {}), run_iter=run_iter)
        for clf in committee:
            clf.fit(X_unlab[labeled_idx], y_unlab[labeled_idx])
        final_eval = evaluate_committee(committee, X_holdout, y_holdout)
        mcc_list.append(final_eval['mcc'])
        f1_list.append(final_eval['f1'])
        roc_auc_list.append(final_eval['roc_auc'])
        dmcc_list.append(final_eval['mcc'] - ref_metrics['mcc'])
        iteration_list.append(iter_idx)
        
        return {
            'mcc_scores': mcc_list,
            'f1_scores': f1_list,
            'roc_auc_scores': roc_auc_list,
            'dmcc_improvements': dmcc_list,
            'n_iterations': len(iteration_list),
            'final_committee': copy.deepcopy(committee)
        }
        
    def save_results(self, filepath: str):
        """
        Save experiment results to file.
        
        Parameters:
        -----------
        filepath : str
            Path to save results
        """
        import joblib
        joblib.dump(self.results, filepath)
        logger.info(f"Results saved to {filepath}")
        
    def load_results(self, filepath: str):
        """
        Load experiment results from file.
        
        Parameters:
        -----------
        filepath : str
            Path to load results from
        """
        import joblib
        self.results = joblib.load(filepath)
        logger.info(f"Results loaded from {filepath}")


# Standalone experiment functions for backward compatibility
def execute_full_model_baseline(X_unlab: np.ndarray, 
                               y_unlab: np.ndarray, 
                               X_holdout: np.ndarray, 
                               y_holdout: np.ndarray, 
                               rf_params: Optional[Dict] = None, 
                               config: Optional[Dict] = None) -> Dict[str, Any]:
    """Trains an RF on the entire unlabeled pool and returns its performance evaluation and model."""
    rf_fact = config.get('rf_factory', rf_factory) if config else rf_factory
    model = rf_fact(rf_params)
    model.fit(X_unlab, y_unlab)
    eval_res = evaluate_model(model, X_holdout, y_holdout)
    return {'eval': eval_res, 'model': model}


def execute_full_model_qbc(base_learner_factory_fn: Callable,
                          X_unlab: np.ndarray, 
                          y_unlab: np.ndarray, 
                          X_holdout: np.ndarray, 
                          y_holdout: np.ndarray, 
                          qbc_params: Optional[Dict] = None, 
                          config: Optional[Dict] = None) -> Dict[str, Any]:
    """Trains a QBC committee on the entire unlabeled pool and returns its performance evaluation and committee."""
    qbc_fact = config.get('qbc_factory', base_learner_factory_fn) if config else base_learner_factory_fn
    committee = qbc_fact(qbc_params)
    for clf in committee:
        clf.fit(X_unlab, y_unlab)
    eval_res = evaluate_committee(committee, X_holdout, y_holdout)
    return {'eval': eval_res, 'committee': committee}