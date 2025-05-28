"""
Parallel processing utilities.

Extracted from the original notebook's parallel processing patterns.
"""

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, List, Any, Dict, Optional
from functools import partial
import logging
from tqdm import tqdm
from joblib import Parallel, delayed

logger = logging.getLogger(__name__)


class ParallelProcessor:
    """
    Utility class for parallel processing of active learning experiments.
    """
    
    def __init__(self, n_jobs: Optional[int] = None):
        """
        Initialize parallel processor.
        
        Parameters:
        -----------
        n_jobs : int, optional
            Number of parallel jobs. If None, uses all available cores
        """
        self.n_jobs = n_jobs if n_jobs is not None else mp.cpu_count()
        logger.info(f"Initialized parallel processor with {self.n_jobs} jobs")
        
    def run_parallel_experiments(self, 
                                experiment_func: Callable,
                                experiment_args: List[tuple],
                                description: str = "Running experiments") -> List[Any]:
        """
        Run experiments in parallel using joblib.
        
        Parameters:
        -----------
        experiment_func : Callable
            Function to run in parallel
        experiment_args : List[tuple]
            List of argument tuples for each experiment
        description : str
            Description for progress bar
            
        Returns:
        --------
        List[Any]
            Results from all experiments
        """
        logger.info(f"Running {len(experiment_args)} experiments in parallel")
        
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(experiment_func)(*args) 
            for args in tqdm(experiment_args, desc=description, ncols=80, leave=True)
        )
        
        logger.info(f"Completed {len(results)} experiments")
        return results
        
    def run_baseline_experiments(self, 
                                baseline_func: Callable,
                                X_unlab: Any, 
                                y_unlab: Any,
                                X_holdout: Any, 
                                y_holdout: Any,
                                n_runs: int,
                                params_list: List[Dict],
                                description: str = "Baseline experiments") -> List[Any]:
        """
        Run baseline experiments in parallel.
        
        Parameters:
        -----------
        baseline_func : Callable
            Baseline experiment function
        X_unlab, y_unlab : Any
            Unlabeled data
        X_holdout, y_holdout : Any
            Holdout test data
        n_runs : int
            Number of runs
        params_list : List[Dict]
            List of parameter dictionaries for each run
        description : str
            Description for progress bar
            
        Returns:
        --------
        List[Any]
            Baseline experiment results
        """
        experiment_args = [
            (X_unlab, y_unlab, X_holdout, y_holdout, params, None)
            for params in params_list
        ]
        
        return self.run_parallel_experiments(
            baseline_func, experiment_args, description
        )
        
    def run_active_learning_experiments(self,
                                      al_func: Callable,
                                      X_unlab: Any,
                                      y_unlab: Any, 
                                      X_holdout: Any,
                                      y_holdout: Any,
                                      sample_fn: Callable,
                                      config: Dict,
                                      ref_metrics: Dict,
                                      n_runs: int,
                                      description: str = "AL experiments") -> List[Any]:
        """
        Run active learning experiments in parallel.
        
        Parameters:
        -----------
        al_func : Callable
            Active learning experiment function
        X_unlab, y_unlab : Any
            Unlabeled data
        X_holdout, y_holdout : Any
            Holdout test data
        sample_fn : Callable
            Sampling function
        config : Dict
            Configuration dictionary
        ref_metrics : Dict
            Reference metrics
        n_runs : int
            Number of runs
        description : str
            Description for progress bar
            
        Returns:
        --------
        List[Any]
            Active learning experiment results
        """
        experiment_args = [
            (X_unlab, y_unlab, X_holdout, y_holdout, sample_fn, config, ref_metrics, i)
            for i in range(n_runs)
        ]
        
        return self.run_parallel_experiments(
            al_func, experiment_args, description
        )
        
    def run_stratified_experiments(self,
                                 al_func: Callable,
                                 X_unlab: Any,
                                 y_unlab: Any,
                                 X_holdout: Any, 
                                 y_holdout: Any,
                                 seeds: List[int],
                                 sample_fn: Callable,
                                 config: Dict,
                                 ref_metrics: Dict,
                                 description: str = "Stratified experiments") -> Dict[int, Any]:
        """
        Run stratified active learning experiments in parallel.
        
        Parameters:
        -----------
        al_func : Callable
            Active learning experiment function
        X_unlab, y_unlab : Any
            Unlabeled data
        X_holdout, y_holdout : Any
            Holdout test data
        seeds : List[int]
            List of random seeds for stratification
        sample_fn : Callable
            Sampling function
        config : Dict
            Configuration dictionary
        ref_metrics : Dict
            Reference metrics
        description : str
            Description for progress bar
            
        Returns:
        --------
        Dict[int, Any]
            Results keyed by seed
        """
        experiment_args = [
            (X_unlab, y_unlab, X_holdout, y_holdout, seed, sample_fn, config, ref_metrics)
            for seed in seeds
        ]
        
        results = self.run_parallel_experiments(
            al_func, experiment_args, description
        )
        
        return dict(zip(seeds, results))
        
    def process_with_executor(self,
                            func: Callable,
                            args_list: List[tuple],
                            max_workers: Optional[int] = None,
                            description: str = "Processing") -> List[Any]:
        """
        Process using ProcessPoolExecutor with progress tracking.
        
        Parameters:
        -----------
        func : Callable
            Function to execute
        args_list : List[tuple]
            List of argument tuples
        max_workers : int, optional
            Maximum number of worker processes
        description : str
            Description for progress bar
            
        Returns:
        --------
        List[Any]
            Results from all processes
        """
        if max_workers is None:
            max_workers = min(self.n_jobs, len(args_list))
            
        results = []
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_args = {
                executor.submit(func, *args): args 
                for args in args_list
            }
            
            # Process completed tasks with progress bar
            with tqdm(total=len(args_list), desc=description, ncols=80) as pbar:
                for future in as_completed(future_to_args):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Task failed: {e}")
                        results.append(None)
                    finally:
                        pbar.update(1)
                        
        return results
        
    def optimize_parameters(self,
                          objective_func: Callable,
                          n_trials: int,
                          study_name: str = "optimization",
                          direction: str = "maximize") -> Dict[str, Any]:
        """
        Run parameter optimization in parallel.
        
        Parameters:
        -----------
        objective_func : Callable
            Objective function for optimization
        n_trials : int
            Number of trials
        study_name : str
            Name for the study
        direction : str
            Optimization direction
            
        Returns:
        --------
        Dict[str, Any]
            Optimization results
        """
        try:
            import optuna
        except ImportError:
            raise ImportError("Optuna is required for parameter optimization")
            
        # Set up environment variables for better parallel performance
        import os
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        
        n_parallel_trials = max(1, self.n_jobs // 2)
        
        study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            sampler=optuna.samplers.TPESampler(
                n_startup_trials=10,
                multivariate=True,
                seed=42
            ),
            pruner=optuna.pruners.HyperbandPruner(
                min_resource=1,
                max_resource=n_trials,
                reduction_factor=3
            )
        )
        
        logger.info(f"Starting optimization with {n_parallel_trials} parallel trials")
        
        try:
            study.optimize(
                objective_func,
                n_trials=n_trials,
                n_jobs=n_parallel_trials,
                gc_after_trial=True,
                show_progress_bar=True
            )
        except KeyboardInterrupt:
            logger.warning("Optimization interrupted by user")
            
        logger.info("Optimization completed")
        logger.info(f"Best value: {study.best_value:.4f}")
        logger.info(f"Best params: {study.best_trial.params}")
        
        return {
            'best_value': study.best_value,
            'best_params': study.best_trial.params,
            'study': study,
            'n_trials': len(study.trials),
            'n_failed': len([t for t in study.trials if t.value == float('-inf')])
        }
        
    @staticmethod
    def setup_parallel_env():
        """
        Setup environment variables for optimal parallel performance.
        """
        import os
        
        # Limit thread creation for better parallel performance
        env_vars = {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1", 
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1"
        }
        
        for var, value in env_vars.items():
            os.environ[var] = value
            
        logger.info("Parallel environment configured")


# Utility functions for common parallel patterns
def parallel_map(func: Callable, 
                args_list: List[Any], 
                n_jobs: Optional[int] = None,
                description: str = "Processing") -> List[Any]:
    """
    Simple parallel map function.
    
    Parameters:
    -----------
    func : Callable
        Function to apply
    args_list : List[Any]
        List of arguments
    n_jobs : int, optional
        Number of parallel jobs
    description : str
        Description for progress bar
        
    Returns:
    --------
    List[Any]
        Results
    """
    if n_jobs is None:
        n_jobs = mp.cpu_count()
        
    return Parallel(n_jobs=n_jobs)(
        delayed(func)(args) 
        for args in tqdm(args_list, desc=description, ncols=80)
    )


def parallel_apply(func: Callable,
                  args_list: List[tuple],
                  n_jobs: Optional[int] = None,
                  description: str = "Processing") -> List[Any]:
    """
    Parallel apply function with multiple arguments.
    
    Parameters:
    -----------
    func : Callable
        Function to apply
    args_list : List[tuple]
        List of argument tuples
    n_jobs : int, optional
        Number of parallel jobs
    description : str
        Description for progress bar
        
    Returns:
    --------
    List[Any]
        Results
    """
    if n_jobs is None:
        n_jobs = mp.cpu_count()
        
    return Parallel(n_jobs=n_jobs)(
        delayed(func)(*args) 
        for args in tqdm(args_list, desc=description, ncols=80)
    )