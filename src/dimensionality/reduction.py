"""
Dimensionality reduction utilities.

Extracted from the original notebook's dimensionality reduction functions.
"""

import warnings
import multiprocessing as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any, Optional
from concurrent.futures import ProcessPoolExecutor

import optuna
import umap
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score


class DimensionalityReducer:
    """
    Class for dimensionality reduction analysis.
    """
    
    def __init__(self):
        """Initialize dimensionality reducer."""
        self.reducers = {}
        self.embeddings = {}
        
    def fit_pca(self, 
                X: np.ndarray, 
                n_components: Optional[int] = None,
                variance_threshold: float = 0.90) -> Tuple[np.ndarray, PCA]:
        """
        Fit PCA and return transformed data.
        
        Parameters:
        -----------
        X : np.ndarray
            Input data
        n_components : int, optional
            Number of components. If None, determined by variance threshold
        variance_threshold : float
            Variance threshold for automatic component selection
            
        Returns:
        --------
        Tuple[np.ndarray, PCA]
            Transformed data and fitted PCA object
        """
        if n_components is None:
            # Determine components needed for variance threshold
            pca_full = PCA()
            pca_full.fit(X)
            
            cum_var = np.cumsum(pca_full.explained_variance_ratio_)
            n_components = np.argmax(cum_var >= variance_threshold) + 1
            
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)
        
        self.reducers['pca'] = pca
        self.embeddings['pca'] = X_pca
        
        return X_pca, pca
        
    def fit_tsne(self, 
                 X: np.ndarray, 
                 n_components: int = 2,
                 random_state: int = 42,
                 **kwargs) -> np.ndarray:
        """
        Fit t-SNE and return transformed data.
        
        Parameters:
        -----------
        X : np.ndarray
            Input data
        n_components : int
            Number of components for t-SNE
        random_state : int
            Random state for reproducibility
        **kwargs
            Additional t-SNE parameters
            
        Returns:
        --------
        np.ndarray
            Transformed data
        """
        tsne_params = {
            'n_components': n_components,
            'random_state': random_state,
            **kwargs
        }
        
        tsne = TSNE(**tsne_params)
        X_tsne = tsne.fit_transform(X)
        
        self.reducers['tsne'] = tsne
        self.embeddings['tsne'] = X_tsne
        
        return X_tsne
        
    def fit_umap(self, 
                 X: np.ndarray, 
                 y: Optional[np.ndarray] = None,
                 n_components: int = 2,
                 optimize_params: bool = False,
                 n_trials: int = 100,
                 **kwargs) -> np.ndarray:
        """
        Fit UMAP and return transformed data.
        
        Parameters:
        -----------
        X : np.ndarray
            Input data
        y : np.ndarray, optional
            Target labels for supervised UMAP
        n_components : int
            Number of components
        optimize_params : bool
            Whether to optimize hyperparameters
        n_trials : int
            Number of optimization trials
        **kwargs
            Additional UMAP parameters
            
        Returns:
        --------
        np.ndarray
            Transformed data
        """
        if optimize_params:
            best_params = self._optimize_umap_params(X, y, n_trials)
            kwargs.update(best_params)
            
        umap_params = {
            'n_components': n_components,
            'random_state': None,  # For parallel processing
            **kwargs
        }
        
        if y is not None:
            umap_params.update({
                'target_metric': 'categorical',
                'target_weight': kwargs.get('target_weight', 0.5)
            })
            
        reducer = umap.UMAP(**umap_params)
        
        if y is not None:
            X_umap = reducer.fit_transform(X, y)
        else:
            X_umap = reducer.fit_transform(X)
            
        self.reducers['umap'] = reducer
        self.embeddings['umap'] = X_umap
        
        return X_umap
        
    def fit_lda(self, 
                X: np.ndarray, 
                y: np.ndarray,
                n_components: Optional[int] = None) -> np.ndarray:
        """
        Fit Linear Discriminant Analysis.
        
        Parameters:
        -----------
        X : np.ndarray
            Input data
        y : np.ndarray
            Target labels
        n_components : int, optional
            Number of components
            
        Returns:
        --------
        np.ndarray
            Transformed data
        """
        if n_components is None:
            n_components = len(np.unique(y)) - 1
            
        lda = LinearDiscriminantAnalysis(n_components=n_components)
        X_lda = lda.fit_transform(X, y)
        
        self.reducers['lda'] = lda
        self.embeddings['lda'] = X_lda
        
        return X_lda
        
    def _optimize_umap_params(self, 
                             X: np.ndarray, 
                             y: Optional[np.ndarray] = None,
                             n_trials: int = 100) -> Dict[str, Any]:
        """
        Optimize UMAP hyperparameters using Optuna.
        
        Parameters:
        -----------
        X : np.ndarray
            Input data
        y : np.ndarray, optional
            Target labels
        n_trials : int
            Number of optimization trials
            
        Returns:
        --------
        Dict[str, Any]
            Best parameters
        """
        def objective(trial):
            # Define search space
            metric = trial.suggest_categorical("metric", ['euclidean', 'manhattan', 'cosine'])
            n_neighbors = trial.suggest_int("n_neighbors", 5, 50)
            min_dist = trial.suggest_float("min_dist", 0.0, 0.9)
            spread = trial.suggest_float("spread", 1.0, 3.0)
            
            if y is not None:
                target_weight = trial.suggest_float("target_weight", 0.0, 1.0)
            else:
                target_weight = 0.0
                
            # Constraint checking
            if spread <= min_dist:
                return float('-inf')
                
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    
                    # Calculate number of threads for this trial
                    n_cores = mp.cpu_count()
                    threads_per_trial = max(1, n_cores // 4)
                    
                    reducer = umap.UMAP(
                        n_neighbors=n_neighbors,
                        min_dist=min_dist,
                        spread=spread,
                        metric=metric,
                        n_components=2,
                        random_state=42,
                        n_jobs=threads_per_trial,
                        low_memory=True
                    )
                    
                    if y is not None:
                        reducer.target_metric = 'categorical'
                        reducer.target_weight = target_weight
                        embedding = reducer.fit_transform(X, y)
                        score = silhouette_score(embedding, y)
                    else:
                        embedding = reducer.fit_transform(X)
                        # Use a different metric for unsupervised case
                        score = -np.mean(np.var(embedding, axis=0))  # Minimize variance
                        
                    if score < 0:
                        return float('-inf')
                        
                    return score
                    
            except Exception as e:
                print(f"Trial failed with error: {str(e)}")
                return float('-inf')
                
        # Set up environment variables for better parallel performance
        import os
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        
        # Run optimization
        n_cores = mp.cpu_count()
        n_parallel_trials = max(1, n_cores // 2)
        
        study = optuna.create_study(
            direction="maximize",
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
        
        study.optimize(objective, n_trials=n_trials, n_jobs=n_parallel_trials)
        
        print(f"Best trial: {study.best_trial.value:.4f}")
        print("Best params:")
        for key, value in study.best_trial.params.items():
            print(f"  {key}: {value}")
            
        return study.best_trial.params
        
    def plot_explained_variance(self, 
                               method: str = 'pca',
                               save_path: Optional[str] = None):
        """
        Plot explained variance for PCA.
        
        Parameters:
        -----------
        method : str
            Method to plot (currently only 'pca')
        save_path : str, optional
            Path to save the plot
        """
        if method == 'pca' and 'pca' in self.reducers:
            pca = self.reducers['pca']
            
            plt.figure(figsize=(10, 6))
            
            # Plot individual explained variance
            plt.subplot(1, 2, 1)
            plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), 
                   pca.explained_variance_ratio_)
            plt.xlabel('Principal Component')
            plt.ylabel('Explained Variance Ratio')
            plt.title('Individual Explained Variance')
            
            # Plot cumulative explained variance
            plt.subplot(1, 2, 2)
            cum_var = np.cumsum(pca.explained_variance_ratio_)
            plt.plot(range(1, len(cum_var) + 1), cum_var, 'bo-')
            plt.axhline(y=0.9, color='r', linestyle='--', alpha=0.7, label='90% threshold')
            plt.xlabel('Number of Components')
            plt.ylabel('Cumulative Explained Variance')
            plt.title('Cumulative Explained Variance')
            plt.legend()
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
        else:
            raise ValueError(f"Method {method} not available or not fitted")
            
    def plot_embeddings(self, 
                       methods: list = ['pca', 'tsne', 'umap'],
                       labels: Optional[np.ndarray] = None,
                       save_path: Optional[str] = None):
        """
        Plot 2D embeddings for multiple methods.
        
        Parameters:
        -----------
        methods : list
            List of methods to plot
        labels : np.ndarray, optional
            Labels for coloring points
        save_path : str, optional
            Path to save the plot
        """
        available_methods = [m for m in methods if m in self.embeddings]
        n_methods = len(available_methods)
        
        if n_methods == 0:
            raise ValueError("No fitted methods available")
            
        fig, axes = plt.subplots(1, n_methods, figsize=(6*n_methods, 5))
        if n_methods == 1:
            axes = [axes]
            
        for i, method in enumerate(available_methods):
            embedding = self.embeddings[method]
            
            if labels is not None:
                scatter = axes[i].scatter(embedding[:, 0], embedding[:, 1], 
                                        c=labels, cmap='viridis', alpha=0.7)
                plt.colorbar(scatter, ax=axes[i])
            else:
                axes[i].scatter(embedding[:, 0], embedding[:, 1], alpha=0.7)
                
            axes[i].set_title(f'{method.upper()} Embedding')
            axes[i].set_xlabel(f'{method.upper()} 1')
            axes[i].set_ylabel(f'{method.upper()} 2')
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def get_embeddings(self, method: str) -> np.ndarray:
        """
        Get embeddings for a specific method.
        
        Parameters:
        -----------
        method : str
            Method name
            
        Returns:
        --------
        np.ndarray
            Embeddings
        """
        if method not in self.embeddings:
            raise ValueError(f"Method {method} not available")
        return self.embeddings[method]
        
    def get_reducer(self, method: str):
        """
        Get fitted reducer for a specific method.
        
        Parameters:
        -----------
        method : str
            Method name
            
        Returns:
        --------
            Fitted reducer object
        """
        if method not in self.reducers:
            raise ValueError(f"Method {method} not available")
        return self.reducers[method]