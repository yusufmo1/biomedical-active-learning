"""
Visualization utilities for biomedical active learning results.

Extracted from the original notebook's plotting functions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path

from sklearn.metrics import roc_curve, auc, confusion_matrix
from .metrics import qbc_predict_proba


class ResultVisualizer:
    """
    Class for visualizing active learning results.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), style: str = "whitegrid"):
        """
        Initialize result visualizer.
        
        Parameters:
        -----------
        figsize : Tuple[int, int]
            Default figure size
        style : str
            Seaborn style
        """
        self.figsize = figsize
        sns.set_style(style)
        plt.rcParams.update({
            'axes.titlesize': 16,
            'axes.labelsize': 14,
            'legend.fontsize': 12,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12
        })
        
    def plot_learning_curves(self, 
                            results_dict: Dict[str, Dict], 
                            metric: str = 'mcc',
                            title: str = "Active Learning Curves",
                            save_path: Optional[str] = None):
        """
        Plot learning curves for multiple experiments.
        
        Parameters:
        -----------
        results_dict : Dict[str, Dict]
            Dictionary of experiment results
        metric : str
            Metric to plot
        title : str
            Plot title
        save_path : str, optional
            Path to save the plot
        """
        plt.figure(figsize=self.figsize)
        
        for exp_name, results in results_dict.items():
            if f'{metric}_scores' in results:
                iterations = range(len(results[f'{metric}_scores']))
                plt.plot(iterations, results[f'{metric}_scores'], 
                        label=exp_name, marker='o', linewidth=2)
                
        plt.xlabel('Iteration')
        plt.ylabel(metric.upper())
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_class_performance(self, 
                              y_true: np.ndarray, 
                              y_pred: np.ndarray,
                              class_names: Optional[List[str]] = None,
                              title: str = "Per-Class Performance",
                              save_path: Optional[str] = None):
        """
        Plot per-class performance metrics.
        
        Parameters:
        -----------
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted labels
        class_names : List[str], optional
            Names for classes
        title : str
            Plot title
        save_path : str, optional
            Path to save the plot
        """
        from sklearn.metrics import classification_report
        
        report = classification_report(y_true, y_pred, output_dict=True)
        
        if class_names is None:
            class_names = [str(i) for i in sorted(np.unique(y_true))]
            
        metrics = ['precision', 'recall', 'f1-score']
        x = np.arange(len(class_names))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        for i, metric in enumerate(metrics):
            values = [report[str(cls)][metric] for cls in sorted(np.unique(y_true))]
            ax.bar(x + i * width, values, width, label=metric.capitalize())
            
        ax.set_xlabel('Class')
        ax.set_ylabel('Score')
        ax.set_title(title)
        ax.set_xticks(x + width)
        ax.set_xticklabels(class_names)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_confusion_matrices(self, 
                               y_true_list: List[np.ndarray],
                               y_pred_list: List[np.ndarray], 
                               model_names: List[str],
                               title: str = "Confusion Matrices",
                               save_path: Optional[str] = None):
        """
        Plot confusion matrices for multiple models.
        
        Parameters:
        -----------
        y_true_list : List[np.ndarray]
            List of true labels
        y_pred_list : List[np.ndarray]
            List of predicted labels
        model_names : List[str]
            List of model names
        title : str
            Plot title
        save_path : str, optional
            Path to save the plot
        """
        n_models = len(model_names)
        cols = min(2, n_models)
        rows = (n_models + 1) // 2
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4))
        if n_models == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
            
        for i, (y_true, y_pred, name) in enumerate(zip(y_true_list, y_pred_list, model_names)):
            cm = confusion_matrix(y_true, y_pred)
            
            ax = axes[i] if n_models > 1 else axes[0]
            im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            ax.set_title(name, fontsize=14)
            
            # Add text annotations
            thresh = cm.max() / 2.
            for row in range(cm.shape[0]):
                for col in range(cm.shape[1]):
                    ax.text(col, row, format(cm[row, col], 'd'),
                           ha="center", va="center",
                           color="white" if cm[row, col] > thresh else "black",
                           fontsize=12)
                           
            ax.set_ylabel('True label')
            ax.set_xlabel('Predicted label')
            
        # Hide empty subplots
        for i in range(n_models, len(axes)):
            axes[i].set_visible(False)
            
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_dmcc_analysis(self, 
                          results_dict: Dict[str, Dict],
                          title: str = "Delta MCC Evolution",
                          save_path: Optional[str] = None):
        """
        Plot Delta MCC (DMCC) evolution over iterations.
        
        Parameters:
        -----------
        results_dict : Dict[str, Dict]
            Dictionary of experiment results
        title : str
            Plot title
        save_path : str, optional
            Path to save the plot
        """
        plt.figure(figsize=self.figsize)
        
        for exp_name, results in results_dict.items():
            if 'dmcc_improvements' in results:
                iterations = range(len(results['dmcc_improvements']))
                plt.plot(iterations, results['dmcc_improvements'], 
                        label=exp_name, marker='o', linewidth=2)
                        
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.xlabel('Iteration')
        plt.ylabel('Delta MCC')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_roc_curves(self, 
                       models_dict: Dict[str, any],
                       X_test: np.ndarray,
                       y_test: np.ndarray,
                       title: str = "ROC Curves",
                       save_path: Optional[str] = None):
        """
        Plot ROC curves for multiple models.
        
        Parameters:
        -----------
        models_dict : Dict[str, any]
            Dictionary of models/committees
        X_test : np.ndarray
            Test features
        y_test : np.ndarray
            Test labels
        title : str
            Plot title
        save_path : str, optional
            Path to save the plot
        """
        plt.figure(figsize=self.figsize)
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(models_dict)))
        
        for (name, model), color in zip(models_dict.items(), colors):
            if model is not None:
                # Get predictions
                if isinstance(model, list):  # Committee
                    y_pred_proba = qbc_predict_proba(model, X_test)
                else:
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                    
                # Calculate ROC curve and AUC
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                roc_auc = auc(fpr, tpr)
                
                # Plot ROC curve
                plt.plot(fpr, tpr, color=color, 
                        label=f'{name} (AUC = {roc_auc:.3f})', linewidth=2)
                        
        # Plot diagonal line
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--', alpha=0.5)
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_feature_importance(self, 
                               model,
                               feature_names: List[str],
                               top_n: int = 20,
                               title: str = "Feature Importance",
                               save_path: Optional[str] = None):
        """
        Plot feature importance for a model.
        
        Parameters:
        -----------
        model : sklearn model
            Trained model with feature_importances_ attribute
        feature_names : List[str]
            Names of features
        top_n : int
            Number of top features to show
        title : str
            Plot title
        save_path : str, optional
            Path to save the plot
        """
        if not hasattr(model, 'feature_importances_'):
            raise ValueError("Model does not have feature_importances_ attribute")
            
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=self.figsize)
        plt.bar(range(top_n), importances[indices], color='skyblue', edgecolor='black')
        
        # Clean feature names for display
        display_names = [feature_names[i].replace("_", " ") for i in indices]
        plt.xticks(range(top_n), display_names, rotation=45, ha='right')
        
        plt.xlabel('Features')
        plt.ylabel('Importance Score')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_class_distributions(self, 
                                datasets: Dict[str, np.ndarray],
                                title: str = "Class Distributions",
                                save_path: Optional[str] = None):
        """
        Plot class distributions for multiple datasets.
        
        Parameters:
        -----------
        datasets : Dict[str, np.ndarray]
            Dictionary of dataset names and labels
        title : str
            Plot title
        save_path : str, optional
            Path to save the plot
        """
        n_datasets = len(datasets)
        fig, axes = plt.subplots(1, n_datasets, figsize=(n_datasets * 5, 4))
        
        if n_datasets == 1:
            axes = [axes]
            
        for i, (dataset_name, labels) in enumerate(datasets.items()):
            unique, counts = np.unique(labels, return_counts=True)
            
            axes[i].bar(range(len(unique)), counts, 
                       color=['skyblue', 'salmon'][i % 2], 
                       edgecolor='black', alpha=0.7)
            axes[i].set_title(dataset_name, fontsize=14)
            axes[i].set_xlabel('Class')
            axes[i].set_ylabel('Count')
            axes[i].set_xticks(range(len(unique)))
            axes[i].set_xticklabels(unique)
            
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()