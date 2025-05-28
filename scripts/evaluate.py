#!/usr/bin/env python3
"""
Evaluation Script for Active Learning Experiments

This script loads experiment results and generates comprehensive evaluation metrics,
visualizations, and comparison reports.

Usage:
    python scripts/evaluate.py --results-dir results/ --output results/figures/
    python scripts/evaluate.py --dataset bbb --metrics mcc f1 roc_auc
"""

import argparse
import logging
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
import json
from datetime import datetime

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from evaluation.visualization import ResultVisualizer
from evaluation.metrics import ModelEvaluator
from utils.helpers import setup_logging

def setup_arguments():
    """Setup command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate active learning experiment results"
    )
    
    parser.add_argument(
        '--results-dir', 
        type=str, 
        default='results/',
        help='Directory containing experiment results (default: results/)'
    )
    
    parser.add_argument(
        '--output', 
        type=str, 
        default='results/figures/',
        help='Output directory for figures and reports (default: results/figures/)'
    )
    
    parser.add_argument(
        '--dataset', 
        choices=['bbb', 'breast_cancer', 'both'], 
        default='both',
        help='Dataset to evaluate (default: both)'
    )
    
    parser.add_argument(
        '--metrics', 
        nargs='+', 
        choices=['mcc', 'f1', 'roc_auc', 'dmcc'],
        default=['mcc', 'f1', 'roc_auc'],
        help='Metrics to include in visualizations (default: mcc f1 roc_auc)'
    )
    
    parser.add_argument(
        '--figure-format', 
        choices=['png', 'pdf', 'svg'],
        default='png',
        help='Figure format for saved plots (default: png)'
    )
    
    parser.add_argument(
        '--dpi', 
        type=int, 
        default=300,
        help='DPI for saved figures (default: 300)'
    )
    
    parser.add_argument(
        '--no-show', 
        action='store_true',
        help='Do not display plots (save only)'
    )
    
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()

def load_experiment_results(results_dir, dataset):
    """Load experiment results for a dataset."""
    results_path = Path(results_dir)
    
    results = {}
    
    # Load baseline results
    baseline_file = results_path / f'{dataset}_baseline_results.pkl'
    if baseline_file.exists():
        results['baseline'] = joblib.load(baseline_file)
        logging.info(f"Loaded baseline results for {dataset}: {len(results['baseline'])} experiments")
    else:
        logging.warning(f"Baseline results not found: {baseline_file}")
        results['baseline'] = []
    
    # Load AL results
    al_file = results_path / f'{dataset}_al_results.pkl'
    if al_file.exists():
        results['active_learning'] = joblib.load(al_file)
        logging.info(f"Loaded AL results for {dataset}: {len(results['active_learning'])} experiments")
    else:
        logging.warning(f"AL results not found: {al_file}")
        results['active_learning'] = []
    
    # Load summary
    summary_file = results_path / f'{dataset}_summary.csv'
    if summary_file.exists():
        results['summary'] = pd.read_csv(summary_file)
        logging.info(f"Loaded summary for {dataset}: {len(results['summary'])} rows")
    else:
        logging.warning(f"Summary not found: {summary_file}")
        results['summary'] = pd.DataFrame()
    
    return results

def create_results_dataframe(all_results):
    """Create consolidated results DataFrame for visualization."""
    rows = []
    
    for dataset, results in all_results.items():
        # Process baseline results
        for result in results.get('baseline', []):
            eval_res = result['results']
            rows.append({
                'Dataset': dataset.upper(),
                'Experiment': f"{dataset.upper()}_{result['strategy'].upper()}_Full",
                'Strategy': result['strategy'].upper(),
                'Type': 'Full',
                'Run': result['run_id'],
                'Iteration': 0,
                'MCC': eval_res['mcc'],
                'F1': eval_res['f1'],
                'ROC_AUC': eval_res['roc_auc'],
                'DMCC': 0.0
            })
        
        # Process AL results
        for result in results.get('active_learning', []):
            strategy = result['strategy'].upper()
            sampling = result['sampling_method'].title()
            
            # Extract iteration data
            res = result['results']
            n_iterations = res['n_iterations']
            
            for i in range(n_iterations):
                rows.append({
                    'Dataset': dataset.upper(),
                    'Experiment': f"{dataset.upper()}_{strategy}_{sampling}",
                    'Strategy': strategy,
                    'Type': sampling,
                    'Run': result['run_id'],
                    'Iteration': i,
                    'MCC': res['mcc_scores'][i],
                    'F1': res['f1_scores'][i],
                    'ROC_AUC': res['roc_auc_scores'][i],
                    'DMCC': res['dmcc_improvements'][i]
                })
    
    return pd.DataFrame(rows)

def plot_learning_curves(df, dataset, metrics, output_dir, args):
    """Plot learning curves for active learning experiments."""
    dataset_df = df[df['Dataset'] == dataset.upper()].copy()
    
    # Filter to AL experiments only
    al_df = dataset_df[dataset_df['Type'].isin(['First5', 'Stratified'])].copy()
    
    if al_df.empty:
        logging.warning(f"No AL data found for {dataset}")
        return
    
    n_metrics = len(metrics)
    n_strategies = len(al_df['Strategy'].unique())
    
    fig, axes = plt.subplots(n_strategies, n_metrics, figsize=(5*n_metrics, 4*n_strategies))
    if n_strategies == 1:
        axes = axes.reshape(1, -1)
    if n_metrics == 1:
        axes = axes.reshape(-1, 1)
    
    for i, strategy in enumerate(sorted(al_df['Strategy'].unique())):
        strategy_df = al_df[al_df['Strategy'] == strategy]
        
        for j, metric in enumerate(metrics):
            ax = axes[i, j]
            
            # Plot each sampling method
            for sampling_type in ['First5', 'Stratified']:
                subset = strategy_df[strategy_df['Type'] == sampling_type]
                if subset.empty:
                    continue
                
                # Aggregate by iteration
                grouped = subset.groupby('Iteration')[metric.upper()].agg(['mean', 'std']).reset_index()
                
                color = 'red' if sampling_type == 'First5' else 'blue'
                ax.plot(grouped['Iteration'], grouped['mean'], '-', color=color, 
                       label=f'{strategy} {sampling_type}')
                
                if sampling_type == 'Stratified':
                    ax.fill_between(grouped['Iteration'],
                                   grouped['mean'] - grouped['std'],
                                   grouped['mean'] + grouped['std'],
                                   color=color, alpha=0.2)
            
            ax.set_title(f"{strategy} {metric.upper()} Evolution")
            ax.set_xlabel("Iteration")
            ax.set_ylabel(metric.upper())
            ax.legend(loc='lower right')
            ax.set_ylim(0, 1)
    
    plt.suptitle(f"{dataset.title()} Active Learning Comparison", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save plot
    output_path = Path(output_dir)
    filename = f"{dataset}_learning_curves.{args.figure_format}"
    plt.savefig(output_path / filename, dpi=args.dpi, bbox_inches='tight')
    logging.info(f"Saved learning curves to {output_path / filename}")
    
    if not args.no_show:
        plt.show()
    plt.close()

def plot_dmcc_evolution(df, dataset, output_dir, args):
    """Plot DMCC (delta MCC) evolution."""
    dataset_df = df[df['Dataset'] == dataset.upper()].copy()
    al_df = dataset_df[dataset_df['Type'].isin(['First5', 'Stratified'])].copy()
    
    if al_df.empty:
        return
    
    strategies = sorted(al_df['Strategy'].unique())
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    axes = axes.flatten()
    
    plot_idx = 0
    for strategy in strategies:
        for sampling_type in ['First5', 'Stratified']:
            if plot_idx >= 4:
                break
                
            subset = al_df[(al_df['Strategy'] == strategy) & (al_df['Type'] == sampling_type)]
            if subset.empty:
                continue
            
            grouped = subset.groupby('Iteration')['DMCC'].agg(['mean', 'std']).reset_index()
            
            ax = axes[plot_idx]
            color = 'red' if sampling_type == 'First5' else 'blue'
            ax.plot(grouped['Iteration'], grouped['mean'], '-', color=color)
            
            if sampling_type == 'Stratified':
                ax.fill_between(grouped['Iteration'],
                               grouped['mean'] - grouped['std'],
                               grouped['mean'] + grouped['std'],
                               color=color, alpha=0.2)
            
            ax.set_title(f"{strategy} {sampling_type}")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("DMCC")
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            
            plot_idx += 1
    
    # Hide unused subplots
    for i in range(plot_idx, 4):
        axes[i].set_visible(False)
    
    plt.suptitle(f"{dataset.title()} DMCC Evolution", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save plot
    output_path = Path(output_dir)
    filename = f"{dataset}_dmcc_evolution.{args.figure_format}"
    plt.savefig(output_path / filename, dpi=args.dpi, bbox_inches='tight')
    logging.info(f"Saved DMCC evolution to {output_path / filename}")
    
    if not args.no_show:
        plt.show()
    plt.close()

def plot_model_comparison(df, metrics, output_dir, args):
    """Plot model comparison across datasets."""
    datasets = df['Dataset'].unique()
    n_datasets = len(datasets)
    
    # Define model order for consistent plotting
    model_order = ['RF_Full', 'QBC_Full', 'RF_First5', 'RF_Stratified', 'QBC_First5', 'QBC_Stratified']
    
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Get final metric values (max iteration for each run)
        final_values = []
        dataset_labels = []
        model_labels = []
        
        for dataset in sorted(datasets):
            dataset_df = df[df['Dataset'] == dataset]
            
            for model_type in model_order:
                # Extract strategy and type from experiment name
                if 'Full' in model_type:
                    strategy = model_type.split('_')[0]
                    type_filter = 'Full'
                    experiment_filter = f"{dataset}_{strategy}_Full"
                else:
                    parts = model_type.split('_')
                    strategy = parts[0]
                    sampling = parts[1]
                    type_filter = sampling
                    experiment_filter = f"{dataset}_{strategy}_{sampling}"
                
                subset = dataset_df[dataset_df['Experiment'] == experiment_filter]
                
                if subset.empty:
                    continue
                
                # Get final values for each run
                if type_filter == 'Full':
                    run_values = subset[metric.upper()].tolist()
                else:
                    # For AL experiments, get the last iteration for each run
                    final_vals = []
                    for run_id in subset['Run'].unique():
                        run_subset = subset[subset['Run'] == run_id]
                        max_iter = run_subset['Iteration'].max()
                        final_val = run_subset[run_subset['Iteration'] == max_iter][metric.upper()].iloc[0]
                        final_vals.append(final_val)
                    run_values = final_vals
                
                if run_values:
                    final_values.extend(run_values)
                    dataset_labels.extend([dataset] * len(run_values))
                    model_labels.extend([model_type] * len(run_values))
        
        # Create DataFrame for plotting
        plot_df = pd.DataFrame({
            'Dataset': dataset_labels,
            'Model': model_labels,
            'Value': final_values
        })
        
        if not plot_df.empty:
            # Create grouped bar plot
            sns.boxplot(data=plot_df, x='Dataset', y='Value', hue='Model', ax=ax)
            ax.set_title(f"Model Comparison - {metric.upper()}")
            ax.set_ylabel(metric.upper())
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        # Save plot
        output_path = Path(output_dir)
        filename = f"model_comparison_{metric}.{args.figure_format}"
        plt.savefig(output_path / filename, dpi=args.dpi, bbox_inches='tight')
        logging.info(f"Saved model comparison to {output_path / filename}")
        
        if not args.no_show:
            plt.show()
        plt.close()

def generate_summary_statistics(df, output_dir):
    """Generate summary statistics table."""
    logging.info("Generating summary statistics...")
    
    # Get final performance for each experiment
    summary_rows = []
    
    for (dataset, experiment), group in df.groupby(['Dataset', 'Experiment']):
        if 'Full' in experiment:
            # Baseline experiments
            metrics = {
                'MCC': group['MCC'].mean(),
                'MCC_std': group['MCC'].std(),
                'F1': group['F1'].mean(),
                'F1_std': group['F1'].std(),
                'ROC_AUC': group['ROC_AUC'].mean(),
                'ROC_AUC_std': group['ROC_AUC'].std(),
                'DMCC': 0.0,
                'DMCC_std': 0.0,
                'Runs': len(group)
            }
        else:
            # AL experiments - get final iteration for each run
            final_vals = []
            for run_id in group['Run'].unique():
                run_group = group[group['Run'] == run_id]
                max_iter = run_group['Iteration'].max()
                final_row = run_group[run_group['Iteration'] == max_iter].iloc[0]
                final_vals.append({
                    'MCC': final_row['MCC'],
                    'F1': final_row['F1'],
                    'ROC_AUC': final_row['ROC_AUC'],
                    'DMCC': final_row['DMCC']
                })
            
            final_df = pd.DataFrame(final_vals)
            metrics = {
                'MCC': final_df['MCC'].mean(),
                'MCC_std': final_df['MCC'].std(),
                'F1': final_df['F1'].mean(),
                'F1_std': final_df['F1'].std(),
                'ROC_AUC': final_df['ROC_AUC'].mean(),
                'ROC_AUC_std': final_df['ROC_AUC'].std(),
                'DMCC': final_df['DMCC'].mean(),
                'DMCC_std': final_df['DMCC'].std(),
                'Runs': len(final_vals)
            }
        
        summary_rows.append({
            'Dataset': dataset,
            'Experiment': experiment,
            **metrics
        })
    
    summary_df = pd.DataFrame(summary_rows)
    
    # Round for readability
    numeric_cols = ['MCC', 'MCC_std', 'F1', 'F1_std', 'ROC_AUC', 'ROC_AUC_std', 'DMCC', 'DMCC_std']
    summary_df[numeric_cols] = summary_df[numeric_cols].round(4)
    
    # Save summary
    output_path = Path(output_dir)
    summary_file = output_path / 'summary_statistics.csv'
    summary_df.to_csv(summary_file, index=False)
    logging.info(f"Saved summary statistics to {summary_file}")
    
    return summary_df

def generate_evaluation_report(df, summary_df, output_dir):
    """Generate comprehensive evaluation report."""
    logging.info("Generating evaluation report...")
    
    report_lines = [
        "# Active Learning Evaluation Report",
        f"Generated at: {datetime.now()}",
        "",
        "## Summary Statistics",
        ""
    ]
    
    # Best performing models
    datasets = df['Dataset'].unique()
    
    for dataset in sorted(datasets):
        dataset_summary = summary_df[summary_df['Dataset'] == dataset]
        
        if dataset_summary.empty:
            continue
        
        report_lines.extend([
            f"### {dataset} Dataset",
            ""
        ])
        
        # Find best MCC performers
        best_mcc = dataset_summary.loc[dataset_summary['MCC'].idxmax()]
        report_lines.extend([
            f"**Best MCC Performance:**",
            f"- Model: {best_mcc['Experiment']}",
            f"- MCC: {best_mcc['MCC']:.4f} ± {best_mcc['MCC_std']:.4f}",
            f"- F1: {best_mcc['F1']:.4f} ± {best_mcc['F1_std']:.4f}",
            f"- ROC AUC: {best_mcc['ROC_AUC']:.4f} ± {best_mcc['ROC_AUC_std']:.4f}",
            ""
        ])
        
        # Best DMCC improvement
        al_summary = dataset_summary[~dataset_summary['Experiment'].str.contains('Full')]
        if not al_summary.empty:
            best_dmcc = al_summary.loc[al_summary['DMCC'].idxmax()]
            report_lines.extend([
                f"**Best DMCC Improvement:**",
                f"- Model: {best_dmcc['Experiment']}",
                f"- DMCC: {best_dmcc['DMCC']:.4f} ± {best_dmcc['DMCC_std']:.4f}",
                f"- Final MCC: {best_dmcc['MCC']:.4f} ± {best_dmcc['MCC_std']:.4f}",
                ""
            ])
        
        # Performance table
        report_lines.extend([
            "#### All Results:",
            ""
        ])
        
        # Format table
        for _, row in dataset_summary.iterrows():
            exp_name = row['Experiment'].replace(f'{dataset}_', '')
            report_lines.append(
                f"| {exp_name} | {row['MCC']:.4f} ± {row['MCC_std']:.4f} | "
                f"{row['F1']:.4f} ± {row['F1_std']:.4f} | "
                f"{row['ROC_AUC']:.4f} ± {row['ROC_AUC_std']:.4f} | "
                f"{row['DMCC']:.4f} ± {row['DMCC_std']:.4f} |"
            )
        
        report_lines.extend(["", "---", ""])
    
    # Key findings
    report_lines.extend([
        "## Key Findings",
        "",
        "### Active Learning Effectiveness",
        "- Compare final performance of AL vs full models",
        "- Analyze convergence patterns",
        "- Evaluate sample efficiency",
        "",
        "### Strategy Comparison", 
        "- Random Forest vs Query-by-Committee",
        "- First5 vs Stratified sampling",
        "- Dataset-specific observations",
        ""
    ])
    
    # Save report
    output_path = Path(output_dir)
    with open(output_path / 'evaluation_report.md', 'w') as f:
        f.write('\n'.join(report_lines))
    
    logging.info(f"Evaluation report saved to {output_path / 'evaluation_report.md'}")

def main():
    """Main function to orchestrate evaluation."""
    args = setup_arguments()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)
    
    # Setup output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Starting evaluation with configuration:")
    logging.info(f"- Results directory: {args.results_dir}")
    logging.info(f"- Output directory: {output_dir}")
    logging.info(f"- Datasets: {args.dataset}")
    logging.info(f"- Metrics: {args.metrics}")
    
    try:
        # Load experiment results
        datasets_to_eval = ['bbb', 'breast_cancer'] if args.dataset == 'both' else [args.dataset]
        all_results = {}
        
        for dataset in datasets_to_eval:
            results = load_experiment_results(args.results_dir, dataset)
            if results['baseline'] or results['active_learning']:
                all_results[dataset] = results
            else:
                logging.warning(f"No results found for {dataset}")
        
        if not all_results:
            logging.error("No experiment results found to evaluate")
            sys.exit(1)
        
        # Create consolidated DataFrame
        df = create_results_dataframe(all_results)
        logging.info(f"Created results DataFrame with {len(df)} rows")
        
        # Generate visualizations
        logging.info("Generating visualizations...")
        
        for dataset in all_results.keys():
            # Learning curves
            plot_learning_curves(df, dataset, args.metrics, output_dir, args)
            
            # DMCC evolution
            if 'dmcc' in [m.lower() for m in args.metrics]:
                plot_dmcc_evolution(df, dataset, output_dir, args)
        
        # Model comparison across datasets
        plot_model_comparison(df, args.metrics, output_dir, args)
        
        # Generate summary statistics
        summary_df = generate_summary_statistics(df, output_dir)
        
        # Generate comprehensive report
        generate_evaluation_report(df, summary_df, output_dir)
        
        # Save consolidated results
        df.to_csv(output_dir / 'consolidated_results.csv', index=False)
        logging.info(f"Saved consolidated results to {output_dir / 'consolidated_results.csv'}")
        
        logging.info("Evaluation completed successfully!")
        logging.info(f"All outputs saved to: {output_dir}")
        
    except Exception as e:
        logging.error(f"Evaluation failed: {e}")
        import traceback
        logging.error(traceback.format_exc())
        sys.exit(1)

if __name__ == '__main__':
    main()