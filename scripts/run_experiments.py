#!/usr/bin/env python3
"""
Active Learning Experiments Runner

This script runs comprehensive active learning experiments comparing Random Forest 
and Query-by-Committee strategies on biomedical datasets.

Usage:
    python scripts/run_experiments.py --dataset bbb --strategies rf qbc
    python scripts/run_experiments.py --dataset breast_cancer --strategies rf
    python scripts/run_experiments.py --dataset both --n-runs 5 --output results/
"""

import argparse
import logging
import sys
import multiprocessing
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from joblib import Parallel, delayed
from tqdm import tqdm
import json
from datetime import datetime

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from active_learning.experiments import ALExperiment
from active_learning.strategies import RandomForestStrategy, QueryByCommitteeStrategy
from active_learning.learners import RandomForestAL, QueryByCommitteeAL
from evaluation.metrics import ModelEvaluator
from utils.config import load_config
from utils.helpers import setup_logging, save_results

def setup_arguments():
    """Setup command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run active learning experiments on biomedical datasets"
    )
    
    parser.add_argument(
        '--dataset', 
        choices=['bbb', 'breast_cancer', 'both'], 
        default='both',
        help='Dataset to run experiments on (default: both)'
    )
    
    parser.add_argument(
        '--strategies', 
        nargs='+', 
        choices=['rf', 'qbc'],
        default=['rf', 'qbc'],
        help='Active learning strategies to evaluate (default: rf qbc)'
    )
    
    parser.add_argument(
        '--sampling-methods',
        nargs='+',
        choices=['first5', 'stratified'],
        default=['first5', 'stratified'],
        help='Initial sampling methods to use (default: first5 stratified)'
    )
    
    parser.add_argument(
        '--data-dir', 
        type=str, 
        default='data/processed/',
        help='Directory containing processed data (default: data/processed/)'
    )
    
    parser.add_argument(
        '--output', 
        type=str, 
        default='results/',
        help='Output directory for results (default: results/)'
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        default='configs/experiment_config.yaml',
        help='Path to experiment configuration file'
    )
    
    parser.add_argument(
        '--n-runs', 
        type=int, 
        default=10,
        help='Number of AL runs to average over (default: 10)'
    )
    
    parser.add_argument(
        '--n-jobs', 
        type=int, 
        default=-1,
        help='Number of parallel jobs (-1 for all cores, default: -1)'
    )
    
    parser.add_argument(
        '--batch-size-bbb', 
        type=int, 
        default=20,
        help='Batch size for BBB dataset (default: 20)'
    )
    
    parser.add_argument(
        '--batch-size-bc', 
        type=int, 
        default=10,
        help='Batch size for Breast Cancer dataset (default: 10)'
    )
    
    parser.add_argument(
        '--force', 
        action='store_true',
        help='Force rerun even if results exist'
    )
    
    parser.add_argument(
        '--baseline-only', 
        action='store_true',
        help='Run only baseline experiments (full model training)'
    )
    
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()

def load_processed_data(data_dir, dataset):
    """Load processed dataset from numpy files."""
    data_path = Path(data_dir)
    
    if dataset == 'bbb':
        X_train = np.load(data_path / 'bbb_X_train.npy')
        X_test = np.load(data_path / 'bbb_X_test.npy') 
        y_train = np.load(data_path / 'bbb_y_train.npy')
        y_test = np.load(data_path / 'bbb_y_test.npy')
        
        return X_train, X_test, y_train, y_test
        
    elif dataset == 'breast_cancer':
        X_train = np.load(data_path / 'bc_X_train.npy')
        X_test = np.load(data_path / 'bc_X_test.npy')
        y_train = np.load(data_path / 'bc_y_train.npy')
        y_test = np.load(data_path / 'bc_y_test.npy')
        
        return X_train, X_test, y_train, y_test
    
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

def run_baseline_experiment(X_train, y_train, X_test, y_test, strategy, dataset, run_id=0):
    """Run baseline experiment (full model on entire training set)."""
    logging.info(f"Running baseline {strategy} experiment for {dataset}, run {run_id}")
    
    if strategy == 'rf':
        learner = RandomForestAL(n_estimators=100, random_state=42+run_id)
    elif strategy == 'qbc':
        learner = QueryByCommitteeAL(n_estimators=100, random_state=42+run_id)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # Train on full training set
    learner.fit(X_train, y_train)
    
    # Evaluate on test set
    evaluator = ModelEvaluator()
    if strategy == 'rf':
        results = evaluator.evaluate_model(learner.model, X_test, y_test)
    else:
        results = evaluator.evaluate_committee(learner.committee, X_test, y_test)
    
    return {
        'strategy': strategy,
        'dataset': dataset,
        'run_id': run_id,
        'type': 'baseline',
        'results': results,
        'model': learner
    }

def run_active_learning_experiment(X_train, y_train, X_test, y_test, strategy, 
                                 sampling_method, config, dataset, run_id=0):
    """Run active learning experiment."""
    logging.info(f"Running AL {strategy} experiment with {sampling_method} sampling for {dataset}, run {run_id}")
    
    # Get dataset-specific config
    if dataset == 'bbb':
        al_config = config.get('active_learning', {}).get('bbb', {})
    else:
        al_config = config.get('active_learning', {}).get('breast_cancer', {})
    
    # Create AL experiment
    experiment = ALExperiment(
        strategy=strategy,
        batch_size=al_config.get('batch_size', 10),
        max_queries=al_config.get('max_queries', -1),
        stop_ratio=al_config.get('stop_ratio', 1.0),
        random_state=42 + run_id
    )
    
    # Run experiment
    results = experiment.run_experiment(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        initial_sampling=sampling_method
    )
    
    return {
        'strategy': strategy,
        'dataset': dataset,
        'sampling_method': sampling_method,
        'run_id': run_id,
        'type': 'active_learning',
        'results': results,
        'final_model': experiment.learner
    }

def run_dataset_experiments(dataset, strategies, sampling_methods, config, args):
    """Run all experiments for a specific dataset."""
    logging.info(f"Starting experiments for {dataset} dataset")
    
    # Load processed data
    try:
        X_train, X_test, y_train, y_test = load_processed_data(args.data_dir, dataset)
        logging.info(f"Loaded {dataset}: train {X_train.shape}, test {X_test.shape}")
    except FileNotFoundError as e:
        logging.error(f"Processed data not found for {dataset}: {e}")
        logging.error("Please run scripts/prepare_data.py first")
        return []
    
    results = []
    n_jobs = args.n_jobs if args.n_jobs > 0 else multiprocessing.cpu_count()
    
    # Run baseline experiments
    logging.info(f"Running baseline experiments for {dataset}...")
    for strategy in strategies:
        baseline_results = Parallel(n_jobs=n_jobs)(
            delayed(run_baseline_experiment)(
                X_train, y_train, X_test, y_test, strategy, dataset, run_id
            ) for run_id in tqdm(range(args.n_runs), desc=f"Baseline {strategy.upper()}")
        )
        results.extend(baseline_results)
    
    # Run active learning experiments (unless baseline-only mode)
    if not args.baseline_only:
        logging.info(f"Running active learning experiments for {dataset}...")
        
        for strategy in strategies:
            for sampling_method in sampling_methods:
                al_results = Parallel(n_jobs=n_jobs)(
                    delayed(run_active_learning_experiment)(
                        X_train, y_train, X_test, y_test, strategy, 
                        sampling_method, config, dataset, run_id
                    ) for run_id in tqdm(range(args.n_runs), 
                                       desc=f"AL {strategy.upper()} {sampling_method}")
                )
                results.extend(al_results)
    
    logging.info(f"Completed experiments for {dataset}: {len(results)} total results")
    return results

def save_experiment_results(results, output_dir, dataset):
    """Save experiment results to files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Separate results by type
    baseline_results = [r for r in results if r['type'] == 'baseline']
    al_results = [r for r in results if r['type'] == 'active_learning']
    
    # Save baseline results
    if baseline_results:
        baseline_file = output_path / f'{dataset}_baseline_results.pkl'
        joblib.dump(baseline_results, baseline_file)
        logging.info(f"Saved baseline results to {baseline_file}")
    
    # Save AL results
    if al_results:
        al_file = output_path / f'{dataset}_al_results.pkl'
        joblib.dump(al_results, al_file)
        logging.info(f"Saved AL results to {al_file}")
    
    # Create summary DataFrame
    summary_rows = []
    for result in results:
        base_row = {
            'dataset': result['dataset'],
            'strategy': result['strategy'],
            'type': result['type'],
            'run_id': result['run_id'],
        }
        
        if result['type'] == 'active_learning':
            base_row['sampling_method'] = result['sampling_method']
            
            # For AL results, extract final metrics
            if 'mcc_scores' in result['results']:
                base_row.update({
                    'final_mcc': result['results']['mcc_scores'][-1],
                    'final_f1': result['results']['f1_scores'][-1],
                    'final_roc_auc': result['results']['roc_auc_scores'][-1],
                    'final_dmcc': result['results']['dmcc_improvements'][-1],
                    'n_iterations': result['results']['n_iterations']
                })
        else:
            # For baseline results
            eval_results = result['results']
            base_row.update({
                'final_mcc': eval_results['mcc'],
                'final_f1': eval_results['f1'], 
                'final_roc_auc': eval_results['roc_auc'],
                'final_dmcc': 0.0,  # Baseline has no improvement
                'n_iterations': 1
            })
        
        summary_rows.append(base_row)
    
    # Save summary
    summary_df = pd.DataFrame(summary_rows)
    summary_file = output_path / f'{dataset}_summary.csv'
    summary_df.to_csv(summary_file, index=False)
    logging.info(f"Saved summary to {summary_file}")
    
    return summary_file

def generate_experiment_report(results, output_dir):
    """Generate a comprehensive experiment report."""
    logging.info("Generating experiment report...")
    
    # Aggregate results by dataset and strategy
    report_lines = [
        "# Active Learning Experiments Report",
        f"Generated at: {datetime.now()}",
        "",
        "## Experiment Configuration",
        ""
    ]
    
    # Group results by dataset
    datasets = set(r['dataset'] for r in results)
    
    for dataset in sorted(datasets):
        dataset_results = [r for r in results if r['dataset'] == dataset]
        report_lines.extend([
            f"### {dataset.upper()} Dataset Results",
            ""
        ])
        
        # Baseline results
        baseline_results = [r for r in dataset_results if r['type'] == 'baseline']
        if baseline_results:
            report_lines.append("#### Baseline (Full Model) Results")
            
            for strategy in ['rf', 'qbc']:
                strategy_baselines = [r for r in baseline_results if r['strategy'] == strategy]
                if strategy_baselines:
                    mccs = [r['results']['mcc'] for r in strategy_baselines]
                    f1s = [r['results']['f1'] for r in strategy_baselines]
                    aucs = [r['results']['roc_auc'] for r in strategy_baselines]
                    
                    report_lines.extend([
                        f"**{strategy.upper()} Full Model:**",
                        f"- MCC: {np.mean(mccs):.4f} ± {np.std(mccs):.4f}",
                        f"- F1: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}",
                        f"- ROC AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}",
                        ""
                    ])
        
        # Active learning results
        al_results = [r for r in dataset_results if r['type'] == 'active_learning']
        if al_results:
            report_lines.append("#### Active Learning Results")
            
            for strategy in ['rf', 'qbc']:
                for sampling in ['first5', 'stratified']:
                    strategy_al = [r for r in al_results 
                                 if r['strategy'] == strategy and r.get('sampling_method') == sampling]
                    
                    if strategy_al:
                        final_mccs = [r['results']['mcc_scores'][-1] for r in strategy_al]
                        final_f1s = [r['results']['f1_scores'][-1] for r in strategy_al]
                        final_aucs = [r['results']['roc_auc_scores'][-1] for r in strategy_al]
                        final_dmccs = [r['results']['dmcc_improvements'][-1] for r in strategy_al]
                        avg_iterations = np.mean([r['results']['n_iterations'] for r in strategy_al])
                        
                        report_lines.extend([
                            f"**{strategy.upper()} {sampling.title()}:**",
                            f"- Final MCC: {np.mean(final_mccs):.4f} ± {np.std(final_mccs):.4f}",
                            f"- Final F1: {np.mean(final_f1s):.4f} ± {np.std(final_f1s):.4f}",
                            f"- Final ROC AUC: {np.mean(final_aucs):.4f} ± {np.std(final_aucs):.4f}",
                            f"- DMCC: {np.mean(final_dmccs):.4f} ± {np.std(final_dmccs):.4f}",
                            f"- Avg iterations: {avg_iterations:.1f}",
                            ""
                        ])
        
        report_lines.append("---\n")
    
    # Save report
    output_path = Path(output_dir)
    with open(output_path / 'experiment_report.md', 'w') as f:
        f.write('\n'.join(report_lines))
    
    logging.info(f"Experiment report saved to {output_path / 'experiment_report.md'}")

def main():
    """Main function to orchestrate experiments."""
    args = setup_arguments()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)
    
    # Setup output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    try:
        config = load_config(args.config)
        logging.info(f"Loaded configuration from {args.config}")
    except Exception as e:
        logging.warning(f"Could not load config file {args.config}: {e}")
        logging.warning("Using default configuration")
        config = {
            'active_learning': {
                'bbb': {'batch_size': args.batch_size_bbb},
                'breast_cancer': {'batch_size': args.batch_size_bc}
            }
        }
    
    logging.info(f"Starting experiments with configuration:")
    logging.info(f"- Datasets: {args.dataset}")
    logging.info(f"- Strategies: {args.strategies}")
    logging.info(f"- Sampling methods: {args.sampling_methods}")
    logging.info(f"- Number of runs: {args.n_runs}")
    logging.info(f"- Parallel jobs: {args.n_jobs}")
    logging.info(f"- Output directory: {output_dir}")
    
    all_results = []
    
    try:
        # Run experiments for each dataset
        datasets_to_run = ['bbb', 'breast_cancer'] if args.dataset == 'both' else [args.dataset]
        
        for dataset in datasets_to_run:
            # Check if results already exist
            existing_files = [
                output_dir / f'{dataset}_baseline_results.pkl',
                output_dir / f'{dataset}_al_results.pkl'
            ]
            
            if all(f.exists() for f in existing_files) and not args.force:
                logging.info(f"Results for {dataset} already exist. Use --force to rerun.")
                continue
            
            # Run experiments
            dataset_results = run_dataset_experiments(
                dataset, args.strategies, args.sampling_methods, config, args
            )
            
            if dataset_results:
                # Save individual dataset results
                save_experiment_results(dataset_results, output_dir, dataset)
                all_results.extend(dataset_results)
            else:
                logging.warning(f"No results generated for {dataset}")
        
        if all_results:
            # Generate comprehensive report
            generate_experiment_report(all_results, output_dir)
            
            # Save experiment metadata
            metadata = {
                'timestamp': datetime.now().isoformat(),
                'arguments': vars(args),
                'total_experiments': len(all_results),
                'datasets': list(set(r['dataset'] for r in all_results)),
                'strategies': list(set(r['strategy'] for r in all_results)),
                'config': config
            }
            
            with open(output_dir / 'experiment_metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logging.info("All experiments completed successfully!")
            logging.info(f"Results saved to: {output_dir}")
            
        else:
            logging.warning("No experiments were run")
            
    except Exception as e:
        logging.error(f"Experiments failed: {e}")
        import traceback
        logging.error(traceback.format_exc())
        sys.exit(1)

if __name__ == '__main__':
    main()