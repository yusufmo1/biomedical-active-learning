#!/usr/bin/env python3
"""
Data Preparation Script for Biomedical Active Learning

This script loads, preprocesses, and prepares datasets for active learning experiments.
Supports both Blood-Brain Barrier Penetration (BBBP) and Breast Cancer datasets.

Usage:
    python scripts/prepare_data.py --dataset bbb --output data/processed/
    python scripts/prepare_data.py --dataset breast_cancer --output data/processed/
    python scripts/prepare_data.py --dataset both --output data/processed/
"""

import argparse
import logging
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data.loader import DatasetLoader
from data.preprocessing import BBBPreprocessor, BreastCancerPreprocessor
from utils.config import load_config
from utils.helpers import setup_logging, save_statistics

def setup_arguments():
    """Setup command line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare datasets for biomedical active learning experiments"
    )
    
    parser.add_argument(
        '--dataset', 
        choices=['bbb', 'breast_cancer', 'both'], 
        default='both',
        help='Dataset to process (default: both)'
    )
    
    parser.add_argument(
        '--output', 
        type=str, 
        default='data/processed/',
        help='Output directory for processed data (default: data/processed/)'
    )
    
    parser.add_argument(
        '--data-dir', 
        type=str, 
        default='data/raw/',
        help='Input directory containing raw data (default: data/raw/)'
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        default='configs/data_config.yaml',
        help='Path to data configuration file'
    )
    
    parser.add_argument(
        '--test-size', 
        type=float, 
        default=0.2,
        help='Test set size ratio (default: 0.2)'
    )
    
    parser.add_argument(
        '--random-state', 
        type=int, 
        default=42,
        help='Random state for reproducibility (default: 42)'
    )
    
    parser.add_argument(
        '--pca-variance', 
        type=float, 
        default=0.90,
        help='PCA variance threshold for BBB dataset (default: 0.90)'
    )
    
    parser.add_argument(
        '--force', 
        action='store_true',
        help='Force reprocessing even if output files exist'
    )
    
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()

def check_mol2vec_model():
    """Download and setup Mol2vec model if needed."""
    import os
    import tarfile
    import requests
    
    model_path = "model_300dim.pkl"
    if os.path.exists(model_path):
        logging.info("Mol2vec model already exists")
        return model_path
        
    logging.info("Downloading Mol2vec model...")
    url = "https://deepchemdata.s3-us-west-1.amazonaws.com/trained_models/mol2vec_model_300dim.tar.gz"
    
    try:
        response = requests.get(url, timeout=300)  # 5 minute timeout
        with open("mol2vec_model_300dim.tar.gz", "wb") as f:
            f.write(response.content)
        
        with tarfile.open("mol2vec_model_300dim.tar.gz", "r:gz") as tar:
            tar.extractall()
        
        os.rename("mol2vec_model_300dim.pkl", model_path)
        os.remove("mol2vec_model_300dim.tar.gz")
        logging.info(f"Mol2vec model downloaded and saved as {model_path}")
        
    except Exception as e:
        logging.error(f"Failed to download Mol2vec model: {e}")
        raise
        
    return model_path

def process_bbb_dataset(data_dir, output_dir, config, args):
    """Process Blood-Brain Barrier Penetration dataset."""
    logging.info("Processing BBB dataset...")
    
    # Check for existing processed files
    output_files = [
        output_dir / 'bbb_X_train.npy',
        output_dir / 'bbb_X_test.npy', 
        output_dir / 'bbb_y_train.npy',
        output_dir / 'bbb_y_test.npy',
        output_dir / 'bbb_scaler.pkl',
        output_dir / 'bbb_pca.pkl',
        output_dir / 'bbb_smiles_train.pkl',
        output_dir / 'bbb_smiles_test.pkl'
    ]
    
    if all(f.exists() for f in output_files) and not args.force:
        logging.info("BBB processed files already exist. Use --force to reprocess.")
        return
    
    # Load raw data
    loader = DatasetLoader(data_dir)
    bbb_df = loader.load_bbb_dataset()
    
    # Ensure Mol2vec model is available
    check_mol2vec_model()
    
    # Process dataset
    preprocessor = BBBPreprocessor()
    bbb_processed = preprocessor.process_full_pipeline(bbb_df)
    
    # Separate features and target
    feature_cols = [col for col in bbb_processed.columns if col not in ['BBB', 'SMILES']]
    X = bbb_processed[feature_cols].values
    y = bbb_processed['BBB'].values
    smiles = bbb_processed['SMILES'].values
    
    logging.info(f"BBB dataset shape: {X.shape}")
    logging.info(f"Class distribution: {np.bincount(y)}")
    
    # Train/test split preserving SMILES alignment
    indices = np.arange(len(X))
    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        X, y, indices, test_size=args.test_size, random_state=args.random_state, stratify=y
    )
    
    smiles_train = smiles[train_idx]
    smiles_test = smiles[test_idx]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=None)
    pca.fit(X_train_scaled)
    
    # Determine number of components for desired variance
    cumsum_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(cumsum_variance >= args.pca_variance) + 1
    
    logging.info(f"PCA components for {args.pca_variance*100:.1f}% variance: {n_components}")
    
    # Refit PCA with selected components
    pca_final = PCA(n_components=n_components)
    X_train_pca = pca_final.fit_transform(X_train_scaled)
    X_test_pca = pca_final.transform(X_test_scaled)
    
    # Save processed data
    np.save(output_dir / 'bbb_X_train.npy', X_train_pca)
    np.save(output_dir / 'bbb_X_test.npy', X_test_pca)
    np.save(output_dir / 'bbb_y_train.npy', y_train)
    np.save(output_dir / 'bbb_y_test.npy', y_test)
    
    joblib.dump(scaler, output_dir / 'bbb_scaler.pkl')
    joblib.dump(pca_final, output_dir / 'bbb_pca.pkl')
    joblib.dump(smiles_train, output_dir / 'bbb_smiles_train.pkl')
    joblib.dump(smiles_test, output_dir / 'bbb_smiles_test.pkl')
    
    # Save statistics
    stats = {
        'dataset': 'BBB',
        'original_shape': X.shape,
        'processed_shape': X_train_pca.shape,
        'test_shape': X_test_pca.shape,
        'pca_components': n_components,
        'explained_variance': float(cumsum_variance[n_components-1]),
        'class_distribution_train': np.bincount(y_train).tolist(),
        'class_distribution_test': np.bincount(y_test).tolist(),
        'feature_columns': feature_cols
    }
    
    save_statistics(output_dir / 'bbb_stats.json', stats)
    logging.info("BBB dataset processing completed successfully")

def process_breast_cancer_dataset(data_dir, output_dir, config, args):
    """Process Breast Cancer dataset."""
    logging.info("Processing Breast Cancer dataset...")
    
    # Check for existing processed files
    output_files = [
        output_dir / 'bc_X_train.npy',
        output_dir / 'bc_X_test.npy',
        output_dir / 'bc_y_train.npy', 
        output_dir / 'bc_y_test.npy',
        output_dir / 'bc_scaler.pkl'
    ]
    
    if all(f.exists() for f in output_files) and not args.force:
        logging.info("Breast Cancer processed files already exist. Use --force to reprocess.")
        return
    
    # Load raw data
    loader = DatasetLoader(data_dir)
    bc_df = loader.load_breast_cancer_dataset()
    
    # Process dataset
    preprocessor = BreastCancerPreprocessor()
    bc_processed = preprocessor.process(bc_df)
    
    # Separate features and target
    X = bc_processed.drop(columns=['target', 'id'], errors='ignore').values
    y = bc_processed['target'].values
    
    logging.info(f"Breast Cancer dataset shape: {X.shape}")
    logging.info(f"Class distribution: {np.bincount(y)}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save processed data
    np.save(output_dir / 'bc_X_train.npy', X_train_scaled)
    np.save(output_dir / 'bc_X_test.npy', X_test_scaled)
    np.save(output_dir / 'bc_y_train.npy', y_train)
    np.save(output_dir / 'bc_y_test.npy', y_test)
    
    joblib.dump(scaler, output_dir / 'bc_scaler.pkl')
    
    # Save statistics
    feature_names = bc_processed.drop(columns=['target', 'id'], errors='ignore').columns.tolist()
    stats = {
        'dataset': 'BreastCancer', 
        'original_shape': X.shape,
        'train_shape': X_train_scaled.shape,
        'test_shape': X_test_scaled.shape,
        'class_distribution_train': np.bincount(y_train).tolist(),
        'class_distribution_test': np.bincount(y_test).tolist(),
        'feature_names': feature_names
    }
    
    save_statistics(output_dir / 'bc_stats.json', stats)
    logging.info("Breast Cancer dataset processing completed successfully")

def generate_data_report(output_dir):
    """Generate a comprehensive data preparation report."""
    logging.info("Generating data preparation report...")
    
    report_lines = [
        "# Data Preparation Report",
        f"Generated at: {pd.Timestamp.now()}",
        "",
        "## Processed Datasets",
        ""
    ]
    
    # Check for BBB dataset
    if (output_dir / 'bbb_stats.json').exists():
        import json
        with open(output_dir / 'bbb_stats.json', 'r') as f:
            bbb_stats = json.load(f)
        
        report_lines.extend([
            "### Blood-Brain Barrier Penetration Dataset",
            f"- Original shape: {bbb_stats['original_shape']}",
            f"- Train shape: {bbb_stats['processed_shape']}",
            f"- Test shape: {bbb_stats['test_shape']}",
            f"- PCA components: {bbb_stats['pca_components']}",
            f"- Explained variance: {bbb_stats['explained_variance']:.3f}",
            f"- Train class distribution: {bbb_stats['class_distribution_train']}",
            f"- Test class distribution: {bbb_stats['class_distribution_test']}",
            ""
        ])
    
    # Check for Breast Cancer dataset
    if (output_dir / 'bc_stats.json').exists():
        import json
        with open(output_dir / 'bc_stats.json', 'r') as f:
            bc_stats = json.load(f)
            
        report_lines.extend([
            "### Breast Cancer Dataset",
            f"- Original shape: {bc_stats['original_shape']}",
            f"- Train shape: {bc_stats['train_shape']}",
            f"- Test shape: {bc_stats['test_shape']}",
            f"- Train class distribution: {bc_stats['class_distribution_train']}",
            f"- Test class distribution: {bc_stats['class_distribution_test']}",
            ""
        ])
    
    report_lines.extend([
        "## Files Generated",
        "",
        "### BBB Dataset Files:",
        "- `bbb_X_train.npy` - Training features (post-PCA)",
        "- `bbb_X_test.npy` - Test features (post-PCA)", 
        "- `bbb_y_train.npy` - Training labels",
        "- `bbb_y_test.npy` - Test labels",
        "- `bbb_scaler.pkl` - Feature scaler",
        "- `bbb_pca.pkl` - PCA transformer",
        "- `bbb_smiles_train.pkl` - Training SMILES strings",
        "- `bbb_smiles_test.pkl` - Test SMILES strings",
        "- `bbb_stats.json` - Processing statistics",
        "",
        "### Breast Cancer Dataset Files:",
        "- `bc_X_train.npy` - Training features (scaled)",
        "- `bc_X_test.npy` - Test features (scaled)",
        "- `bc_y_train.npy` - Training labels", 
        "- `bc_y_test.npy` - Test labels",
        "- `bc_scaler.pkl` - Feature scaler",
        "- `bc_stats.json` - Processing statistics",
        ""
    ])
    
    # Save report
    with open(output_dir / 'data_preparation_report.md', 'w') as f:
        f.write('\n'.join(report_lines))
    
    logging.info(f"Data preparation report saved to {output_dir / 'data_preparation_report.md'}")

def main():
    """Main function to orchestrate data preparation."""
    args = setup_arguments()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)
    
    # Setup directories
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        logging.warning(f"Could not load config file {args.config}: {e}")
        config = {}
    
    logging.info(f"Starting data preparation for: {args.dataset}")
    logging.info(f"Input directory: {data_dir}")
    logging.info(f"Output directory: {output_dir}")
    
    try:
        # Process datasets based on selection
        if args.dataset in ['bbb', 'both']:
            process_bbb_dataset(data_dir, output_dir, config, args)
            
        if args.dataset in ['breast_cancer', 'both']:
            process_breast_cancer_dataset(data_dir, output_dir, config, args)
        
        # Generate comprehensive report
        generate_data_report(output_dir)
        
        logging.info("Data preparation completed successfully!")
        
    except Exception as e:
        logging.error(f"Data preparation failed: {e}")
        import traceback
        logging.error(traceback.format_exc())
        sys.exit(1)

if __name__ == '__main__':
    main()