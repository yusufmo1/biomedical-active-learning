"""
Integration tests for the biomedical active learning pipeline.

Tests full pipeline execution and experiment reproducibility.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, Mock
import sys
import os

# Add src to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.data.loader import DatasetLoader
from src.data.preprocessing import BreastCancerPreprocessor
from src.active_learning.strategies import LeastConfidenceSampling, RandomSampling
from src.active_learning.learners import RandomForestAL, QueryByCommitteeAL
from src.evaluation.metrics import ModelEvaluator


class TestFullPipelineExecution:
    """Test complete pipeline from data loading to evaluation."""
    
    def test_breast_cancer_pipeline(self):
        """Test full pipeline with Breast Cancer dataset."""
        # Use fixture data
        fixtures_dir = Path(__file__).parent / "fixtures"
        
        # Load data
        loader = DatasetLoader(fixtures_dir)
        with patch('src.data.loader.inspect_dataframe'):
            df = loader.load_breast_cancer_dataset("mock_breast_cancer.csv")
        
        # Preprocess
        preprocessor = BreastCancerPreprocessor()
        features_df, metadata_df = preprocessor.preprocess(df)
        
        # Convert to arrays
        X = features_df.values
        y = metadata_df['target'].values
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Initialize learner
        learner = RandomForestAL(n_estimators=10, random_state=42)
        learner.fit(X_train, y_train)
        
        # Evaluate
        evaluator = ModelEvaluator()
        results = evaluator.evaluate_single_model(learner, X_test, y_test, "rf_test")
        
        # Verify pipeline completed successfully
        assert 'mcc' in results
        assert 'f1' in results
        assert -1 <= results['mcc'] <= 1
        assert 0 <= results['f1'] <= 1
        assert len(features_df.columns) == 30  # Expected BC features
    
    def test_active_learning_simulation(self, train_test_split_data):
        """Test simplified active learning simulation."""
        data = train_test_split_data
        X_train, y_train = data['X_train'], data['y_train']
        X_test, y_test = data['X_test'], data['y_test']
        
        # Start with small labeled set
        initial_size = 5
        X_labeled = X_train[:initial_size]
        y_labeled = y_train[:initial_size]
        X_pool = X_train[initial_size:initial_size+20]  # Small pool
        y_pool = y_train[initial_size:initial_size+20]
        
        # Initialize learner and strategy
        learner = RandomForestAL(n_estimators=10, random_state=42)
        strategy = LeastConfidenceSampling()
        evaluator = ModelEvaluator()
        
        # Track performance
        performances = []
        
        # Active learning loop (3 iterations)
        for iteration in range(3):
            # Train on current labeled data
            learner.fit(X_labeled, y_labeled)
            
            # Evaluate
            results = evaluator.evaluate_single_model(
                learner, X_test, y_test, f"iter_{iteration}"
            )
            performances.append(results['mcc'])
            
            # Query next sample if pool not empty
            if len(X_pool) > 0:
                query_idx = strategy.select_sample(learner, X_pool)
                
                # Add queried sample to labeled set
                X_labeled = np.vstack([X_labeled, X_pool[query_idx:query_idx+1]])
                y_labeled = np.append(y_labeled, y_pool[query_idx])
                
                # Remove from pool
                X_pool = np.delete(X_pool, query_idx, axis=0)
                y_pool = np.delete(y_pool, query_idx)
        
        # Verify simulation ran successfully
        assert len(performances) == 3
        assert all(-1 <= mcc <= 1 for mcc in performances)
        assert len(X_labeled) == initial_size + 2  # Added 2 samples
    
    def test_qbc_vs_rf_comparison(self, train_test_split_data):
        """Test comparison between QBC and RF active learning."""
        data = train_test_split_data
        X_train, y_train = data['X_train'], data['y_train']
        X_test, y_test = data['X_test'], data['y_test']
        
        # Train both learners
        rf_learner = RandomForestAL(n_estimators=20, random_state=42)
        qbc_learner = QueryByCommitteeAL(random_state=42)
        
        rf_learner.fit(X_train, y_train)
        qbc_learner.fit(X_train, y_train)
        
        # Evaluate both
        evaluator = ModelEvaluator()
        rf_results = evaluator.evaluate_single_model(rf_learner, X_test, y_test, "rf")
        qbc_results = evaluator.evaluate_committee(qbc_learner.committee, X_test, y_test, "qbc")
        
        # Compare
        comparison = evaluator.compare_models(['mcc', 'f1'])
        
        # Verify comparison completed
        assert 'rf' in comparison['mcc']
        assert 'qbc' in comparison['mcc']
        assert isinstance(comparison['mcc']['rf'], (float, np.floating))
        assert isinstance(comparison['mcc']['qbc'], (float, np.floating))


class TestExperimentReproducibility:
    """Test experiment reproducibility and consistency."""
    
    def test_random_seed_reproducibility(self, train_test_split_data):
        """Test that results are reproducible with same random seed."""
        data = train_test_split_data
        X_train, y_train = data['X_train'], data['y_train']
        X_test, y_test = data['X_test'], data['y_test']
        
        # Run experiment twice with same seed
        results1 = self._run_experiment(X_train, y_train, X_test, y_test, seed=42)
        results2 = self._run_experiment(X_train, y_train, X_test, y_test, seed=42)
        
        # Results should be identical
        assert results1['mcc'] == results2['mcc']
        assert results1['f1'] == results2['f1']
        np.testing.assert_array_equal(results1['predictions'], results2['predictions'])
    
    def test_different_seed_variation(self, train_test_split_data):
        """Test that different seeds produce different results."""
        data = train_test_split_data
        X_train, y_train = data['X_train'], data['y_train']
        X_test, y_test = data['X_test'], data['y_test']
        
        # Run experiment with different seeds
        results1 = self._run_experiment(X_train, y_train, X_test, y_test, seed=42)
        results2 = self._run_experiment(X_train, y_train, X_test, y_test, seed=123)
        
        # Results should be different (with high probability)
        # Note: They could be the same by chance, but very unlikely
        assert (results1['mcc'] != results2['mcc'] or 
                not np.array_equal(results1['predictions'], results2['predictions']))
    
    def _run_experiment(self, X_train, y_train, X_test, y_test, seed):
        """Helper method to run a single experiment."""
        learner = RandomForestAL(n_estimators=10, random_state=seed)
        learner.fit(X_train, y_train)
        
        evaluator = ModelEvaluator()
        results = evaluator.evaluate_single_model(learner, X_test, y_test)
        
        predictions = learner.predict(X_test)
        
        return {
            'mcc': results['mcc'],
            'f1': results['f1'],
            'predictions': predictions
        }
    
    def test_strategy_consistency(self, train_test_split_data):
        """Test that sampling strategies are consistent."""
        data = train_test_split_data
        X_train, y_train = data['X_train'], data['y_train']
        X_pool = data['X_test'][:10]  # Small pool
        
        # Train learner
        learner = RandomForestAL(n_estimators=10, random_state=42)
        learner.fit(X_train, y_train)
        
        # Test least confidence strategy consistency
        lc_strategy = LeastConfidenceSampling()
        
        # Should select same sample multiple times
        idx1 = lc_strategy.select_sample(learner, X_pool)
        idx2 = lc_strategy.select_sample(learner, X_pool)
        
        assert idx1 == idx2
        assert 0 <= idx1 < len(X_pool)
    
    def test_random_strategy_variation(self, train_test_split_data):
        """Test that random strategy produces variation."""
        data = train_test_split_data
        X_pool = data['X_test'][:10]
        
        random_strategy = RandomSampling()
        
        # Collect multiple selections
        selections = []
        np.random.seed(42)
        for _ in range(20):
            idx = random_strategy.select_sample(None, X_pool)
            selections.append(idx)
        
        # Should have some variation (not all the same)
        unique_selections = set(selections)
        assert len(unique_selections) > 1
        assert all(0 <= idx < len(X_pool) for idx in selections)


class TestConfigurationIntegration:
    """Test integration with configuration files."""
    
    def test_yaml_config_loading(self):
        """Test loading configuration from YAML files."""
        import yaml
        
        # Test data config structure
        data_config = {
            'datasets': {
                'breast_cancer': {
                    'name': 'Breast Cancer',
                    'path': 'data/raw/breast-cancer.csv',
                    'target_column': 'diagnosis'
                }
            }
        }
        
        # Test experiment config structure
        experiment_config = {
            'active_learning': {
                'strategies': ['random_forest', 'query_by_committee'],
                'initial_samples': ['first_5', 'stratified_5'],
                'batch_size': 10,
                'n_runs': 10,
                'stop_ratio': 1.0
            }
        }
        
        # Test model config structure
        model_config = {
            'random_forest': {
                'n_estimators': 100,
                'random_state': 42
            },
            'qbc': {
                'n_committee_members': 5,
                'base_learners': ['RandomForest', 'ExtraTrees', 'GradientBoosting']
            }
        }
        
        # Verify configs can be processed
        assert 'datasets' in data_config
        assert 'active_learning' in experiment_config
        assert 'random_forest' in model_config
        
        # Test that RF can be created with config
        rf_params = model_config['random_forest']
        learner = RandomForestAL(**rf_params)
        assert learner.model.n_estimators == 100
        assert learner.model.random_state == 42


class TestDataPipelineIntegration:
    """Test integration of data processing components."""
    
    def test_data_loading_preprocessing_chain(self):
        """Test complete data processing chain."""
        fixtures_dir = Path(__file__).parent / "fixtures"
        
        # Load breast cancer data
        loader = DatasetLoader(fixtures_dir)
        with patch('src.data.loader.inspect_dataframe'):
            df = loader.load_breast_cancer_dataset("mock_breast_cancer.csv")
        
        # Verify initial data structure
        assert 'diagnosis' in df.columns
        assert len(df) > 0
        
        # Preprocess
        preprocessor = BreastCancerPreprocessor()
        features_df, metadata_df = preprocessor.preprocess(df)
        
        # Verify preprocessing results
        assert 'target' in metadata_df.columns
        assert 'diagnosis' not in features_df.columns
        assert len(features_df) == len(metadata_df)
        assert set(metadata_df['target'].unique()) == {0, 1}
    
    def test_mock_bbb_data_processing(self):
        """Test BBB data processing with mock data."""
        fixtures_dir = Path(__file__).parent / "fixtures"
        
        # Load BBB data
        loader = DatasetLoader(fixtures_dir)
        with patch('src.data.loader.inspect_dataframe'):
            df = loader.load_bbb_dataset("mock_bbb.csv")
        
        # Verify basic structure
        assert 'smiles' in df.columns
        assert 'p_np' in df.columns
        assert len(df) == 20  # Mock data has 20 compounds
        
        # Verify SMILES are valid strings
        assert all(isinstance(smiles, str) for smiles in df['smiles'])


class TestErrorHandlingIntegration:
    """Test error handling in integrated scenarios."""
    
    def test_invalid_data_handling(self):
        """Test handling of invalid data in pipeline."""
        # Create invalid data
        invalid_df = pd.DataFrame({
            'invalid_column': [1, 2, 3],
            'another_invalid': ['a', 'b', 'c']
        })
        
        # Should raise appropriate errors
        preprocessor = BreastCancerPreprocessor()
        with pytest.raises(KeyError):
            preprocessor.preprocess(invalid_df)
    
    def test_empty_data_handling(self):
        """Test handling of empty datasets."""
        empty_df = pd.DataFrame(columns=['diagnosis', 'radius_mean'])
        
        preprocessor = BreastCancerPreprocessor()
        
        # Should handle empty data gracefully or raise appropriate error
        with pytest.raises((ValueError, IndexError, KeyError)):
            preprocessor.preprocess(empty_df)
    
    def test_single_class_data(self):
        """Test handling of single-class datasets."""
        # Create data with only one class
        single_class_df = pd.DataFrame({
            'diagnosis': ['M', 'M', 'M'],  # Only malignant
            'radius_mean': [12.0, 13.0, 14.0],
            'texture_mean': [15.0, 16.0, 17.0]
        })
        
        preprocessor = BreastCancerPreprocessor()
        features_df, metadata_df = preprocessor.preprocess(single_class_df)
        
        # Should process without error
        assert len(features_df) == 3
        assert len(metadata_df) == 3
        assert set(metadata_df['target'].unique()) == {1}  # Only class 1


class TestPerformanceIntegration:
    """Test performance characteristics of integrated pipeline."""
    
    @pytest.mark.slow
    def test_large_dataset_pipeline(self, performance_monitor):
        """Test pipeline performance with larger dataset."""
        from sklearn.datasets import make_classification
        
        # Create larger dataset
        X, y = make_classification(
            n_samples=500, n_features=30, n_informative=15,
            n_redundant=5, random_state=42
        )
        
        # Convert to DataFrame format
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=feature_names)
        df['diagnosis'] = ['M' if label == 1 else 'B' for label in y]
        
        performance_monitor.start()
        
        # Run preprocessing
        preprocessor = BreastCancerPreprocessor()
        features_df, metadata_df = preprocessor.preprocess(df)
        
        # Train learner
        X_processed = features_df.values
        y_processed = metadata_df['target'].values
        
        learner = RandomForestAL(n_estimators=50, random_state=42)
        learner.fit(X_processed, y_processed)
        
        # Evaluate
        evaluator = ModelEvaluator()
        results = evaluator.evaluate_single_model(
            learner, X_processed[:100], y_processed[:100], "large_test"
        )
        
        stats = performance_monitor.stop()
        
        # Should complete in reasonable time
        assert stats['execution_time'] < 30.0  # Less than 30 seconds
        assert 'mcc' in results
        assert len(features_df) == 500
    
    @pytest.mark.slow 
    def test_committee_performance_integration(self, performance_monitor, train_test_split_data):
        """Test QBC committee performance in full pipeline."""
        data = train_test_split_data
        X_train, y_train = data['X_train'], data['y_train']
        X_test, y_test = data['X_test'], data['y_test']
        
        performance_monitor.start()
        
        # Create and train QBC learner
        qbc_learner = QueryByCommitteeAL(
            rf={'n_estimators': 30},
            extratrees={'n_estimators': 30},
            gb={'n_estimators': 20},
            random_state=42
        )
        qbc_learner.fit(X_train, y_train)
        
        # Comprehensive evaluation
        evaluator = ModelEvaluator()
        results = evaluator.evaluate_committee(
            qbc_learner.committee, X_test, y_test, "qbc_perf_test"
        )
        
        stats = performance_monitor.stop()
        
        # Should complete in reasonable time
        assert stats['execution_time'] < 20.0  # Less than 20 seconds
        assert 'member_results' in results
        assert len(results['member_results']) == 5  # Default committee size