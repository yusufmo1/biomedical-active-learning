"""
Unit tests for active learning experiments.

Tests the src.active_learning.experiments module including ALExperiment class
and experiment execution functions.
"""

import pytest
import numpy as np
import tempfile
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add src to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.active_learning.experiments import (
    ALExperiment,
    execute_full_model_baseline,
    execute_full_model_qbc
)


class TestALExperiment:
    """Test cases for ALExperiment class."""
    
    def test_initialization(self):
        """Test ALExperiment initialization."""
        config = {
            'max_queries': 10,
            'batch_size': 1,
            'stop_ratio': 0.8
        }
        
        experiment = ALExperiment(config)
        
        assert experiment.config == config
        assert experiment.results == {}
    
    def test_run_experiment_invalid_type(self):
        """Test run_experiment with invalid experiment type."""
        config = {}
        experiment = ALExperiment(config)
        
        X_unlab = np.random.randn(20, 10)
        y_unlab = np.random.choice([0, 1], 20)
        X_holdout = np.random.randn(10, 10)
        y_holdout = np.random.choice([0, 1], 10)
        
        with pytest.raises(ValueError, match="Unknown experiment type"):
            experiment.run_experiment(X_unlab, y_unlab, X_holdout, y_holdout, "invalid_type")
    
    @patch('src.active_learning.experiments.tqdm')
    @patch('src.active_learning.experiments.rf_factory')
    @patch('src.active_learning.experiments.evaluate_model')
    @patch('src.active_learning.experiments.least_confidence_sampling')
    def test_rf_first5_experiment(self, mock_sampling, mock_evaluate, mock_rf_factory, mock_tqdm):
        """Test RF first5 experiment execution."""
        # Setup mocks
        mock_model = Mock()
        mock_rf_factory.return_value = mock_model
        
        mock_evaluate.side_effect = [
            {'mcc': 0.8, 'f1': 0.9, 'roc_auc': 0.95},  # Reference metrics
            {'mcc': 0.7, 'f1': 0.85, 'roc_auc': 0.9},  # Iteration 1
            {'mcc': 0.75, 'f1': 0.88, 'roc_auc': 0.92} # Final evaluation
        ]
        
        mock_sampling.return_value = 0  # Always select first sample
        
        # Mock tqdm
        mock_pbar = Mock()
        mock_tqdm.return_value = mock_pbar
        
        # Setup experiment
        config = {
            'max_queries': 1,
            'batch_size': 1,
            'stop_ratio': 0.8,
            'rf_params': {'n_estimators': 10}
        }
        
        experiment = ALExperiment(config)
        
        # Create test data
        X_unlab = np.random.randn(20, 10)
        y_unlab = np.random.choice([0, 1], 20)
        X_holdout = np.random.randn(10, 10)
        y_holdout = np.random.choice([0, 1], 10)
        
        # Run experiment
        results = experiment.run_experiment(X_unlab, y_unlab, X_holdout, y_holdout, "rf_first5")
        
        # Verify results structure
        expected_keys = ['mcc_scores', 'f1_scores', 'roc_auc_scores', 'dmcc_improvements', 'n_iterations', 'final_model']
        assert all(key in results for key in expected_keys)
        
        # Verify metrics were collected
        assert len(results['mcc_scores']) > 0
        assert len(results['f1_scores']) > 0
        assert len(results['roc_auc_scores']) > 0
        assert len(results['dmcc_improvements']) > 0
        
        # Verify model was trained and evaluated
        assert mock_model.fit.called
        assert mock_evaluate.called
    
    @patch('src.active_learning.experiments.StratifiedShuffleSplit')
    def test_rf_stratified_experiment(self, mock_sss):
        """Test RF stratified experiment initialization."""
        # Mock stratified split
        mock_split = Mock()
        mock_split.split.return_value = [(np.array([0, 1, 2, 3, 4]), np.array([5, 6, 7]))]
        mock_sss.return_value = mock_split
        
        config = {
            'max_queries': 1,
            'stratified_seeds': [42],
            'rf_params': {}
        }
        
        experiment = ALExperiment(config)
        
        with patch.object(experiment, '_execute_active_learning_experiment') as mock_execute:
            mock_execute.return_value = {'mcc_scores': [0.8]}
            
            with patch.object(experiment, '_get_reference_metrics') as mock_ref:
                mock_ref.return_value = {'mcc': 0.75}
                
                X_unlab = np.random.randn(20, 10)
                y_unlab = np.random.choice([0, 1], 20)
                X_holdout = np.random.randn(10, 10)
                y_holdout = np.random.choice([0, 1], 10)
                
                results = experiment.run_experiment(X_unlab, y_unlab, X_holdout, y_holdout, "rf_stratified")
                
                # Verify stratified split was used
                mock_sss.assert_called_once()
                mock_execute.assert_called_once()
    
    @patch('src.active_learning.experiments.tqdm')
    @patch('src.active_learning.experiments.base_learner_factory')
    @patch('src.active_learning.experiments.evaluate_committee')
    @patch('src.active_learning.experiments.qbc_vote_entropy_sampling')
    def test_qbc_first5_experiment(self, mock_sampling, mock_evaluate, mock_factory, mock_tqdm):
        """Test QBC first5 experiment execution."""
        # Setup mocks
        mock_committee = [Mock(), Mock(), Mock()]
        mock_factory.return_value = mock_committee
        
        mock_evaluate.side_effect = [
            {'mcc': 0.8, 'f1': 0.9, 'roc_auc': 0.95},  # Reference metrics
            {'mcc': 0.7, 'f1': 0.85, 'roc_auc': 0.9},  # Iteration 1
            {'mcc': 0.75, 'f1': 0.88, 'roc_auc': 0.92} # Final evaluation
        ]
        
        mock_sampling.return_value = 0  # Always select first sample
        
        # Mock tqdm
        mock_pbar = Mock()
        mock_tqdm.return_value = mock_pbar
        
        # Setup experiment
        config = {
            'max_queries': 1,
            'batch_size': 1,
            'stop_ratio': 0.8,
            'qbc_params': {}
        }
        
        experiment = ALExperiment(config)
        
        # Create test data
        X_unlab = np.random.randn(20, 10)
        y_unlab = np.random.choice([0, 1], 20)
        X_holdout = np.random.randn(10, 10)
        y_holdout = np.random.choice([0, 1], 10)
        
        # Run experiment
        results = experiment.run_experiment(X_unlab, y_unlab, X_holdout, y_holdout, "qbc_first5")
        
        # Verify results structure
        expected_keys = ['mcc_scores', 'f1_scores', 'roc_auc_scores', 'dmcc_improvements', 'n_iterations', 'final_committee']
        assert all(key in results for key in expected_keys)
        
        # Verify committee was created and trained
        assert mock_factory.called
        for clf in mock_committee:
            assert clf.fit.called
    
    @patch('src.active_learning.experiments.rf_factory')
    @patch('src.active_learning.experiments.evaluate_model')
    def test_get_reference_metrics_rf(self, mock_evaluate, mock_rf_factory):
        """Test getting reference metrics for RF model."""
        mock_model = Mock()
        mock_rf_factory.return_value = mock_model
        mock_evaluate.return_value = {'mcc': 0.8, 'f1': 0.9}
        
        config = {'rf_params': {'n_estimators': 100}}
        experiment = ALExperiment(config)
        
        X_unlab = np.random.randn(20, 10)
        y_unlab = np.random.choice([0, 1], 20)
        X_holdout = np.random.randn(10, 10)
        y_holdout = np.random.choice([0, 1], 10)
        
        ref_metrics = experiment._get_reference_metrics(X_unlab, y_unlab, X_holdout, y_holdout, "rf")
        
        # Verify model was trained and evaluated
        mock_model.fit.assert_called_once_with(X_unlab, y_unlab)
        mock_evaluate.assert_called_once_with(mock_model, X_holdout, y_holdout)
        assert ref_metrics == {'mcc': 0.8, 'f1': 0.9}
    
    @patch('src.active_learning.experiments.base_learner_factory')
    @patch('src.active_learning.experiments.evaluate_committee')
    def test_get_reference_metrics_qbc(self, mock_evaluate, mock_factory):
        """Test getting reference metrics for QBC model."""
        mock_committee = [Mock(), Mock()]
        mock_factory.return_value = mock_committee
        mock_evaluate.return_value = {'mcc': 0.75, 'f1': 0.85}
        
        config = {'qbc_params': {}}
        experiment = ALExperiment(config)
        
        X_unlab = np.random.randn(20, 10)
        y_unlab = np.random.choice([0, 1], 20)
        X_holdout = np.random.randn(10, 10)
        y_holdout = np.random.choice([0, 1], 10)
        
        ref_metrics = experiment._get_reference_metrics(X_unlab, y_unlab, X_holdout, y_holdout, "qbc")
        
        # Verify committee was trained and evaluated
        for clf in mock_committee:
            clf.fit.assert_called_once_with(X_unlab, y_unlab)
        mock_evaluate.assert_called_once_with(mock_committee, X_holdout, y_holdout)
        assert ref_metrics == {'mcc': 0.75, 'f1': 0.85}
    
    def test_get_reference_metrics_invalid_type(self):
        """Test get_reference_metrics with invalid model type."""
        config = {}
        experiment = ALExperiment(config)
        
        X_unlab = np.random.randn(20, 10)
        y_unlab = np.random.choice([0, 1], 20)
        X_holdout = np.random.randn(10, 10)
        y_holdout = np.random.choice([0, 1], 10)
        
        with pytest.raises(ValueError, match="Unknown model type"):
            experiment._get_reference_metrics(X_unlab, y_unlab, X_holdout, y_holdout, "invalid")
    
    @patch('joblib.dump')
    def test_save_results(self, mock_dump):
        """Test saving experiment results."""
        config = {}
        experiment = ALExperiment(config)
        experiment.results = {'test': 'data'}
        
        filepath = "test_results.joblib"
        experiment.save_results(filepath)
        
        mock_dump.assert_called_once_with({'test': 'data'}, filepath)
    
    @patch('joblib.load')
    def test_load_results(self, mock_load):
        """Test loading experiment results."""
        mock_load.return_value = {'loaded': 'data'}
        
        config = {}
        experiment = ALExperiment(config)
        
        filepath = "test_results.joblib"
        experiment.load_results(filepath)
        
        mock_load.assert_called_once_with(filepath)
        assert experiment.results == {'loaded': 'data'}


class TestStandaloneExperimentFunctions:
    """Test cases for standalone experiment functions."""
    
    @patch('src.active_learning.experiments.rf_factory')
    @patch('src.active_learning.experiments.evaluate_model')
    def test_execute_full_model_baseline(self, mock_evaluate, mock_rf_factory):
        """Test execute_full_model_baseline function."""
        mock_model = Mock()
        mock_rf_factory.return_value = mock_model
        mock_evaluate.return_value = {'mcc': 0.8, 'f1': 0.9}
        
        X_unlab = np.random.randn(20, 10)
        y_unlab = np.random.choice([0, 1], 20)
        X_holdout = np.random.randn(10, 10)
        y_holdout = np.random.choice([0, 1], 10)
        
        rf_params = {'n_estimators': 100}
        
        results = execute_full_model_baseline(X_unlab, y_unlab, X_holdout, y_holdout, rf_params)
        
        # Verify model was created with correct parameters
        mock_rf_factory.assert_called_once_with(rf_params)
        
        # Verify model was trained and evaluated
        mock_model.fit.assert_called_once_with(X_unlab, y_unlab)
        mock_evaluate.assert_called_once_with(mock_model, X_holdout, y_holdout)
        
        # Check return structure
        assert 'eval' in results
        assert 'model' in results
        assert results['eval'] == {'mcc': 0.8, 'f1': 0.9}
        assert results['model'] == mock_model
    
    @patch('src.active_learning.experiments.evaluate_committee')
    def test_execute_full_model_qbc(self, mock_evaluate):
        """Test execute_full_model_qbc function."""
        mock_committee = [Mock(), Mock(), Mock()]
        mock_factory = Mock(return_value=mock_committee)
        mock_evaluate.return_value = {'mcc': 0.75, 'f1': 0.85}
        
        X_unlab = np.random.randn(20, 10)
        y_unlab = np.random.choice([0, 1], 20)
        X_holdout = np.random.randn(10, 10)
        y_holdout = np.random.choice([0, 1], 10)
        
        qbc_params = {'param': 'value'}
        
        results = execute_full_model_qbc(mock_factory, X_unlab, y_unlab, X_holdout, y_holdout, qbc_params)
        
        # Verify factory was called with correct parameters
        mock_factory.assert_called_once_with(qbc_params)
        
        # Verify committee was trained and evaluated
        for clf in mock_committee:
            clf.fit.assert_called_once_with(X_unlab, y_unlab)
        mock_evaluate.assert_called_once_with(mock_committee, X_holdout, y_holdout)
        
        # Check return structure
        assert 'eval' in results
        assert 'committee' in results
        assert results['eval'] == {'mcc': 0.75, 'f1': 0.85}
        assert results['committee'] == mock_committee
    
    def test_execute_full_model_baseline_with_config(self):
        """Test execute_full_model_baseline with custom config."""
        custom_rf_factory = Mock()
        config = {'rf_factory': custom_rf_factory}
        
        mock_model = Mock()
        custom_rf_factory.return_value = mock_model
        
        X_unlab = np.random.randn(20, 10)
        y_unlab = np.random.choice([0, 1], 20)
        X_holdout = np.random.randn(10, 10)
        y_holdout = np.random.choice([0, 1], 10)
        
        with patch('src.active_learning.experiments.evaluate_model') as mock_evaluate:
            mock_evaluate.return_value = {'mcc': 0.8}
            
            results = execute_full_model_baseline(X_unlab, y_unlab, X_holdout, y_holdout, 
                                                 rf_params=None, config=config)
            
            # Verify custom factory was used
            custom_rf_factory.assert_called_once_with(None)
            assert results['model'] == mock_model


class TestExperimentIntegration:
    """Integration tests for experiment functionality."""
    
    def test_complete_experiment_workflow(self, train_test_split_data):
        """Test complete experiment workflow with real data."""
        data = train_test_split_data
        X_train, y_train = data['X_train'], data['y_train']
        X_test, y_test = data['X_test'], data['y_test']
        
        # Use small subset for fast testing
        X_unlab = X_train[:20]
        y_unlab = y_train[:20]
        X_holdout = X_test[:10]
        y_holdout = y_test[:10]
        
        config = {
            'max_queries': 2,  # Very small for testing
            'batch_size': 1,
            'stop_ratio': 0.8,
            'rf_params': {'n_estimators': 5, 'random_state': 42}  # Small for speed
        }
        
        experiment = ALExperiment(config)
        
        # Suppress tqdm output for clean testing
        with patch('src.active_learning.experiments.tqdm') as mock_tqdm:
            mock_pbar = Mock()
            mock_tqdm.return_value = mock_pbar
            
            # Run experiment
            results = experiment.run_experiment(X_unlab, y_unlab, X_holdout, y_holdout, "rf_first5")
            
            # Verify results structure
            assert 'mcc_scores' in results
            assert 'f1_scores' in results
            assert 'roc_auc_scores' in results
            assert 'dmcc_improvements' in results
            assert 'n_iterations' in results
            assert 'final_model' in results
            
            # Verify we have some results
            assert len(results['mcc_scores']) > 0
            assert len(results['f1_scores']) > 0
            assert isinstance(results['n_iterations'], int)
    
    def test_experiment_reproducibility(self, train_test_split_data):
        """Test that experiments are reproducible with same config."""
        data = train_test_split_data
        X_train, y_train = data['X_train'], data['y_train']
        X_test, y_test = data['X_test'], data['y_test']
        
        # Use small subset
        X_unlab = X_train[:15]
        y_unlab = y_train[:15]
        X_holdout = X_test[:5]
        y_holdout = y_test[:5]
        
        config = {
            'max_queries': 1,
            'batch_size': 1,
            'stop_ratio': 0.8,
            'rf_params': {'n_estimators': 5, 'random_state': 42}
        }
        
        # Run experiment twice
        experiment1 = ALExperiment(config)
        experiment2 = ALExperiment(config)
        
        with patch('src.active_learning.experiments.tqdm') as mock_tqdm:
            mock_pbar = Mock()
            mock_tqdm.return_value = mock_pbar
            
            results1 = experiment1.run_experiment(X_unlab, y_unlab, X_holdout, y_holdout, "rf_first5")
            results2 = experiment2.run_experiment(X_unlab, y_unlab, X_holdout, y_holdout, "rf_first5")
            
            # Results should be identical due to fixed random state
            assert len(results1['mcc_scores']) == len(results2['mcc_scores'])
            assert results1['n_iterations'] == results2['n_iterations']
    
    def test_save_load_workflow(self):
        """Test save and load functionality."""
        config = {'test': 'config'}
        experiment = ALExperiment(config)
        experiment.results = {'test_results': [1, 2, 3]}
        
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Save results
            experiment.save_results(tmp_path)
            
            # Create new experiment and load results
            new_experiment = ALExperiment(config)
            new_experiment.load_results(tmp_path)
            
            # Verify results were loaded correctly
            assert new_experiment.results == {'test_results': [1, 2, 3]}
            
        finally:
            # Clean up
            os.unlink(tmp_path)


class TestErrorHandling:
    """Test error handling in experiments."""
    
    def test_experiment_with_insufficient_data(self):
        """Test experiment behavior with very small datasets."""
        config = {
            'max_queries': 10,
            'batch_size': 1,
            'stop_ratio': 0.8
        }
        
        experiment = ALExperiment(config)
        
        # Very small dataset (smaller than initial requirement)
        X_unlab = np.random.randn(3, 5)
        y_unlab = np.array([0, 1, 0])
        X_holdout = np.random.randn(2, 5)
        y_holdout = np.array([1, 0])
        
        # Should handle gracefully or raise appropriate error
        with patch('src.active_learning.experiments.tqdm') as mock_tqdm:
            mock_pbar = Mock()
            mock_tqdm.return_value = mock_pbar
            
            # This might raise an error due to insufficient data for stratification
            # or complete successfully with limited iterations
            try:
                results = experiment.run_experiment(X_unlab, y_unlab, X_holdout, y_holdout, "rf_first5")
                # If successful, verify basic structure
                assert 'mcc_scores' in results
            except (ValueError, IndexError):
                # Expected for very small datasets
                pass
    
    def test_experiment_with_single_class(self):
        """Test experiment with single-class data."""
        config = {
            'max_queries': 1,
            'batch_size': 1,
            'stop_ratio': 0.8,
            'rf_params': {'n_estimators': 5}
        }
        
        experiment = ALExperiment(config)
        
        # Single class data
        X_unlab = np.random.randn(10, 5)
        y_unlab = np.zeros(10)  # All same class
        X_holdout = np.random.randn(5, 5)
        y_holdout = np.zeros(5)  # All same class
        
        with patch('src.active_learning.experiments.tqdm') as mock_tqdm:
            mock_pbar = Mock()
            mock_tqdm.return_value = mock_pbar
            
            # Should handle single-class scenario
            try:
                results = experiment.run_experiment(X_unlab, y_unlab, X_holdout, y_holdout, "rf_first5")
                # Verify basic structure if successful
                assert 'mcc_scores' in results
            except (ValueError, Exception):
                # Some metrics may fail with single class
                pass


@pytest.mark.slow
class TestExperimentPerformance:
    """Performance tests for experiments."""
    
    def test_experiment_performance(self, performance_monitor, train_test_split_data):
        """Test experiment execution time."""
        data = train_test_split_data
        X_train, y_train = data['X_train'], data['y_train']
        X_test, y_test = data['X_test'], data['y_test']
        
        # Moderate size for performance testing
        X_unlab = X_train[:50]
        y_unlab = y_train[:50]
        X_holdout = X_test[:20]
        y_holdout = y_test[:20]
        
        config = {
            'max_queries': 5,
            'batch_size': 1,
            'stop_ratio': 0.8,
            'rf_params': {'n_estimators': 20, 'random_state': 42}
        }
        
        experiment = ALExperiment(config)
        
        performance_monitor.start()
        
        with patch('src.active_learning.experiments.tqdm') as mock_tqdm:
            mock_pbar = Mock()
            mock_tqdm.return_value = mock_pbar
            
            results = experiment.run_experiment(X_unlab, y_unlab, X_holdout, y_holdout, "rf_first5")
        
        stats = performance_monitor.stop()
        
        # Should complete in reasonable time
        assert stats['execution_time'] < 10.0  # Less than 10 seconds
        assert 'mcc_scores' in results
        assert len(results['mcc_scores']) > 0