"""
Unit tests for evaluation metrics and functions.

Tests the src.evaluation.metrics module including ModelEvaluator class
and metric calculation functions.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
import sys
import os

# Add src to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.evaluation.metrics import (
    evaluate_model,
    evaluate_committee,
    calculate_per_class_accuracy,
    qbc_predict,
    qbc_predict_proba,
    ModelEvaluator
)


class TestEvaluateModel:
    """Test cases for evaluate_model function."""
    
    def test_evaluate_model_basic(self, train_test_split_data):
        """Test basic model evaluation."""
        from sklearn.ensemble import RandomForestClassifier
        
        data = train_test_split_data
        X_train, y_train = data['X_train'], data['y_train']
        X_test, y_test = data['X_test'], data['y_test']
        
        # Train a model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        results = evaluate_model(model, X_test, y_test)
        
        # Check all expected keys are present
        expected_keys = ['mcc', 'f1', 'roc_auc', 'roc_curve', 'accuracy', 'confusion_matrix']
        assert all(key in results for key in expected_keys)
        
        # Check value ranges
        assert -1 <= results['mcc'] <= 1
        assert 0 <= results['f1'] <= 1
        assert 0 <= results['roc_auc'] <= 1
        assert 0 <= results['accuracy'] <= 1
        
        # Check ROC curve format
        fpr, tpr, thresholds = results['roc_curve']
        assert isinstance(fpr, np.ndarray)
        assert isinstance(tpr, np.ndarray)
        assert isinstance(thresholds, np.ndarray)
        
        # Check confusion matrix shape
        cm = results['confusion_matrix']
        assert cm.shape == (2, 2)  # Binary classification
    
    def test_evaluate_model_without_predict_proba(self):
        """Test evaluation with model that doesn't support predict_proba."""
        # Create mock model without predict_proba
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0, 1, 0, 1])
        # No predict_proba method
        
        X_test = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y_test = np.array([0, 1, 0, 1])
        
        results = evaluate_model(mock_model, X_test, y_test)
        
        # Should handle missing predict_proba gracefully
        assert 'mcc' in results
        assert 'f1' in results
        assert np.isnan(results['roc_auc'])
        assert results['roc_curve'] == (None, None, None)
    
    def test_evaluate_model_perfect_predictions(self):
        """Test evaluation with perfect predictions."""
        mock_model = Mock()
        y_test = np.array([0, 1, 0, 1])
        mock_model.predict.return_value = y_test  # Perfect predictions
        mock_model.predict_proba.return_value = np.array([
            [1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]
        ])
        
        X_test = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        
        results = evaluate_model(mock_model, X_test, y_test)
        
        # Perfect predictions should give high scores
        assert results['mcc'] == 1.0
        assert results['f1'] == 1.0
        assert results['roc_auc'] == 1.0
        assert results['accuracy'] == 1.0


class TestEvaluateCommittee:
    """Test cases for evaluate_committee function."""
    
    @patch('src.evaluation.metrics.qbc_predict')
    def test_evaluate_committee_basic(self, mock_qbc_predict, train_test_split_data):
        """Test basic committee evaluation."""
        data = train_test_split_data
        X_test, y_test = data['X_test'], data['y_test']
        
        # Mock committee predictions
        mock_qbc_predict.return_value = np.array([0, 1, 0, 1])
        
        # Create mock committee with predict_proba
        mock_clf1 = Mock()
        mock_clf1.predict_proba.return_value = np.array([
            [0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.4, 0.6]
        ])
        
        mock_clf2 = Mock()
        mock_clf2.predict_proba.return_value = np.array([
            [0.7, 0.3], [0.2, 0.8], [0.8, 0.2], [0.3, 0.7]
        ])
        
        committee = [mock_clf1, mock_clf2]
        
        results = evaluate_committee(committee, X_test[:4], y_test[:4])
        
        # Check all expected keys
        expected_keys = ['mcc', 'f1', 'roc_auc', 'roc_curve', 'accuracy', 'confusion_matrix']
        assert all(key in results for key in expected_keys)
        
        # Verify qbc_predict was called
        mock_qbc_predict.assert_called_once()
    
    def test_evaluate_committee_no_predict_proba(self):
        """Test committee evaluation when members don't support predict_proba."""
        # Create mock committee without predict_proba
        mock_clf1 = Mock()
        mock_clf1.predict.return_value = np.array([0, 1, 0])
        # No predict_proba method
        
        mock_clf2 = Mock()
        mock_clf2.predict.return_value = np.array([1, 1, 0])
        
        committee = [mock_clf1, mock_clf2]
        X_test = np.array([[1, 2], [3, 4], [5, 6]])
        y_test = np.array([0, 1, 0])
        
        with patch('src.evaluation.metrics.qbc_predict') as mock_qbc:
            mock_qbc.return_value = np.array([0, 1, 0])
            
            results = evaluate_committee(committee, X_test, y_test)
            
            # Should handle missing predict_proba
            assert np.isnan(results['roc_auc'])
            assert results['roc_curve'] == (None, None, None)


class TestCalculatePerClassAccuracy:
    """Test cases for calculate_per_class_accuracy function."""
    
    def test_per_class_accuracy_basic(self):
        """Test basic per-class accuracy calculation."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0])
        
        per_class_acc = calculate_per_class_accuracy(y_true, y_pred)
        
        # Class 0: true=[0,0,0], pred=[0,1,0] -> 2/3 correct
        # Class 1: true=[1,1,1], pred=[1,1,0] -> 2/3 correct
        assert len(per_class_acc) == 2
        assert per_class_acc[0] == pytest.approx(2/3, rel=1e-3)
        assert per_class_acc[1] == pytest.approx(2/3, rel=1e-3)
    
    def test_per_class_accuracy_perfect(self):
        """Test per-class accuracy with perfect predictions."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        
        per_class_acc = calculate_per_class_accuracy(y_true, y_pred)
        
        assert per_class_acc[0] == 1.0
        assert per_class_acc[1] == 1.0
    
    def test_per_class_accuracy_empty_class(self):
        """Test handling of empty classes."""
        y_true = np.array([0, 0, 0])  # Only class 0
        y_pred = np.array([0, 1, 0])
        
        per_class_acc = calculate_per_class_accuracy(y_true, y_pred)
        
        # Only class 0 should be present
        assert len(per_class_acc) == 1
        assert 0 in per_class_acc
        assert per_class_acc[0] == pytest.approx(2/3, rel=1e-3)


class TestQBCFunctions:
    """Test cases for QBC prediction functions."""
    
    @patch('src.evaluation.metrics.Parallel')
    def test_qbc_predict(self, mock_parallel):
        """Test qbc_predict function."""
        # Mock parallel execution
        mock_parallel.return_value = [
            np.array([0, 1, 0]),
            np.array([1, 1, 0]),
            np.array([0, 0, 1])
        ]
        
        # Create mock committee
        committee = [Mock(), Mock(), Mock()]
        X_test = np.array([[1, 2], [3, 4], [5, 6]])
        
        predictions = qbc_predict(committee, X_test)
        
        # Expected majority votes: [0,1,0], [1,1,0], [0,0,1] -> [0,1,0]
        expected = np.array([0, 1, 0])
        np.testing.assert_array_equal(predictions, expected)
        
        # Verify parallel was called
        mock_parallel.assert_called_once_with(n_jobs=3)
    
    def test_qbc_predict_proba(self):
        """Test qbc_predict_proba function."""
        # Create mock committee with predict_proba
        mock_clf1 = Mock()
        mock_clf1.predict_proba.return_value = np.array([
            [0.8, 0.2], [0.3, 0.7]
        ])
        
        mock_clf2 = Mock()
        mock_clf2.predict_proba.return_value = np.array([
            [0.6, 0.4], [0.2, 0.8]
        ])
        
        committee = [mock_clf1, mock_clf2]
        X_test = np.array([[1, 2], [3, 4]])
        
        probas = qbc_predict_proba(committee, X_test)
        
        # Expected: mean of [0.2, 0.7] and [0.4, 0.8] = [0.3, 0.75]
        expected = np.array([0.3, 0.75])
        np.testing.assert_array_almost_equal(probas, expected)


class TestModelEvaluator:
    """Test cases for ModelEvaluator class."""
    
    def test_initialization(self):
        """Test ModelEvaluator initialization."""
        evaluator = ModelEvaluator()
        assert evaluator.results == {}
    
    def test_evaluate_single_model(self, train_test_split_data):
        """Test single model evaluation."""
        from sklearn.ensemble import RandomForestClassifier
        
        data = train_test_split_data
        X_train, y_train = data['X_train'], data['y_train']
        X_test, y_test = data['X_test'], data['y_test']
        
        # Train model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        evaluator = ModelEvaluator()
        results = evaluator.evaluate_single_model(model, X_test, y_test, "test_rf")
        
        # Check results structure
        expected_keys = ['mcc', 'f1', 'roc_auc', 'accuracy', 'per_class_accuracy', 'classification_report']
        assert all(key in results for key in expected_keys)
        
        # Check results are stored
        assert "test_rf" in evaluator.results
        assert evaluator.results["test_rf"] == results
        
        # Check per-class accuracy format
        assert isinstance(results['per_class_accuracy'], dict)
        assert isinstance(results['classification_report'], dict)
    
    def test_evaluate_committee(self, train_test_split_data):
        """Test committee evaluation."""
        from sklearn.tree import DecisionTreeClassifier
        
        data = train_test_split_data
        X_train, y_train = data['X_train'], data['y_train']
        X_test, y_test = data['X_test'], data['y_test']
        
        # Create and train committee
        committee = [DecisionTreeClassifier(random_state=i) for i in range(2)]
        for clf in committee:
            clf.fit(X_train, y_train)
        
        evaluator = ModelEvaluator()
        results = evaluator.evaluate_committee(committee, X_test, y_test, "test_committee")
        
        # Check results structure
        expected_keys = ['mcc', 'f1', 'roc_auc', 'member_results', 'member_mcc_std', 'member_f1_std']
        assert all(key in results for key in expected_keys)
        
        # Check member results
        assert len(results['member_results']) == 2
        assert all('mcc' in member for member in results['member_results'])
        
        # Check standard deviations
        assert isinstance(results['member_mcc_std'], (float, np.floating))
        assert isinstance(results['member_f1_std'], (float, np.floating))
        
        # Check results are stored
        assert "test_committee" in evaluator.results
    
    def test_compare_models(self, train_test_split_data):
        """Test model comparison."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.tree import DecisionTreeClassifier
        
        data = train_test_split_data
        X_train, y_train = data['X_train'], data['y_train']
        X_test, y_test = data['X_test'], data['y_test']
        
        # Train models
        rf = RandomForestClassifier(n_estimators=10, random_state=42)
        rf.fit(X_train, y_train)
        
        dt = DecisionTreeClassifier(random_state=42)
        dt.fit(X_train, y_train)
        
        # Evaluate with ModelEvaluator
        evaluator = ModelEvaluator()
        evaluator.evaluate_single_model(rf, X_test, y_test, "random_forest")
        evaluator.evaluate_single_model(dt, X_test, y_test, "decision_tree")
        
        # Compare models
        comparison = evaluator.compare_models(['mcc', 'f1'])
        
        assert 'mcc' in comparison
        assert 'f1' in comparison
        assert 'random_forest' in comparison['mcc']
        assert 'decision_tree' in comparison['mcc']
        assert 'random_forest' in comparison['f1']
        assert 'decision_tree' in comparison['f1']
    
    def test_get_best_model(self, train_test_split_data):
        """Test finding best model."""
        evaluator = ModelEvaluator()
        
        # Add mock results
        evaluator.results = {
            'model_a': {'mcc': 0.8, 'f1': 0.9},
            'model_b': {'mcc': 0.9, 'f1': 0.8},
            'model_c': {'mcc': 0.7, 'f1': 0.95}
        }
        
        # Test best by MCC
        best_name, best_score = evaluator.get_best_model('mcc')
        assert best_name == 'model_b'
        assert best_score == 0.9
        
        # Test best by F1
        best_name, best_score = evaluator.get_best_model('f1')
        assert best_name == 'model_c'
        assert best_score == 0.95
    
    def test_get_best_model_empty_results(self):
        """Test get_best_model with no results."""
        evaluator = ModelEvaluator()
        
        best_name, best_score = evaluator.get_best_model('mcc')
        
        assert best_name is None
        assert best_score == -np.inf


class TestEvaluationIntegration:
    """Integration tests for evaluation functionality."""
    
    def test_full_evaluation_workflow(self, train_test_split_data):
        """Test complete evaluation workflow."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.tree import DecisionTreeClassifier
        
        data = train_test_split_data
        X_train, y_train = data['X_train'], data['y_train']
        X_test, y_test = data['X_test'], data['y_test']
        
        # Train models
        rf = RandomForestClassifier(n_estimators=10, random_state=42)
        rf.fit(X_train, y_train)
        
        committee = [DecisionTreeClassifier(random_state=i) for i in range(2)]
        for clf in committee:
            clf.fit(X_train, y_train)
        
        # Comprehensive evaluation
        evaluator = ModelEvaluator()
        
        # Evaluate single model
        rf_results = evaluator.evaluate_single_model(rf, X_test, y_test, "random_forest")
        
        # Evaluate committee
        committee_results = evaluator.evaluate_committee(committee, X_test, y_test, "dt_committee")
        
        # Compare
        comparison = evaluator.compare_models()
        best_model, best_score = evaluator.get_best_model()
        
        # Verify workflow completed successfully
        assert len(evaluator.results) == 2
        assert "random_forest" in evaluator.results
        assert "dt_committee" in evaluator.results
        assert len(comparison) == 3  # mcc, f1, roc_auc
        assert best_model in ["random_forest", "dt_committee"]
        assert isinstance(best_score, (float, np.floating))


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_evaluate_model_with_single_class(self):
        """Test evaluation when predictions contain only one class."""
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0, 0, 0, 0])  # All same class
        mock_model.predict_proba.return_value = np.array([
            [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]
        ])
        
        X_test = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y_test = np.array([0, 1, 0, 1])  # True labels have both classes
        
        # Should handle gracefully (might produce warnings)
        results = evaluate_model(mock_model, X_test, y_test)
        
        # Basic structure should still be there
        assert 'mcc' in results
        assert 'f1' in results
    
    def test_empty_committee_evaluation(self):
        """Test evaluation with empty committee."""
        committee = []
        X_test = np.array([[1, 2], [3, 4]])
        y_test = np.array([0, 1])
        
        with patch('src.evaluation.metrics.qbc_predict') as mock_qbc:
            mock_qbc.return_value = np.array([])
            
            # Should handle empty committee
            results = evaluate_committee(committee, X_test, y_test)
            assert 'mcc' in results