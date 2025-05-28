"""
Unit tests for active learning learners and model factories.

Tests the src.active_learning.learners module including ActiveLearner base class,
RandomForestAL, QueryByCommitteeAL, and factory functions.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
import sys
import os

# Add src to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.active_learning.learners import (
    ActiveLearner,
    RandomForestAL,
    QueryByCommitteeAL,
    rf_factory,
    base_learner_factory,
    qbc_predict
)


class TestActiveLearner:
    """Test cases for the abstract ActiveLearner base class."""
    
    def test_cannot_instantiate_abstract_class(self):
        """Test that ActiveLearner cannot be instantiated directly."""
        with pytest.raises(TypeError):
            ActiveLearner()
    
    def test_concrete_implementation_must_implement_methods(self):
        """Test that concrete implementations must implement all abstract methods."""
        class IncompleteActiveLearner(ActiveLearner):
            def fit(self, X, y):
                pass
            # Missing predict and predict_proba methods
        
        with pytest.raises(TypeError):
            IncompleteActiveLearner()
    
    def test_complete_implementation_works(self):
        """Test that complete implementations can be instantiated."""
        class CompleteActiveLearner(ActiveLearner):
            def fit(self, X, y):
                return self
            
            def predict(self, X):
                return np.zeros(len(X))
            
            def predict_proba(self, X):
                return np.ones((len(X), 2)) / 2
        
        learner = CompleteActiveLearner()
        X = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])
        
        learner.fit(X, y)
        predictions = learner.predict(X)
        probas = learner.predict_proba(X)
        
        assert len(predictions) == len(X)
        assert probas.shape == (len(X), 2)


class TestRandomForestAL:
    """Test cases for RandomForestAL class."""
    
    def test_initialization_default_params(self):
        """Test initialization with default parameters."""
        rf_al = RandomForestAL()
        
        assert isinstance(rf_al, ActiveLearner)
        assert rf_al.model.n_estimators == 100
        assert rf_al.model.random_state == 42
    
    def test_initialization_custom_params(self):
        """Test initialization with custom parameters."""
        custom_params = {'n_estimators': 50, 'max_depth': 10, 'random_state': 123}
        rf_al = RandomForestAL(**custom_params)
        
        assert rf_al.model.n_estimators == 50
        assert rf_al.model.max_depth == 10
        assert rf_al.model.random_state == 123
    
    def test_initialization_override_defaults(self):
        """Test that custom parameters override defaults."""
        rf_al = RandomForestAL(n_estimators=200, random_state=999)
        
        assert rf_al.model.n_estimators == 200
        assert rf_al.model.random_state == 999
    
    def test_fit_method(self, train_test_split_data):
        """Test the fit method."""
        data = train_test_split_data
        X_train, y_train = data['X_train'], data['y_train']
        
        rf_al = RandomForestAL(n_estimators=10)  # Small for speed
        result = rf_al.fit(X_train, y_train)
        
        # Should return self
        assert result is rf_al
        # Model should be trained
        assert hasattr(rf_al.model, 'estimators_')
    
    def test_predict_method(self, train_test_split_data):
        """Test the predict method."""
        data = train_test_split_data
        X_train, y_train = data['X_train'], data['y_train']
        X_test = data['X_test']
        
        rf_al = RandomForestAL(n_estimators=10)
        rf_al.fit(X_train, y_train)
        
        predictions = rf_al.predict(X_test)
        
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(X_test)
        assert set(predictions).issubset({0, 1})  # Binary classification
    
    def test_predict_proba_method(self, train_test_split_data):
        """Test the predict_proba method."""
        data = train_test_split_data
        X_train, y_train = data['X_train'], data['y_train']
        X_test = data['X_test']
        
        rf_al = RandomForestAL(n_estimators=10)
        rf_al.fit(X_train, y_train)
        
        probas = rf_al.predict_proba(X_test)
        
        assert isinstance(probas, np.ndarray)
        assert probas.shape == (len(X_test), 2)  # Binary classification
        assert np.allclose(probas.sum(axis=1), 1.0)  # Probabilities sum to 1
        assert np.all(probas >= 0) and np.all(probas <= 1)  # Valid probabilities
    
    def test_predict_before_fit_raises_error(self):
        """Test that predict raises error when called before fit."""
        rf_al = RandomForestAL()
        X = np.array([[1, 2], [3, 4]])
        
        with pytest.raises(Exception):  # sklearn raises NotFittedError
            rf_al.predict(X)


class TestQueryByCommitteeAL:
    """Test cases for QueryByCommitteeAL class."""
    
    def test_initialization_default_committee(self):
        """Test initialization with default committee."""
        qbc_al = QueryByCommitteeAL()
        
        assert isinstance(qbc_al, ActiveLearner)
        assert len(qbc_al.committee) == 5
        # Check that we have the expected learner types
        learner_types = [type(clf).__name__ for clf in qbc_al.committee]
        expected_types = ['RandomForestClassifier', 'ExtraTreesClassifier', 
                         'GradientBoostingClassifier', 'LogisticRegression', 
                         'KNeighborsClassifier']
        assert learner_types == expected_types
    
    def test_initialization_custom_committee(self):
        """Test initialization with custom committee."""
        from sklearn.tree import DecisionTreeClassifier
        custom_committee = [DecisionTreeClassifier(), DecisionTreeClassifier()]
        
        qbc_al = QueryByCommitteeAL(committee=custom_committee)
        
        assert len(qbc_al.committee) == 2
        assert all(isinstance(clf, DecisionTreeClassifier) for clf in qbc_al.committee)
    
    def test_initialization_with_params(self):
        """Test initialization with parameters for default committee."""
        params = {
            'random_state': 123,
            'run_iter': 5,
            'rf': {'n_estimators': 50},
            'lr': {'max_iter': 500}
        }
        
        qbc_al = QueryByCommitteeAL(**params)
        
        # Check that random states are set correctly
        rf_clf = qbc_al.committee[0]  # RandomForest is first
        assert rf_clf.random_state == 123 + 5  # base + run_iter
        assert rf_clf.n_estimators == 50
        
        lr_clf = qbc_al.committee[3]  # LogisticRegression is fourth
        assert lr_clf.random_state == 123 + 5
        assert lr_clf.max_iter == 500
    
    def test_fit_method(self, train_test_split_data):
        """Test the fit method."""
        data = train_test_split_data
        X_train, y_train = data['X_train'], data['y_train']
        
        # Use smaller committee for speed
        from sklearn.tree import DecisionTreeClassifier
        small_committee = [DecisionTreeClassifier() for _ in range(2)]
        qbc_al = QueryByCommitteeAL(committee=small_committee)
        
        result = qbc_al.fit(X_train, y_train)
        
        # Should return self
        assert result is qbc_al
        # All committee members should be fitted
        for clf in qbc_al.committee:
            assert hasattr(clf, 'tree_')  # DecisionTree specific
    
    def test_predict_method(self, train_test_split_data):
        """Test the predict method with majority voting."""
        data = train_test_split_data
        X_train, y_train = data['X_train'], data['y_train']
        X_test = data['X_test']
        
        # Create small committee for controlled testing
        from sklearn.tree import DecisionTreeClassifier
        committee = [DecisionTreeClassifier(random_state=i) for i in range(3)]
        qbc_al = QueryByCommitteeAL(committee=committee)
        qbc_al.fit(X_train, y_train)
        
        predictions = qbc_al.predict(X_test)
        
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(X_test)
        assert set(predictions).issubset({0, 1})  # Binary classification
    
    def test_predict_proba_method(self, train_test_split_data):
        """Test the predict_proba method with committee averaging."""
        data = train_test_split_data
        X_train, y_train = data['X_train'], data['y_train']
        X_test = data['X_test']
        
        # Use small committee for speed
        from sklearn.tree import DecisionTreeClassifier
        committee = [DecisionTreeClassifier(random_state=i) for i in range(2)]
        qbc_al = QueryByCommitteeAL(committee=committee)
        qbc_al.fit(X_train, y_train)
        
        probas = qbc_al.predict_proba(X_test)
        
        assert isinstance(probas, np.ndarray)
        assert probas.shape == (len(X_test), 2)  # Binary classification
        assert np.allclose(probas.sum(axis=1), 1.0)  # Probabilities sum to 1
        assert np.all(probas >= 0) and np.all(probas <= 1)  # Valid probabilities
    
    def test_predict_proba_no_proba_support_raises_error(self, train_test_split_data):
        """Test predict_proba raises error when committee doesn't support it."""
        data = train_test_split_data
        X_train, y_train = data['X_train'], data['y_train']
        X_test = data['X_test']
        
        # Create committee with classifiers that don't support predict_proba
        class NoProbaBinaryClassifier:
            def fit(self, X, y):
                self.classes_ = [0, 1]
                return self
            
            def predict(self, X):
                return np.zeros(len(X))
        
        committee = [NoProbaBinaryClassifier() for _ in range(2)]
        qbc_al = QueryByCommitteeAL(committee=committee)
        qbc_al.fit(X_train, y_train)
        
        with pytest.raises(ValueError, match="No committee members support predict_proba"):
            qbc_al.predict_proba(X_test)
    
    def test_create_default_committee_parameters(self):
        """Test the _create_default_committee method with various parameters."""
        params = {
            'random_state': 100,
            'run_iter': 2,
            'rf': {'n_estimators': 30},
            'extratrees': {'n_estimators': 40},
            'gb': {'n_estimators': 20},
            'lr': {'max_iter': 800},
            'knn': {'n_neighbors': 3}
        }
        
        qbc_al = QueryByCommitteeAL()
        committee = qbc_al._create_default_committee(**params)
        
        assert len(committee) == 5
        
        # Check RF parameters
        rf = committee[0]
        assert rf.n_estimators == 30
        assert rf.random_state == 102  # 100 + 2
        
        # Check ExtraTrees parameters
        et = committee[1]
        assert et.n_estimators == 40
        assert et.random_state == 102
        
        # Check GradientBoosting parameters
        gb = committee[2]
        assert gb.n_estimators == 20
        assert gb.random_state == 102
        
        # Check LogisticRegression parameters
        lr = committee[3]
        assert lr.max_iter == 800
        assert lr.random_state == 102
        
        # Check KNN parameters
        knn = committee[4]
        assert knn.n_neighbors == 3


class TestFactoryFunctions:
    """Test cases for factory functions."""
    
    def test_rf_factory_default_params(self):
        """Test rf_factory with default parameters."""
        rf = rf_factory()
        
        assert rf.n_estimators == 100
        assert rf.random_state == 42
    
    def test_rf_factory_custom_params(self):
        """Test rf_factory with custom parameters."""
        params = {'n_estimators': 200, 'max_depth': 5, 'random_state': 999}
        rf = rf_factory(params)
        
        assert rf.n_estimators == 200
        assert rf.max_depth == 5
        assert rf.random_state == 999
    
    def test_rf_factory_none_params(self):
        """Test rf_factory with None parameters."""
        rf = rf_factory(None)
        
        assert rf.n_estimators == 100
        assert rf.random_state == 42
    
    def test_base_learner_factory_default(self):
        """Test base_learner_factory with default parameters."""
        learners = base_learner_factory()
        
        assert len(learners) == 5
        
        # Check types
        types = [type(clf).__name__ for clf in learners]
        expected_types = ['RandomForestClassifier', 'ExtraTreesClassifier',
                         'GradientBoostingClassifier', 'LogisticRegression',
                         'KNeighborsClassifier']
        assert types == expected_types
        
        # Check default random states
        assert learners[0].random_state == 42  # RF
        assert learners[1].random_state == 42  # ET
        assert learners[2].random_state == 42  # GB
        assert learners[3].random_state == 42  # LR
    
    def test_base_learner_factory_with_run_iter(self):
        """Test base_learner_factory with run_iter parameter."""
        learners = base_learner_factory(run_iter=5)
        
        # Check that random states are adjusted by run_iter
        assert learners[0].random_state == 47  # 42 + 5
        assert learners[1].random_state == 47
        assert learners[2].random_state == 47
        assert learners[3].random_state == 47
    
    def test_base_learner_factory_with_custom_params(self):
        """Test base_learner_factory with custom parameters."""
        qbc_params = {
            'rf': {'n_estimators': 150},
            'extratrees': {'n_estimators': 120},
            'gb': {'n_estimators': 80},
            'lr': {'max_iter': 2000},
            'knn': {'n_neighbors': 7}
        }
        
        learners = base_learner_factory(qbc_params, run_iter=3)
        
        # Check RF
        assert learners[0].n_estimators == 150
        assert learners[0].random_state == 45  # 42 + 3
        
        # Check ExtraTrees
        assert learners[1].n_estimators == 120
        assert learners[1].random_state == 45
        
        # Check GradientBoosting
        assert learners[2].n_estimators == 80
        assert learners[2].random_state == 45
        
        # Check LogisticRegression
        assert learners[3].max_iter == 2000
        assert learners[3].random_state == 45
        
        # Check KNN
        assert learners[4].n_neighbors == 7


class TestQBCPredict:
    """Test cases for qbc_predict function."""
    
    def test_qbc_predict_majority_vote(self, train_test_split_data):
        """Test majority voting in qbc_predict."""
        data = train_test_split_data
        X_train, y_train = data['X_train'], data['y_train']
        X_test = data['X_test'][:5]  # Small test set
        
        # Create and train committee
        from sklearn.tree import DecisionTreeClassifier
        committee = [DecisionTreeClassifier(random_state=i) for i in range(3)]
        for clf in committee:
            clf.fit(X_train, y_train)
        
        predictions = qbc_predict(committee, X_test)
        
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(X_test)
        assert set(predictions).issubset({0, 1})  # Binary classification
    
    def test_qbc_predict_deterministic_committee(self):
        """Test qbc_predict with deterministic committee results."""
        X_test = np.array([[1, 2], [3, 4], [5, 6]])
        
        # Create mock committee with known predictions
        mock_clf1 = Mock()
        mock_clf1.predict.return_value = np.array([0, 1, 0])
        
        mock_clf2 = Mock()
        mock_clf2.predict.return_value = np.array([0, 0, 1])
        
        mock_clf3 = Mock()
        mock_clf3.predict.return_value = np.array([1, 1, 0])
        
        committee = [mock_clf1, mock_clf2, mock_clf3]
        
        # Expected majority votes:
        # Sample 0: [0, 0, 1] -> 0 (majority)
        # Sample 1: [1, 0, 1] -> 1 (majority)  
        # Sample 2: [0, 1, 0] -> 0 (majority)
        expected = np.array([0, 1, 0])
        
        predictions = qbc_predict(committee, X_test)
        
        np.testing.assert_array_equal(predictions, expected)
        
        # Verify all classifiers were called
        for clf in committee:
            clf.predict.assert_called_once_with(X_test)
    
    @patch('src.active_learning.learners.Parallel')
    def test_qbc_predict_uses_parallel(self, mock_parallel, train_test_split_data):
        """Test that qbc_predict uses parallel processing."""
        data = train_test_split_data
        X_train, y_train = data['X_train'], data['y_train']
        X_test = data['X_test'][:3]
        
        # Setup mock parallel to return known results
        mock_parallel.return_value = [
            np.array([0, 1, 0]),
            np.array([1, 1, 0]),
            np.array([0, 0, 1])
        ]
        
        # Create committee
        from sklearn.tree import DecisionTreeClassifier
        committee = [DecisionTreeClassifier(random_state=i) for i in range(3)]
        for clf in committee:
            clf.fit(X_train, y_train)
        
        predictions = qbc_predict(committee, X_test)
        
        # Verify Parallel was called
        mock_parallel.assert_called_once_with(n_jobs=3)  # 3 committee members
        assert isinstance(predictions, np.ndarray)
    
    def test_qbc_predict_single_classifier(self, train_test_split_data):
        """Test qbc_predict with single classifier (edge case)."""
        data = train_test_split_data
        X_train, y_train = data['X_train'], data['y_train']
        X_test = data['X_test'][:3]
        
        # Single classifier committee
        from sklearn.tree import DecisionTreeClassifier
        clf = DecisionTreeClassifier(random_state=42)
        clf.fit(X_train, y_train)
        committee = [clf]
        
        predictions = qbc_predict(committee, X_test)
        
        # Should be the same as single classifier prediction
        expected = clf.predict(X_test)
        np.testing.assert_array_equal(predictions, expected)
    
    def test_qbc_predict_empty_committee(self):
        """Test qbc_predict with empty committee."""
        X_test = np.array([[1, 2], [3, 4]])
        committee = []
        
        # Should handle empty committee gracefully
        predictions = qbc_predict(committee, X_test)
        
        # Empty committee results in empty predictions array
        assert len(predictions) == 0


class TestLearnersIntegration:
    """Integration tests with realistic scenarios."""
    
    def test_rf_al_vs_qbc_al_comparison(self, train_test_split_data):
        """Test comparison between RandomForestAL and QueryByCommitteeAL."""
        data = train_test_split_data
        X_train, y_train = data['X_train'], data['y_train']
        X_test = data['X_test']
        
        # Train both learners
        rf_al = RandomForestAL(n_estimators=20, random_state=42)
        qbc_al = QueryByCommitteeAL(random_state=42)
        
        rf_al.fit(X_train, y_train)
        qbc_al.fit(X_train, y_train)
        
        # Get predictions and probabilities
        rf_pred = rf_al.predict(X_test)
        rf_proba = rf_al.predict_proba(X_test)
        
        qbc_pred = qbc_al.predict(X_test)
        qbc_proba = qbc_al.predict_proba(X_test)
        
        # Both should produce valid outputs
        assert len(rf_pred) == len(qbc_pred) == len(X_test)
        assert rf_proba.shape == qbc_proba.shape == (len(X_test), 2)
        
        # Results might be different due to different approaches
        # But both should be reasonable predictions
        assert set(rf_pred).issubset({0, 1})
        assert set(qbc_pred).issubset({0, 1})
    
    def test_learners_with_active_learning_workflow(self, train_test_split_data):
        """Test learners in a simplified active learning workflow."""
        data = train_test_split_data
        X_train, y_train = data['X_train'], data['y_train']
        X_pool = data['X_test']
        
        # Start with small labeled set
        initial_size = 5
        X_labeled = X_train[:initial_size]
        y_labeled = y_train[:initial_size]
        X_unlabeled = X_train[initial_size:initial_size+10]  # Small pool
        
        # Train RF learner
        rf_al = RandomForestAL(n_estimators=10, random_state=42)
        rf_al.fit(X_labeled, y_labeled)
        
        # Get uncertainties for active learning
        probas = rf_al.predict_proba(X_unlabeled)
        uncertainties = 1 - np.max(probas, axis=1)
        
        # Select most uncertain sample
        query_idx = np.argmax(uncertainties)
        
        assert 0 <= query_idx < len(X_unlabeled)
        assert isinstance(uncertainties[query_idx], (float, np.floating))
    
    def test_committee_diversity(self, train_test_split_data):
        """Test that QBC committee members produce diverse predictions."""
        data = train_test_split_data
        X_train, y_train = data['X_train'], data['y_train']
        X_test = data['X_test'][:10]  # Small test set
        
        qbc_al = QueryByCommitteeAL(random_state=42)
        qbc_al.fit(X_train, y_train)
        
        # Get individual predictions from each committee member
        individual_predictions = []
        for clf in qbc_al.committee:
            pred = clf.predict(X_test)
            individual_predictions.append(pred)
        
        individual_predictions = np.array(individual_predictions)
        
        # Check that there's some diversity (not all identical)
        # At least some classifiers should disagree on some samples
        disagreements = []
        for i in range(len(X_test)):
            sample_preds = individual_predictions[:, i]
            disagreements.append(len(np.unique(sample_preds)) > 1)
        
        # At least some samples should have disagreement
        assert any(disagreements), "Committee members show no diversity in predictions"


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_rf_al_with_invalid_params(self):
        """Test RandomForestAL with invalid parameters."""
        # sklearn will handle invalid parameters
        with pytest.raises((ValueError, TypeError)):
            rf_al = RandomForestAL(n_estimators=-1)
            # The error might not occur until fit is called
            X = np.array([[1, 2], [3, 4]])
            y = np.array([0, 1])
            rf_al.fit(X, y)
    
    def test_qbc_al_with_empty_committee(self):
        """Test QueryByCommitteeAL with empty committee."""
        qbc_al = QueryByCommitteeAL(committee=[])
        X = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])
        
        # Should work for fit (no committee members to fit)
        qbc_al.fit(X, y)
        
        # But predict should handle empty committee
        predictions = qbc_al.predict(X)
        assert len(predictions) == 0  # No predictions from empty committee
    
    def test_single_sample_prediction(self, train_test_split_data):
        """Test predictions with single sample."""
        data = train_test_split_data
        X_train, y_train = data['X_train'], data['y_train']
        X_single = data['X_test'][:1]  # Single sample
        
        rf_al = RandomForestAL(n_estimators=10)
        rf_al.fit(X_train, y_train)
        
        pred = rf_al.predict(X_single)
        proba = rf_al.predict_proba(X_single)
        
        assert len(pred) == 1
        assert proba.shape == (1, 2)
        assert pred[0] in {0, 1}
        assert np.isclose(proba[0].sum(), 1.0)


@pytest.mark.slow
class TestLearnersPerformance:
    """Performance tests for learners."""
    
    def test_large_dataset_performance(self, performance_monitor):
        """Test learners performance with larger datasets."""
        from sklearn.datasets import make_classification
        
        # Create moderately large dataset
        X, y = make_classification(
            n_samples=1000, n_features=20, n_informative=10,
            n_redundant=5, random_state=42
        )
        
        rf_al = RandomForestAL(n_estimators=50, random_state=42)
        
        performance_monitor.start()
        rf_al.fit(X, y)
        predictions = rf_al.predict(X[:100])  # Predict on subset
        stats = performance_monitor.stop()
        
        # Should complete in reasonable time
        assert stats['execution_time'] < 10.0  # Less than 10 seconds
        assert len(predictions) == 100
    
    def test_qbc_committee_performance(self, performance_monitor, train_test_split_data):
        """Test QBC performance with default committee."""
        data = train_test_split_data
        X_train, y_train = data['X_train'], data['y_train']
        X_test = data['X_test']
        
        # Reduce committee size for faster testing
        params = {
            'rf': {'n_estimators': 20},
            'extratrees': {'n_estimators': 20},
            'gb': {'n_estimators': 20}
        }
        
        qbc_al = QueryByCommitteeAL(**params)
        
        performance_monitor.start()
        qbc_al.fit(X_train, y_train)
        predictions = qbc_al.predict(X_test)
        stats = performance_monitor.stop()
        
        # Should complete in reasonable time
        assert stats['execution_time'] < 15.0  # Less than 15 seconds
        assert len(predictions) == len(X_test)