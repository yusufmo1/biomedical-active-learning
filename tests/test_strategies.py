"""
Unit tests for active learning sampling strategies.

Tests the src.active_learning.strategies module including all sampling
strategies (least confidence, QBC vote entropy, random) and the abstract base class.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
import sys
import os

# Add src to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.active_learning.strategies import (
    SamplingStrategy,
    LeastConfidenceSampling,
    QBCVoteEntropySampling,
    RandomSampling,
    least_confidence_sampling,
    qbc_vote_entropy_sampling,
    random_sample
)


class TestSamplingStrategy:
    """Test cases for the abstract SamplingStrategy base class."""
    
    def test_cannot_instantiate_abstract_class(self):
        """Test that SamplingStrategy cannot be instantiated directly."""
        with pytest.raises(TypeError):
            SamplingStrategy()
    
    def test_concrete_implementation_must_implement_select_sample(self):
        """Test that concrete implementations must implement select_sample method."""
        class IncompleteStrategy(SamplingStrategy):
            pass
        
        with pytest.raises(TypeError):
            IncompleteStrategy()
    
    def test_complete_implementation_works(self):
        """Test that complete implementations can be instantiated."""
        class CompleteStrategy(SamplingStrategy):
            def select_sample(self, learner, X_pool):
                return 0
        
        strategy = CompleteStrategy()
        assert strategy.select_sample(None, np.array([[1, 2]])) == 0


class TestLeastConfidenceSampling:
    """Test cases for LeastConfidenceSampling strategy."""
    
    def test_initialization(self):
        """Test strategy initialization."""
        strategy = LeastConfidenceSampling()
        assert isinstance(strategy, SamplingStrategy)
    
    def test_select_sample_basic(self):
        """Test basic sample selection functionality."""
        # Create mock learner
        mock_learner = Mock()
        mock_learner.predict_proba.return_value = np.array([
            [0.8, 0.2],   # High confidence (max prob = 0.8, uncertainty = 0.2)
            [0.6, 0.4],   # Medium confidence (max prob = 0.6, uncertainty = 0.4)  
            [0.51, 0.49], # Low confidence (max prob = 0.51, uncertainty = 0.49) <- most uncertain
            [0.9, 0.1]    # Very high confidence (max prob = 0.9, uncertainty = 0.1)
        ])
        
        X_pool = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        
        strategy = LeastConfidenceSampling()
        selected_idx = strategy.select_sample(mock_learner, X_pool)
        
        # Should select index 2 (most uncertain)
        assert selected_idx == 2
        mock_learner.predict_proba.assert_called_once_with(X_pool)
    
    def test_select_sample_multiclass(self):
        """Test with multiclass predictions."""
        mock_learner = Mock()
        mock_learner.predict_proba.return_value = np.array([
            [0.7, 0.2, 0.1],   # Max prob = 0.7, uncertainty = 0.3
            [0.4, 0.4, 0.2],   # Max prob = 0.4, uncertainty = 0.6 <- most uncertain
            [0.6, 0.3, 0.1],   # Max prob = 0.6, uncertainty = 0.4
        ])
        
        X_pool = np.array([[1, 2], [3, 4], [5, 6]])
        
        strategy = LeastConfidenceSampling()
        selected_idx = strategy.select_sample(mock_learner, X_pool)
        
        assert selected_idx == 1
    
    def test_select_sample_single_sample(self):
        """Test with single sample pool."""
        mock_learner = Mock()
        mock_learner.predict_proba.return_value = np.array([[0.7, 0.3]])
        
        X_pool = np.array([[1, 2]])
        
        strategy = LeastConfidenceSampling()
        selected_idx = strategy.select_sample(mock_learner, X_pool)
        
        assert selected_idx == 0
    
    def test_select_sample_tie_breaking(self):
        """Test behavior when multiple samples have same uncertainty."""
        mock_learner = Mock()
        mock_learner.predict_proba.return_value = np.array([
            [0.6, 0.4],   # Uncertainty = 0.4
            [0.6, 0.4],   # Uncertainty = 0.4 (tie)
            [0.8, 0.2],   # Uncertainty = 0.2
        ])
        
        X_pool = np.array([[1, 2], [3, 4], [5, 6]])
        
        strategy = LeastConfidenceSampling()
        selected_idx = strategy.select_sample(mock_learner, X_pool)
        
        # Should select first occurrence (index 0)
        assert selected_idx == 0


class TestQBCVoteEntropySampling:
    """Test cases for QBCVoteEntropySampling strategy."""
    
    def test_initialization(self):
        """Test strategy initialization."""
        strategy = QBCVoteEntropySampling()
        assert isinstance(strategy, SamplingStrategy)
    
    def test_select_sample_basic(self):
        """Test basic QBC vote entropy sample selection."""
        # Create mock committee
        mock_clf1 = Mock()
        mock_clf1.predict.return_value = np.array([0, 1, 0])  # Predictions for 3 samples
        
        mock_clf2 = Mock()
        mock_clf2.predict.return_value = np.array([0, 0, 1])
        
        mock_clf3 = Mock()
        mock_clf3.predict.return_value = np.array([1, 1, 0])
        
        committee = [mock_clf1, mock_clf2, mock_clf3]
        X_pool = np.array([[1, 2], [3, 4], [5, 6]])
        
        # Vote analysis:
        # Sample 0: votes [0, 0, 1] -> counts [2, 1] -> proportions [2/3, 1/3] -> entropy ≈ 0.636
        # Sample 1: votes [1, 0, 1] -> counts [1, 2] -> proportions [1/3, 2/3] -> entropy ≈ 0.636  
        # Sample 2: votes [0, 1, 0] -> counts [2, 1] -> proportions [2/3, 1/3] -> entropy ≈ 0.636
        # All have same entropy, should select first (index 0)
        
        strategy = QBCVoteEntropySampling()
        selected_idx = strategy.select_sample(committee, X_pool)
        
        # Verify all classifiers were called
        for clf in committee:
            clf.predict.assert_called_once_with(X_pool)
        
        assert selected_idx in [0, 1, 2]  # All have similar entropy
    
    def test_select_sample_high_disagreement(self):
        """Test selection when one sample has high disagreement."""
        # Create committee with clear disagreement on middle sample
        mock_clf1 = Mock()
        mock_clf1.predict.return_value = np.array([0, 0, 0])
        
        mock_clf2 = Mock()
        mock_clf2.predict.return_value = np.array([0, 1, 0])
        
        mock_clf3 = Mock()
        mock_clf3.predict.return_value = np.array([0, 1, 0])
        
        mock_clf4 = Mock()
        mock_clf4.predict.return_value = np.array([0, 0, 0])
        
        committee = [mock_clf1, mock_clf2, mock_clf3, mock_clf4]
        X_pool = np.array([[1, 2], [3, 4], [5, 6]])
        
        # Vote analysis:
        # Sample 0: votes [0, 0, 0, 0] -> all agree -> entropy = 0
        # Sample 1: votes [0, 1, 1, 0] -> counts [2, 2] -> proportions [0.5, 0.5] -> max entropy ≈ 0.693
        # Sample 2: votes [0, 0, 0, 0] -> all agree -> entropy = 0
        
        strategy = QBCVoteEntropySampling()
        selected_idx = strategy.select_sample(committee, X_pool)
        
        assert selected_idx == 1  # Should select the sample with highest disagreement
    
    def test_select_sample_unanimous_committee(self):
        """Test when committee is unanimous on all samples."""
        mock_clf1 = Mock()
        mock_clf1.predict.return_value = np.array([0, 1, 0])
        
        mock_clf2 = Mock()
        mock_clf2.predict.return_value = np.array([0, 1, 0])
        
        committee = [mock_clf1, mock_clf2]
        X_pool = np.array([[1, 2], [3, 4], [5, 6]])
        
        strategy = QBCVoteEntropySampling()
        selected_idx = strategy.select_sample(committee, X_pool)
        
        # All samples have entropy = 0, should select first
        assert selected_idx == 0
    
    def test_select_sample_single_classifier(self):
        """Test with single classifier in committee."""
        mock_clf = Mock()
        mock_clf.predict.return_value = np.array([0, 1, 0])
        
        committee = [mock_clf]
        X_pool = np.array([[1, 2], [3, 4], [5, 6]])
        
        strategy = QBCVoteEntropySampling()
        selected_idx = strategy.select_sample(committee, X_pool)
        
        # With single classifier, all entropies are 0, should select first
        assert selected_idx == 0
    
    def test_select_sample_single_sample(self):
        """Test with single sample in pool."""
        mock_clf1 = Mock()
        mock_clf1.predict.return_value = np.array([0])
        
        mock_clf2 = Mock() 
        mock_clf2.predict.return_value = np.array([1])
        
        committee = [mock_clf1, mock_clf2]
        X_pool = np.array([[1, 2]])
        
        strategy = QBCVoteEntropySampling()
        selected_idx = strategy.select_sample(committee, X_pool)
        
        assert selected_idx == 0


class TestRandomSampling:
    """Test cases for RandomSampling strategy."""
    
    def test_initialization(self):
        """Test strategy initialization."""
        strategy = RandomSampling()
        assert isinstance(strategy, SamplingStrategy)
    
    def test_select_sample_basic(self):
        """Test basic random sample selection."""
        X_pool = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        
        strategy = RandomSampling()
        
        # Set random seed for reproducibility
        np.random.seed(42)
        selected_idx = strategy.select_sample(None, X_pool)
        
        assert isinstance(selected_idx, (int, np.integer))
        assert 0 <= selected_idx < len(X_pool)
    
    def test_select_sample_distribution(self):
        """Test that random sampling follows uniform distribution."""
        X_pool = np.array([[1, 2], [3, 4], [5, 6]])
        strategy = RandomSampling()
        
        # Collect many samples to check distribution
        samples = []
        np.random.seed(42)
        for _ in range(1000):
            idx = strategy.select_sample(None, X_pool)
            samples.append(idx)
        
        # Check that all indices are selected with roughly equal frequency
        unique, counts = np.unique(samples, return_counts=True)
        
        assert len(unique) == 3  # All indices should be selected
        assert set(unique) == {0, 1, 2}
        
        # Check roughly uniform distribution (within reasonable tolerance)
        expected_count = 1000 / 3
        for count in counts:
            assert abs(count - expected_count) < 100  # Allow some variation
    
    def test_select_sample_single_sample(self):
        """Test with single sample pool."""
        X_pool = np.array([[1, 2]])
        
        strategy = RandomSampling()
        selected_idx = strategy.select_sample(None, X_pool)
        
        assert selected_idx == 0
    
    def test_learner_parameter_ignored(self):
        """Test that learner parameter is ignored in random sampling."""
        X_pool = np.array([[1, 2], [3, 4]])
        strategy = RandomSampling()
        
        # Should work with any learner value (including None)
        np.random.seed(42)
        idx1 = strategy.select_sample(None, X_pool)
        
        np.random.seed(42)
        idx2 = strategy.select_sample("dummy_learner", X_pool)
        
        np.random.seed(42)
        idx3 = strategy.select_sample(Mock(), X_pool)
        
        # Results should be identical regardless of learner
        assert idx1 == idx2 == idx3


class TestFunctionalInterface:
    """Test cases for the functional interface (backward compatibility)."""
    
    def test_least_confidence_sampling_function(self):
        """Test least_confidence_sampling function."""
        mock_learner = Mock()
        mock_learner.predict_proba.return_value = np.array([
            [0.8, 0.2],
            [0.51, 0.49],  # Most uncertain
            [0.9, 0.1]
        ])
        
        X_pool = np.array([[1, 2], [3, 4], [5, 6]])
        
        selected_idx = least_confidence_sampling(mock_learner, X_pool)
        
        assert selected_idx == 1
        mock_learner.predict_proba.assert_called_once_with(X_pool)
    
    def test_qbc_vote_entropy_sampling_function(self):
        """Test qbc_vote_entropy_sampling function."""
        mock_clf1 = Mock()
        mock_clf1.predict.return_value = np.array([0, 0, 1])
        
        mock_clf2 = Mock()
        mock_clf2.predict.return_value = np.array([0, 1, 0])
        
        committee = [mock_clf1, mock_clf2]
        X_pool = np.array([[1, 2], [3, 4], [5, 6]])
        
        selected_idx = qbc_vote_entropy_sampling(committee, X_pool)
        
        assert isinstance(selected_idx, (int, np.integer))
        assert 0 <= selected_idx < len(X_pool)
        
        for clf in committee:
            clf.predict.assert_called_once_with(X_pool)
    
    def test_random_sample_function(self):
        """Test random_sample function."""
        X_pool = np.array([[1, 2], [3, 4], [5, 6]])
        
        np.random.seed(42)
        selected_idx = random_sample(None, X_pool)
        
        assert isinstance(selected_idx, (int, np.integer))
        assert 0 <= selected_idx < len(X_pool)


class TestStrategiesIntegration:
    """Integration tests with realistic scenarios."""
    
    def test_strategies_with_sklearn_models(self, train_test_split_data):
        """Test strategies with actual sklearn models."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        
        data = train_test_split_data
        X_train, y_train = data['X_train'], data['y_train']
        X_test = data['X_test'][:5]  # Small pool for testing
        
        # Train models
        rf = RandomForestClassifier(n_estimators=10, random_state=42)
        lr = LogisticRegression(random_state=42)
        
        rf.fit(X_train, y_train)
        lr.fit(X_train, y_train)
        
        # Test least confidence sampling
        lc_strategy = LeastConfidenceSampling()
        lc_idx = lc_strategy.select_sample(rf, X_test)
        
        assert isinstance(lc_idx, (int, np.integer))
        assert 0 <= lc_idx < len(X_test)
        
        # Test QBC vote entropy sampling
        committee = [rf, lr]
        qbc_strategy = QBCVoteEntropySampling()
        qbc_idx = qbc_strategy.select_sample(committee, X_test)
        
        assert isinstance(qbc_idx, (int, np.integer))
        assert 0 <= qbc_idx < len(X_test)
        
        # Test random sampling
        random_strategy = RandomSampling()
        random_idx = random_strategy.select_sample(rf, X_test)
        
        assert isinstance(random_idx, (int, np.integer))
        assert 0 <= random_idx < len(X_test)
    
    def test_reproducibility_with_seed(self):
        """Test that results are reproducible when random seed is set."""
        X_pool = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        
        # Test random sampling reproducibility
        random_strategy = RandomSampling()
        
        np.random.seed(42)
        idx1 = random_strategy.select_sample(None, X_pool)
        
        np.random.seed(42)
        idx2 = random_strategy.select_sample(None, X_pool)
        
        assert idx1 == idx2
    
    def test_strategies_comparison(self, train_test_split_data):
        """Test different strategies on same data to ensure they behave differently."""
        from sklearn.ensemble import RandomForestClassifier
        
        data = train_test_split_data
        X_train, y_train = data['X_train'], data['y_train']
        X_test = data['X_test'][:10]  # Small pool
        
        # Train model
        rf = RandomForestClassifier(n_estimators=10, random_state=42)
        rf.fit(X_train, y_train)
        
        # Get selections from different strategies
        lc_strategy = LeastConfidenceSampling()
        lc_idx = lc_strategy.select_sample(rf, X_test)
        
        np.random.seed(42)
        random_strategy = RandomSampling()
        random_idx = random_strategy.select_sample(rf, X_test)
        
        # Strategies should potentially select different samples
        # (though they might occasionally select the same one by chance)
        assert isinstance(lc_idx, (int, np.integer))
        assert isinstance(random_idx, (int, np.integer))
        assert 0 <= lc_idx < len(X_test)
        assert 0 <= random_idx < len(X_test)


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_empty_pool(self):
        """Test behavior with empty sample pool."""
        X_pool = np.array([]).reshape(0, 2)
        
        mock_learner = Mock()
        mock_learner.predict_proba.return_value = np.array([]).reshape(0, 2)
        
        lc_strategy = LeastConfidenceSampling()
        
        # Should handle empty arrays gracefully or raise appropriate error
        with pytest.raises((ValueError, IndexError)):
            lc_strategy.select_sample(mock_learner, X_pool)
    
    def test_invalid_probabilities(self):
        """Test handling of invalid probability predictions."""
        mock_learner = Mock()
        mock_learner.predict_proba.return_value = np.array([
            [1.1, -0.1],  # Invalid probabilities (sum > 1, negative)
            [0.5, 0.5]
        ])
        
        X_pool = np.array([[1, 2], [3, 4]])
        
        lc_strategy = LeastConfidenceSampling()
        # Should still work (numpy max will handle it)
        selected_idx = lc_strategy.select_sample(mock_learner, X_pool)
        
        assert isinstance(selected_idx, (int, np.integer))
        assert 0 <= selected_idx < len(X_pool)
    
    def test_committee_prediction_mismatch(self):
        """Test QBC when committee members return different length predictions."""
        mock_clf1 = Mock()
        mock_clf1.predict.return_value = np.array([0, 1])
        
        mock_clf2 = Mock()
        mock_clf2.predict.return_value = np.array([0])  # Different length!
        
        committee = [mock_clf1, mock_clf2]
        X_pool = np.array([[1, 2], [3, 4]])
        
        qbc_strategy = QBCVoteEntropySampling()
        
        # Should raise an error due to shape mismatch
        with pytest.raises((ValueError, IndexError)):
            qbc_strategy.select_sample(committee, X_pool)


@pytest.mark.slow
class TestStrategiesPerformance:
    """Performance tests for sampling strategies."""
    
    def test_large_pool_performance(self, performance_monitor):
        """Test performance with large sample pools."""
        # Create large pool
        X_pool = np.random.randn(1000, 50)
        
        # Mock learner with realistic predictions
        mock_learner = Mock()
        mock_learner.predict_proba.return_value = np.random.rand(1000, 2)
        
        lc_strategy = LeastConfidenceSampling()
        
        performance_monitor.start()
        selected_idx = lc_strategy.select_sample(mock_learner, X_pool)
        stats = performance_monitor.stop()
        
        # Should complete quickly
        assert stats['execution_time'] < 1.0  # Less than 1 second
        assert isinstance(selected_idx, (int, np.integer))
        assert 0 <= selected_idx < len(X_pool)
    
    def test_large_committee_performance(self, performance_monitor):
        """Test QBC performance with large committee."""
        X_pool = np.random.randn(100, 20)
        
        # Create large committee
        committee = []
        for i in range(20):
            mock_clf = Mock()
            mock_clf.predict.return_value = np.random.choice([0, 1], size=100)
            committee.append(mock_clf)
        
        qbc_strategy = QBCVoteEntropySampling()
        
        performance_monitor.start()
        selected_idx = qbc_strategy.select_sample(committee, X_pool)
        stats = performance_monitor.stop()
        
        # Should complete in reasonable time
        assert stats['execution_time'] < 2.0  # Less than 2 seconds
        assert isinstance(selected_idx, (int, np.integer))
        assert 0 <= selected_idx < len(X_pool)