"""
Unit tests for data preprocessing functionality.

Tests the src.data.preprocessing module including SMILES validation,
BBBPreprocessor and BreastCancerPreprocessor classes.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, Mock, MagicMock
import sys

# Add src to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.data.preprocessing import (
    is_valid_smiles, 
    filter_valid_smiles, 
    BasePreprocessor,
    BBBPreprocessor, 
    BreastCancerPreprocessor
)


class TestSmilesValidation:
    """Test cases for SMILES validation functions."""
    
    def test_is_valid_smiles_valid_cases(self):
        """Test valid SMILES strings."""
        valid_smiles = [
            "CCO",  # ethanol
            "CC(C)O",  # isopropanol
            "C1=CC=CC=C1",  # benzene
            "CC(=O)O",  # acetic acid
            "CCN"  # ethylamine
        ]
        
        for smiles in valid_smiles:
            assert is_valid_smiles(smiles), f"Expected {smiles} to be valid"
    
    def test_is_valid_smiles_invalid_cases(self):
        """Test invalid SMILES strings."""
        invalid_smiles = [
            "InvalidSMILES",
            "",
            "ABC123XYZ",
            "C1=CC=CC=C",  # Incomplete ring
            None,
            123,  # Non-string input
            []  # Non-string input
        ]
        
        for smiles in invalid_smiles:
            assert not is_valid_smiles(smiles), f"Expected {smiles} to be invalid"
    
    def test_filter_valid_smiles_success(self):
        """Test filtering valid SMILES from DataFrame."""
        df = pd.DataFrame({
            'SMILES': ['CCO', 'InvalidSMILES', 'C1=CC=CC=C1', ''],
            'Class': ['BBB+', 'BBB-', 'BBB+', 'BBB-'],
            'name': ['ethanol', 'invalid', 'benzene', 'empty']
        })
        
        filtered_df = filter_valid_smiles(df)
        
        # Should only keep valid SMILES
        assert len(filtered_df) == 2
        assert 'CCO' in filtered_df['SMILES'].values
        assert 'C1=CC=CC=C1' in filtered_df['SMILES'].values
        assert 'InvalidSMILES' not in filtered_df['SMILES'].values
    
    def test_filter_valid_smiles_custom_column(self):
        """Test filtering with custom SMILES column name."""
        df = pd.DataFrame({
            'smiles': ['CCO', 'InvalidSMILES'],
            'target': [1, 0]
        })
        
        filtered_df = filter_valid_smiles(df, smiles_col='smiles')
        
        assert len(filtered_df) == 1
        assert filtered_df.iloc[0]['smiles'] == 'CCO'
    
    def test_filter_valid_smiles_missing_column(self):
        """Test error when SMILES column doesn't exist."""
        df = pd.DataFrame({'other_col': [1, 2, 3]})
        
        with pytest.raises(ValueError, match="SMILES column 'SMILES' not found"):
            filter_valid_smiles(df)
    
    def test_filter_valid_smiles_no_valid_smiles(self):
        """Test error when no valid SMILES found."""
        df = pd.DataFrame({
            'SMILES': ['InvalidSMILES1', 'InvalidSMILES2', ''],
            'Class': ['BBB+', 'BBB-', 'BBB+']
        })
        
        with pytest.raises(ValueError, match="No valid SMILES strings found"):
            filter_valid_smiles(df)


class TestBreastCancerPreprocessor:
    """Test cases for BreastCancerPreprocessor."""
    
    def test_init(self):
        """Test initialization of BreastCancerPreprocessor."""
        preprocessor = BreastCancerPreprocessor()
        assert preprocessor.scaler is not None
    
    def test_preprocess_basic(self):
        """Test basic preprocessing of breast cancer data."""
        # Create mock breast cancer data
        df = pd.DataFrame({
            'diagnosis': ['M', 'B', 'M', 'B'],
            'id': [1, 2, 3, 4],
            'radius_mean': [17.99, 20.57, 19.69, 11.42],
            'texture_mean': [10.38, 17.77, 21.25, 20.38],
            'perimeter_mean': [122.8, 132.9, 130.0, 77.58]
        })
        
        preprocessor = BreastCancerPreprocessor()
        features_df, metadata_df = preprocessor.preprocess(df)
        
        # Check output structure
        assert 'target' in metadata_df.columns
        assert 'diagnosis' not in features_df.columns
        assert 'id' not in features_df.columns
        assert 'target' not in features_df.columns
        
        # Check target mapping
        expected_targets = [1, 0, 1, 0]  # M->1, B->0
        assert metadata_df['target'].tolist() == expected_targets
        
        # Check feature columns are preserved
        expected_features = ['radius_mean', 'texture_mean', 'perimeter_mean']
        for feature in expected_features:
            assert feature in features_df.columns
    
    def test_preprocess_missing_columns(self):
        """Test preprocessing when some columns are missing."""
        df = pd.DataFrame({
            'diagnosis': ['M', 'B'],
            'radius_mean': [17.99, 20.57],
            'texture_mean': [10.38, 17.77]
            # No 'id' column
        })
        
        preprocessor = BreastCancerPreprocessor()
        features_df, metadata_df = preprocessor.preprocess(df)
        
        # Should work fine without 'id' column
        assert len(features_df) == 2
        assert len(metadata_df) == 2
        assert 'target' in metadata_df.columns


class TestBBBPreprocessor:
    """Test cases for BBBPreprocessor."""
    
    def test_init(self):
        """Test initialization of BBBPreprocessor."""
        preprocessor = BBBPreprocessor()
        assert preprocessor.mol2vec_model_path == "model_300dim.pkl"
        assert preprocessor.mol2vec_model is None
        assert preprocessor.scaler is not None
    
    def test_init_custom_model_path(self):
        """Test initialization with custom model path."""
        custom_path = "custom_model.pkl"
        preprocessor = BBBPreprocessor(custom_path)
        assert preprocessor.mol2vec_model_path == custom_path
    
    @patch('src.data.preprocessing.requests.get')
    @patch('src.data.preprocessing.tarfile.open')
    @patch('src.data.preprocessing.os.path.exists')
    @patch('src.data.preprocessing.os.rename')
    @patch('src.data.preprocessing.os.remove')
    def test_download_mol2vec_model(self, mock_remove, mock_rename, mock_exists, mock_tar, mock_get, temp_dir):
        """Test downloading Mol2vec model."""
        # Setup mocks
        mock_exists.return_value = False
        mock_response = Mock()
        mock_response.content = b"fake model content"
        mock_get.return_value = mock_response
        
        mock_tar_context = Mock()
        mock_tar.return_value.__enter__.return_value = mock_tar_context
        
        preprocessor = BBBPreprocessor("test_model.pkl")
        
        # Change to temp directory for test
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            preprocessor.download_mol2vec_model()
            
            # Verify download was attempted
            mock_get.assert_called_once()
            mock_tar.assert_called_once()
            mock_rename.assert_called_once()
            mock_remove.assert_called_once()
        finally:
            os.chdir(original_cwd)
    
    @patch('src.data.preprocessing.os.path.exists')
    def test_download_mol2vec_model_exists(self, mock_exists, caplog):
        """Test when Mol2vec model already exists."""
        mock_exists.return_value = True
        
        preprocessor = BBBPreprocessor()
        preprocessor.download_mol2vec_model()
        
        assert "already exists" in caplog.text
    
    @patch('src.data.preprocessing.Word2Vec.load')
    @patch.object(BBBPreprocessor, 'download_mol2vec_model')
    def test_load_mol2vec_model(self, mock_download, mock_load):
        """Test loading Mol2vec model."""
        mock_model = Mock()
        mock_load.return_value = mock_model
        
        preprocessor = BBBPreprocessor()
        result = preprocessor.load_mol2vec_model()
        
        mock_download.assert_called_once()
        mock_load.assert_called_once()
        assert result == mock_model
        assert preprocessor.mol2vec_model == mock_model
    
    @patch('src.data.preprocessing.Word2Vec.load')
    @patch.object(BBBPreprocessor, 'download_mol2vec_model')
    def test_load_mol2vec_model_error(self, mock_download, mock_load):
        """Test error handling when loading Mol2vec model fails."""
        mock_load.side_effect = Exception("Load failed")
        
        preprocessor = BBBPreprocessor()
        
        with pytest.raises(Exception, match="Load failed"):
            preprocessor.load_mol2vec_model()
    
    def test_attach_class_and_map(self):
        """Test attaching class labels and mapping."""
        # Create test data
        desc_df = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0],
            'feature2': [0.5, 1.5, 2.5]
        }, index=[0, 1, 2])
        
        original_df = pd.DataFrame({
            'Class': ['BBB+', 'BBB-', 'BBB+'],
            'other_col': ['a', 'b', 'c']
        }, index=[0, 1, 2])
        
        smiles_series = pd.Series(['CCO', 'CCN', 'CCC'], index=[0, 1, 2])
        
        preprocessor = BBBPreprocessor()
        result = preprocessor.attach_class_and_map(desc_df, original_df, smiles_series)
        
        # Check structure
        assert 'BBB' in result.columns
        assert 'SMILES' in result.columns
        assert 'Class' not in result.columns
        assert 'feature1' in result.columns
        assert 'feature2' in result.columns
        
        # Check mapping
        expected_bbb = [1, 0, 1]  # BBB+ -> 1, BBB- -> 0
        assert result['BBB'].tolist() == expected_bbb
        
        # Check SMILES
        assert result['SMILES'].tolist() == ['CCO', 'CCN', 'CCC']
    
    def test_attach_class_and_map_missing_class_column(self):
        """Test error when Class column is missing."""
        desc_df = pd.DataFrame({'feature1': [1.0, 2.0]})
        original_df = pd.DataFrame({'other_col': ['a', 'b']})
        smiles_series = pd.Series(['CCO', 'CCN'])
        
        preprocessor = BBBPreprocessor()
        
        with pytest.raises(ValueError, match="'Class' column not found"):
            preprocessor.attach_class_and_map(desc_df, original_df, smiles_series)
    
    def test_attach_class_and_map_invalid_labels(self):
        """Test error with invalid class labels."""
        desc_df = pd.DataFrame({'feature1': [1.0, 2.0]})
        original_df = pd.DataFrame({'Class': ['INVALID', 'BBB+']})
        smiles_series = pd.Series(['CCO', 'CCN'])
        
        preprocessor = BBBPreprocessor()
        
        with pytest.raises(ValueError, match="Invalid class labels found"):
            preprocessor.attach_class_and_map(desc_df, original_df, smiles_series)
    
    def test_clean_and_preprocess(self):
        """Test data cleaning and preprocessing."""
        # Create test data with various issues
        df = pd.DataFrame({
            'feature1': [1.0, 2.0, np.nan, 4.0],  # Missing values
            'feature2': [0.5, 0.5, 0.5, 0.5],     # Constant feature
            'feature3': [1.0, 1.001, 1.002, 1.003],  # Low variance
            'feature4': [np.nan] * 4,               # All missing
            'feature5': [1.0, 2.0, 3.0, 4.0],     # Good feature
            'BBB': [1, 0, 1, 0],
            'SMILES': ['CCO', 'CCN', 'CCC', 'CCCC']
        })
        
        preprocessor = BBBPreprocessor()
        result = preprocessor.clean_and_preprocess(df)
        
        # Check SMILES is preserved
        assert 'SMILES' in result.columns
        assert result['SMILES'].tolist() == ['CCO', 'CCN', 'CCC', 'CCCC']
        
        # Check missing values are handled
        assert result.isnull().sum().sum() == 0
        
        # Check that constant and low variance features are removed
        # feature2 (constant) and feature4 (all missing) should be removed
        assert 'feature2' not in result.columns
        assert 'feature4' not in result.columns
        
        # Good features should remain
        assert 'feature5' in result.columns
        assert 'BBB' in result.columns
    
    def test_clean_and_preprocess_no_issues(self):
        """Test preprocessing with clean data."""
        df = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0, 4.0],
            'feature2': [0.1, 0.5, 0.8, 1.2],
            'BBB': [1, 0, 1, 0],
            'SMILES': ['CCO', 'CCN', 'CCC', 'CCCC']
        })
        
        preprocessor = BBBPreprocessor()
        result = preprocessor.clean_and_preprocess(df)
        
        # All features should be preserved
        assert result.shape == df.shape
        assert all(col in result.columns for col in df.columns)
    
    @patch('src.data.preprocessing.filter_valid_smiles')
    @patch('src.data.preprocessing.compute_descriptors')
    @patch.object(BBBPreprocessor, 'load_mol2vec_model')
    @patch.object(BBBPreprocessor, 'attach_class_and_map')
    @patch.object(BBBPreprocessor, 'clean_and_preprocess')
    def test_preprocess_integration(self, mock_clean, mock_attach, mock_load_model, 
                                   mock_compute_desc, mock_filter_smiles):
        """Test the full preprocessing pipeline."""
        # Setup mocks
        mock_filter_smiles.return_value = pd.DataFrame({'SMILES': ['CCO'], 'Class': ['BBB+']})
        mock_compute_desc.return_value = (
            pd.DataFrame({'feature1': [1.0]}),
            pd.Series(['CCO'])
        )
        mock_attach.return_value = pd.DataFrame({
            'feature1': [1.0], 'BBB': [1], 'SMILES': ['CCO']
        })
        mock_clean.return_value = pd.DataFrame({
            'feature1': [1.0], 'BBB': [1], 'SMILES': ['CCO']
        })
        mock_load_model.return_value = Mock()
        
        # Test
        preprocessor = BBBPreprocessor()
        input_df = pd.DataFrame({'SMILES': ['CCO'], 'Class': ['BBB+']})
        
        features_df, metadata_df = preprocessor.preprocess(input_df)
        
        # Verify pipeline calls
        mock_filter_smiles.assert_called_once()
        mock_compute_desc.assert_called_once()
        mock_attach.assert_called_once()
        mock_clean.assert_called_once()
        
        # Check output structure
        assert 'feature1' in features_df.columns
        assert 'BBB' not in features_df.columns
        assert 'SMILES' not in features_df.columns
        
        assert 'BBB' in metadata_df.columns
        assert 'SMILES' in metadata_df.columns


class TestBasePreprocessor:
    """Test cases for BasePreprocessor abstract class."""
    
    def test_cannot_instantiate_abstract_class(self):
        """Test that BasePreprocessor cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BasePreprocessor()
    
    def test_concrete_implementation_must_implement_preprocess(self):
        """Test that concrete implementations must implement preprocess method."""
        class IncompletePreprocessor(BasePreprocessor):
            pass
        
        with pytest.raises(TypeError):
            IncompletePreprocessor()


# Integration tests with mock data
class TestPreprocessingIntegration:
    """Integration tests using mock datasets."""
    
    def test_bbb_preprocessing_with_mock_data(self, mock_mol2vec_model):
        """Test BBB preprocessing with mock data."""
        # Use the fixture data
        fixtures_dir = Path(__file__).parent / "fixtures"
        df = pd.read_csv(fixtures_dir / "mock_bbb.csv")
        
        # Rename columns to match expected format
        df = df.rename(columns={'p_np': 'Class'})
        df['Class'] = df['Class'].map({1: 'BBB+', 0: 'BBB-'})
        df = df.rename(columns={'smiles': 'SMILES'})
        
        # Mock the heavy dependencies
        with patch('src.data.preprocessing.compute_descriptors') as mock_compute:
            mock_compute.return_value = (
                pd.DataFrame(np.random.randn(len(df), 10)), 
                df['SMILES']
            )
            
            with patch.object(BBBPreprocessor, 'load_mol2vec_model') as mock_load:
                mock_load.return_value = mock_mol2vec_model
                
                preprocessor = BBBPreprocessor()
                features_df, metadata_df = preprocessor.preprocess(df)
        
        # Check basic structure
        assert len(features_df) == len(metadata_df)
        assert 'BBB' in metadata_df.columns
        assert 'SMILES' in metadata_df.columns
        assert features_df.shape[1] > 0  # Should have features
    
    def test_bc_preprocessing_with_mock_data(self):
        """Test breast cancer preprocessing with mock data."""
        # Use the fixture data
        fixtures_dir = Path(__file__).parent / "fixtures"
        df = pd.read_csv(fixtures_dir / "mock_breast_cancer.csv")
        
        # Map diagnosis to expected format
        df['diagnosis'] = df['diagnosis'].map({1: 'M', 0: 'B'})
        
        preprocessor = BreastCancerPreprocessor()
        features_df, metadata_df = preprocessor.preprocess(df)
        
        # Check structure
        assert len(features_df) == len(metadata_df)
        assert 'target' in metadata_df.columns
        assert features_df.shape[1] == 30  # Expected number of features
        
        # Check target values
        assert set(metadata_df['target'].unique()) == {0, 1}


# Error handling and edge cases
class TestPreprocessingErrorHandling:
    """Test error handling and edge cases."""
    
    def test_empty_dataframe_handling(self):
        """Test handling of empty DataFrames."""
        empty_df = pd.DataFrame()
        
        bc_preprocessor = BreastCancerPreprocessor()
        
        with pytest.raises((KeyError, AttributeError)):
            bc_preprocessor.preprocess(empty_df)
    
    def test_malformed_data_handling(self):
        """Test handling of malformed data."""
        malformed_df = pd.DataFrame({
            'invalid_column': [1, 2, 3],
            'another_invalid': ['a', 'b', 'c']
        })
        
        bc_preprocessor = BreastCancerPreprocessor()
        
        with pytest.raises(KeyError):
            bc_preprocessor.preprocess(malformed_df)
    
    def test_single_row_data(self):
        """Test handling of single-row data."""
        single_row_df = pd.DataFrame({
            'diagnosis': ['M'],
            'radius_mean': [17.99],
            'texture_mean': [10.38]
        })
        
        bc_preprocessor = BreastCancerPreprocessor()
        features_df, metadata_df = bc_preprocessor.preprocess(single_row_df)
        
        assert len(features_df) == 1
        assert len(metadata_df) == 1
        assert metadata_df.iloc[0]['target'] == 1  # M -> 1


@pytest.mark.slow
class TestPreprocessingPerformance:
    """Performance tests for preprocessing (marked as slow)."""
    
    def test_large_dataset_performance(self, performance_monitor):
        """Test preprocessing performance with larger datasets."""
        # Create a moderately large dataset
        large_df = pd.DataFrame({
            'diagnosis': np.random.choice(['M', 'B'], 1000),
            **{f'feature_{i}': np.random.randn(1000) for i in range(50)}
        })
        
        preprocessor = BreastCancerPreprocessor()
        
        performance_monitor.start()
        features_df, metadata_df = preprocessor.preprocess(large_df)
        stats = performance_monitor.stop()
        
        # Should complete in reasonable time
        assert stats['execution_time'] < 10.0  # Less than 10 seconds
        assert len(features_df) == 1000
        assert len(metadata_df) == 1000