"""
Unit tests for data loading functionality.

Tests the src.data.loader module including load_dataset function,
inspect_dataframe function, and DatasetLoader class.
"""

import pytest
import pandas as pd
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, call
import sys
import os

# Add src to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.data.loader import load_dataset, inspect_dataframe, DatasetLoader


class TestLoadDataset:
    """Test cases for the load_dataset function."""
    
    def test_load_csv_file(self, temp_dir):
        """Test loading a CSV file."""
        # Create a temporary CSV file
        csv_file = Path(temp_dir) / "test.csv"
        test_data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })
        test_data.to_csv(csv_file, index=False)
        
        # Load the file
        loaded_data = load_dataset(csv_file)
        
        # Verify the data
        pd.testing.assert_frame_equal(loaded_data, test_data)
    
    def test_load_excel_file(self, temp_dir):
        """Test loading an Excel file."""
        # Create a temporary Excel file
        excel_file = Path(temp_dir) / "test.xlsx"
        test_data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })
        test_data.to_excel(excel_file, index=False)
        
        # Load the file
        loaded_data = load_dataset(excel_file)
        
        # Verify the data
        pd.testing.assert_frame_equal(loaded_data, test_data)
    
    def test_load_nonexistent_file(self):
        """Test loading a file that doesn't exist."""
        with pytest.raises(FileNotFoundError, match="File not found"):
            load_dataset("nonexistent_file.csv")
    
    def test_load_unsupported_format(self, temp_dir):
        """Test loading a file with unsupported format."""
        # Create a file with unsupported extension
        unsupported_file = Path(temp_dir) / "test.txt"
        unsupported_file.write_text("some text")
        
        with pytest.raises(ValueError, match="Unsupported file format"):
            load_dataset(unsupported_file)
    
    def test_load_with_string_path(self, temp_dir):
        """Test loading with string path instead of Path object."""
        # Create a temporary CSV file
        csv_file = Path(temp_dir) / "test.csv"
        test_data = pd.DataFrame({'col1': [1, 2, 3]})
        test_data.to_csv(csv_file, index=False)
        
        # Load using string path
        loaded_data = load_dataset(str(csv_file))
        
        # Verify the data
        pd.testing.assert_frame_equal(loaded_data, test_data)


class TestInspectDataframe:
    """Test cases for the inspect_dataframe function."""
    
    def test_inspect_basic_dataframe(self, capsys):
        """Test inspecting a basic DataFrame."""
        df = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': ['a', 'b', 'c', 'd', 'e'],
            'col3': [1.1, 2.2, 3.3, 4.4, 5.5]
        })
        
        inspect_dataframe(df, "Test Dataset")
        
        captured = capsys.readouterr()
        output = captured.out
        
        # Check that key information is in output
        assert "Test Dataset" in output
        assert "DataFrame shape: (5, 3)" in output
        assert "col1" in output
        assert "col2" in output
        assert "col3" in output
        assert "NaN count: 0" in output
    
    def test_inspect_dataframe_with_nans(self, capsys):
        """Test inspecting a DataFrame with NaN values."""
        df = pd.DataFrame({
            'col1': [1, 2, None, 4, 5],
            'col2': ['a', None, 'c', 'd', 'e']
        })
        
        inspect_dataframe(df, "NaN Test")
        
        captured = capsys.readouterr()
        output = captured.out
        
        # Check that NaN counts are reported
        assert "NaN count: 1" in output
    
    def test_inspect_empty_dataframe(self, capsys):
        """Test inspecting an empty DataFrame."""
        df = pd.DataFrame()
        
        inspect_dataframe(df, "Empty Dataset")
        
        captured = capsys.readouterr()
        output = captured.out
        
        assert "Empty Dataset" in output
        assert "DataFrame shape: (0, 0)" in output


class TestDatasetLoader:
    """Test cases for the DatasetLoader class."""
    
    def test_init_default_directory(self):
        """Test DatasetLoader initialization with default directory."""
        loader = DatasetLoader()
        assert loader.data_dir == Path("data/raw")
    
    def test_init_custom_directory(self):
        """Test DatasetLoader initialization with custom directory."""
        custom_dir = "custom/data/path"
        loader = DatasetLoader(custom_dir)
        assert loader.data_dir == Path(custom_dir)
    
    def test_init_with_path_object(self):
        """Test DatasetLoader initialization with Path object."""
        custom_dir = Path("custom/data/path")
        loader = DatasetLoader(custom_dir)
        assert loader.data_dir == custom_dir
    
    @patch('src.data.loader.load_dataset')
    @patch('src.data.loader.inspect_dataframe')
    def test_load_bbb_dataset(self, mock_inspect, mock_load, temp_dir):
        """Test loading BBB dataset."""
        # Setup
        loader = DatasetLoader(temp_dir)
        mock_df = pd.DataFrame({'smiles': ['CCO'], 'p_np': [1]})
        mock_load.return_value = mock_df
        
        # Execute
        result = loader.load_bbb_dataset("test_bbb.xlsx")
        
        # Verify
        expected_path = Path(temp_dir) / "test_bbb.xlsx"
        mock_load.assert_called_once_with(expected_path)
        mock_inspect.assert_called_once_with(mock_df, "Blood-Brain Barrier Penetration")
        assert result is mock_df
    
    @patch('src.data.loader.load_dataset')
    @patch('src.data.loader.inspect_dataframe')
    def test_load_breast_cancer_dataset(self, mock_inspect, mock_load, temp_dir):
        """Test loading breast cancer dataset."""
        # Setup
        loader = DatasetLoader(temp_dir)
        mock_df = pd.DataFrame({'diagnosis': [0, 1], 'radius_mean': [10.0, 12.0]})
        mock_load.return_value = mock_df
        
        # Execute
        result = loader.load_breast_cancer_dataset("test_bc.csv")
        
        # Verify
        expected_path = Path(temp_dir) / "test_bc.csv"
        mock_load.assert_called_once_with(expected_path)
        mock_inspect.assert_called_once_with(mock_df, "Breast Cancer")
        assert result is mock_df
    
    def test_load_bbb_dataset_default_filename(self, temp_dir):
        """Test loading BBB dataset with default filename."""
        # Create mock BBB file
        bbb_file = Path(temp_dir) / "BBBP.xlsx"
        test_data = pd.DataFrame({'smiles': ['CCO'], 'p_np': [1]})
        test_data.to_excel(bbb_file, index=False)
        
        loader = DatasetLoader(temp_dir)
        
        with patch('src.data.loader.inspect_dataframe'):
            result = loader.load_bbb_dataset()
        
        assert len(result) == 1
        assert 'smiles' in result.columns
        assert 'p_np' in result.columns
    
    def test_load_breast_cancer_dataset_default_filename(self, temp_dir):
        """Test loading breast cancer dataset with default filename."""
        # Create mock breast cancer file
        bc_file = Path(temp_dir) / "breast-cancer.csv"
        test_data = pd.DataFrame({'diagnosis': [0, 1], 'radius_mean': [10.0, 12.0]})
        test_data.to_csv(bc_file, index=False)
        
        loader = DatasetLoader(temp_dir)
        
        with patch('src.data.loader.inspect_dataframe'):
            result = loader.load_breast_cancer_dataset()
        
        assert len(result) == 2
        assert 'diagnosis' in result.columns
        assert 'radius_mean' in result.columns
    
    @patch('src.data.loader.DatasetLoader.load_bbb_dataset')
    @patch('src.data.loader.DatasetLoader.load_breast_cancer_dataset')
    def test_load_both_datasets(self, mock_load_bc, mock_load_bbb):
        """Test loading both datasets."""
        # Setup
        mock_bbb_df = pd.DataFrame({'smiles': ['CCO']})
        mock_bc_df = pd.DataFrame({'diagnosis': [1]})
        mock_load_bbb.return_value = mock_bbb_df
        mock_load_bc.return_value = mock_bc_df
        
        loader = DatasetLoader()
        
        # Execute
        bbb_result, bc_result = loader.load_both_datasets()
        
        # Verify
        mock_load_bbb.assert_called_once()
        mock_load_bc.assert_called_once()
        assert bbb_result is mock_bbb_df
        assert bc_result is mock_bc_df


class TestDataLoaderIntegration:
    """Integration tests for data loader with actual mock files."""
    
    def test_load_mock_bbb_dataset(self):
        """Test loading the mock BBB dataset file."""
        # Use the fixture mock data
        fixtures_dir = Path(__file__).parent / "fixtures"
        loader = DatasetLoader(fixtures_dir)
        
        with patch('src.data.loader.inspect_dataframe'):
            df = loader.load_bbb_dataset("mock_bbb.csv")
        
        # Verify structure
        assert 'smiles' in df.columns
        assert 'p_np' in df.columns
        assert 'name' in df.columns
        assert len(df) == 20  # Check we have 20 mock compounds
        
        # Verify some content
        assert 'CCO' in df['smiles'].values  # ethanol
        assert 'C1=CC=CC=C1' in df['smiles'].values  # benzene
        assert set(df['p_np'].unique()) == {0, 1}  # Binary classification
    
    def test_load_mock_breast_cancer_dataset(self):
        """Test loading the mock breast cancer dataset file."""
        # Use the fixture mock data
        fixtures_dir = Path(__file__).parent / "fixtures"
        loader = DatasetLoader(fixtures_dir)
        
        with patch('src.data.loader.inspect_dataframe'):
            df = loader.load_breast_cancer_dataset("mock_breast_cancer.csv")
        
        # Verify structure
        assert 'diagnosis' in df.columns
        assert len(df.columns) == 31  # 30 features + diagnosis
        assert len(df) == 20  # Check we have 20 mock samples
        
        # Verify some content
        assert set(df['diagnosis'].unique()) == {0, 1}  # Binary classification
        expected_features = ['radius_mean', 'texture_mean', 'perimeter_mean']
        for feature in expected_features:
            assert feature in df.columns


# Error handling tests
class TestDataLoaderErrorHandling:
    """Test error handling scenarios."""
    
    def test_load_bbb_file_not_found(self):
        """Test error when BBB file doesn't exist."""
        loader = DatasetLoader("nonexistent_dir")
        
        with pytest.raises(FileNotFoundError):
            loader.load_bbb_dataset("nonexistent.xlsx")
    
    def test_load_bc_file_not_found(self):
        """Test error when breast cancer file doesn't exist."""
        loader = DatasetLoader("nonexistent_dir")
        
        with pytest.raises(FileNotFoundError):
            loader.load_breast_cancer_dataset("nonexistent.csv")
    
    def test_corrupted_file_handling(self, temp_dir):
        """Test handling of corrupted files."""
        # Create a corrupted CSV file
        corrupted_file = Path(temp_dir) / "corrupted.csv"
        corrupted_file.write_text("invalid,csv,content\nthis is not proper CSV")
        
        loader = DatasetLoader(temp_dir)
        
        # This should handle pandas parsing errors gracefully
        with pytest.raises((pd.errors.ParserError, pd.errors.EmptyDataError)):
            loader.load_breast_cancer_dataset("corrupted.csv")


@pytest.mark.requires_data
class TestDataLoaderWithRealData:
    """Tests that require actual dataset files (marked as requiring data)."""
    
    def test_load_real_datasets_if_available(self, data_dir):
        """Test loading real datasets if they exist."""
        # Skip if data files don't exist
        bbb_file = data_dir / "raw" / "BBBP.xlsx"
        bc_file = data_dir / "raw" / "breast-cancer.csv"
        
        if not (bbb_file.exists() and bc_file.exists()):
            pytest.skip("Real dataset files not available")
        
        loader = DatasetLoader(data_dir / "raw")
        
        with patch('src.data.loader.inspect_dataframe'):
            bbb_df, bc_df = loader.load_both_datasets()
        
        # Basic validation
        assert len(bbb_df) > 0
        assert len(bc_df) > 0
        assert 'smiles' in bbb_df.columns
        assert 'diagnosis' in bc_df.columns