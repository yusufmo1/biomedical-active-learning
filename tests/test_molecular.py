"""
Unit tests for molecular featurization functionality.

Tests the src.features.molecular module including SMILES processing,
molecular descriptor calculation, and Mol2vec embedding functions.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add src to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.features.molecular import (
    mol2alt_sentence,
    compute_mol2vec_embedding,
    compute_descriptors,
    MolecularFeaturizer
)

# Try to import RDKit, skip tests if not available
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not available")
class TestMol2AltSentence:
    """Test cases for mol2alt_sentence function."""
    
    def test_simple_molecule(self):
        """Test alternating sentence generation for simple molecule."""
        mol = Chem.MolFromSmiles("CCO")  # ethanol
        sentence = mol2alt_sentence(mol, radius=1)
        
        # Should return a list of string identifiers
        assert isinstance(sentence, list)
        assert all(isinstance(s, str) for s in sentence)
        assert len(sentence) > 0
    
    def test_different_radii(self):
        """Test with different fingerprint radii."""
        mol = Chem.MolFromSmiles("C1=CC=CC=C1")  # benzene
        
        sentence_r0 = mol2alt_sentence(mol, radius=0)
        sentence_r1 = mol2alt_sentence(mol, radius=1)
        sentence_r2 = mol2alt_sentence(mol, radius=2)
        
        # Higher radius should generally give more identifiers
        assert len(sentence_r0) <= len(sentence_r1)
        assert len(sentence_r1) <= len(sentence_r2)
    
    def test_empty_molecule(self):
        """Test with minimal molecule."""
        mol = Chem.MolFromSmiles("C")  # methane
        sentence = mol2alt_sentence(mol, radius=1)
        
        assert isinstance(sentence, list)
        assert len(sentence) > 0
    
    def test_complex_molecule(self):
        """Test with more complex molecule."""
        mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")  # aspirin
        sentence = mol2alt_sentence(mol, radius=1)
        
        assert isinstance(sentence, list)
        assert len(sentence) > 0
        # Complex molecule should have many unique identifiers
        assert len(set(sentence)) > 1


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not available")
class TestComputeMol2vecEmbedding:
    """Test cases for compute_mol2vec_embedding function."""
    
    def test_basic_embedding(self, mock_mol2vec_model):
        """Test basic Mol2vec embedding computation."""
        mol = Chem.MolFromSmiles("CCO")
        
        embedding = compute_mol2vec_embedding(mol, mock_mol2vec_model, radius=1)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (300,)  # Mock model has 300 dimensions
        assert not np.all(embedding == 0)  # Should not be all zeros
    
    def test_embedding_with_unseen_substructures(self, mock_mol2vec_model):
        """Test embedding when molecule contains unseen substructures."""
        # Mock the model to have limited vocabulary
        mock_mol2vec_model.wv.key_to_index = {'123': 0, 'UNK': 1}  # Very limited vocab
        
        mol = Chem.MolFromSmiles("CC(C)O")  # isopropanol
        
        embedding = compute_mol2vec_embedding(mol, mock_mol2vec_model, radius=1, unseen='UNK')
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (300,)
    
    def test_embedding_without_unk_token(self, mock_mol2vec_model):
        """Test embedding when UNK token is not in vocabulary."""
        # Mock model without UNK token
        mock_mol2vec_model.wv.key_to_index = {'123': 0, '456': 1}
        
        mol = Chem.MolFromSmiles("CCO")
        
        embedding = compute_mol2vec_embedding(mol, mock_mol2vec_model, radius=1, unseen='UNK')
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (300,)
    
    def test_different_radii_produce_different_embeddings(self, mock_mol2vec_model):
        """Test that different radii produce different embeddings."""
        mol = Chem.MolFromSmiles("C1=CC=CC=C1")  # benzene
        
        embedding_r1 = compute_mol2vec_embedding(mol, mock_mol2vec_model, radius=1)
        embedding_r2 = compute_mol2vec_embedding(mol, mock_mol2vec_model, radius=2)
        
        # Different radii should produce different embeddings
        assert not np.array_equal(embedding_r1, embedding_r2)


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not available")
class TestComputeDescriptors:
    """Test cases for compute_descriptors function."""
    
    def test_basic_descriptor_computation(self, mock_mol2vec_model):
        """Test basic descriptor computation for valid SMILES."""
        df = pd.DataFrame({
            'SMILES': ['CCO', 'CC(C)O', 'C1=CC=CC=C1'],
            'other_col': ['a', 'b', 'c']
        })
        
        desc_df, smiles_series = compute_descriptors(df, 'SMILES', mock_mol2vec_model)
        
        # Check output structure
        assert isinstance(desc_df, pd.DataFrame)
        assert isinstance(smiles_series, pd.Series)
        assert len(desc_df) == len(smiles_series) == 3
        
        # Check that we have RDKit descriptors and Mol2vec features
        mol2vec_cols = [col for col in desc_df.columns if col.startswith('mol2vec_')]
        rdkit_cols = [col for col in desc_df.columns if not col.startswith('mol2vec_')]
        
        assert len(mol2vec_cols) == 300  # Mock model has 300 dimensions
        assert len(rdkit_cols) > 0  # Should have RDKit descriptors
        
        # Check SMILES series
        assert smiles_series.name == 'SMILES'
        assert all(smiles in ['CCO', 'CC(C)O', 'C1=CC=CC=C1'] for smiles in smiles_series.values)
    
    def test_invalid_smiles_handling(self, mock_mol2vec_model):
        """Test handling of invalid SMILES strings."""
        df = pd.DataFrame({
            'SMILES': ['CCO', 'InvalidSMILES', 'CC(C)O', ''],
            'other_col': ['a', 'b', 'c', 'd']
        })
        
        desc_df, smiles_series = compute_descriptors(df, 'SMILES', mock_mol2vec_model)
        
        # Only valid SMILES should be processed
        assert len(desc_df) == 2  # Only CCO and CC(C)O are valid
        assert len(smiles_series) == 2
        assert 'InvalidSMILES' not in smiles_series.values
        assert '' not in smiles_series.values
    
    def test_no_model_provided(self):
        """Test error when no Mol2vec model is provided."""
        df = pd.DataFrame({'SMILES': ['CCO']})
        
        with pytest.raises(ValueError, match="Mol2vec model must be provided"):
            compute_descriptors(df, 'SMILES', model=None)
    
    def test_custom_smiles_column(self, mock_mol2vec_model):
        """Test with custom SMILES column name."""
        df = pd.DataFrame({
            'custom_smiles': ['CCO', 'CC(C)O'],
            'other_col': ['a', 'b']
        })
        
        desc_df, smiles_series = compute_descriptors(df, 'custom_smiles', mock_mol2vec_model)
        
        assert len(desc_df) == 2
        assert smiles_series.name == 'custom_smiles'
    
    def test_constant_columns_removal(self, mock_mol2vec_model):
        """Test removal of constant descriptor columns."""
        # Mock descriptor calculator to return some constant values
        df = pd.DataFrame({'SMILES': ['CCO', 'CCN']})
        
        with patch('src.features.molecular.MoleculeDescriptors.MolecularDescriptorCalculator') as mock_calc_class:
            mock_calc = Mock()
            mock_calc_class.return_value = mock_calc
            mock_calc.GetDescriptorNames.return_value = ['desc1', 'desc2', 'desc3']
            # Make desc2 constant across molecules
            mock_calc.CalcDescriptors.side_effect = [
                (1.0, 5.0, 3.0),  # First molecule
                (2.0, 5.0, 4.0)   # Second molecule (desc2 is constant)
            ]
            
            desc_df, smiles_series = compute_descriptors(df, 'SMILES', mock_mol2vec_model)
            
            # desc2 should be removed as it's constant
            rdkit_cols = [col for col in desc_df.columns if not col.startswith('mol2vec_')]
            assert 'desc2' not in rdkit_cols
    
    def test_missing_values_handling(self, mock_mol2vec_model):
        """Test handling of missing values in descriptors."""
        df = pd.DataFrame({'SMILES': ['CCO', 'CCN']})
        
        with patch('src.features.molecular.MoleculeDescriptors.MolecularDescriptorCalculator') as mock_calc_class:
            mock_calc = Mock()
            mock_calc_class.return_value = mock_calc
            mock_calc.GetDescriptorNames.return_value = ['desc1', 'desc2']
            # Return NaN for second molecule
            mock_calc.CalcDescriptors.side_effect = [
                (1.0, 2.0),      # First molecule
                (np.nan, 3.0)    # Second molecule with NaN
            ]
            
            desc_df, smiles_series = compute_descriptors(df, 'SMILES', mock_mol2vec_model)
            
            # Rows with NaN should be dropped
            assert len(desc_df) == 1
            assert len(smiles_series) == 1


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not available")
class TestMolecularFeaturizer:
    """Test cases for MolecularFeaturizer class."""
    
    def test_init_without_model(self):
        """Test initialization without Mol2vec model."""
        featurizer = MolecularFeaturizer()
        
        assert featurizer.mol2vec_model is None
        assert featurizer.radius == 1
        assert featurizer.descriptor_calculator is not None
    
    def test_init_with_model(self, mock_mol2vec_model):
        """Test initialization with Mol2vec model."""
        featurizer = MolecularFeaturizer(mock_mol2vec_model, radius=2)
        
        assert featurizer.mol2vec_model is mock_mol2vec_model
        assert featurizer.radius == 2
        assert featurizer.descriptor_calculator is not None
    
    def test_set_mol2vec_model(self, mock_mol2vec_model):
        """Test setting Mol2vec model."""
        featurizer = MolecularFeaturizer()
        featurizer.set_mol2vec_model(mock_mol2vec_model)
        
        assert featurizer.mol2vec_model is mock_mol2vec_model
    
    def test_featurize_molecule(self, mock_mol2vec_model):
        """Test featurizing a single molecule."""
        featurizer = MolecularFeaturizer(mock_mol2vec_model)
        mol = Chem.MolFromSmiles("CCO")
        
        features = featurizer.featurize_molecule(mol)
        
        assert isinstance(features, dict)
        
        # Check that we have both RDKit descriptors and Mol2vec features
        mol2vec_keys = [k for k in features.keys() if k.startswith('mol2vec_')]
        rdkit_keys = [k for k in features.keys() if not k.startswith('mol2vec_')]
        
        assert len(mol2vec_keys) == 300  # Mock model has 300 dimensions
        assert len(rdkit_keys) > 0  # Should have RDKit descriptors
    
    def test_featurize_molecule_without_model(self):
        """Test error when featurizing without model."""
        featurizer = MolecularFeaturizer()
        mol = Chem.MolFromSmiles("CCO")
        
        with pytest.raises(ValueError, match="Mol2vec model not set"):
            featurizer.featurize_molecule(mol)
    
    def test_featurize_smiles(self, mock_mol2vec_model):
        """Test featurizing a SMILES string."""
        featurizer = MolecularFeaturizer(mock_mol2vec_model)
        
        features = featurizer.featurize_smiles("CCO")
        
        assert isinstance(features, dict)
        assert len(features) > 0
    
    def test_featurize_invalid_smiles(self, mock_mol2vec_model):
        """Test error with invalid SMILES."""
        featurizer = MolecularFeaturizer(mock_mol2vec_model)
        
        with pytest.raises(ValueError, match="Invalid SMILES"):
            featurizer.featurize_smiles("InvalidSMILES")
    
    @patch('src.features.molecular.compute_descriptors')
    def test_featurize_dataframe(self, mock_compute_descriptors, mock_mol2vec_model):
        """Test featurizing a dataframe."""
        # Setup mock
        mock_desc_df = pd.DataFrame({'feature1': [1.0, 2.0]})
        mock_smiles_series = pd.Series(['CCO', 'CCN'])
        mock_compute_descriptors.return_value = (mock_desc_df, mock_smiles_series)
        
        featurizer = MolecularFeaturizer(mock_mol2vec_model)
        df = pd.DataFrame({'SMILES': ['CCO', 'CCN']})
        
        desc_df, smiles_series = featurizer.featurize_dataframe(df)
        
        # Verify compute_descriptors was called correctly
        mock_compute_descriptors.assert_called_once_with(df, 'SMILES', mock_mol2vec_model)
        
        # Check outputs
        assert desc_df is mock_desc_df
        assert smiles_series is mock_smiles_series
    
    @patch('src.features.molecular.compute_descriptors')
    def test_featurize_dataframe_custom_column(self, mock_compute_descriptors, mock_mol2vec_model):
        """Test featurizing dataframe with custom SMILES column."""
        mock_compute_descriptors.return_value = (pd.DataFrame(), pd.Series())
        
        featurizer = MolecularFeaturizer(mock_mol2vec_model)
        df = pd.DataFrame({'custom_smiles': ['CCO']})
        
        featurizer.featurize_dataframe(df, smiles_col='custom_smiles')
        
        mock_compute_descriptors.assert_called_once_with(df, 'custom_smiles', mock_mol2vec_model)


# Integration tests
@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not available")
class TestMolecularIntegration:
    """Integration tests for molecular featurization with mock data."""
    
    def test_end_to_end_featurization(self, mock_mol2vec_model):
        """Test complete featurization pipeline."""
        # Load mock BBB data
        fixtures_dir = Path(__file__).parent / "fixtures" 
        df = pd.read_csv(fixtures_dir / "mock_bbb.csv")
        df = df.rename(columns={'smiles': 'SMILES'})
        
        # Test with MolecularFeaturizer
        featurizer = MolecularFeaturizer(mock_mol2vec_model)
        desc_df, smiles_series = featurizer.featurize_dataframe(df)
        
        # Basic validation
        assert len(desc_df) > 0
        assert len(smiles_series) > 0
        assert len(desc_df) == len(smiles_series)
        
        # Check feature types
        mol2vec_cols = [col for col in desc_df.columns if col.startswith('mol2vec_')]
        rdkit_cols = [col for col in desc_df.columns if not col.startswith('mol2vec_')]
        
        assert len(mol2vec_cols) == 300
        assert len(rdkit_cols) > 0
    
    def test_featurization_consistency(self, mock_mol2vec_model):
        """Test that featurization is consistent across runs."""
        smiles = "CCO"
        featurizer = MolecularFeaturizer(mock_mol2vec_model)
        
        # Featurize the same molecule twice
        features1 = featurizer.featurize_smiles(smiles)
        features2 = featurizer.featurize_smiles(smiles)
        
        # Results should be identical
        assert features1.keys() == features2.keys()
        for key in features1.keys():
            assert abs(features1[key] - features2[key]) < 1e-10


# Error handling and edge cases
@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not available")
class TestMolecularErrorHandling:
    """Test error handling and edge cases."""
    
    def test_empty_dataframe(self, mock_mol2vec_model):
        """Test handling of empty DataFrame."""
        empty_df = pd.DataFrame(columns=['SMILES'])
        
        desc_df, smiles_series = compute_descriptors(empty_df, 'SMILES', mock_mol2vec_model)
        
        assert len(desc_df) == 0
        assert len(smiles_series) == 0
    
    def test_all_invalid_smiles(self, mock_mol2vec_model):
        """Test DataFrame with all invalid SMILES."""
        df = pd.DataFrame({
            'SMILES': ['InvalidSMILES1', 'InvalidSMILES2', '']
        })
        
        desc_df, smiles_series = compute_descriptors(df, 'SMILES', mock_mol2vec_model)
        
        assert len(desc_df) == 0
        assert len(smiles_series) == 0
    
    def test_descriptor_calculation_error(self, mock_mol2vec_model):
        """Test handling of descriptor calculation errors."""
        df = pd.DataFrame({'SMILES': ['CCO']})
        
        with patch('src.features.molecular.MoleculeDescriptors.MolecularDescriptorCalculator') as mock_calc_class:
            mock_calc = Mock()
            mock_calc_class.return_value = mock_calc
            mock_calc.CalcDescriptors.side_effect = Exception("Calculation failed")
            
            desc_df, smiles_series = compute_descriptors(df, 'SMILES', mock_mol2vec_model)
            
            # Should handle error gracefully and return empty results
            assert len(desc_df) == 0
            assert len(smiles_series) == 0


# Performance tests
@pytest.mark.slow
@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not available")
class TestMolecularPerformance:
    """Performance tests for molecular featurization."""
    
    def test_large_dataset_performance(self, mock_mol2vec_model, performance_monitor):
        """Test featurization performance with larger dataset."""
        # Create moderately large dataset
        valid_smiles = ['CCO', 'CC(C)O', 'C1=CC=CC=C1', 'CC(=O)O', 'CCN']
        large_smiles = valid_smiles * 20  # 100 molecules
        
        df = pd.DataFrame({'SMILES': large_smiles})
        
        featurizer = MolecularFeaturizer(mock_mol2vec_model)
        
        performance_monitor.start()
        desc_df, smiles_series = featurizer.featurize_dataframe(df)
        stats = performance_monitor.stop()
        
        # Should complete in reasonable time
        assert stats['execution_time'] < 30.0  # Less than 30 seconds
        assert len(desc_df) == 100
        assert len(smiles_series) == 100


# Utility to import Path
from pathlib import Path