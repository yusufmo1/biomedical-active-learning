"""
Molecular featurization utilities.

Extracted from the original notebook's molecular feature computation functions.
"""

import logging
import numpy as np
import pandas as pd
from typing import Tuple, Optional, List

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from gensim.models import Word2Vec

logger = logging.getLogger(__name__)


def mol2alt_sentence(mol, radius: int = 1) -> List[str]:
    """
    Generate an alternating sentence (a list of substructure identifiers) from an RDKit molecule.
    This function uses the Morgan fingerprint algorithm (ECFP) to extract substructure identifiers.

    Args:
        mol: RDKit Mol object
        radius (int): Radius used for fingerprinting (default is 1)

    Returns:
        List[str]: A list of substructure identifiers (as strings)
    """
    radii = list(range(int(radius) + 1))
    info = {}
    AllChem.GetMorganFingerprint(mol, radius, bitInfo=info)
    
    mol_atoms = [atom.GetIdx() for atom in mol.GetAtoms()]
    dict_atoms = {idx: {r: None for r in radii} for idx in mol_atoms}
    
    for identifier, atom_infos in info.items():
        for atom_idx, rad in atom_infos:
            dict_atoms[atom_idx][rad] = identifier

    sentence = []
    for idx in sorted(dict_atoms.keys()):
        for r in radii:
            if dict_atoms[idx][r] is not None:
                sentence.append(str(dict_atoms[idx][r]))
    return sentence


def compute_mol2vec_embedding(mol, model: Word2Vec, radius: int = 1, unseen: str = 'UNK') -> np.ndarray:
    """
    Compute the Mol2vec embedding for a molecule by summing the substructure vectors.
    
    Args:
        mol: RDKit Mol object
        model: Pretrained gensim Word2Vec model (the Mol2vec model)
        radius (int): Fingerprint radius
        unseen (str): Token to use for unseen substructures
    
    Returns:
        np.ndarray: A 1D vector representing the Mol2vec embedding
    """
    sentence = mol2alt_sentence(mol, radius)
    keys = set(model.wv.key_to_index.keys())
    
    if unseen in keys:
        unseen_vec = model.wv.get_vector(unseen)
    else:
        unseen_vec = np.zeros(model.vector_size)
    
    vec = np.zeros(model.vector_size)
    for word in sentence:
        if word in keys:
            vec += model.wv.get_vector(word)
        else:
            vec += unseen_vec
    return vec


def compute_descriptors(df: pd.DataFrame, 
                       smiles_col: str = 'SMILES', 
                       model: Optional[Word2Vec] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Calculate RDKit molecular descriptors for SMILES strings and integrate Mol2vec embeddings.
    
    Args:
        df (pd.DataFrame): Input dataframe with valid SMILES
        smiles_col (str): Name of the SMILES column
        model: A pretrained Mol2vec Word2Vec model. If None, model must be provided
        
    Returns:
        Tuple[pd.DataFrame, pd.Series]: DataFrame containing descriptors and Series with SMILES
    """
    if model is None:
        raise ValueError("Mol2vec model must be provided")
        
    descriptor_names = [desc[0] for desc in Descriptors._descList]
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
    
    descriptor_list = []
    valid_indices = []
    valid_smiles = []
    
    for idx, row in df.iterrows():
        smi = row[smiles_col]
        mol = Chem.MolFromSmiles(smi)
        try:
            desc_values = calc.CalcDescriptors(mol)
            desc_dict = dict(zip(calc.GetDescriptorNames(), desc_values))
            
            embedding = compute_mol2vec_embedding(mol, model, radius=1, unseen='UNK')
            mol2vec_dict = {f'mol2vec_{i}': float(embedding[i]) for i in range(len(embedding))}
            
            combined_dict = {**desc_dict, **mol2vec_dict}
            descriptor_list.append(combined_dict)
            valid_indices.append(idx)
            valid_smiles.append(smi)
        except Exception as e:
            logger.warning(f"Error processing SMILES {smi}: {str(e)}")
            continue

    desc_df = pd.DataFrame(descriptor_list, index=valid_indices)
    smiles_series = pd.Series(valid_smiles, index=valid_indices, name=smiles_col)
    
    constant_cols = [col for col in desc_df.columns if desc_df[col].nunique() == 1]
    if constant_cols:
        logger.info(f"Removing {len(constant_cols)} constant columns")
        desc_df.drop(columns=constant_cols, inplace=True)
    
    if desc_df.isna().any().any():
        logger.warning("Missing values found in descriptors/embeddings; dropping missing rows.")
        desc_df.dropna(inplace=True)
        smiles_series = smiles_series[desc_df.index]
        
    logger.info(f"Final descriptor shape: {desc_df.shape}")
    return desc_df, smiles_series


class MolecularFeaturizer:
    """
    Class for molecular featurization using RDKit descriptors and Mol2vec embeddings.
    """
    
    def __init__(self, mol2vec_model: Optional[Word2Vec] = None, radius: int = 1):
        """
        Initialize molecular featurizer.
        
        Parameters:
        -----------
        mol2vec_model : Word2Vec, optional
            Pretrained Mol2vec model
        radius : int
            Radius for Morgan fingerprint calculation
        """
        self.mol2vec_model = mol2vec_model
        self.radius = radius
        self.descriptor_calculator = None
        self._setup_descriptor_calculator()
        
    def _setup_descriptor_calculator(self):
        """Setup RDKit descriptor calculator."""
        descriptor_names = [desc[0] for desc in Descriptors._descList]
        self.descriptor_calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
        
    def set_mol2vec_model(self, model: Word2Vec):
        """
        Set the Mol2vec model.
        
        Parameters:
        -----------
        model : Word2Vec
            Pretrained Mol2vec model
        """
        self.mol2vec_model = model
        
    def featurize_molecule(self, mol) -> dict:
        """
        Featurize a single molecule.
        
        Parameters:
        -----------
        mol : rdkit.Chem.Mol
            RDKit molecule object
            
        Returns:
        --------
        dict
            Dictionary containing molecular descriptors and Mol2vec embeddings
        """
        if self.mol2vec_model is None:
            raise ValueError("Mol2vec model not set. Use set_mol2vec_model() first.")
            
        # Calculate RDKit descriptors
        desc_values = self.descriptor_calculator.CalcDescriptors(mol)
        desc_dict = dict(zip(self.descriptor_calculator.GetDescriptorNames(), desc_values))
        
        # Calculate Mol2vec embedding
        embedding = compute_mol2vec_embedding(mol, self.mol2vec_model, self.radius)
        mol2vec_dict = {f'mol2vec_{i}': float(embedding[i]) for i in range(len(embedding))}
        
        return {**desc_dict, **mol2vec_dict}
        
    def featurize_smiles(self, smiles: str) -> dict:
        """
        Featurize a SMILES string.
        
        Parameters:
        -----------
        smiles : str
            SMILES string
            
        Returns:
        --------
        dict
            Dictionary containing molecular descriptors and Mol2vec embeddings
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        return self.featurize_molecule(mol)
        
    def featurize_dataframe(self, df: pd.DataFrame, smiles_col: str = 'SMILES') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Featurize all molecules in a dataframe.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe with SMILES
        smiles_col : str
            Name of the SMILES column
            
        Returns:
        --------
        Tuple[pd.DataFrame, pd.Series]
            DataFrame containing descriptors and Series with SMILES
        """
        return compute_descriptors(df, smiles_col, self.mol2vec_model)