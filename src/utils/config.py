"""
Configuration management utilities.
"""

import yaml
import json
from typing import Dict, Any, Union
from pathlib import Path


class ConfigManager:
    """
    Configuration manager for biomedical active learning experiments.
    """
    
    def __init__(self, config_path: Union[str, Path]):
        """
        Initialize configuration manager.
        
        Parameters:
        -----------
        config_path : str or Path
            Path to configuration file
        """
        self.config_path = Path(config_path)
        self.config = {}
        
        if self.config_path.exists():
            self.load()
            
    def load(self):
        """Load configuration from file."""
        if self.config_path.suffix.lower() in ['.yaml', '.yml']:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        elif self.config_path.suffix.lower() == '.json':
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {self.config_path.suffix}")
            
    def save(self):
        """Save configuration to file."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.config_path.suffix.lower() in ['.yaml', '.yml']:
            with open(self.config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
        elif self.config_path.suffix.lower() == '.json':
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        else:
            raise ValueError(f"Unsupported config format: {self.config_path.suffix}")
            
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Parameters:
        -----------
        key : str
            Configuration key (supports dot notation)
        default : Any
            Default value if key not found
            
        Returns:
        --------
        Any
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
            
    def set(self, key: str, value: Any):
        """
        Set configuration value.
        
        Parameters:
        -----------
        key : str
            Configuration key (supports dot notation)
        value : Any
            Value to set
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
            
        config[keys[-1]] = value
        
    def update(self, other_config: Dict[str, Any]):
        """
        Update configuration with another dictionary.
        
        Parameters:
        -----------
        other_config : Dict[str, Any]
            Configuration to merge
        """
        def deep_update(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = deep_update(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
            
        deep_update(self.config, other_config)
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Get configuration as dictionary.
        
        Returns:
        --------
        Dict[str, Any]
            Configuration dictionary
        """
        return self.config.copy()


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from file.
    
    Parameters:
    -----------
    config_path : str or Path
        Path to configuration file
        
    Returns:
    --------
    Dict[str, Any]
        Configuration dictionary
    """
    manager = ConfigManager(config_path)
    return manager.to_dict()


def save_config(config: Dict[str, Any], config_path: Union[str, Path]):
    """
    Save configuration to file.
    
    Parameters:
    -----------
    config : Dict[str, Any]
        Configuration to save
    config_path : str or Path
        Path to save configuration
    """
    manager = ConfigManager(config_path)
    manager.config = config
    manager.save()


# Default configurations
DEFAULT_AL_CONFIG_BBBP = {
    'max_queries': -1,
    'stop_ratio': 1.0,
    'batch_size': 20,
    'stratified_seeds': [42, 10, 50, 100],
    'rf_params': {'n_estimators': 100},
    'qbc_params': {},
    'n_runs': 10,
    'n_jobs': -1
}

DEFAULT_AL_CONFIG_BC = {
    'max_queries': -1,
    'stop_ratio': 1.0,
    'batch_size': 10,
    'stratified_seeds': [42, 10, 50, 100],
    'rf_params': {'n_estimators': 100},
    'qbc_params': {},
    'n_runs': 10,
    'n_jobs': -1
}

DEFAULT_DATA_CONFIG = {
    'datasets': {
        'bbb': {
            'name': 'Blood-Brain Barrier Penetration',
            'path': 'data/raw/BBBP.xlsx',
            'features': {
                'molecular_descriptors': True,
                'mol2vec': True,
                'mol2vec_model': 'model_300dim.pkl'
            }
        },
        'breast_cancer': {
            'name': 'Breast Cancer',
            'path': 'data/raw/breast-cancer.csv',
            'target_column': 'diagnosis'
        }
    }
}

DEFAULT_MODEL_CONFIG = {
    'random_forest': {
        'n_estimators': 100,
        'random_state': 42
    },
    'qbc': {
        'n_committee_members': 5,
        'base_learners': [
            'RandomForest',
            'ExtraTrees', 
            'GradientBoosting',
            'LogisticRegression',
            'KNN'
        ]
    }
}