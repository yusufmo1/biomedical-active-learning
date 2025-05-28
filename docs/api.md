# API Documentation: Biomedical Active Learning

## Overview

This document provides comprehensive API documentation for all modules in the biomedical active learning project. The codebase is organized into modular components that handle data processing, active learning strategies, model evaluation, and utility functions.

## Table of Contents

1. [Package Structure](#package-structure)
2. [Data Module](#data-module)
3. [Active Learning Module](#active-learning-module)
4. [Features Module](#features-module)
5. [Evaluation Module](#evaluation-module)
6. [Dimensionality Module](#dimensionality-module)
7. [Utils Module](#utils-module)
8. [Usage Examples](#usage-examples)
9. [Configuration](#configuration)

## Package Structure

```
src/
├── __init__.py                 # Main package initialization
├── active_learning/            # Active learning algorithms and experiments
│   ├── __init__.py
│   ├── strategies.py          # Sampling strategies
│   ├── learners.py           # Active learning models
│   └── experiments.py        # Experiment orchestration
├── data/                      # Data loading and preprocessing
│   ├── __init__.py
│   ├── loader.py             # Dataset loading utilities
│   └── preprocessing.py      # Data preprocessing pipelines
├── features/                  # Feature engineering
│   ├── __init__.py
│   └── molecular.py          # Molecular featurization
├── evaluation/               # Model evaluation and visualization
│   ├── __init__.py
│   ├── metrics.py           # Evaluation metrics
│   └── visualization.py     # Plotting and visualization
├── dimensionality/           # Dimensionality reduction
│   ├── __init__.py
│   └── reduction.py         # PCA, t-SNE, UMAP, LDA
└── utils/                    # Utility functions
    ├── __init__.py
    ├── config.py            # Configuration management
    ├── helpers.py           # General utilities
    └── parallel.py          # Parallel processing
```

## Data Module

### `src.data.loader`

#### Classes

##### `DatasetLoader`

Main class for loading and preparing datasets.

```python
class DatasetLoader:
    """Centralized dataset loading and preparation."""
    
    def __init__(self, data_dir: str = "data/raw"):
        """Initialize loader with data directory path."""
```

**Methods:**

```python
def load_bbb_dataset(self, file_path: Optional[str] = None) -> Tuple[pd.DataFrame, str]:
    """
    Load Blood-Brain Barrier Penetration dataset.
    
    Args:
        file_path: Optional path to BBBP.xlsx file
        
    Returns:
        Tuple of (dataframe, target_column_name)
        
    Raises:
        FileNotFoundError: If dataset file not found
        ValueError: If required columns missing
    """

def load_breast_cancer_dataset(self, file_path: Optional[str] = None) -> Tuple[pd.DataFrame, str]:
    """
    Load Breast Cancer Wisconsin dataset.
    
    Args:
        file_path: Optional path to breast-cancer.csv file
        
    Returns:
        Tuple of (dataframe, target_column_name)
        
    Raises:
        FileNotFoundError: If dataset file not found
        ValueError: If required columns missing
    """

def prepare_train_test_split(self, X: np.ndarray, y: np.ndarray, 
                           test_size: float = 0.2, random_state: int = 42) -> Tuple:
    """
    Create stratified train-test split.
    
    Args:
        X: Feature matrix
        y: Target vector
        test_size: Proportion for test set
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
```

**Usage Example:**

```python
from src.data.loader import DatasetLoader

loader = DatasetLoader(data_dir="data/raw")
df, target_col = loader.load_breast_cancer_dataset()
print(f"Loaded {len(df)} samples with target: {target_col}")
```

#### Functions

```python
def load_dataset(dataset_name: str, data_dir: str = "data/raw") -> Tuple[pd.DataFrame, str]:
    """
    Generic dataset loading function.
    
    Args:
        dataset_name: Name of dataset ('bbb' or 'breast_cancer')
        data_dir: Directory containing raw data files
        
    Returns:
        Tuple of (dataframe, target_column_name)
    """

def inspect_dataframe(df: pd.DataFrame, dataset_name: str = "Dataset") -> Dict[str, Any]:
    """
    Generate comprehensive dataset inspection report.
    
    Args:
        df: Input dataframe
        dataset_name: Name for reporting
        
    Returns:
        Dictionary containing dataset statistics
    """
```

### `src.data.preprocessing`

#### Classes

##### `BBBPreprocessor`

Specialized preprocessing for Blood-Brain Barrier dataset.

```python
class BBBPreprocessor:
    """Blood-Brain Barrier dataset preprocessing pipeline."""
    
    def __init__(self, mol2vec_model_path: Optional[str] = None):
        """Initialize with optional Mol2vec model path."""
```

**Methods:**

```python
def validate_smiles(self, smiles_list: List[str]) -> Tuple[List[str], List[int]]:
    """
    Validate SMILES strings using RDKit.
    
    Args:
        smiles_list: List of SMILES strings
        
    Returns:
        Tuple of (valid_smiles, valid_indices)
    """

def compute_molecular_descriptors(self, smiles_list: List[str]) -> np.ndarray:
    """
    Compute RDKit molecular descriptors.
    
    Args:
        smiles_list: List of valid SMILES strings
        
    Returns:
        Array of molecular descriptors
    """

def compute_mol2vec_embeddings(self, smiles_list: List[str]) -> np.ndarray:
    """
    Compute Mol2vec embeddings for molecules.
    
    Args:
        smiles_list: List of SMILES strings
        
    Returns:
        Array of Mol2vec embeddings (n_molecules, 300)
    """

def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Complete preprocessing pipeline.
    
    Args:
        df: Raw BBB dataframe
        
    Returns:
        Tuple of (features, targets)
    """
```

##### `BreastCancerPreprocessor`

Preprocessing for Breast Cancer dataset.

```python
class BreastCancerPreprocessor:
    """Breast Cancer dataset preprocessing pipeline."""
    
    def __init__(self):
        """Initialize preprocessor."""
```

**Methods:**

```python
def encode_target(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Encode target variable (M->1, B->0).
    
    Args:
        df: Input dataframe
        target_col: Name of target column
        
    Returns:
        Dataframe with encoded target
    """

def scale_features(self, X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scale features using StandardScaler.
    
    Args:
        X_train: Training features
        X_test: Test features
        
    Returns:
        Tuple of (scaled_X_train, scaled_X_test)
    """

def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Complete preprocessing pipeline.
    
    Args:
        df: Raw Breast Cancer dataframe
        
    Returns:
        Tuple of (features, targets)
    """
```

## Active Learning Module

### `src.active_learning.strategies`

#### Abstract Base Class

```python
class SamplingStrategy(ABC):
    """Abstract base class for active learning sampling strategies."""
    
    @abstractmethod
    def query(self, clf, X_pool: np.ndarray, n_samples: int) -> np.ndarray:
        """
        Query most informative samples.
        
        Args:
            clf: Trained classifier
            X_pool: Pool of unlabeled samples
            n_samples: Number of samples to select
            
        Returns:
            Array of selected sample indices
        """
```

#### Sampling Strategy Classes

##### `UncertaintySampling`

```python
class UncertaintySampling(SamplingStrategy):
    """Uncertainty-based sampling for single classifiers."""
    
    def __init__(self, method: str = "least_confidence"):
        """
        Initialize uncertainty sampling.
        
        Args:
            method: Uncertainty measure ('least_confidence', 'margin', 'entropy')
        """

    def query(self, clf, X_pool: np.ndarray, n_samples: int) -> np.ndarray:
        """Query samples with highest uncertainty."""
```

##### `QueryByCommittee`

```python
class QueryByCommittee(SamplingStrategy):
    """Query-by-Committee sampling strategy."""
    
    def __init__(self, committee_size: int = 5):
        """
        Initialize QBC sampling.
        
        Args:
            committee_size: Number of committee members
        """

    def query(self, committee: List, X_pool: np.ndarray, n_samples: int) -> np.ndarray:
        """Query samples with maximum committee disagreement."""
```

#### Functions

```python
def least_confidence_sampling(clf, X_pool: np.ndarray, n_samples: int) -> np.ndarray:
    """
    Select samples with lowest prediction confidence.
    
    Args:
        clf: Trained classifier with predict_proba method
        X_pool: Pool of unlabeled samples
        n_samples: Number of samples to query
        
    Returns:
        Indices of selected samples
    """

def qbc_vote_entropy_sampling(committee: List, X_pool: np.ndarray, n_samples: int) -> np.ndarray:
    """
    Select samples with maximum vote entropy among committee.
    
    Args:
        committee: List of trained classifiers
        X_pool: Pool of unlabeled samples
        n_samples: Number of samples to query
        
    Returns:
        Indices of selected samples
    """

def random_sampling(X_pool: np.ndarray, n_samples: int, random_state: int = None) -> np.ndarray:
    """
    Random baseline sampling strategy.
    
    Args:
        X_pool: Pool of unlabeled samples
        n_samples: Number of samples to query
        random_state: Random seed
        
    Returns:
        Indices of randomly selected samples
    """
```

### `src.active_learning.learners`

#### Classes

##### `ActiveLearner`

```python
class ActiveLearner(ABC):
    """Abstract base class for active learning algorithms."""
    
    def __init__(self, strategy: SamplingStrategy, batch_size: int = 10):
        """
        Initialize active learner.
        
        Args:
            strategy: Sampling strategy instance
            batch_size: Number of samples to query per iteration
        """

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the model on labeled data."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data."""

    @abstractmethod
    def query(self, X_pool: np.ndarray) -> np.ndarray:
        """Query most informative samples from pool."""
```

##### `RandomForestAL`

```python
class RandomForestAL(ActiveLearner):
    """Random Forest with active learning."""
    
    def __init__(self, strategy: SamplingStrategy, **rf_params):
        """
        Initialize RF active learner.
        
        Args:
            strategy: Sampling strategy
            **rf_params: RandomForestClassifier parameters
        """

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit Random Forest model."""

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using Random Forest."""

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""

    def query(self, X_pool: np.ndarray) -> np.ndarray:
        """Query using uncertainty sampling."""
```

##### `QueryByCommitteeAL`

```python
class QueryByCommitteeAL(ActiveLearner):
    """Query-by-Committee active learner."""
    
    def __init__(self, committee_specs: List[Dict], strategy: SamplingStrategy):
        """
        Initialize QBC active learner.
        
        Args:
            committee_specs: List of classifier specifications
            strategy: Sampling strategy
        """

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit all committee members."""

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Majority vote prediction."""

    def query(self, X_pool: np.ndarray) -> np.ndarray:
        """Query using committee disagreement."""
```

#### Factory Functions

```python
def rf_factory(n_estimators: int = 100, random_state: int = 42, **kwargs) -> RandomForestClassifier:
    """
    Create Random Forest classifier with standard parameters.
    
    Args:
        n_estimators: Number of trees
        random_state: Random seed
        **kwargs: Additional parameters
        
    Returns:
        Configured RandomForestClassifier
    """

def base_learner_factory(learner_type: str, **kwargs):
    """
    Create base learner for committee.
    
    Args:
        learner_type: Type of learner ('rf', 'et', 'gb', 'lr', 'knn')
        **kwargs: Learner-specific parameters
        
    Returns:
        Configured classifier instance
    """
```

### `src.active_learning.experiments`

#### Classes

##### `ALExperiment`

```python
class ALExperiment:
    """Active learning experiment orchestration."""
    
    def __init__(self, strategy: str, sampling_method: str, 
                 batch_size: int = 10, n_runs: int = 10,
                 max_queries: int = -1, stop_ratio: float = 1.0):
        """
        Initialize AL experiment.
        
        Args:
            strategy: AL strategy ('rf' or 'qbc')
            sampling_method: Initial sampling ('first_5', 'stratified')
            batch_size: Samples per iteration
            n_runs: Number of experimental runs
            max_queries: Maximum queries (-1 for unlimited)
            stop_ratio: Stop when this proportion of pool is labeled
        """
```

**Methods:**

```python
def run_experiment(self, X_train: np.ndarray, y_train: np.ndarray,
                  X_test: np.ndarray, y_test: np.ndarray,
                  dataset_name: str) -> Dict[str, Any]:
    """
    Run complete active learning experiment.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets
        dataset_name: Name of dataset for logging
        
    Returns:
        Dictionary containing experiment results
    """

def evaluate_iteration(self, model, X_test: np.ndarray, y_test: np.ndarray,
                      iteration: int, n_labeled: int) -> Dict[str, float]:
    """
    Evaluate model performance at current iteration.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
        iteration: Current iteration number
        n_labeled: Number of labeled samples
        
    Returns:
        Dictionary of evaluation metrics
    """

def save_results(self, results: Dict, output_dir: str, 
                experiment_id: str) -> str:
    """
    Save experiment results to file.
    
    Args:
        results: Experiment results dictionary
        output_dir: Output directory path
        experiment_id: Unique experiment identifier
        
    Returns:
        Path to saved results file
    """
```

#### Functions

```python
def run_active_learning_experiment(X_train: np.ndarray, y_train: np.ndarray,
                                 X_test: np.ndarray, y_test: np.ndarray,
                                 strategy: str, sampling_method: str,
                                 config: Dict, dataset: str,
                                 run_id: int = 0) -> Dict[str, Any]:
    """
    Run single active learning experiment.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets
        strategy: AL strategy ('rf' or 'qbc')
        sampling_method: Initial sampling method
        config: Configuration dictionary
        dataset: Dataset name
        run_id: Run identifier
        
    Returns:
        Experiment results dictionary
    """

def aggregate_experiment_results(results_list: List[Dict]) -> Dict[str, Any]:
    """
    Aggregate results from multiple experimental runs.
    
    Args:
        results_list: List of individual experiment results
        
    Returns:
        Aggregated results with means and standard deviations
    """
```

## Features Module

### `src.features.molecular`

#### Classes

##### `MolecularFeaturizer`

```python
class MolecularFeaturizer:
    """Molecular featurization using RDKit and Mol2vec."""
    
    def __init__(self, mol2vec_model_path: Optional[str] = None,
                 descriptor_list: Optional[List[str]] = None):
        """
        Initialize molecular featurizer.
        
        Args:
            mol2vec_model_path: Path to Mol2vec model
            descriptor_list: List of RDKit descriptors to compute
        """
```

**Methods:**

```python
def compute_rdkit_descriptors(self, smiles_list: List[str]) -> np.ndarray:
    """
    Compute RDKit molecular descriptors.
    
    Args:
        smiles_list: List of SMILES strings
        
    Returns:
        Array of molecular descriptors
    """

def compute_mol2vec_embeddings(self, smiles_list: List[str]) -> np.ndarray:
    """
    Compute Mol2vec embeddings.
    
    Args:
        smiles_list: List of SMILES strings
        
    Returns:
        Array of Mol2vec embeddings
    """

def featurize(self, smiles_list: List[str]) -> np.ndarray:
    """
    Complete featurization pipeline.
    
    Args:
        smiles_list: List of SMILES strings
        
    Returns:
        Combined feature matrix
    """
```

#### Functions

```python
def mol2alt_sentence(mol, radius: int = 1) -> List[str]:
    """
    Convert molecule to sentence for Mol2vec training.
    
    Args:
        mol: RDKit molecule object
        radius: Radius for Morgan fingerprints
        
    Returns:
        List of molecular "words"
    """

def compute_mol2vec_embedding(smiles: str, model) -> np.ndarray:
    """
    Compute single molecule Mol2vec embedding.
    
    Args:
        smiles: SMILES string
        model: Trained Mol2vec model
        
    Returns:
        Mol2vec embedding vector
    """

def compute_descriptors(smiles_list: List[str], 
                       descriptor_names: List[str]) -> np.ndarray:
    """
    Compute specified RDKit descriptors for molecules.
    
    Args:
        smiles_list: List of SMILES strings
        descriptor_names: List of descriptor names
        
    Returns:
        Descriptor matrix
    """
```

## Evaluation Module

### `src.evaluation.metrics`

#### Classes

##### `ModelEvaluator`

```python
class ModelEvaluator:
    """Comprehensive model evaluation utilities."""
    
    def __init__(self, metrics: List[str] = None):
        """
        Initialize evaluator.
        
        Args:
            metrics: List of metrics to compute
        """
```

**Methods:**

```python
def evaluate_single_model(self, y_true: np.ndarray, y_pred: np.ndarray,
                         y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Evaluate single model performance.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Prediction probabilities
        
    Returns:
        Dictionary of evaluation metrics
    """

def evaluate_committee(self, committee: List, X_test: np.ndarray,
                      y_test: np.ndarray) -> Dict[str, Any]:
    """
    Evaluate committee performance.
    
    Args:
        committee: List of trained models
        X_test: Test features
        y_test: Test targets
        
    Returns:
        Committee evaluation results
    """

def calculate_confidence_intervals(self, scores: List[float],
                                 confidence: float = 0.95) -> Tuple[float, float]:
    """
    Calculate confidence intervals for scores.
    
    Args:
        scores: List of performance scores
        confidence: Confidence level
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
```

#### Functions

```python
def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray,
                  y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Evaluate model with standard metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Prediction probabilities
        
    Returns:
        Dictionary of metrics (MCC, F1, ROC AUC, etc.)
    """

def calculate_per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate per-class accuracy metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Per-class accuracy dictionary
    """

def compute_statistical_significance(scores1: List[float], 
                                   scores2: List[float]) -> Dict[str, float]:
    """
    Compare two sets of scores for statistical significance.
    
    Args:
        scores1: First set of scores
        scores2: Second set of scores
        
    Returns:
        Statistical test results
    """
```

### `src.evaluation.visualization`

#### Classes

##### `ResultVisualizer`

```python
class ResultVisualizer:
    """Comprehensive result visualization utilities."""
    
    def __init__(self, style: str = "seaborn", figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize visualizer.
        
        Args:
            style: Matplotlib style
            figsize: Default figure size
        """
```

**Methods:**

```python
def plot_learning_curves(self, results_df: pd.DataFrame, 
                        metrics: List[str] = None,
                        save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot active learning curves.
    
    Args:
        results_df: DataFrame with experiment results
        metrics: List of metrics to plot
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure object
    """

def plot_confusion_matrix_comparison(self, y_true: np.ndarray,
                                   y_pred_al: np.ndarray,
                                   y_pred_full: np.ndarray,
                                   save_path: Optional[str] = None) -> plt.Figure:
    """
    Compare confusion matrices side by side.
    
    Args:
        y_true: True labels
        y_pred_al: Active learning predictions
        y_pred_full: Full model predictions
        save_path: Optional save path
        
    Returns:
        Matplotlib figure object
    """

def plot_dmcc_evolution(self, results_df: pd.DataFrame,
                       save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot Delta MCC evolution over iterations.
    
    Args:
        results_df: DataFrame with DMCC values
        save_path: Optional save path
        
    Returns:
        Matplotlib figure object
    """

def create_performance_comparison_table(self, results_dict: Dict[str, Dict],
                                      save_path: Optional[str] = None) -> pd.DataFrame:
    """
    Create formatted performance comparison table.
    
    Args:
        results_dict: Dictionary of experiment results
        save_path: Optional save path for CSV
        
    Returns:
        Formatted DataFrame
    """
```

## Dimensionality Module

### `src.dimensionality.reduction`

#### Classes

##### `DimensionalityReducer`

```python
class DimensionalityReducer:
    """Comprehensive dimensionality reduction utilities."""
    
    def __init__(self):
        """Initialize dimensionality reducer."""
```

**Methods:**

```python
def apply_pca(self, X: np.ndarray, n_components: Union[int, float] = 0.9,
             return_transformer: bool = False) -> Union[np.ndarray, Tuple]:
    """
    Apply Principal Component Analysis.
    
    Args:
        X: Input data matrix
        n_components: Number of components or variance ratio
        return_transformer: Whether to return fitted transformer
        
    Returns:
        Transformed data or (transformed_data, transformer)
    """

def apply_tsne(self, X: np.ndarray, n_components: int = 2,
              perplexity: float = 30.0, random_state: int = 42) -> np.ndarray:
    """
    Apply t-SNE dimensionality reduction.
    
    Args:
        X: Input data matrix
        n_components: Number of output dimensions
        perplexity: t-SNE perplexity parameter
        random_state: Random seed
        
    Returns:
        t-SNE transformed data
    """

def apply_umap(self, X: np.ndarray, n_components: int = 2,
              n_neighbors: int = 15, min_dist: float = 0.1,
              optimize_params: bool = False) -> np.ndarray:
    """
    Apply UMAP dimensionality reduction.
    
    Args:
        X: Input data matrix
        n_components: Number of output dimensions
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter
        optimize_params: Whether to optimize hyperparameters
        
    Returns:
        UMAP transformed data
    """

def apply_lda(self, X: np.ndarray, y: np.ndarray,
             n_components: Optional[int] = None) -> np.ndarray:
    """
    Apply Linear Discriminant Analysis.
    
    Args:
        X: Input data matrix
        y: Target labels
        n_components: Number of components (max: n_classes-1)
        
    Returns:
        LDA transformed data
    """

def compare_methods(self, X: np.ndarray, y: np.ndarray,
                   methods: List[str] = None) -> Dict[str, np.ndarray]:
    """
    Compare multiple dimensionality reduction methods.
    
    Args:
        X: Input data matrix
        y: Target labels
        methods: List of methods to compare
        
    Returns:
        Dictionary of transformed data for each method
    """
```

## Utils Module

### `src.utils.config`

#### Classes

##### `ConfigManager`

```python
class ConfigManager:
    """Configuration management utilities."""
    
    def __init__(self, config_dir: str = "configs"):
        """
        Initialize config manager.
        
        Args:
            config_dir: Directory containing configuration files
        """
```

**Methods:**

```python
def load_config(self, config_name: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_name: Name of config file (without .yaml)
        
    Returns:
        Configuration dictionary
    """

def save_config(self, config: Dict[str, Any], config_name: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_name: Name of config file
    """

def merge_configs(self, *configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple configuration dictionaries.
    
    Args:
        *configs: Variable number of config dictionaries
        
    Returns:
        Merged configuration
    """
```

### `src.utils.helpers`

#### Functions

```python
def set_random_seeds(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """

def create_experiment_id(strategy: str, dataset: str, timestamp: bool = True) -> str:
    """
    Create unique experiment identifier.
    
    Args:
        strategy: AL strategy name
        dataset: Dataset name
        timestamp: Whether to include timestamp
        
    Returns:
        Unique experiment ID
    """

def ensure_dir_exists(dir_path: str):
    """
    Ensure directory exists, create if necessary.
    
    Args:
        dir_path: Directory path to check/create
    """

def save_pickle(obj: Any, file_path: str):
    """
    Save object to pickle file.
    
    Args:
        obj: Object to save
        file_path: Output file path
    """

def load_pickle(file_path: str) -> Any:
    """
    Load object from pickle file.
    
    Args:
        file_path: Input file path
        
    Returns:
        Loaded object
    """
```

### `src.utils.parallel`

#### Functions

```python
def parallel_experiment_runner(experiment_configs: List[Dict],
                             n_jobs: int = -1) -> List[Dict]:
    """
    Run multiple experiments in parallel.
    
    Args:
        experiment_configs: List of experiment configurations
        n_jobs: Number of parallel jobs (-1 for all CPUs)
        
    Returns:
        List of experiment results
    """

def parallel_cross_validation(model, X: np.ndarray, y: np.ndarray,
                            cv_folds: int = 5, n_jobs: int = -1) -> List[float]:
    """
    Parallel cross-validation scoring.
    
    Args:
        model: Model to evaluate
        X: Feature matrix
        y: Target vector
        cv_folds: Number of CV folds
        n_jobs: Number of parallel jobs
        
    Returns:
        List of CV scores
    """
```

## Usage Examples

### Complete Workflow Example

```python
# 1. Load and preprocess data
from src.data.loader import DatasetLoader
from src.data.preprocessing import BreastCancerPreprocessor

loader = DatasetLoader()
df, target_col = loader.load_breast_cancer_dataset()

preprocessor = BreastCancerPreprocessor()
X, y = preprocessor.fit_transform(df)
X_train, X_test, y_train, y_test = loader.prepare_train_test_split(X, y)

# 2. Set up active learning experiment
from src.active_learning.experiments import ALExperiment

experiment = ALExperiment(
    strategy='qbc',
    sampling_method='first_5',
    batch_size=10,
    n_runs=5
)

# 3. Run experiment
results = experiment.run_experiment(
    X_train, y_train, X_test, y_test, 
    dataset_name='breast_cancer'
)

# 4. Evaluate and visualize results
from src.evaluation.visualization import ResultVisualizer
from src.evaluation.metrics import ModelEvaluator

visualizer = ResultVisualizer()
evaluator = ModelEvaluator()

# Plot learning curves
fig = visualizer.plot_learning_curves(results['learning_curves'])

# Evaluate final performance
final_metrics = evaluator.evaluate_single_model(
    results['y_true'], results['y_pred'], results['y_proba']
)

print(f"Final MCC: {final_metrics['mcc']:.4f}")
```

### Molecular Featurization Example

```python
from src.features.molecular import MolecularFeaturizer
from src.data.loader import DatasetLoader

# Load BBB dataset
loader = DatasetLoader()
df, target_col = loader.load_bbb_dataset()

# Initialize featurizer
featurizer = MolecularFeaturizer(
    mol2vec_model_path='data/external/mol2vec_model_300dim.pkl'
)

# Extract SMILES and featurize
smiles_list = df['smiles'].tolist()
features = featurizer.featurize(smiles_list)

print(f"Generated {features.shape[1]} features for {features.shape[0]} molecules")
```

### Custom Active Learning Strategy

```python
from src.active_learning.strategies import SamplingStrategy
from src.active_learning.learners import ActiveLearner
import numpy as np

class CustomUncertainty(SamplingStrategy):
    """Custom uncertainty sampling strategy."""
    
    def query(self, clf, X_pool, n_samples):
        # Get prediction probabilities
        probabilities = clf.predict_proba(X_pool)
        
        # Calculate custom uncertainty measure
        uncertainty = np.sum(probabilities * np.log(probabilities + 1e-10), axis=1)
        
        # Select most uncertain samples
        uncertain_indices = np.argsort(uncertainty)[-n_samples:]
        return uncertain_indices

# Use custom strategy
from src.active_learning.learners import RandomForestAL

strategy = CustomUncertainty()
al_model = RandomForestAL(strategy=strategy, n_estimators=100)
```

## Configuration

### Default Configuration Structure

#### Data Configuration (`configs/data_config.yaml`)
```yaml
datasets:
  bbb:
    name: "Blood-Brain Barrier Penetration"
    path: "data/raw/BBBP.xlsx"
    features:
      molecular_descriptors: true
      mol2vec: true
      mol2vec_model: "model_300dim.pkl"
  
  breast_cancer:
    name: "Breast Cancer"
    path: "data/raw/breast-cancer.csv"
    target_column: "diagnosis"
```

#### Experiment Configuration (`configs/experiment_config.yaml`)
```yaml
active_learning:
  strategies:
    - random_forest
    - query_by_committee
  
  initial_samples:
    - first_5
    - stratified_5
  
  batch_size: 10
  n_runs: 10
  stop_ratio: 1.0
```

#### Model Configuration (`configs/model_config.yaml`)
```yaml
random_forest:
  n_estimators: 100
  random_state: 42

qbc:
  n_committee_members: 5
  base_learners:
    - RandomForest
    - ExtraTrees
    - GradientBoosting
    - LogisticRegression
    - KNN
```

### Environment Variables

```bash
# Data paths
export DATA_DIR="data/raw"
export PROCESSED_DATA_DIR="data/processed"
export RESULTS_DIR="results"

# Model paths
export MOL2VEC_MODEL_PATH="data/external/mol2vec_model_300dim.pkl"

# Computation
export N_JOBS="-1"  # Use all available CPUs
export RANDOM_STATE="42"
```

## Error Handling and Logging

All modules implement comprehensive error handling and logging:

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Example error handling in data loading
try:
    df, target_col = loader.load_breast_cancer_dataset()
except FileNotFoundError as e:
    logging.error(f"Dataset file not found: {e}")
    raise
except ValueError as e:
    logging.error(f"Invalid dataset format: {e}")
    raise
```

## Performance Considerations

- **Memory Usage**: Large datasets may require chunked processing
- **Parallel Processing**: Use `n_jobs=-1` for CPU-intensive operations
- **Caching**: Results are cached to avoid recomputation
- **GPU Support**: TensorFlow operations support GPU acceleration

## Testing

All modules include comprehensive unit tests:

```python
# Example test structure
import pytest
from src.data.loader import DatasetLoader

def test_dataset_loader():
    loader = DatasetLoader()
    df, target_col = loader.load_breast_cancer_dataset()
    
    assert len(df) == 569
    assert target_col == 'diagnosis'
    assert df[target_col].nunique() == 2

# Run tests
pytest tests/test_data.py -v
```

## Production Deployment

### Docker API

The entire framework is containerized for production deployment:

```bash
# Build production container
docker build -t biomedical-al:latest .

# Deploy with Docker Compose
docker-compose up -d

# API Health Check
curl http://localhost:8501/_stcore/health

# Container Management
docker-compose logs -f biomedical-al  # View logs
docker-compose scale biomedical-al=3  # Scale instances
```

### Streamlit Web API

Interactive web application provides GUI access to all functionality:

```python
# Main application entry point
streamlit run app.py

# Available endpoints:
# http://localhost:8501/         # Home dashboard
# http://localhost:8501/data     # Data exploration
# http://localhost:8501/demo     # Active learning demo
# http://localhost:8501/results  # Results comparison
# http://localhost:8501/predict  # Model predictions
```

### REST API Integration

For programmatic access, the framework can be extended with REST endpoints:

```python
from flask import Flask, jsonify, request
from src.active_learning.experiments import ALExperiment

app = Flask(__name__)

@app.route('/api/experiment', methods=['POST'])
def run_experiment():
    """Run active learning experiment via REST API."""
    config = request.json
    experiment = ALExperiment(**config)
    results = experiment.run_experiment(
        config['X_train'], config['y_train'],
        config['X_test'], config['y_test'],
        config['dataset_name']
    )
    return jsonify(results)

@app.route('/api/health', methods=['GET'])
def health_check():
    """API health check endpoint."""
    return jsonify({'status': 'healthy', 'version': '1.0.0'})
```

### Cloud Deployment

#### AWS ECS
```yaml
# Deploy to AWS ECS
task_definition:
  family: biomedical-al
  cpu: 1024
  memory: 4096
  container_definitions:
    - name: biomedical-al
      image: your-registry/biomedical-al:latest
      ports: [{containerPort: 8501}]
      healthCheck:
        command: ["CMD-SHELL", "curl -f http://localhost:8501/_stcore/health"]
```

#### Google Cloud Run
```bash
# Deploy to Google Cloud Run
gcloud run deploy biomedical-al \
  --image gcr.io/PROJECT_ID/biomedical-al \
  --platform managed \
  --memory 4Gi --cpu 2 --port 8501
```

### Monitoring and Observability

```python
# Integration with monitoring systems
import prometheus_client
from loguru import logger

# Metrics collection
EXPERIMENT_COUNTER = prometheus_client.Counter(
    'biomedical_al_experiments_total',
    'Total number of experiments run'
)

REQUEST_DURATION = prometheus_client.Histogram(
    'biomedical_al_request_duration_seconds',
    'Time spent processing requests'
)

# Structured logging
logger.add("logs/biomedical_al.log", rotation="500 MB", retention="10 days")
logger.info("Experiment started", dataset="breast_cancer", strategy="qbc")
```

### Security and Authentication

```python
# API authentication example
from functools import wraps
import jwt

def require_auth(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token or not validate_token(token):
            return jsonify({'error': 'Unauthorized'}), 401
        return f(*args, **kwargs)
    return decorated_function

@app.route('/api/protected-experiment', methods=['POST'])
@require_auth
def protected_experiment():
    """Authenticated experiment endpoint."""
    return run_experiment()
```

### Performance Optimization

```python
# Caching for repeated requests
from functools import lru_cache
import redis

# Redis cache for expensive operations
redis_client = redis.Redis(host='localhost', port=6379, db=0)

@lru_cache(maxsize=128)
def cached_feature_extraction(smiles_hash):
    """Cache molecular feature extraction."""
    cached_result = redis_client.get(f"features:{smiles_hash}")
    if cached_result:
        return pickle.loads(cached_result)
    
    # Compute features if not cached
    features = compute_features(smiles_hash)
    redis_client.setex(f"features:{smiles_hash}", 3600, pickle.dumps(features))
    return features
```

This API documentation provides comprehensive coverage of all modules and their functionality, from development to production deployment. Each class and function includes detailed parameter descriptions, return values, and usage examples to facilitate easy integration and extension of the biomedical active learning framework in both research and production environments.