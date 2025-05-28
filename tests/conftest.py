import pytest
import numpy as np
import pandas as pd
import tempfile
import shutil
import os
from pathlib import Path
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


@pytest.fixture(scope="session")
def temp_dir():
    """Create a temporary directory for testing that gets cleaned up after session."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture(scope="session")
def project_root():
    """Get the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def config_dir(project_root):
    """Path to configuration directory."""
    return project_root / "configs"


@pytest.fixture(scope="session")
def data_dir(project_root):
    """Path to data directory."""
    return project_root / "data"


@pytest.fixture
def mock_bbb_smiles():
    """Mock BBB dataset with valid SMILES strings."""
    smiles_data = [
        "CCO",  # ethanol
        "CC(C)O",  # isopropanol  
        "C1=CC=CC=C1",  # benzene
        "CCN(CC)CC",  # triethylamine
        "CC(=O)O",  # acetic acid
        "CCCCO",  # butanol
        "C1=CC=C(C=C1)O",  # phenol
        "CCN",  # ethylamine
        "CC(C)(C)O",  # tert-butanol
        "C1=CC=C(C=C1)N"  # aniline
    ]
    
    df = pd.DataFrame({
        'smiles': smiles_data,
        'p_np': np.random.choice([0, 1], size=len(smiles_data)),
        'name': [f'compound_{i}' for i in range(len(smiles_data))]
    })
    return df


@pytest.fixture
def mock_breast_cancer_data():
    """Mock breast cancer dataset with realistic features."""
    np.random.seed(42)
    
    # Generate synthetic data similar to breast cancer dataset
    X, y = make_classification(
        n_samples=100,
        n_features=30,
        n_informative=15,
        n_redundant=5,
        n_clusters_per_class=1,
        random_state=42
    )
    
    # Create realistic feature names
    feature_names = [
        'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
        'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
        'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
        'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se', 'fractal_dimension_se',
        'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
        'compactness_worst', 'concavity_worst', 'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst'
    ]
    
    df = pd.DataFrame(X, columns=feature_names)
    df['diagnosis'] = y
    
    return df


@pytest.fixture
def mock_bbb_features():
    """Mock processed BBB features for testing."""
    np.random.seed(42)
    n_samples = 50
    n_features = 100  # Simulating molecular descriptors + mol2vec
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.choice([0, 1], size=n_samples)
    
    return X, y


@pytest.fixture
def mock_bc_features():
    """Mock processed breast cancer features for testing."""
    np.random.seed(42)
    X, y = make_classification(
        n_samples=50,
        n_features=30,
        n_informative=15,
        n_redundant=5,
        random_state=42
    )
    return X, y


@pytest.fixture
def train_test_split_data(mock_bc_features):
    """Pre-split training and testing data."""
    X, y = mock_bc_features
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }


@pytest.fixture
def sample_config():
    """Sample configuration dictionary for testing."""
    return {
        'active_learning': {
            'strategies': ['random_forest', 'query_by_committee'],
            'initial_samples': ['first_5', 'stratified_5'],
            'batch_size': 5,
            'n_runs': 3,
            'stop_ratio': 1.0
        },
        'random_forest': {
            'n_estimators': 10,  # Small for fast testing
            'random_state': 42
        },
        'qbc': {
            'n_committee_members': 3,  # Smaller committee for testing
            'base_learners': ['RandomForest', 'ExtraTrees', 'LogisticRegression']
        }
    }


@pytest.fixture
def mock_experiment_results():
    """Mock active learning experiment results."""
    n_iterations = 10
    results = {
        'iterations': list(range(n_iterations)),
        'n_samples': [5 + i * 5 for i in range(n_iterations)],
        'mcc_scores': np.random.uniform(0.5, 0.95, n_iterations),
        'f1_scores': np.random.uniform(0.7, 0.98, n_iterations),
        'roc_auc_scores': np.random.uniform(0.8, 0.99, n_iterations),
        'query_indices': [list(range(i*5, (i+1)*5)) for i in range(n_iterations)],
        'training_time': np.random.uniform(0.1, 2.0, n_iterations)
    }
    return results


@pytest.fixture
def mock_mol2vec_model():
    """Mock Mol2vec model for testing."""
    class MockMol2vecModel:
        def __init__(self):
            self.vector_size = 300
            
        def infer_vector(self, sentence):
            # Return a random 300-dimensional vector
            np.random.seed(hash(str(sentence)) % 2**32)
            return np.random.randn(self.vector_size)
    
    return MockMol2vecModel()


@pytest.fixture
def sample_smiles():
    """Sample SMILES strings for testing."""
    return [
        "CCO",  # ethanol - valid
        "CC(C)O",  # isopropanol - valid
        "InvalidSMILES",  # invalid
        "C1=CC=CC=C1",  # benzene - valid
        "",  # empty - invalid
        "CC(=O)O"  # acetic acid - valid
    ]


@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch, temp_dir):
    """Set up test environment variables and paths."""
    # Set environment variables for testing
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).parent.parent / "src"))
    monkeypatch.setenv("TEST_MODE", "1")
    
    # Change to temp directory for tests that create files
    original_cwd = os.getcwd()
    os.chdir(temp_dir)
    
    yield
    
    # Restore original directory
    os.chdir(original_cwd)


@pytest.fixture
def suppress_plots(monkeypatch):
    """Suppress matplotlib plots during testing."""
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    
    # Mock plt.show() to prevent plots from displaying
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, 'show', lambda: None)


# Utility functions for tests
def assert_dataframe_equal(df1, df2, check_dtype=True):
    """Custom assertion for DataFrame equality with better error messages."""
    pd.testing.assert_frame_equal(df1, df2, check_dtype=check_dtype)


def assert_array_shape(arr, expected_shape):
    """Assert array has expected shape."""
    assert arr.shape == expected_shape, f"Expected shape {expected_shape}, got {arr.shape}"


def assert_valid_predictions(predictions, n_samples):
    """Assert predictions are valid (correct length, binary values)."""
    assert len(predictions) == n_samples
    assert set(predictions).issubset({0, 1}), "Predictions should be binary (0 or 1)"


# Performance testing utilities
@pytest.fixture
def performance_monitor():
    """Monitor memory and time usage during tests."""
    import psutil
    import time
    
    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.start_memory = None
            
        def start(self):
            self.start_time = time.time()
            self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
        def stop(self):
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            return {
                'execution_time': end_time - self.start_time,
                'memory_used': end_memory - self.start_memory
            }
    
    return PerformanceMonitor()