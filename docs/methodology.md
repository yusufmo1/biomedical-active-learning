# Methodology: Biomedical Active Learning

## Overview

This document provides a comprehensive explanation of the active learning methodology employed in our biomedical machine learning project. Our approach demonstrates that intelligent sample selection can match or exceed the performance of models trained on entire datasets, with particular success on the Breast Cancer dataset where Query-by-Committee with 5 initial samples achieved superior performance (MCC 0.942 vs 0.9253).

## Table of Contents

1. [Active Learning Theory](#active-learning-theory)
2. [Dataset Descriptions](#dataset-descriptions)
3. [Feature Engineering](#feature-engineering)
4. [Active Learning Strategies](#active-learning-strategies)
5. [Experimental Design](#experimental-design)
6. [Evaluation Metrics](#evaluation-metrics)
7. [Statistical Analysis](#statistical-analysis)

## Active Learning Theory

### Foundation

Active learning is a machine learning paradigm where the algorithm can interactively query a human annotator (or oracle) to obtain labels for new data points. The key insight is that a learning algorithm can achieve better performance with fewer labeled examples if it can choose which examples to learn from.

### Core Principles

1. **Pool-based Sampling**: We maintain a pool of unlabeled instances and select the most informative samples for labeling
2. **Query Strategy**: The algorithm uses uncertainty measures to identify samples that would most improve the model
3. **Iterative Learning**: The process continues iteratively, adding new labeled samples and retraining the model

### Theoretical Motivation

In biomedical domains, data annotation is expensive and time-consuming. Active learning addresses this by:
- Reducing annotation costs while maintaining performance
- Identifying critical samples that maximize information gain
- Handling class imbalance through strategic sampling

## Dataset Descriptions

### Blood-Brain Barrier Penetration (BBB) Dataset

**Source**: BBBP dataset from MoleculeNet
**Size**: 2,790 molecules (2,232 after preprocessing)
**Task**: Binary classification (BBB+ vs BBB-)
**Format**: SMILES molecular representations

#### Characteristics:
- **Class Distribution**: Moderately imbalanced
- **Feature Space**: High-dimensional molecular descriptors
- **Challenge**: Complex chemical space with nonlinear relationships
- **Domain**: Drug discovery and pharmaceutical research

#### Preprocessing Pipeline:
```python
1. SMILES Validation
   - Input: Raw SMILES strings
   - Validation: RDKit Chem.MolFromSmiles()
   - Filter: Remove invalid molecular structures
   
2. Molecular Descriptor Calculation
   - Tool: RDKit MolecularDescriptorCalculator
   - Features: 200+ physicochemical properties
   - Cleaning: Remove constant and highly correlated features
   
3. Mol2vec Embeddings
   - Model: Pre-trained 300-dimensional embeddings
   - Method: Morgan fingerprint-based training
   - Integration: Concatenate with RDKit descriptors
   
4. Dimensionality Reduction
   - Method: Principal Component Analysis (PCA)
   - Target: Retain 90% of variance
   - Result: ~50-80 principal components
```

### Breast Cancer Wisconsin Dataset

**Source**: UCI Machine Learning Repository
**Size**: 569 samples
**Task**: Binary classification (Malignant vs Benign)
**Features**: 30 real-valued features

#### Characteristics:
- **Class Distribution**: 
  - Malignant: 212 samples (37.3%)
  - Benign: 357 samples (62.7%)
- **Feature Types**: Cell nucleus measurements
- **Quality**: High-quality, well-curated dataset
- **Domain**: Medical diagnosis and cancer screening

#### Feature Categories:
1. **Radius**: Mean distances from center to perimeter points
2. **Texture**: Standard deviation of gray-scale values
3. **Perimeter**: Size measurements
4. **Area**: Cell nucleus area measurements
5. **Smoothness**: Local variation in radius lengths
6. **Compactness**: (perimeter² / area) - 1.0
7. **Concavity**: Severity of concave portions
8. **Concave points**: Number of concave portions
9. **Symmetry**: Cell nucleus symmetry
10. **Fractal dimension**: "Coastline approximation" - 1

Each feature includes: **mean**, **standard error**, and **worst** (largest) values.

#### Preprocessing Pipeline:
```python
1. Target Encoding
   - Malignant (M) → 1
   - Benign (B) → 0
   
2. Feature Scaling
   - Method: StandardScaler
   - Apply: Fit on training, transform on test
   
3. Train/Test Split
   - Ratio: 80/20 (455 train, 114 test)
   - Method: Stratified split
   - Random state: 42 (reproducibility)
```

## Feature Engineering

### Molecular Featurization (BBB Dataset)

#### 1. SMILES Processing
```python
def validate_smiles(smiles_list):
    """Validate SMILES strings using RDKit"""
    valid_molecules = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            valid_molecules.append(smiles)
    return valid_molecules
```

#### 2. RDKit Descriptor Calculation
```python
from rdkit.Chem import Descriptors

descriptor_names = [desc[0] for desc in Descriptors._descList]
# Results in 200+ molecular descriptors including:
# - MolWt (Molecular Weight)
# - LogP (Lipophilicity)
# - NumHDonors/NumHAcceptors
# - TPSA (Topological Polar Surface Area)
# - NumRotatableBonds
# - And many more...
```

#### 3. Mol2vec Embeddings
```python
from gensim.models import Word2Vec

# Convert molecules to "sentences" using Morgan fingerprints
def mol2alt_sentence(mol, radius):
    """Convert molecule to sentence for Mol2vec"""
    fingerprints = []
    for atom_idx in range(mol.GetNumAtoms()):
        env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_idx)
        fingerprints.append(env)
    return fingerprints

# Use pre-trained 300-dimensional model
model = Word2Vec.load('mol2vec_model_300dim.pkl')
embeddings = compute_mol2vec_embedding(molecules, model)
```

#### 4. Feature Integration
```python
# Combine RDKit descriptors with Mol2vec embeddings
combined_features = np.concatenate([
    rdkit_descriptors,      # ~200 features
    mol2vec_embeddings      # 300 features
], axis=1)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=0.9)  # Retain 90% variance
final_features = pca.fit_transform(combined_features)
```

### Clinical Feature Processing (Breast Cancer Dataset)

The Breast Cancer dataset requires minimal preprocessing due to its high quality:

```python
# Simple but effective preprocessing
def preprocess_breast_cancer(data):
    # Encode target variable
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    
    # Separate features and target
    X = data.drop(['diagnosis'], axis=1)
    y = data['diagnosis']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y
```

## Active Learning Strategies

### 1. Random Forest with Uncertainty Sampling

#### Algorithm
```python
def least_confidence_sampling(clf, X_pool, n_samples):
    """Select samples with lowest prediction confidence"""
    probabilities = clf.predict_proba(X_pool)
    confidence = np.max(probabilities, axis=1)
    uncertainty = 1 - confidence
    
    # Select top uncertain samples
    uncertain_indices = np.argsort(uncertainty)[-n_samples:]
    return uncertain_indices
```

#### Configuration
- **Base Model**: RandomForestClassifier
- **Parameters**: 
  - n_estimators=100
  - random_state=42
  - class_weight='balanced'
- **Query Strategy**: Least confidence
- **Batch Size**: 20 samples (BBB), 10 samples (BC)

#### Theoretical Justification
Random Forest uncertainty sampling works by:
1. **Ensemble Variance**: RF naturally provides uncertainty estimates through tree voting
2. **Low Confidence**: Samples with ~0.5 probability are most uncertain
3. **Decision Boundaries**: Focuses sampling near class boundaries

### 2. Query-by-Committee (QBC)

#### Algorithm
```python
def qbc_vote_entropy_sampling(committee, X_pool, n_samples):
    """Select samples with maximum committee disagreement"""
    vote_entropy = []
    
    for x in X_pool:
        # Get predictions from all committee members
        votes = [clf.predict([x])[0] for clf in committee]
        
        # Calculate vote entropy
        vote_counts = np.bincount(votes, minlength=2)
        probabilities = vote_counts / len(committee)
        entropy = -np.sum(p * np.log2(p) for p in probabilities if p > 0)
        vote_entropy.append(entropy)
    
    # Select samples with highest disagreement
    disagreement_indices = np.argsort(vote_entropy)[-n_samples:]
    return disagreement_indices
```

#### Committee Composition
```python
committee = [
    RandomForestClassifier(n_estimators=100, random_state=42),
    ExtraTreesClassifier(n_estimators=100, random_state=42),
    GradientBoostingClassifier(n_estimators=100, random_state=42),
    LogisticRegression(random_state=42),
    KNeighborsClassifier(n_neighbors=5)
]
```

#### Theoretical Justification
QBC works through:
1. **Diverse Perspectives**: Different algorithms capture different aspects of data
2. **Disagreement Measure**: High entropy indicates uncertain regions
3. **Complementary Strengths**: Ensemble covers various learning biases

### Initialization Strategies

#### First-5 Sampling
```python
def first_5_sampling(X_train, y_train):
    """Select first 5 samples from dataset"""
    return np.arange(5), np.arange(5, len(X_train))
```

#### Stratified Sampling
```python
def stratified_sampling(X_train, y_train, n_samples=5):
    """Maintain class distribution in initial samples"""
    stratified_indices = []
    for class_label in np.unique(y_train):
        class_indices = np.where(y_train == class_label)[0]
        n_class_samples = max(1, int(n_samples * np.mean(y_train == class_label)))
        selected = np.random.choice(class_indices, n_class_samples, replace=False)
        stratified_indices.extend(selected)
    
    return stratified_indices[:n_samples]
```

## Experimental Design

### Experimental Protocol

1. **Data Preparation**
   - Load and preprocess datasets
   - Apply train/test split (80/20)
   - Initialize feature scaling/encoding

2. **Active Learning Loop**
   ```python
   for iteration in range(max_iterations):
       # Train model on current labeled set
       model.fit(X_labeled, y_labeled)
       
       # Query most informative samples
       query_indices = query_strategy(model, X_pool, batch_size)
       
       # Add to labeled set
       X_labeled = np.vstack([X_labeled, X_pool[query_indices]])
       y_labeled = np.hstack([y_labeled, y_pool[query_indices]])
       
       # Remove from pool
       X_pool = np.delete(X_pool, query_indices, axis=0)
       y_pool = np.delete(y_pool, query_indices)
       
       # Evaluate current model
       metrics = evaluate_model(model, X_test, y_test)
       results.append(metrics)
   ```

3. **Multiple Runs**
   - Repeat each experiment 10 times
   - Use different random seeds
   - Calculate mean and standard deviation

### Configuration Parameters

```yaml
# Active Learning Configuration
active_learning:
  initial_samples: [5]
  sampling_methods: ['first_5', 'stratified']
  batch_size: 
    bbb: 20
    breast_cancer: 10
  max_queries: -1  # Continue until pool exhausted
  stop_ratio: 1.0
  n_runs: 10

# Model Configuration
models:
  random_forest:
    n_estimators: 100
    random_state: 42
    class_weight: 'balanced'
  
  qbc_committee:
    size: 5
    diversity: ['RandomForest', 'ExtraTrees', 'GradientBoosting', 
                'LogisticRegression', 'KNN']

# Evaluation Configuration
evaluation:
  metrics: ['mcc', 'f1', 'roc_auc', 'accuracy', 'precision', 'recall']
  test_size: 0.2
  stratify: true
  random_state: 42
```

## Evaluation Metrics

### Primary Metrics

#### 1. Matthews Correlation Coefficient (MCC)
```python
def matthews_correlation_coefficient(y_true, y_pred):
    """Calculate MCC - primary metric for imbalanced datasets"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    numerator = (tp * tn) - (fp * fn)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    
    if denominator == 0:
        return 0
    return numerator / denominator
```

**Range**: [-1, +1]
**Interpretation**: 
- +1: Perfect prediction
- 0: Random prediction
- -1: Completely wrong prediction

**Advantages**: 
- Balanced measure for imbalanced datasets
- Considers all confusion matrix elements
- More robust than accuracy

#### 2. F1 Score
```python
def f1_score(y_true, y_pred):
    """Harmonic mean of precision and recall"""
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * (precision * recall) / (precision + recall)
```

**Focus**: Balances precision and recall
**Use Case**: When both false positives and false negatives are important

#### 3. ROC AUC
```python
def roc_auc_score(y_true, y_scores):
    """Area under ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    return auc(fpr, tpr)
```

**Range**: [0, 1]
**Interpretation**: Probability that model ranks random positive higher than random negative

### Secondary Metrics

- **Accuracy**: Overall correctness
- **Precision**: True positive rate among positive predictions
- **Recall**: True positive rate among actual positives
- **Specificity**: True negative rate among actual negatives

### Delta MCC (DMCC)

```python
def calculate_dmcc(al_mcc, full_mcc):
    """Calculate improvement over baseline"""
    return al_mcc - full_mcc
```

**Interpretation**:
- Positive DMCC: AL outperforms full model
- Negative DMCC: AL underperforms full model
- Zero DMCC: Equivalent performance

## Statistical Analysis

### Confidence Intervals

```python
def calculate_confidence_interval(data, confidence=0.95):
    """Calculate confidence interval for metric"""
    n = len(data)
    mean = np.mean(data)
    sem = stats.sem(data)  # Standard error of mean
    
    # t-distribution for small samples
    t_value = stats.t.ppf((1 + confidence) / 2, n - 1)
    margin_error = t_value * sem
    
    return mean - margin_error, mean + margin_error
```

### Statistical Significance Testing

#### Two-sample t-test
```python
def compare_performance(al_scores, full_scores):
    """Compare AL vs Full model performance"""
    statistic, p_value = stats.ttest_ind(al_scores, full_scores)
    
    effect_size = (np.mean(al_scores) - np.mean(full_scores)) / \
                  np.sqrt((np.var(al_scores) + np.var(full_scores)) / 2)
    
    return {
        'statistic': statistic,
        'p_value': p_value,
        'effect_size': effect_size,
        'significant': p_value < 0.05
    }
```

### Performance Analysis

#### Learning Curve Analysis
- Plot MCC vs number of labeled samples
- Identify convergence points
- Compare AL strategies

#### Error Analysis
- Confusion matrix comparison
- Per-class performance analysis
- Failure case investigation

## Reproducibility

### Random Seed Management
```python
# Ensure reproducible results
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# For multiple runs
seeds = [42, 10, 50, 100, 200, 300, 400, 500, 600, 700]
```

### Experimental Controls
- Fixed train/test splits
- Consistent preprocessing
- Identical model parameters
- Standardized evaluation procedures

## Limitations and Considerations

### Methodological Limitations
1. **Dataset Size**: Limited to two datasets
2. **Domain Scope**: Biomedical focus only
3. **Strategy Coverage**: Two AL strategies tested
4. **Computational Cost**: Full pool querying approach

### Practical Considerations
1. **Annotation Cost**: Assumes unit cost per sample
2. **Oracle Assumption**: Perfect labeling assumed
3. **Batch Size**: Fixed batch sizes may not be optimal
4. **Stopping Criteria**: Simple exhaustive approach

### Future Improvements
1. **Multi-domain Validation**: Test on diverse datasets
2. **Cost-sensitive Learning**: Incorporate labeling costs
3. **Advanced Strategies**: Explore uncertainty-diversity trade-offs
4. **Real-world Validation**: Clinical annotation studies

## Conclusion

Our methodology demonstrates that carefully designed active learning can achieve superior performance with minimal labeled data. The key findings include:

1. **QBC First5 Excellence**: Outperformed full model on Breast Cancer (MCC 0.942 vs 0.9253)
2. **Statistical Parity**: AL approaches match full-data performance consistently
3. **Rapid Convergence**: Peak performance achieved within 5-10 iterations
4. **Robust Results**: Consistent across multiple runs and evaluation metrics

This methodology provides a foundation for efficient biomedical machine learning with reduced annotation requirements.