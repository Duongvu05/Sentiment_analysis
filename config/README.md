# ⚙️ Configuration Management

This directory contains configuration files and templates for managing the sentiment analysis project's various components and experiments.

## 📂 Directory Purpose

The `config/` directory is designed to centralize all configuration management for:
- **Model architectures** and hyperparameters
- **Training procedures** and optimization settings
- **Data processing** pipelines and parameters
- **Evaluation metrics** and reporting configurations
- **Experiment tracking** and reproducibility settings

## 🎯 Configuration Philosophy

This project follows configuration best practices:
- **Separation of concerns**: Logic separate from parameters
- **Reproducibility**: All experiments can be replicated
- **Flexibility**: Easy parameter tuning and experimentation
- **Documentation**: Self-documenting configuration files

## 📋 Recommended Configuration Files

### `model_config.yaml`
**Purpose**: Define model architectures and hyperparameters
```yaml
# Custom Logistic Regression Configuration
custom_logistic_regression:
  learning_rate: 1e-9
  max_iterations: 1000
  convergence_threshold: 1e-8
  regularization: null
  random_seed: 42

# Scikit-learn Model Configurations
sklearn_models:
  logistic_regression:
    penalty: 'l2'
    solver: 'lbfgs'
    max_iter: 1000
    random_state: 42
  
  random_forest:
    n_estimators: 100
    max_depth: 10
    min_samples_split: 5
    random_state: 42
```

### `training_config.yaml`
**Purpose**: Control training procedures and optimization
```yaml
# Training Configuration
training:
  batch_size: 32
  validation_split: 0.2
  shuffle: true
  stratify: true
  random_seed: 42

# Optimization Settings
optimization:
  gradient_clipping: true
  clip_value: 1.0
  early_stopping: true
  patience: 100
  numerical_stability: true

# Feature Engineering
feature_engineering:
  normalization:
    method: 'sentence_length'
    apply: false
  
  scaling:
    method: 'standard'  # 'minmax', 'standard', 'robust'
    apply: true
```

### `data_config.yaml`
**Purpose**: Define data processing parameters
```yaml
# Data Sources
data_sources:
  primary: 'dataset/raw/twitter_samples.json'
  backup: 'dataset/raw/twitter_samples.pkl'
  processed_dir: 'dataset/processed/'

# Preprocessing Settings
preprocessing:
  text_cleaning:
    lowercase: true
    remove_punctuation: true
    remove_stopwords: true
    remove_urls: true
    remove_mentions: true
  
  tokenization:
    method: 'nltk'
    language: 'english'
    preserve_case: false

# Dataset Splitting
splitting:
  test_size: 0.2
  validation_size: 0.0
  stratify: true
  random_state: 42
```

### `evaluation_config.yaml`
**Purpose**: Configure evaluation metrics and reporting
```yaml
# Evaluation Metrics
metrics:
  primary: 'accuracy'
  secondary: ['precision', 'recall', 'f1_score']
  averaging: 'binary'
  pos_label: 1

# Reporting Configuration
reporting:
  save_results: true
  output_dir: 'results/'
  generate_plots: true
  plot_formats: ['png', 'pdf']
  
# Comparison Settings
comparison:
  baseline_models: ['custom_lr', 'sklearn_lr']
  advanced_models: ['random_forest', 'svm', 'neural_network']
  significance_test: true
  confidence_level: 0.95
```

### `experiment_config.yaml`
**Purpose**: Track experiments and ensure reproducibility
```yaml
# Experiment Tracking
experiment:
  name: 'sentiment_analysis_homework'
  version: '1.0.0'
  description: 'Comprehensive homework solutions'
  author: 'Duongvu05'
  date: '2025-09-28'

# Reproducibility Settings
reproducibility:
  random_seeds:
    global: 42
    data_split: 42
    model_init: 42
    numpy: 42
  
  environment:
    python_version: '3.8+'
    key_packages:
      - 'numpy>=1.21.0'
      - 'pandas>=1.3.0'
      - 'scikit-learn>=1.0.0'
      - 'matplotlib>=3.4.0'
      - 'seaborn>=0.11.0'
      - 'nltk>=3.6.0'

# Logging Configuration
logging:
  level: 'INFO'
  format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
  save_logs: true
  log_dir: 'logs/'
```

## 🚀 Usage Examples

### Loading Configuration in Python
```python
import yaml
from pathlib import Path

# Load model configuration
with open('config/model_config.yaml', 'r') as f:
    model_config = yaml.safe_load(f)

# Access specific parameters
lr = model_config['custom_logistic_regression']['learning_rate']
max_iter = model_config['custom_logistic_regression']['max_iterations']

# Use in model initialization
model = LogisticRegressionModel(
    learning_rate=lr,
    max_iterations=max_iter
)
```

### Configuration-Driven Training
```python
# Load training configuration
with open('config/training_config.yaml', 'r') as f:
    train_config = yaml.safe_load(f)

# Configure training pipeline
trainer = ModelTrainer(
    batch_size=train_config['training']['batch_size'],
    validation_split=train_config['training']['validation_split'],
    gradient_clipping=train_config['optimization']['gradient_clipping']
)
```

## 🎯 Integration with Homework Solutions

While the main homework solutions notebook contains all implementations inline for educational purposes, these configuration templates demonstrate:

1. **Professional development practices**
2. **Scalable experiment management**
3. **Reproducible research methodology**
4. **Parameter organization strategies**
5. **Configuration-driven development**

## 📊 Benefits of Configuration Management

### For Learning:
- **Clear parameter documentation** for understanding model behavior
- **Easy experimentation** with different hyperparameters
- **Systematic comparison** across configurations

### For Development:
- **Code reusability** across different experiments
- **Version control** of experimental parameters
- **Team collaboration** with shared configurations

### For Research:
- **Reproducible results** through version-controlled configs
- **Systematic ablation studies** by modifying single parameters
- **Publication-ready** experiment documentation

## 🔧 Best Practices

1. **Use semantic names** for configuration parameters
2. **Include comments** explaining parameter purposes
3. **Version control** all configuration files
4. **Validate configurations** before running experiments
5. **Document dependencies** between parameters
6. **Use environment-specific** configurations when needed

## 💡 Future Extensions

This configuration system can be extended for:
- **Hyperparameter optimization** (GridSearch, RandomSearch)
- **Multi-dataset experiments** with dataset-specific configs
- **Distributed training** configurations
- **Model deployment** settings
- **A/B testing** parameter management

---
*This configuration system enables systematic, reproducible, and scalable sentiment analysis experiments while maintaining educational clarity.*