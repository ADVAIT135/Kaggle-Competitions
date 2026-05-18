# 🏎️ F1 Pit Stops Prediction using LightGBM

## Project Overview

This project leverages advanced machine learning techniques to predict the probability of a Formula 1 driver making a pit stop on the next lap. Using a LightGBM classifier trained on real F1 race strategy data, the model achieves a **ROC AUC score of 0.945**, demonstrating strong predictive performance.

## Problem Statement

In Formula 1 racing, strategic pit stop decisions can determine race outcomes. This project aims to build a predictive model that estimates the likelihood of a driver making a pit stop on their next lap, considering various race conditions and driver-specific factors.

## Dataset

The training and test datasets contain the following features:
- **Driver**: Encoded identifier for the F1 driver
- **Compound**: Tire compound type (Soft, Medium, Hard) - label encoded
- **Race**: The specific F1 race event - label encoded
- **Target Variable (PitNextLap)**: Binary indicator (1 = pit stop on next lap, 0 = no pit stop)
- Additional race-related features for context

### Dataset Statistics
- **Training samples**: 351,312 records
- **Class distribution**: 19.9% positive cases (pit stops), 80.1% negative cases
- **Features**: 15 numeric features after preprocessing

## Methodology

### Data Preprocessing
1. **Categorical Encoding**: Label encoding applied to categorical columns (Driver, Compound, Race)
2. **Train-Validation Split**: 80-20 stratified split to maintain class distribution
3. **Feature Engineering**: Utilized all available features for model training

### Model Architecture

**LightGBM Classifier Configuration:**
- **Number of estimators**: 500 trees
- **Learning rate**: 0.05 (conservative boosting rate for stability)
- **Objective**: Binary classification
- **Boosting type**: GBDT (Gradient Boosting Decision Tree)

### Key Performance Metrics
- **Validation ROC AUC**: 0.945 (94.5% - excellent discrimination between pit stop and non-pit stop scenarios)
- **Model capabilities**: Probabilistic predictions suitable for strategic decision-making

## Model Performance

```
Validation Results:
- ROC AUC Score: 0.9453486992922306
- This indicates strong model discrimination power between the two classes
```

## Files in This Project

- **f1-pit-stops.ipynb**: Main Jupyter notebook containing:
  - Data loading and exploration
  - Data transformation and preprocessing
  - Model training and evaluation
  - Predictions on test set
  - Advanced optimization techniques (StratifiedKFold, CalibratedClassifierCV)
  
- **submission.csv**: Generated predictions for test set with format:
  - `id`: Test sample identifier
  - `PitNextLap`: Predicted probability of pit stop on next lap

## Installation & Requirements

```bash
# Required libraries
pandas
scikit-learn
lightgbm
numpy
```

Install via:
```bash
pip install pandas scikit-learn lightgbm numpy
```

## Usage

### Running the Notebook

1. Place `train.csv` and `test.csv` in the same directory as the notebook
2. Open `f1-pit-stops.ipynb` in Jupyter Notebook or JupyterLab
3. Run cells sequentially to:
   - Load and preprocess data
   - Train the LightGBM model
   - Generate predictions
   - Create submission file

### Key Code Sections

**Model Training:**
```python
model = LGBMClassifier(n_estimators=500, learning_rate=0.05)
model.fit(X_train, y_train)
```

**Generating Predictions:**
```python
preds_test = model.predict_proba(test)[:,1]
```

**Validation:**
```python
preds_val = model.predict_proba(X_val)[:,1]
roc_auc = roc_auc_score(y_val, preds_val)
```

## Advanced Optimization Techniques

The notebook includes implementations for further optimization:
- **StratifiedKFold Cross-Validation**: Ensures balanced class representation across folds
- **CalibratedClassifierCV**: Calibrates probability predictions for better reliability
- **LabelEncoder**: Handles categorical features consistently across train and test sets

## Model Insights

### Why LightGBM?
1. **Efficiency**: Fast training on large datasets (350K+ records)
2. **Performance**: Superior to traditional models on tabular data
3. **Interpretability**: Feature importance rankings available
4. **Robustness**: Handles class imbalance naturally through gradient boosting

### Class Imbalance Handling
The dataset exhibits natural class imbalance (≈20% pit stops). LightGBM's gradient boosting inherently handles this through:
- Weighted loss functions
- Focus on hard-to-predict samples
- Stratified train-test splitting

## Results & Deliverables

✅ **Trained LightGBM model** with 94.5% ROC AUC  
✅ **Probabilistic predictions** on test set  
✅ **submission.csv** formatted for Kaggle submission  
✅ **Reproducible notebook** with clear documentation  

## Future Enhancements

Potential improvements for next iterations:
- Hyperparameter tuning using Optuna or GridSearchCV
- Feature engineering (lap position, fuel load, weather conditions)
- Ensemble methods combining multiple models
- SHAP values for model explainability
- Time-series cross-validation if temporal data is available

## References

- **LightGBM Documentation**: https://lightgbm.readthedocs.io/
- **Scikit-learn**: https://scikit-learn.org/
- **Kaggle Competition**: Predicting F1 Pit Stops

## Author

**ADVAIT135**  
Kaggle Competitions Repository  
Created: 2026

---

## License

This project is part of the Kaggle Competitions collection. Please refer to the repository license for usage terms.

## Questions & Support

For questions or issues related to this project, please refer to the main repository or open an issue on GitHub.

