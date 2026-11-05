# Titanic and Diabetes Classification

Supervised binary classification with scikit-learn on two datasets:
- **Titanic**: predict passenger survival (`Survived`)
- **Diabetes**: predict diabetes (`Outcome`)

## What's inside
- Data exploration and a majority-class baseline
- Preprocessing (imputation, encoding, scaling fitted on training data only)
- Stratified 80/20 train/test split
- Comparison of Logistic Regression, Decision Tree and Random Forest
- Evaluation with accuracy, confusion matrix, ROC/AUC and 5-fold cross-validation
- Hyperparameter tuning with `GridSearchCV`
- Models saved with `joblib`

## Results (diabetes)
| Model | Accuracy | AUC |
|---|---|---|
| Baseline | 0.651 | n/a |
| Logistic Regression | 0.727 | 0.817 |
| Tuned Logistic Regression | 0.747 | n/a |

Note: the diabetes CSV was imputed with class-specific medians, which leaks the label into tree-based models, so Logistic Regression is the more reliable model here.

## Run
```bash
pip install numpy pandas scikit-learn matplotlib seaborn joblib
jupyter notebook
```
Place `train.csv` (Titanic, from Kaggle) and `diabetes.csv` next to the notebook.

## Stack
Python, pandas, scikit-learn, matplotlib, seaborn
