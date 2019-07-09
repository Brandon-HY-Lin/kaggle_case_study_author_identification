# kaggle_case_study_author_identification
Case study of Kaggle competition "Spooky Author Identification"

# Process of Tuning Hyperparameters of XGBoost
  - step 1: tune learning_rate and n_estimator
    - learning_rate:
      - default value: 0.1
      - suggested initial value: 0.1
      - suggested range: [0.05, 0.3]
    - n_estimators:
      - default value: 100
      - suggested initial value: 1000
      - suggested range: [100, 1000]
  - step 2: tune max_depth and min_child_weight
    - max_depth:
      - default value: 3
      - suggested initial value: 5
      - suggested range: [3, 10]
    - min_child_weight:
      - default value: 1
      - suggested initial value: 1
      - suggested range: [1, 6]
  - step 3: tune gamma
    - gamma:
      - default value: 0
      - suggested initial value: 0
      - suggested range: [0.0, 0.5]
  - step 4: tune subsample and colsample_bytree:
    - subsample:
      - default value: 1.0
      - suggested initial value: 0.8
      - suggested range: [0.6, 1.0]
    - min_child_weight:
      - default value: 1.0
      - suggested initial value: 0.8
      - suggested range: [0.6, 1.0]
  - step 5: tune regularization (reg_alpha):
    - reg_alpha:
      - default value: 0.0
      - suggested initial value: 0.0
      - suggested values: [1e-5, 1e-2, 0.0, 0.1, 1.0, 100.0]
      

# Key APIs
- Split Dataset
  - sklearn.model_selection.train_test_split()
- XGBoost (Extreme Gradient Boost)
  - xgboost.XGBClassifier()
  - xgboost.XGBRegressor()
- Metrics
  - sklearn.metrics.classification_report(y, prediction)
  - sklearn.metrics.accuracy_score(y, prediction)
- Grid Search
  - sklearn.model_selection.GridSearchCV()
