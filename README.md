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
  - K-Fold
    - sklearn.model_selection.StratifiedKFold()
    - sklearn.model_selection.KFold()
- XGBoost (Extreme Gradient Boost)
  - xgboost.XGBClassifier()
    - Get Softmax Result
        - xgboost.XGBClassifier().fit(data).predict_proba(X)
    - Get Prediction
        - xgboost.XGBClassifier().fit(data).predict(X)
  - xgboost.XGBRegressor()
  - Print Importance of Features
    - xgboost.plot_importance()
- Naive Bayes
    - sklearn.naive_bayes.MultinominalNB()
    - sklearn.naive_bayes.GaussianNB()
- Metrics
  - sklearn.metrics.classification_report(y, prediction)
  - sklearn.metrics.accuracy_score(y, prediction)
  - sklearn.metrics.log_loss(y, pred_prob)
- Grid Search
  - sklearn.model_selection.GridSearchCV()
- Text-Document-Based Vectorization
    - BOW:
        - sklearn.feature_extraction.text.CountVectorizer()
    - TF-IDF:
        - sklearn.feature_extraction.text.TfidfVectorizer()
        - sklearn.feature_extraction.text.TfidfVectorizer().fit().vocabulary_
- Reduce the Feature Size
  - SVD (Singular Value Decomposition)
    - sklearn.decomposition.TruncatedSVD(n_components, n_iter).fit()
    - sklearn.decomposition.TruncatedSVD(n_components, n_iter).fit().transform()
- Plot
  - seaborn.pairplot()
  - Histogram
    - seaborn.distplot()
    - pandas.dataframe.plot.hist()
    - pandas.dataframe.plot(kind='hist')
  - Box Plot
    - seaborn.boxplot()
  - Scatter Plot
    - seaborn.scatterplot()
  - Plot Encodded String
    - pandas.series.Series.value_counts().plot.bar()
  - Violin Plot
    - seaborn.violinplot(x='author', y='text', data=train_df)
  - Heat Map for Plotting Confusion Matrix
    - sklearn.metrics.confusion_matrix()
    - seaborn.heatmap()
- Stopwords/Punctuations
    - Stopwords
        - nltk.corpora.stopwords.words('english')
    - Punctuations
        - string.punctuation
        
- Data Manipulation
    - Group Data
        - pandas.DataFrame.groupby(column_name)
            ```
            # get all groups
            groups = train_df['author'].groupby()
            
            # access each group and its column_name
            for name, g in groups:
                
                for index, row in g:
                    print(row['text'])
            ```
    - Encode String to Number
        - pandas.DataFrame.map({'str1': 0, 'str2': 1})
            ```
            train_df['author'].map({'EAP':0, 'HPL':1, 'MWS':2})
            ```


# Reference:
- [Complete Guide to Parameter Tuning in XGBoost with codes in Python](https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/)
