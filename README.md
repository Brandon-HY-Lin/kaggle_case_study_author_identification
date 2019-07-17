# kaggle_case_study_author_identification
Case study of Kaggle competition "Spooky Author Identification"

# Feature Engineering
- [notebook](https://github.com/Brandon-HY-Lin/kaggle_case_study_author_identification/blob/master/step_2_feature_engineering.ipynb)
- Meta Features
    - Features that are extracted from the text like number of words, stopwords, and punctuations.
        1. Number of words.
        1. Number of unique words.
        1. Number of characters.
        1. Number of stopwords.
        1. Number of punctuations.
        1. Number of upper case words.
        1. Number of title case words.
        1. Average length of words.
- Text-Based Features
    - BOW
    - TF-IDF
- Results
    
<table>
  <tr>
    <th>Features</th>
    <th>Meta</th>
    <th colspan="2">TF-IDF</th>
    <th colspan="2">SVD</th>
    <th colspan="3">SVD+Meta</th>
  </tr>
  <tr>
    <td>Method</td>
    <td>XGBoost</td>
    <td>XGBoost</td>
    <td>MultinomialNB</td>
    <td>XGBoost</td>
    <td>GaussianNB</td>
    <td>XGBoost</td>
    <td>GaussianNB</td>
    <td>DNN (2 FCs)</td>
  </tr>
  <tr>
    <td>Accuracy</td>
    <td>0.518</td>
    <td>0.606</td>
    <td>0.989</td>
    <td>0.697</td>
    <td>0.451</td>
    <td>0.716</td>
    <td>0.453</td>
    <td>0.6645</td>
  </tr>
  <tr>
    <td>Log Loss</td>
    <td>0.97</td>
    <td>0.912</td>
    <td>0.466</td>
    <td>0.737</td>
    <td>6.027</td>
    <td>0.709</td>
    <td>5.795</td>
    <td>0.7784</td>
  </tr>
</table>
 
 # Deep Learning Models with Word Embedding 
- [notebook](https://github.com/Brandon-HY-Lin/kaggle_case_study_author_identification/blob/master/step_3_word_embedding_and_keras.ipynb)

- Results


|          | DNN + <br>Average Pooling | RNN     | Bidirectional<br>RNN |
|----------|-----------------------|---------|-------------------|
| Accuracy | 0.9916                | 0.4037  | 0.9892            |
| Log Loss | 0.0327                | 1.0878  | 0.0325            |
| Epochs   | 100                   | 5       | 20                |
| CPU Time | 14min 19s             | 3min 8s | 35min 43s         |
 

# XGBoost
- [notebook](https://github.com/Brandon-HY-Lin/kaggle_case_study_author_identification/blob/master/step_0_xgboost_classification_tuning_params.ipynb)

-Process of Tuning Hyperparameters of XGBoost
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
- Encodding
    - keras.utils.to_categorical()
- Pad Sequence:
    - keras.preprocessing.sequence.pad_sequences()
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
