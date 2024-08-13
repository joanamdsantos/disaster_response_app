# Disaster Response Pipeline Project

In this project, I analyzed disaster data from Appen(formerly Figure 8) to build a model for an API that classifies disaster messages.

A data set containing real messages that were sent during disaster events (disaster_messages.csv) is used. A machine learning pipeline to categorize these events, using disaster_categories.csv, is used so that an app can send the messages to an appropriate disaster relief agency.

The project provides a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.

### Constrains observed in the project

During the process of building the model for this project, it was observed that most categories are imbalanced. While training a classification model with this data, the model is going to be biased towards the majority class because machine learning models learn from the examples and most of the examples in your dataset belong to a single class.

There  are options to handle imbalanced data in machine classification problems. Options include using random under-sampling, random over-sampling, and synthetic over-sampling: SMOTE. Other methods using random forests and bagging methods may also be used. However this dataset is multilabel.

In this case the best approach is choosing a proper evaluation metric. The accuracy of a classifier is the total number of correct predictions divided by the total number of predictions. This may be good enough for a well-balanced class but not ideal for an imbalanced class problem. Other metrics, such as precision, measure how accurate the classifier’s prediction of a specific class, and recall measures the classifier’s ability to identify a class.

For an imbalanced class dataset we can use class weight on the classifier. The F1 score is a more appropriate metric as well. It is the harmonic mean of precision and recall and the expression is f1 score. So, if the classifier predicts the minority class but the prediction is erroneous and the false-positive increases, the precision metric will be low, and so will the F1 score. Also, if the classifier identifies the minority class poorly, i.e., more of this class wrongfully predicted as the majority class, then false negatives will increase, so recall and F1 score will be low. The F1 score only increases if the number and prediction quality improve. F1 score keeps the balance between precision and recall and improves the score only if the classifier identifies more of a certain class correctly.
Also, classifiers like LightGBM and XGBoost are preferable in these situations. In this way I used the Random Forest classifier, first and the XGBoost after, with the weight F1 score as primary selector in the gridsearch CV, to deal with the imbalanced problem in the dataset.

### Best machine learning model

In this project two machine learning models were tested, as indicated in the constrains section: Random Forests and XGBoost.
The parameters grid used for each were th following:

Random Forests: n_estimators': [50, 100, 200], min_samples_split': [2, 3, 4]
XGBoost: learning_rate': [0.01,0.05,0.1], gamma': [0, 0.5, 1]

The classification report for both classifiers was the following:

##### Random Forest

| Category               | Precision | Recal | F1-score | Accuracy |
| ---------------------- | --------- | ----- | -------- | -------- |
| related                | 0.80      | 0.81  | 0.78     | 0.81     |
| request                | 0.89      | 0.90  | 0.88     | 0.90     |
| offer                  | 0.99      | 1.00  | 0.99     | 1.00     |
| aid_related            | 0.77      | 0.78  | 0.77     | 0.78     |
| medical_help           | 0.92      | 0.93  | 0.90     | 0.93     |
| medical_products       | 0.95      | 0.95  | 0.94     | 0.95     |
| search_and_rescue      | 0.96      | 0.97  | 0.96     | 0.97     |
| security               | 0.97      | 0.98  | 0.97     | 0.98     |
| military               | 0.96      | 0.96  | 0.95     | 0.96     |
| child_alone            | 1.00      | 1.00  | 1.00     | 1.00     |
| water                  | 0.96      | 0.96  | 0.95     | 0.96     |
| food                   | 0.94      | 0.95  | 0.95     | 0.95     |
| shelter                | 0.93      | 0.94  | 0.93     | 0.94     |
| clothing               | 0.98      | 0.99  | 0.98     | 0.99     |
| money                  | 0.98      | 0.98  | 0.96     | 0.98     |
| missing_people         | 0.97      | 0.99  | 0.98     | 0.99     |
| refugees               | 0.96      | 0.97  | 0.96     | 0.97     |
| death                  | 0.96      | 0.96  | 0.95     | 0.96     |
| other_aid              | 0.84      | 0.87  | 0.81     | 0.87     |
| infrastructure_related | 0.87      | 0.93  | 0.90     | 0.93     |
| transport              | 0.95      | 0.96  | 0.95     | 0.96     |
| buildings              | 0.95      | 0.96  | 0.94     | 0.96     |
| electricity            | 0.98      | 0.98  | 0.97     | 0.98     |
| tools                  | 0.98      | 0.99  | 0.99     | 0.99     |
| hospitals              | 0.98      | 0.99  | 0.98     | 0.99     |
| shops                  | 0.99      | 0.99  | 0.99     | 0.99     |
| aid_centers            | 0.97      | 0.99  | 0.98     | 0.99     |
| other_infrastructure   | 0.91      | 0.95  | 0.93     | 0.95     |
| weather_related        | 0.88      | 0.88  | 0.88     | 0.88     |
| floods                 | 0.96      | 0.96  | 0.95     | 0.96     |
| storm                  | 0.95      | 0.95  | 0.95     | 0.95     |
| fire                   | 0.98      | 0.99  | 0.98     | 0.99     |
| earthquake             | 0.97      | 0.97  | 0.97     | 0.97     |
| cold                   | 0.98      | 0.98  | 0.97     | 0.98     |
| other_weather          | 0.94      | 0.95  | 0.92     | 0.95     |
| direct_report          | 0.87      | 0.87  | 0.85     | 0.87     |


##### XGBoost

| Category               | Precision | Recal | F1-score | Accuracy |
| ---------------------- | --------- | ----- | -------- | -------- |
| related                | 0.80      | 0.81  | 0.78     | 0.81     |
| request                | 0.90      | 0.90  | 0.90     | 0.90     |
| offer                  | 0.99      | 0.99  | 0.99     | 0.99     |
| aid_related            | 0.77      | 0.76  | 0.76     | 0.76     |
| medical_help           | 0.92      | 0.93  | 0.91     | 0.93     |
| medical_products       | 0.95      | 0.96  | 0.95     | 0.96     |
| search_and_rescue      | 0.97      | 0.98  | 0.97     | 0.98     |
| security               | 0.98      | 0.98  | 0.97     | 0.98     |
| military               | 0.97      | 0.97  | 0.97     | 0.97     |
| child_alone            | 1.00      | 1.00  | 1.00     | 1.00     |
| water                  | 0.96      | 0.97  | 0.97     | 0.97     |
| food                   | 0.95      | 0.95  | 0.95     | 0.95     |
| shelter                | 0.94      | 0.95  | 0.94     | 0.95     |
| clothing               | 0.99      | 0.99  | 0.99     | 0.99     |
| money                  | 0.97      | 0.98  | 0.98     | 0.98     |
| missing_people         | 0.99      | 0.99  | 0.99     | 0.99     |
| refugees               | 0.96      | 0.97  | 0.96     | 0.97     |
| death                  | 0.97      | 0.97  | 0.97     | 0.97     |
| other_aid              | 0.86      | 0.88  | 0.85     | 0.88     |
| infrastructure_related | 0.91      | 0.94  | 0.91     | 0.94     |
| transport              | 0.96      | 0.96  | 0.95     | 0.96     |
| buildings              | 0.95      | 0.96  | 0.96     | 0.96     |
| electricity            | 0.97      | 0.98  | 0.97     | 0.98     |
| tools                  | 0.99      | 0.99  | 0.99     | 0.99     |
| hospitals              | 0.99      | 0.99  | 0.99     | 0.99     |
| shops                  | 0.99      | 1.00  | 0.99     | 1.00     |
| aid_centers            | 0.99      | 0.99  | 0.99     | 0.99     |
| other_infrastructure   | 0.93      | 0.95  | 0.94     | 0.95     |
| weather_related        | 0.88      | 0.88  | 0.87     | 0.88     |
| floods                 | 0.96      | 0.96  | 0.95     | 0.96     |
| storm                  | 0.94      | 0.95  | 0.94     | 0.95     |
| fire                   | 0.99      | 0.99  | 0.99     | 0.99     |
| earthquake             | 0.97      | 0.97  | 0.97     | 0.97     |
| cold                   | 0.98      | 0.98  | 0.98     | 0.98     |
| other_weather          | 0.93      | 0.94  | 0.93     | 0.94     |
| direct_report          | 0.87      | 0.88  | 0.86     | 0.88     |

We can observe from the weighted F1-score that XGboost surpasses the Random Forest classifier in almost every category, except for the categories related, aid_related, weather_related and storm. Given these results I opted to use the XGBoost Classifier in this project.


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
