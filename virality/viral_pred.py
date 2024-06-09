import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from xgboost import XGBClassifier

# Load the dataset
df = pd.read_csv("data.csv")
relevant_cols = []  # List your relevant columns here
data = df[relevant_cols]

# Preprocess the data
data = data.dropna()
data.to_csv("data/test.csv")

# Define the target variable
sentiment = data['target_col']
data = data.drop(columns=['target_col'])

# Split the data
x_train, x_test, y_train, y_test = train_test_split(
    data, sentiment, random_state=42, test_size=0.2, shuffle=True)

target_names = ['class 0', 'class 1']

# K-Nearest Neighbors model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
print("K-Nearest Neighbors:")
print(classification_report(y_test, y_pred, target_names=target_names))

# Random Forest model
rf = RandomForestRegressor(random_state=42)
n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
max_features = ['sqrt', 'log2', None]
max_depth = [int(x) for x in np.linspace(start=10, stop=100, num=10)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=500, cv=5, verbose=2, n_jobs=-1)
rf_random.fit(x_train, y_train)
print("Random Forest best parameters:")
print(rf_random.best_params_)

random_forest = RandomForestClassifier(**rf_random.best_params_)
random_forest.fit(x_train, y_train)
y_pred = random_forest.predict(x_test)
print("Random Forest:")
print(classification_report(y_test, y_pred, target_names=target_names))

# Decision Tree model
decision_tree = DecisionTreeClassifier()
decision_tree.fit(x_train, y_train)
y_pred = decision_tree.predict(x_test)
print("Decision Tree:")
print(classification_report(y_test, y_pred, target_names=target_names))

# Neural Network model
neural_network = MLPClassifier(hidden_layer_sizes=(100, 100, 100), max_iter=1000)
neural_network.fit(x_train, y_train.values.ravel())
y_pred = neural_network.predict(x_test)
print("Neural Network:")
print(classification_report(y_test, y_pred, target_names=target_names))

# XGBoost model
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
xgb_model.fit(x_train, y_train)
y_pred = xgb_model.predict(x_test)
print("XGBoost:")
print(classification_report(y_test, y_pred, target_names=target_names))

# Generate a viral score using a weighted sum of the various ML classifier prediction scores
neural_network_preds = neural_network.predict(x_test)
random_forest_preds = random_forest.predict(x_test)
decision_tree_preds = decision_tree.predict(x_test)
knn_preds = knn.predict(x_test)
xgb_preds = xgb_model.predict(x_test)

# Adjust the weights as needed
weights = [0.2, 0.2, 0.2, 0.2, 0.2]  # Equal weights for all models
viral_score = (weights[0] * neural_network_preds +
               weights[1] * random_forest_preds +
               weights[2] * decision_tree_preds +
               weights[3] * knn_preds +
               weights[4] * xgb_preds)

# Output the viral score (as probabilities or classifications)
print("Viral Score:")
print(viral_score)
