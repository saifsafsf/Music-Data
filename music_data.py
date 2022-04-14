import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
import joblib

# reading csv into a dataframe
music_data = pd.read_csv('music.csv')

# Genre is output, other 2 columns are inputs
X = music_data.drop(columns=['genre'])
y = music_data['genre']

# 80% data for training the model, 20% for testing
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Decision Tree used
model = tree.DecisionTreeClassifier()
model.fit(X, y)
# model.fit(X_train, y_train)

# using previously stored model
# model = joblib.load('music_recommender.joblib')

# predictions & accuracy score using the model & testing data
# predictions = model.predict(X_test)
# acc_score = accuracy_score(y_test, predictions)

# graphic visualization of the model
tree.export_graphviz(model, out_file='music_recommender.dot',
                      feature_names=['age', 'gender'],
                      class_names=sorted(y.unique()),
                      label='all', rounded=True,
                      filled=True)