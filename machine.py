import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

train = pd.read_csv("/home/atticus/Desktop/Machine/austrain.csv", header=0, delimiter=",")
#train.columns.values
test = pd.read_csv("/home/atticus/Desktop/Machine/austest.csv", delimiter=",", header = 0)

# bag of words tool.
vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = 'english',   \
                             max_features = 10000)

# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of
# strings.
# Numpy arrays are easy to work with, so convert the result to an
# array
train_data_features = vectorizer.fit_transform(train['ProductName'])
train_data_features = train_data_features.toarray()
# Initialize a Random Forest classifier with 200 trees
forest = RandomForestClassifier(n_estimators=200)
forest = forest.fit( train_data_features, train['TC'])

# Get a bag of words for the test set, and convert to a numpy array
test_data_features = (vectorizer.transform(test['ProductName'])).toarray()

predictions = forest.predict(test_data_features)
print(accuracy_score(test['TC'], predictions))
print(confusion_matrix(test['TC'], predictions))
print(classification_report(test['TC'], predictions))
