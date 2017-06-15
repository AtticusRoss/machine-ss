import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

import nltk
from nltk.corpus import stopwords # Import the stop word list
stops = set(stopwords.words("english"))
import re
def review_to_words( raw ):
    letters_only = re.sub("[^a-zA-Z]", " ", raw)
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]
    #
    # 6. Join the words back into one string separated by space,
    # and return the result.
    return( " ".join( meaningful_words ))

train = pd.read_csv("/home/atticus/Desktop/Machine/austrain.csv", header=0, delimiter=",")
#train.columns.values
test = pd.read_csv("/home/atticus/Desktop/Machine/austest.csv", delimiter=",", header = 0)

clean_pn = []

# Loop over each review; create an index i that goes from 0 to the length
# of the movie review list
for item in train['ProductName']:
    # Call our function for each one, and add the result to the list of
    # clean reviews
    clean_pn.append( review_to_words( item))
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
train_data_features = vectorizer.fit_transform(clean_pn).toarray()

# Initialize a Random Forest classifier with 200 trees
forest = RandomForestClassifier(n_estimators=500)
forest = forest.fit( train_data_features, train['LC'])

clean_pn_test = []
for item in test['ProductName'].values:
    # Call our function for each one, and add the result to the list of
    # clean reviews
    clean_pn_test.append( review_to_words( item))
clean_pn_test
# Get a bag of words for the test set, and convert to a numpy array
test_data_features = (vectorizer.transform(clean_pn_test)).toarray()

predictions = forest.predict(test_data_features)
print(accuracy_score(test['LC'], predictions))
print(confusion_matrix(test['LC'], predictions))
print(classification_report(test['LC'], predictions))
