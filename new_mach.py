
# coding: utf-8

# In[461]:

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.svm import SVC
import numpy as np
import scipy as sp


# In[474]:

df = pd.read_csv("perfect.csv", encoding = "ISO-8859-1")


# In[476]:

msk = np.random.rand(len(df)) < 0.8
train = df[msk]
test = df[~msk]


# In[477]:

# Convert string column to integer
def str_column_to_int(dataset,column): 
    unique = set(dataset.ix[:,column])
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for item in dataset.ix[:,column]:
        try:
            dataset.ix[:,column] = dataset.ix[:,column].replace(item, lookup[item])
        except KeyError:
            pass
    return lookup


# In[479]:

brands_dict = str_column_to_int(train, 2)
man_dict = str_column_to_int(train, 3)


# In[480]:

test_brands_dict = str_column_to_int(test, 2)
test_man_dict = str_column_to_int(test, 3)


# In[481]:

# bag of words tool.
vectorizer = CountVectorizer(analyzer = "word",                                tokenizer = None,                                 preprocessor = None,                              stop_words = 'english',                                max_features = 10000)


# In[482]:

train_data_vect = vectorizer.fit_transform(train.ProductName)
test_data_vect = vectorizer.transform(test.ProductName)


# In[483]:

compressed_values_train = sp.sparse.hstack((vectorizer.fit_transform(train.ProductName),train[['Brand','Manufacturer','Barcode']].values))
compressed_values_test = sp.sparse.hstack((vectorizer.transform(test.ProductName),test[['Brand','Manufacturer','Barcode']].values))


# In[484]:

forest = RandomForestClassifier(n_estimators = 500)
forest = forest.fit(compressed_values_train, train['TC'])
train.columns


# In[454]:

predictions = forest.predict(compressed_values_test)
print(accuracy_score(test['TC'], predictions))
print(confusion_matrix(test['TC'], predictions))
print(classification_report(test['TC'], predictions))

