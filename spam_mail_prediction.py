

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

data = pd.read_csv("spamham.csv")

data

data.isnull().sum()

data.isnull().sum().sum()

data.shape

data.head()

## Spam mail = 0
## Ham mail = 1

data.loc[data["Category"] == "spam", 'Category',] = 0
data.loc[data["Category"] == "ham", 'Category',] = 1

data

X = data["Message"]
Y = data["Category"]

print(X)

print(Y)





X_train, X_test, Y_train, Y_test = train_test_split(X,Y, train_size = 0.8, test_size = 0.2, random_state =3)

feature_extraction = TfidfVectorizer(lowercase=True, min_df = 1, stop_words= "english" )

X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

Y_train_int = Y_train.astype("int")
Y_test_int = Y_test.astype("int")





model = LinearSVC()

model.fit(X_train_features,Y_train_int)

prediction_test_data = model.predict(X_test_features)

type(prediction_test_data)

type(Y_test)

type(Y_test_int)

accurcay_on_test_data = accuracy_score(Y_test_int,prediction_test_data)

print("Accuracy on test data = ", accurcay_on_test_data)

print("Accuracy percentage  on test data = ", accurcay_on_test_data * 100, "%")



input_mail_1 = ["hpl nom for may 17 , 2001"]

input_mail_2 = ["Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's"]

input_mail_features = feature_extraction.transform(input_mail_1)

input_mail_features

prediction_on_new_data = model.predict(input_mail_features)

if(prediction_on_new_data[0] == 1):
    print("It is a HAM MAIL")
else:
    print("It is a SPAM MAIL")

input_mail_features = feature_extraction.transform(input_mail_2)
prediction_on_new_data = model.predict(input_mail_features)
if(prediction_on_new_data[0] == 1):
    print("It is a HAM MAIL")
else:
    print("It is a SPAM MAIL")

