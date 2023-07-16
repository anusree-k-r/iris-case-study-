# DECISION TREE MODEL

import pandas as pd 

df = pd.read_excel('iris.xls')

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state = 42)
X_train
X_test
y_train

from sklearn.tree import DecisionTreeClassifier

DT = DecisionTreeClassifier(max_depth=4)
DT.fit(X_train,y_train)
DT.score(X_train, y_train)
DT.score(X_test, y_test)

y_test_pred = DT.predict(X_test)

X_test,y_test

# SAVE THE MODEL AS A PKL FILE
import pickle
pickle.dump(DT, open('model.pkl', 'wb'))  