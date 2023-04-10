import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv('DataPreprocessing.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values
print("X : ", X)
print("Y : ", Y)

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
print("X : ", X)

ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print("X : ", X)

le = LabelEncoder()
Y = le.fit_transform(Y)
print("Y : ", Y)

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=1)
print("X_train : ", X_train)
print("X_test : ", X_test)
print("Y_train : ", Y_train)
print("Y_test : ", Y_test)

sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])
print("X_train : ", X_train)
print("X_test:", X_test)
