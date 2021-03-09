import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


dataset = pd.read_csv('weatherAUS.csv')

independent = dataset[['Location','MinTemp','MaxTemp','Rainfall','WindGustSpeed']].values
dependent = dataset.iloc[:,-1].values

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")

imputer.fit(independent[:, 1:4])

independent[:, 1:4] = imputer.transform(independent[:, 1:4])


transformer = ColumnTransformer(transformers=[('encoder' , OneHotEncoder(), [0])], remainder='passthrough')
independent = np.array((transformer.fit_transform(independent)))

ind_train, ind_test, dep_train, dep_test = train_test_split(independent, dependent, test_size=0.2, random_state=0)
linearRegression = LinearRegression()
linearRegression.fit(ind_train, dep_train)

dep_pred = linearRegression.predict(ind_test)

np.set_printoptions(precision=2)

print (np.concatenate((dep_pred.reshape(len(dep_pred), 1), dep_test.reshape(len(dep_pred), 1) ), axis=1))

print (linearRegression.predict([['Albury', 13.4, 25.4, 2.2, 46]]))

for c in linearRegression.coef_:
    print (f'{c:.2f} ')
print (f'{linearRegression.intercept_:.2f}')

