import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

d1 = pd.read_csv("C:/Users/vs_00/Desktop/Internship/SAS/Diabetes/archive/diabetes.csv")
# print(d1.head())
# print(d1.info())
#
# print(type(d1['Glucose']))
#
# print(d1.isna())

#indexing 5th row and 1st column
# print(d1.iloc[4,0])

imputer = SimpleImputer(missing_values =np.nan,strategy ='mean')
# d1['SkinThickness'].fill6na(d1['SkinThickness'].mean())
# d1['BloodPressure'].fillna(d1['BloodPressure'].mean())
# d1['Insulin'].fillna(d1['Insulin'].mean())
# d1['BMI'].fillna(d1['BMI'].mean())

# print(d1['SkinThickness'].isnull().values.any())
#printing the correlation matrix
corrMatrix = d1.corr()

##plotting a heat map of correlation matrix
# sn.heatmap(corrMatrix, annot=True)
# plt.show()

# print(sn.countplot(x='Outcome',data=d1))
# plt.show()

x_train,x_test,y_train,y_test = train_test_split(d1.drop('Outcome', axis=1),d1['Outcome'], test_size=0.3,random_state=101)

logmodel = LogisticRegression(max_iter = 150)
logmodel.fit(x_train,y_train)
predictions = logmodel.predict(x_test)
print(classification_report(y_test,predictions))