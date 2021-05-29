import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.impute import SimpleImputer

d1 = pd.read_csv("C:/Users/vs_00/Desktop/Internship/SAS/Diabetes/archive/diabetes.csv")
# print(d1.head())
print(d1.info())
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

print(d1['SkinThickness'].isnull().values.any())
#printing the correlation matrix
corrMatrix = d1.corr()

##plotting a heat map of correlation matrix
# sn.heatmap(corrMatrix, annot=True)
# plt.show()