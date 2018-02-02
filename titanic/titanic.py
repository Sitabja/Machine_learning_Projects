# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 10:46:22 2018

@author: Sitabja Ukil
"""
# pandas
import pandas as pd
import seaborn as sns
sns.set_style('whitegrid')
import matplotlib.pyplot as plt

from sklearn.preprocessing import Imputer

# get titanic & test csv files as a DataFrame

#developmental data (train)
titanic_df = pd.read_csv("./data/train.csv")

#cross validation data (hold-out testing)
test_df    = pd.read_csv("./data/test.csv")

# preview developmental data
titanic_df.head(5)

# preview the test data
test_df.head(5)

# check missing values in train dataset
titanic_df.isnull().sum()

sum(pd.isnull(titanic_df['Age']))

# proportion of "Age" missing
round(177/(len(titanic_df["PassengerId"])),4)

ax = titanic_df["Age"].hist(bins=15, color="teal" , alpha=0.8)
ax.set(xlabel="Age" , ylabel="Count")
plt.show()

# median age is 28 (as compared to mean which is ~30)
titanic_df["Age"].median(skipna=True)

# proportion of "cabin" missing
round(687/len(titanic_df["PassengerId"]),4)

#77% of records are missing, which means that imputing information and using this variable for prediction is probably not wise. 
#We'll ignore this variable in our model.

# proportion of "Embarked" missing
round(2/len(titanic_df["PassengerId"]),4)

sns.countplot(x='Embarked',data=titanic_df,palette='Set2')
plt.show()

#final adjustment

train_data = titanic_df
train_data["Age"].fillna(titanic_df["Age"].median(skipna=True), inplace=True)
train_data["Embarked"].fillna("S", inplace=True)
train_data.drop('Cabin', axis=1, inplace=True)

## Create categorical variable for traveling alone
train_data['TravelBuds']=train_data["SibSp"]+train_data["Parch"]
# 1 if travel alone else 0
train_data['TravelAlone']=np.where(train_data['TravelBuds']>0, 0, 1)

train_data.drop('SibSp', axis=1, inplace=True)
train_data.drop('Parch', axis=1, inplace=True)
train_data.drop('TravelBuds', axis=1, inplace=True)

#create categorical variable for Pclass

train2 = pd.get_dummies(train_data, columns=["Pclass"])
train3 = pd.get_dummies(train2, columns=["Embarked"])
train4 = pd.get_dummies(train3, columns=["Sex"])
train4.drop('Sex_female', axis=1, inplace=True)

train4.drop('PassengerId', axis=1, inplace=True)
train4.drop('Name', axis=1, inplace=True)
train4.drop('Ticket', axis=1, inplace=True)

train_final = train4

#same process should be applied to test set
test_df["Age"].fillna(titanic_df["Age"].median(skipna=True), inplace=True)
test_df["Fare"].fillna(test_df["Fare"].median(skipna=True), inplace=True)

test_df.drop('Cabin', axis=1, inplace=True)

test_df.drop("PassengerId", axis=1, inplace=True)
test_df.drop("Name", axis=1, inplace=True)
test_df.drop("Ticket", axis=1 , inplace=True)

test_df['TravelBuds']=test_df["SibSp"]+test_df["Parch"]
test_df['TravelAlone']=np.where(test_df['TravelBuds']>0, 0, 1)

test_df.drop('SibSp', axis=1, inplace=True)
test_df.drop('Parch', axis=1, inplace=True)
test_df.drop('TravelBuds', axis=1, inplace=True)

test2 = pd.get_dummies(test_df, columns=["Pclass"])
test3 = pd.get_dummies(test2, columns=["Embarked"])

test4=pd.get_dummies(test3, columns=["Sex"])
test4.drop('Sex_female', axis=1, inplace=True)

test_final = test4


train_final.head(10)
col = ["Age","Fare","TravelAlone","Pclass_1", "Pclass_2", "Pclass_3", "Embarked_C", "Embarked_Q", "Embarked_S", "Sex_male"]
X2 = train_final[col]
Y =  train_final['Survived']
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(X2, Y)

print(logreg.score(X2, Y))