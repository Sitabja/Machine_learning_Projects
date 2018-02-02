ra# Loan Prediction
"""
Created on Mon Jan 22 19:06:25 2018

@author: Sitabja Ukil
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Loan_prediction_data.csv')

#Dimensions of Dataset
# shape
print(dataset.shape)

# Peek at the Data
# head
print(dataset.head(20))

# Statistical Summary
# descriptions
print(dataset.describe())

#Class Distribution
print(dataset.groupby('Loan_Status').size())

#Data Visualization
#Univariate Plots to better understand each attribute.
# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(3,2), sharex=False, sharey=False)
plt.show()

# histograms
dataset.hist( sharex=False, sharey=False)
plt.show()

from pandas.tools.plotting import scatter_matrix
scatter_matrix(dataset)
plt.show()
#handle missing values
dataset.isnull().sum()
dataset.info()
# distribution of gender
my_tab = pd.crosstab(index = dataset["Gender"],  # Make a crosstab
                              columns="count")   # Name the count column
my_tab.plot.bar()
#forward-fill for gender
dataset['Gender'].fillna(method='ffill',inplace = True)

# distribution of married
my_tab = pd.crosstab(index = dataset["Married"],  # Make a crosstab
                              columns="count")      # Name the count column
my_tab.plot.bar()
 
#filling married and dependents with most frequent value
dataset['Married'].fillna(dataset['Married'].value_counts().index[0], inplace = True)
dataset['Dependents'].fillna(dataset['Dependents'].value_counts().index[0], inplace = True)

# distribution of Self_Employed
my_tab = pd.crosstab(index = dataset["Self_Employed"],  # Make a crosstab
                              columns="count")      # Name the count column
my_tab.plot.bar()

#filling self_employed with most frequent value
dataset['Self_Employed'].fillna(dataset['Self_Employed'].value_counts().index[0], inplace = True)

# distribution for ApplicantIncome
# histograms
dataset['ApplicantIncome'].hist()
plt.show()

# skewed distribution , taking median to replace missing values
dataset['ApplicantIncome'].fillna(dataset['ApplicantIncome'].median(), inplace=True)

# distribution of Loan amount
# histograms
dataset['LoanAmount'].hist()
plt.show()

# skewed distribution , taking median to replace missing values
dataset['LoanAmount'].fillna(dataset.groupby(['Gender','Married','Education','Self_Employed','Property_Area'])['LoanAmount'].transform('mean'), inplace=True)   

# distribution of Loan amount
# histograms
dataset['Loan_Amount_Term'].hist()
plt.show()

# replacing missing values using mode
dataset['Loan_Amount_Term'].fillna(dataset['Loan_Amount_Term'].mode()[0], inplace=True) 

# distribution of Self_Employed
my_tab = pd.crosstab(index = dataset["Credit_History"],  # Make a crosstab
                              columns="count")      # Name the count column
my_tab.plot.bar()

# replacing missing values using mode
dataset['Credit_History'].fillna(dataset['Credit_History'].mode()[0], inplace=True) 

#Evaluate Some Algorithms
#Create a Validation Dataset
X = dataset.iloc[:, 1:-1].values #ignoring loan id because id in unique
y = dataset.iloc[:, -1].values

# Encoding categorical data
# 1 if 3+ dependents else 0
X[:,2]=np.where(X[:,2] == '3+', 1, 0)
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:, 0])
X[:,1] = labelencoder_X.fit_transform(X[:, 1])
X[:,3] = labelencoder_X.fit_transform(X[:, 3])
X[:,4] = labelencoder_X.fit_transform(X[:, 4])
X[:,-1] = labelencoder_X.fit_transform(X[:, -1])
onehotencoder = OneHotEncoder(categorical_features = [-1])
X = onehotencoder.fit_transform(X).toarray()
# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
validation_size = 0.20
seed = 42
X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size = 0.20, random_state = seed)



# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_validation = sc.transform(X_validation)

#Test Harness
#10-fold cross validation
# Test options and evaluation metric
seed = 7
scoring = 'accuracy'

# Spot Check Algorithms
models = []
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
models.append(('Gradient Boosting', GradientBoostingClassifier(n_estimators = 100)))
##models.append(('Random Forest',RandomForestClassifier(n_estimators = 500, criterion = 'entropy', random_state = 42)))
# evaluate each model in turn
results = []
names = []

from sklearn import model_selection

for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
    

# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

# Make predictions on validation dataset
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


classifier = SVC()
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_validation)
print(accuracy_score(y_validation, predictions))
print(confusion_matrix(y_validation, predictions))
print(classification_report(y_validation, predictions))

