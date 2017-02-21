import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve
import ipdb
import seaborn as sns
sns.set()

def import_Data(filepath):
    '''
    INPUT: STRING of filepath to data file
    OUTPUT: X and y data, including the engineered 'churn' columns

    ***dropped the last_trip_date and signup_date for now***
    ***GOT RID OF ALL NA VALUES***
    '''
    fullData = pd.read_csv(filepath)
    fullData = fullData.dropna()
    fullData['last_trip_date'] = pd.to_datetime(fullData['last_trip_date'])
    condition = fullData['last_trip_date'] < '2014-06-01'
    fullData.drop(fullData[['last_trip_date', 'signup_date']], axis=1,inplace=True)
    X = fullData
    fullData['churn'] = 1
    fullData.ix[~condition, 'churn'] = 0
    y = fullData['churn']
    return X, y

def mult_Binary_Features(X):
    '''
    INPUT: pandas dataframe - X data (BEFORE SPLIT!!!!)
    OUTPUT: pandas dataframe - X data with engineered binary feature columns
    '''
    X.drop('churn',axis=1,inplace=True)
    X['city'].unique()
    df_city = pd.get_dummies(X['city'])
    X['phone'].unique()
    df_phone = pd.get_dummies(X['phone'])
    X = pd.concat([X,df_city], axis = 1)
    X = pd.concat([X,df_phone], axis = 1)
    X.drop(X[['city','phone']], axis=1, inplace=True)
    return X

def log_Reg(X_train,y_train,X_test):
    '''
    INPUT: numpy arrays of X_train, y_train from train_test_split module
    OUTPUT: fitted model object and probabilities created by model
    '''
    model = LogisticRegression().fit(X_train, y_train)
    return model, model.predict_proba(X_test)

def random_Forest(X_train,y_train,X_test):
    '''
    INPUT: numpy arrays of X_train, y_train from train_test_split module
    OUTPUT: fitted model object and probabilities created by model
    '''
    model = RandomForestClassifier(n_estimators=300).fit(X_train, y_train)
    return model, model.predict_proba(X_test)

def plot_ROC(probabilities, labels):
    '''
    INPUT: Probabilities from any given model, numpy array of y_test data from
    test_train_split
    OUTPUT: Plotted ROC curve

    NOTE: uses sk_learn's roc_curve module to produce fpr and tpr
    '''
    fpr, tpr, thresholds = roc_curve(labels, probabilities[:,1])
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity, Recall)")
    plt.title("ROC plot of fake data")
    plt.plot(np.linspace(0,1,100), np.linspace(0,1,100),'k-', zorder=0)
    plt.show()
    return


if __name__ == '__main__':
    full, y = import_Data('data/churn_train.csv')
    X = mult_Binary_Features(full)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=101)
    model, probabilities = random_Forest(X_train, y_train,X_test)
    plot_ROC(probabilities, y_test.values)
    #print X.head()
