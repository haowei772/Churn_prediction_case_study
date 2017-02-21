import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve
import ipdb

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
    X['city'].unique()
    df_city = pd.get_dummies(X['city'])
    X['phone'].unique()
    df_phone = pd.get_dummies(X['phone'])
    X = pd.concat([X,df_city], axis = 1)
    X = pd.concat([X,df_phone], axis = 1)
    X.drop(X[['city','phone']], axis=1, inplace=True)
    return X

def Log_Reg(X_train,y_train,X_test):
    '''
    INPUT: numpy arrays of X_train, y_train from train_test_split module
    OUTPUT: fitted model object and probabilities created by model
    '''
    model = LogisticRegression().fit(X_train, y_train)
    return model, model.predict_proba(X_test)

def ROC_curve(probabilities, labels):
    '''
    Probabilites: LIST of values between (0,1) returned from logistic regression
    Labels: LIST of true values either 0 or 1
    The indexes are synced, so probabilites[0] is assigned to labels[0]
    Function returns:
    LIST of FLOATS - True Positive Rate (TPR = TP/(TP + FN)),
    LIST of FLOATS - False Positive Rate (FPR = FP/(TN + FP)),
    LIST of FLOATS - thresholds which is the probabilities sorted
    '''
    thresholds = probabilities[:,1]
    thresholds.sort()
    TPR = []
    FPR = []
    for t in thresholds:
        TP = 0
        FN = 0
        FP = 0
        TN = 0
        for n1,p in enumerate(probabilities[:,1]):
            if t > p:
                if labels[n1] == 0:
                    TN += 1.
                else:
                    FN += 1.
            else:
                if labels[n1] == 0:
                    FP += 1.
                else:
                    TP += 1.
        if (TP + FN) == 0:
            TPR.append(0)
        else:
            TPR.append(TP/(TP + FN))
        if (FP + TN) == 0:
            FPR.append(0)
        else:
            FPR.append(FP/(FP + TN))
    return TPR, FPR, thresholds

def plot_ROC(probabilities, labels):
    '''
    INPUT: Probabilities from any given model, numpy array of y_test data from
    test_train_split
    OUTPUT: Plotted ROC curve

    NOTE: uses sk_learn's roc_curve module to produce fpr and tpr
    '''
    fpr, tpr, thresholds = ROC_curve(probabilities, labels)
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity, Recall)")
    plt.title("ROC plot of fake data")
    plt.show()
    return


if __name__ == '__main__':
    full, y = import_Data('data/churn_train.csv')
    #print len(full)
    X = mult_Binary_Features(full)
    #print len(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=101)
    model, probabilities = LogisticRegression(X_train, y_train,X_test)
    plot_ROC(probabilities, y_test)
