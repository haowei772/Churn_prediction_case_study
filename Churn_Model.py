import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import roc_curve, precision_recall_fscore_support, precision_score, recall_score
import ipdb
import seaborn as sns
sns.set()


def import_Data(filepath):
    '''
    INPUT: STRING of filepath to data file
    OUTPUT: X and y data, including the engineered 'churn' column

    ***dropped the last_trip_date and signup_date for now***
    ***GOT RID OF ALL NA VALUES***
    '''
    fullData = pd.read_csv(filepath)
    fullData['last_trip_date'] = pd.to_datetime(fullData['last_trip_date'])
    fullData['signup_date'] = pd.to_datetime(fullData['signup_date'])
    X = fullData
    return X


def feature_engineer(X):
    '''
    INPUT: pandas dataframe - X data (BEFORE SPLIT!!!!)
    OUTPUT: pandas dataframe - X data with engineered binary feature columns
    '''
    # creating dependent churn variables
    # called people churned if they hadn't used the service in the last month
    condition = X['last_trip_date'] < '2014-06-01'
    X['churn'] = 1
    X.ix[~condition, 'churn'] = 0

    # X['no_ratings'] = X['avg_rating_of_driver'].isnull()*1 # attempted
    # feature left out

    # creating a feature of people who only used the service once
    used_once = []
    for num in xrange(len(X)):
        used_once.append(
            ((X['last_trip_date'][num] - X['signup_date'][num]).days > 2) * 1)
    X['used_once'] = pd.Series(used_once)

    X.drop(X[['last_trip_date', 'signup_date']], axis=1, inplace=True)

    # Filling missing values, check presentation for our
    condition3 = X['avg_rating_of_driver'].isnull()
    X.ix[condition3, 'avg_rating_of_driver'] = 0.5 * \
        X['avg_rating_of_driver'].mean()

    condition4 = X['avg_rating_by_driver'].isnull()
    X.ix[condition4, 'avg_rating_by_driver'] = 0.5 * \
        X['avg_rating_by_driver'].mean()

    # I think this is pretty self expanatory, check git for our presentation
    # on why we chose our features
    X['total_distance'] = X['trips_in_first_30_days'] * X['avg_dist']

    #X['compatability'] = X['avg_rating_of_driver']*X['avg_rating_by_driver']

    # X.dropna(inplace=True)
    # Creating dependent variable object and dropping it from independent
    # variable df
    y = X['churn']
    X.drop('churn', axis=1, inplace=True)

    # creating dummy variables
    X['city'].unique()
    df_city = pd.get_dummies(X['city'])
    X['phone'].unique()
    df_phone = pd.get_dummies(X['phone'])
    X = pd.concat([X, df_city], axis=1)
    X = pd.concat([X, df_phone], axis=1)

    # Dropping all extra columns not being used
    X.drop(X[['city', 'phone', 'avg_surge', 'iPhone', 'Winterfell']],
           axis=1, inplace=True)
    return X, y


def log_Reg(X_train, y_train, X_test):
    '''
    INPUT: numpy arrays of X_train, y_train from train_test_split module
    OUTPUT: fitted model object and probabilities created by model
    '''
    model = LogisticRegression().fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)
    return model, probabilities

def random_Forest(X_train, y_train, X_test):
    '''
    INPUT: numpy arrays of X_train, y_train from train_test_split module
    OUTPUT: fitted model object and probabilities created by model
    '''
    model = RandomForestClassifier(n_estimators=50).fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)
    return model, probabilities

def gradient_Boosting(X_train, y_train, X_test):
    model = GradientBoostingClassifier(n_estimators=200, max_features=1.0,
                                       learning_rate=0.1, max_depth=4, min_samples_leaf=17).fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)
    '''predict_proba() give 2 completentary probabilities'''
    #print probabilities
    return model, probabilities
''' Maybe useful
class init:
    def __init__(self, est):
        self.est = est
    def predict(self, X):
        return self.est.predict_proba(X)[:,1][:,numpy.newaxis]
    def fit(self, X, y):
        self.est.fit(X, y)

'''
def plot_ROC(probabilities, labels):
    '''
    INPUT: probabilities from any given model, numpy array of y_test data from
    test_train_split
    OUTPUT: Plotted ROC curve

    NOTE: uses sk_learn's roc_curve module to produce fpr and tpr
    '''
    fpr, tpr, thresholds = roc_curve(labels, probabilities[:, 1])
    '''fpr, tpr are calculated based on the thresholds (the probabilities)'''
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity, Recall)")
    plt.title("Rideshare Gradient Boost ROC plot")
    plt.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), 'k-', zorder=0)
    plt.show()
    return


def plot_feature_importance(model, X):
    '''
    INPUT: fitted model object, pandas df of indicator variables
    OUTPUT: Graph of feature importances for model
    '''
    importances = model.feature_importances_
    std = np.std([model.feature_importances_ for tree in model.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X.shape[1]):
        print("%d. feature %d (%f)" %
              (f + 1, indices[f], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices],
            color="r", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), list(X), rotation='vertical')
    plt.xlim([-1, X.shape[1]])
    plt.show()
    return


def GridSearch(X, y):
    '''
    Grid Search you can figure this one out, I believe in you
    '''
    param_grid = {'n_estimators': [100, 200], 'learning_rate': [0.1, 0.05], 'max_depth': [
        1, 2, 4], 'min_samples_leaf': [9, 17], 'max_features': [1.0, 0.3]}
    gsearch1 = GridSearchCV(GradientBoostingClassifier(), param_grid).fit(X, y)
    return gsearch1.best_params_, gsearch1.best_score_

if __name__ == '__main__':
    file_path1 = '/Users/haowei/Documents/gal/Day_37_case_study2/case_study2/Churn/churn_train.csv'
    file_path2 = '/Users/haowei/Documents/gal/Day_37_case_study2/case_study2/Churn/churn_test.csv'
    full = import_Data(file_path1)
    full2 = import_Data(file_path2)
    X, y = feature_engineer(full)
    X2, y2 = feature_engineer(full2)
    #X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=101)
    model, probabilities = gradient_Boosting(X, y, X2)
    #print model.score(X2, y2), precision_score(y2, model.predict(X2)), recall_score(y2, model.predict(X2))
    plot_ROC(probabilities, y2.values)
    plot_feature_importance(model, X)
