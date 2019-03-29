import sys
sys.path.append('/Users/hiro./.pyenv/versions/3.6.5/lib/python3.6/site-packages')


import matplotlib.pyplot as plt
import pandas as pd
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def preprocess(df):
    for i in df:
        print(type(df[i][0]) is str)
        if type(df[i][0]) is str:
            df = df.drop(i,axis=1)
    [print(type(df[i])) for i in df]
    print('-----------')
    [print(df[i][0]) for i in df]

    X = df.drop(target_column,axis=1)
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.1, random_state = 666)
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    return X_train_std, X_test_std, y_train, y_test

def main():
    global target_column
    df = pd.read_csv("train.csv")
    #for i in df:
        #print(df[i][9])

    X_train_std, X_test_std, y_train, y_test = preprocess(df)

    params = {
        'kernel': 'rbf',
        'C': 5,
        'gamma': 0.1,
    }

    mdl = SVC(**params)
    mdl.fit(X_train_std, y_train)

    pred_train = mdl.predict(X_train_std)
    accuracy_train = accuracy_score(y_train, pred_train)
    pred_test = mdl.predict(X_test_std)
    accuracy_test = accuracy_score(y_test, pred_test)
    print('train score :',accuracy_train)
    print('test score :',accuracy_test)

if __name__ == '__main__':
    target_column = 'SalePrice'
    main()