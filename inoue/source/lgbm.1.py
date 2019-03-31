import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error

from functools import partial
import optuna


def preprocess(train_x, test):

    data = pd.concat([train_x, test], axis=0)

    cat_df =  data.select_dtypes(include=['object'])
    #cat_df = df.iloc[:, [12, 15, 16, 30, 41, 53, 57, 58, 63]]
    cat_df = cat_df.fillna("missing")
    cat_df = pd.get_dummies(cat_df)
    cat_df = cat_df.drop("GarageQual_TA", axis=1)

    num_df = data.select_dtypes(exclude=['object'])
    #num_df = df.iloc[:, [1, 4, 17, 18, 19, 20, 43, 44, 46, 47, 51, 54, 56, 59, 61]]
    num_df = num_df.fillna(-999)

    X = pd.concat([num_df, cat_df], axis=1)
    X = X.drop(['Id','PoolArea'], axis=1)

    train_X = X.iloc[:train_x.shape[0], :]
    test_X = X.iloc[train_x.shape[0]:, :]

    return train_X, test_X


def objective(df_x, df_y, test_x, test_y, trial):
    
    (test_x, x, test_y, y) = train_test_split(test_x, test_y, test_size = 0.5, random_state = None)

    #目的関数
    params = {
        'num_leaves' : trial.suggest_int('num_leaves',2,8),
        'learning_rate' : trial.suggest_uniform('learning_rate',0.01,0.2), 
        'n_estimators' : trial.suggest_int('n_estimators', 50, 5000),
        'num_iteration' : trial.suggest_int('num_iteration', 50, 150),
        'feature_fraction' : trial.suggest_uniform('feature_fraction',0.1,0.9),
        'bagging_fraction' : trial.suggest_uniform('bagging_fraction',0.1,0.9),
        'bagging_freq' : trial.suggest_int('bagging_freq', 0, 10),
        'max_bin' : trial.suggest_int('max_bin', 150, 250),
        'bagging_seed' : trial.suggest_int('bagging_seed', 7, 7),
        'feature_fraction_seed' : trial.suggest_int('feature_fraction_seed', 7, 7),
        'verbose' : trial.suggest_int('verbose', -1,-1)
    }
    lightgbm = LGBMRegressor(**params)

    lightgbm.fit(df_x, df_y)

    answer = lightgbm.predict(test_x)
    score = np.sqrt(mean_squared_log_error(answer,test_y))
    return score


def main(train_original, test_original):

    target_column = 'SalePrice'
    train_original['MSSubClass'] = train_original['MSSubClass'].apply(str)
    train_original = train_original[~((train_original['GrLivArea'] > 4000))]
    train_original = train_original[~((train_original['GrLivArea'] > 4000) & (train_original['SalePrice'] < 300000))]
    train_original = train_original[~(train_original['LotFrontage'] > 300)]
    train_original = train_original[~(train_original['LotArea'] > 100000)]
    train_original = train_original[~((train_original['OverallQual'] == 10) & (train_original['SalePrice'] < 200000))]
    train_original = train_original[~((train_original['OverallCond'] == 2) & (train_original['SalePrice'] > 300000))]
    train_original = train_original[~((train_original['OverallCond'] > 4) & (train_original['SalePrice'] > 700000))]
    train_original = train_original[~((train_original['YearBuilt'] < 1900) & (train_original['SalePrice'] > 200000))]
    train_original = train_original[~((train_original['YearRemodAdd'] < 2010) & (train_original['SalePrice'] > 600000))]
    train_original = train_original[~(train_original['BsmtFinSF1'] > 5000)]
    train_original = train_original[~(train_original['TotalBsmtSF'] > 5000)]


    df_y = train_original[target_column]
    df_train, df_test = preprocess(train_original.drop([target_column], axis = 1), test_original)
    #(train_x, train_t_x, train_y, train_t_y) = train_test_split(df_train, df_y, test_size = 0.1, random_state = None)
    #(train_t_x, x, train_t_y, y) = train_test_split(train_t_x, train_t_y, test_size = 0.5, random_state = None)


    optuna_fg = False

    if optuna_fg:
        obj_f = partial(objective, train_x, train_y, train_t_x, train_t_y)
        # セッション作成
        study = optuna.create_study()
        # 回数
        study.optimize(obj_f, n_trials=30)
        params = study.best_params
    else:
        params = {
            "num_leaves":4,
            "learning_rate":0.01, 
            "n_estimators":5000,
            "max_bin":200, 
            "bagging_fraction":0.75,
            "bagging_freq":5, 
            "bagging_seed":7,
            "feature_fraction":0.2,
            "feature_fraction_seed":7,
            "verbose":-1
        }

    lightgbm = LGBMRegressor(**params)
    lightgbm.fit(df_train, df_y)

    #train_answer = lightgbm.predict(df_test)
    #print('train score :',np.sqrt(mean_squared_log_error(train_answer,train_y)))
    #train_t_answer = lightgbm.predict(train_t_x)
    #print('train_t score :',np.sqrt(mean_squared_log_error(train_t_answer,train_t_y)))
    #t_answer = lightgbm.predict(train_t_x)
    #print('test score :',np.sqrt(mean_squared_log_error(t_answer,train_t_y)))
    

    sub = pd.read_csv("sample_submission.csv")
    sub["SalePrice"] = lightgbm.predict(df_test)
    return sub



if __name__ == '__main__':
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    answer = main(train, test)

    answer.to_csv("lgbm_1.csv", index=False)
    '''
    f, ax = plt.subplots(figsize=[7,10])
    lgb.plot_importance(lightgbm, max_num_features=85, ax=ax)
    plt.title("Light GBM Feature Importance")
    plt.savefig('feature_import.png')
    '''



