import sys
sys.path.append('/Users/hiro./.pyenv/versions/3.6.5/lib/python3.6/site-packages')
print(sys.path)
import csv

from functools import partial
import optuna
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error


def preprocess(train_x, test):

    data = pd.concat([train_x, test], axis=0)

    #地域平均に対する特徴量
    data["RatioArea_Frontage"] = data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x/x.mean())
    data["RatioArea_Lot"] = data.groupby("Neighborhood")["LotArea"].transform(lambda x: x/x.mean())
    data["RatioArea_Old"] = data.groupby("Neighborhood")["YearBuilt"].transform(lambda x: x/x.mean())
    data["RatioArea_1stSF"] = data.groupby("Neighborhood")["1stFlrSF"].transform(lambda x: x/x.mean())
    data["RatioArea_Rms"] = data.groupby("Neighborhood")["TotRmsAbvGrd"].transform(lambda x: x/x.mean())


    #cat_df =  data.select_dtypes(include=['object'])
    cat_df = data.iloc[:, [12, 15, 16, 30, 41, 53, 57, 58, 63]]
    cat_df = cat_df.fillna("missing")
    cat_df = pd.get_dummies(cat_df)
    cat_df = cat_df.drop("GarageQual_TA", axis=1)

    #num_df = data.select_dtypes(exclude=['object'])
    num_df = data.iloc[:, [1, 4, 17, 18, 19, 20, 43, 44, 46, 47, 51, 54, 56, 59, 61]]
    num_df = num_df.fillna(-999)

    '''
    #各種比率
    num_df["TotBath"] = num_df["FullBath"]+num_df["HalfBath"]
    num_df["ratio_fi"] = num_df["2ndFlrSF"]/num_df["GrLivArea"]

    num_df["diff_build_Reno"] = num_df["YearBuilt"] - num_df["YearRemodAdd"]
    #num_df["GarageRes"] = num_df["GarageArea"] - num_df["GarageCars"]
    num_df["GarageOld"] = num_df["YrSold"] - num_df["GarageYrBlt"]
    num_df["HouseOld"] = num_df["YrSold"] - num_df["YearBuilt"]
    num_df["HouseOld"] = num_df["YrSold"] - num_df["YearBuilt"]
    num_df["SF_open_ratio"] = num_df["WoodDeckSF"] / num_df["EnclosedPorch"] 
    num_df["RoomFireplacesRatio"] = num_df["Fireplaces"] / num_df["TotRmsAbvGrd"]

    num_df["BadRoomRatio"] =num_df["LowQualFinSF"]/(num_df["1stFlrSF"]+num_df["2ndFlrSF"])
    num_df["ratio_Bsmt"] = num_df["TotalBsmtSF"] / (num_df["1stFlrSF"] + num_df["2ndFlrSF"])

    num_df["ratio_kitchen"] = num_df["KitchenAbvGr"] / num_df["TotRmsAbvGrd"]
    num_df["ratio_Bedroom"] = num_df["BedroomAbvGr"] / num_df["TotRmsAbvGrd"]
    num_df["ratio_Bathroom"] = num_df["TotBath"] / num_df["TotRmsAbvGrd"]
    num_df["OtherRooms"] = num_df["TotRmsAbvGrd"] - num_df["KitchenAbvGr"] - num_df["BedroomAbvGr"]
    num_df["TotBsmtBath"] = num_df["BsmtFullBath"]+num_df["BsmtHalfBath"]
    '''

    X = pd.concat([num_df, cat_df], axis=1)
    #X = X.drop('Id', axis=1)

    train_X = X.iloc[:train_x.shape[0], :]
    test_X = X.iloc[train_x.shape[0]:, :]

    return train_X, test_X

def predict(bst,df):
    return bst.predict(xgb.DMatrix(df.as_matrix()))


def objective(dtrain, train_t_x, train_t_y, trial):

    (train_t_x, x, train_t_y, y) = train_test_split(train_t_x, train_t_y, test_size = 0.5, random_state = None)

    #目的関数
    params = {
        'objective' : 'reg:linear',
        'max_depth' : trial.suggest_int('max_depth',3,10),
        'learning_rate' : trial.suggest_uniform('learning_rate',0.001,0.5),
        'num_round' : trial.suggest_int('round_num',3,10),
        'gamma' : trial.suggest_uniform('gamma',0.0,10.0),
        'colsample_bytree' : trial.suggest_uniform('colsample_bytree',0.5,1.0),
        'min_childe_weigh' : trial.suggest_uniform('min_childe_weigh',0.9,1.0),
        'alpha' : trial.suggest_uniform('alpha',0.0,10.0)
    }
    bst = xgb.train(params, dtrain)
    answer = predict(bst, train_t_x)
    score = np.sqrt(mean_squared_log_error(answer,train_t_y))
    return score


def main(train_original, test_original):
    target_column = 'SalePrice'
    '''
    train = train_original
    train['MSSubClass'] = train['MSSubClass'].apply(str)
    train = train[~((train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000))]
    train = train[~(train['LotFrontage'] > 300)]
    train = train[~(train['LotArea'] > 100000)]
    train = train[~((train['OverallQual'] == 10) & (train['SalePrice'] < 200000))]
    train = train[~((train['OverallCond'] == 2) & (train['SalePrice'] > 300000))]
    train = train[~((train['OverallCond'] > 4) & (train['SalePrice'] > 700000))]
    train = train[~((train['YearBuilt'] < 1900) & (train['SalePrice'] > 200000))]
    train = train[~((train['YearRemodAdd'] < 2010) & (train['SalePrice'] > 600000))]
    train = train[~(train['BsmtFinSF1'] > 5000)]
    train = train[~(train['TotalBsmtSF'] > 5000)]
    '''
    df_y = train_original[target_column]
    df_train, df_test = preprocess(train_original.drop([target_column], axis = 1), test_original)
    (train_x, train_t_x, train_y, train_t_y) = train_test_split(df_train, df_y, test_size = 0.2, random_state = None)
    #(train_t_x, x, train_t_y, y) = train_test_split(train_t_x, train_t_y, test_size = 0.5, random_state = None)

    print(train_x.shape)
    print(train_t_x.shape)
    print(train_y.shape)
    print(train_t_y.shape)

    dtrain = xgb.DMatrix(train_x.as_matrix(),label=train_y)

    optuna_fg = True

    if optuna_fg:
        obj_f = partial(objective, dtrain, train_t_x, train_t_y)
        # セッション作成
        study = optuna.create_study()
        # 回数
        study.optimize(obj_f, n_trials=50)
        params = study.best_params
    else:
        params = {
            'objective' : 'reg:linear',
            'max_depth' : 6,
            'learning_rate' : 0.1,
            'num_round' : 3,
            'colsample_bytree' : 1,
            'min_childe_weigh' : 1,
            'gamma' : 1,
            'alpha' : 1,
            'silent' : 1
        }

    bst = xgb.train(params, dtrain)


    #print('\nparams :',study.best_params)
    answer = predict(bst,train_x)
    train_t_answer = predict(bst,train_t_x)
    t_answer = predict(bst,train_t_x)

    print('train score :',np.sqrt(mean_squared_log_error(answer,train_y)))    
    print('train_t score :',np.sqrt(mean_squared_log_error(train_t_answer,train_t_y)))
    print('test score :',np.sqrt(mean_squared_log_error(t_answer,train_t_y)))

    
    sub = pd.read_csv("sample_submission.csv")
    sub["SalePrice"] = predict(bst,df_test)
    return sub


if __name__ == '__main__':
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    answer = main(train, test)

    answer.to_csv("xgb.csv", index=False)
    '''
    f, ax = plt.subplots(figsize=[7,10])
    lgb.plot_importance(lightgbm, max_num_features=85, ax=ax)
    plt.title("Light GBM Feature Importance")
    plt.savefig('feature_import.png')
    '''



