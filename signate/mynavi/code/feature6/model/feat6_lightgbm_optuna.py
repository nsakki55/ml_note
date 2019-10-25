import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.options.display.max_columns = 200
import japanize_matplotlib

from time import time
import seaborn as sns
import pandas_profiling as pdp
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from IPython.display import display
import gc
import pickle

import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import optuna 

from logging import StreamHandler, DEBUG, Formatter, FileHandler, getLogger    

def rmse(y_true,y_pred):
    return np.sqrt(mean_squared_error(y_true,y_pred))


def opt(trial):
    n_estimators = 10000
    max_depth = trial.suggest_int('max_depth', 4, 10)
    min_child_weight = trial.suggest_int('min_child_weight', 1, 32)
    subsample = trial.suggest_discrete_uniform('subsample', 0.5, 0.9, 0.1)
    colsample_bytree = trial.suggest_discrete_uniform('colsample_bytree', 0.5, 0.9, 0.1)



    train=pd.read_feather('../data/train_feat6_all.ftr')
    test=pd.read_feather('../data/test_feat6_all.ftr')

    y_train=train.rent_log

    X_train=train.drop(['id','rent_log'],axis=1)
    X_test=test.drop(['id','rent_log'],axis=1)

    kfold=KFold(n_splits=4,shuffle=True,random_state=0)

    f=open('../data/feat6_category.pickle','rb')
    categorical_features=pickle.load(f)

    splits=5
    feature_importances = pd.DataFrame()

    cv={}
    y_preds = np.zeros(X_test.shape[0])

    for col in X_train.select_dtypes(include='category').columns:
        X_train[col]=X_train[col].astype(int)
        X_test[col]=X_test[col].astype(int)

    # クロスバリデーションのfoldごとにtarget encodingをやり直す
    for fold_n,(train_index,val_index) in enumerate(kfold.split(X_train)):
        cv_fold_start_time = time()
        print ('** Training fold {}'.format(fold_n + 1))

        X_trn,X_val=X_train.iloc[train_index],X_train.iloc[val_index]
        y_trn,y_val=y_train[train_index],y_train[val_index]
        eval_set  = [(X_trn,y_trn), (X_val, y_val)]
        
        for c in categorical_features:

            # 訓練データから、一時的に目的変数をもったDFを作成
            data_tmp_test=pd.DataFrame({c:X_train[c],'target':y_train})

            # 訓練データの各カテゴリの目的変数の平均値をとる
            targe_mean_test=data_tmp_test.groupby(c)['target'].mean()

            # テストデータに訓練データ全体での平均を入れる
            X_test[c+'_target']=X_test[c].map(targe_mean_test)


            # 訓練データから、一時的に目的変数をもったDFを作成
            data_tmp=pd.DataFrame({c : X_trn[c],'target' : y_trn})

            # 訓練データの各カテゴリの目的変数の平均値をとる
            targe_mean=data_tmp.groupby(c)['target'].mean()

            # テストデータに訓練データ全体での平均を入れる
            X_val.loc[:,c+'_target']=X_val[c].map(targe_mean)

            tmp=np.repeat(np.nan,X_trn.shape[0])
            
            # クロスバリデーションごとに訓練データのTarget Encoding 用のfold
            kfold_enc=KFold(n_splits=4,shuffle=True,random_state=0)
            
            for idx_1,idx_2 in kfold_enc.split(X_trn):
                
                # out-of-foldで各カテゴリにおける目的変数の平均を計算
                target_mean=data_tmp.iloc[idx_1].groupby(c)['target'].mean()
        
                # 変換後の値を一時配列に保存
                tmp[idx_2]=X_train[c].iloc[idx_2].map(target_mean)
            
            X_trn.loc[:,c+'_target']=tmp


        
        
        model_opt = lgb.LGBMRegressor(
            random_state=42,
            n_estimators = n_estimators,
            max_depth = max_depth,
            min_child_weight = min_child_weight,
            subsample = subsample,
            colsample_bytree = colsample_bytree,
            )

        model_opt.fit(X_trn, y_trn,
                eval_set=eval_set, 
                eval_metric="rmse",
                early_stopping_rounds=100,
                verbose= 500)




        val_pred=model_opt.predict(X_val)

        del X_trn, y_trn
        
        del model_opt, X_val
        val_rmse=rmse(np.exp(y_val)-1,np.exp(val_pred)-1)
        cv[fold_n+1]=val_rmse
        del val_pred,y_val,val_rmse

        gc.collect()
        
    cv=pd.DataFrame(cv,index=['cv',])
    print('CV RMSE:{}'.format(cv.mean(axis=1)[0]))
    return cv.mean(axis=1)[0]



def main():
    study = optuna.create_study()   
    study.optimize(opt, n_trials=30)

    print(study.best_params)
    print(study.best_value)

    logger.debug('best params:{}'.format(study.best_params))    
    logger.debug('best values:{}'.format(study.best_value))



if __name__=='__main__':

    

    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')
    
    # ハンドラはログ記録の適切な送り先等を決める
    handler = StreamHandler()

    # level よりも深刻でないログメッセージは無視される
    handler.setLevel('INFO')
    logger = getLogger(__name__)
    logger.addHandler(handler)

    # ログの保存先
    handler = FileHandler('feat6_lightgbm_optuna.py.log', 'a')

    # ログレベルをDEBUGに設定することで、コマンドラインにログが出力される
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)

    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    main()