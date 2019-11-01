import pandas as pd
import numpy as np
from IPython.display import display
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from logging import StreamHandler, DEBUG, Formatter, FileHandler, getLogger
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from time import time 
import gc, pickle, os 
from argparse import ArgumentParser
from sklearn.model_selection import GridSearchCV
import optuna 
from fillna_feat6 import make_data
import lightgbm as lgb 


# Ridge
# Standard 
# value=27198.02311947381,
#'alpha': 0.010076789091771243

# Lasso
# Standard
# value=35010.66791745882,
# 'alpha': 0.010048643249333071}

FOLD = 4
SEED = 77
SCALER = 'standard'


def main():
    
    # optuna
    study=optuna.create_study()
    study.optimize(objective, n_trials=30)

    # 最適解
    print(study.best_params)
    print(study.best_value)
    print(study.best_trial)


def objective(trial):

    train, test = make_data().load_data()

    y_train = train.rent_log
    X_train = train.drop(['id','rent_log'],axis=1)
    X_test = test.drop(['id','rent_log'],axis=1)

    for col in X_train.select_dtypes(include = 'category').columns:
        X_train[col] = X_train[col].astype(int)
        X_test[col] = X_test[col].astype(int)

    n_estimators = 10000
    max_depth = trial.suggest_int('max_depth', 4, 10)
    min_child_weight = trial.suggest_int('min_child_weight', 1, 32)
    subsample = trial.suggest_discrete_uniform('subsample', 0.5, 0.9, 0.1)
    colsample_bytree = trial.suggest_discrete_uniform('colsample_bytree', 0.5, 0.9, 0.1)

    cv_rmse = {}
    cv_preds = np.zeros(X_train.shape[0])
    y_preds = np.zeros(X_test.shape[0])
      
    kfold=KFold(n_splits=FOLD,shuffle=True,random_state=SEED)

    category_path = '/Users/satsuki/kaggle/signate/mynavi/code/feature6/data/feat6_category.pickle'
    f=open(category_path,'rb')
    categorical_features=pickle.load(f)

    # クロスバリデーションのfoldごとにtarget encodingをやり直す
    for fold_n,(train_index,val_index) in enumerate(kfold.split(X_train)):
        cv_fold_start_time = time()
        X_test_fold = X_test.copy()
        logger.info('** Training fold {} **'.format(fold_n + 1))

        X_trn,X_val=X_train.iloc[train_index],X_train.iloc[val_index]
        y_trn,y_val=y_train[train_index],y_train[val_index]
        
        for c in categorical_features:

            # 訓練データから、一時的に目的変数をもったDFを作成
            data_tmp_test=pd.DataFrame({c:X_train[c],'target':y_train})

            # 訓練データの各カテゴリの目的変数の平均値をとる
            targe_mean_test=data_tmp_test.groupby(c)['target'].mean()

            # テストデータに訓練データ全体での平均を入れる
            X_test_fold[c+'_target']=X_test_fold[c].map(targe_mean_test)


            # 訓練データから、一時的に目的変数をもったDFを作成
            data_tmp=pd.DataFrame({c : X_trn[c],'target' : y_trn})

            # 訓練データの各カテゴリの目的変数の平均値をとる
            targe_mean=data_tmp.groupby(c)['target'].mean()

            # テストデータに訓練データ全体での平均を入れる
            X_val.loc[:,c+'_target']=X_val[c].map(targe_mean)

            tmp=np.repeat(np.nan,X_trn.shape[0])
            
            # クロスバリデーションごとに訓練データのTarget Encoding 用のfold
            kfold_enc = KFold(n_splits=4,shuffle=True,random_state=0)
            
            for idx_1,idx_2 in kfold_enc.split(X_trn):
                
                # out-of-foldで各カテゴリにおける目的変数の平均を計算
                target_mean=data_tmp.iloc[idx_1].groupby(c)['target'].mean()
        
                # 変換後の値を一時配列に保存
                tmp[idx_2]=X_train[c].iloc[idx_2].map(target_mean)
            
            X_trn.loc[:,c+'_target']=tmp
            
        # Target Encoding の欠損値を訓練データの平均で補完
        X_trn = X_trn.fillna(X_trn.mean())
        X_val = X_val.fillna(X_trn.mean())

        X_test_fold = X_test_fold.fillna(X_trn.mean())
        
        if SCALER == 'standard':
            sc = StandardScaler()
        elif SCALER == 'robust':
            sc = RobustScaler()
        elif SCALER == 'minmax':
            sc = MinMaxScaler()

        logger.info('train shape{}'.format(X_trn.shape))
        logger.info('val shape{}'.format(X_val.shape))
        logger.info('test shape{}'.format(X_test_fold.shape))

        sc.fit(X_trn)
        X_trn = sc.transform(X_trn)
        X_val = sc.transform(X_val) 
        X_test_fold = sc.transform(X_test_fold)
        
        eval_set  = [(X_trn,y_trn), (X_val, y_val)]
 
        reg = lgb.LGBMRegressor(
            random_state=42,
            n_estimators = n_estimators,
            max_depth = max_depth,
            min_child_weight = min_child_weight,
            subsample = subsample,
            colsample_bytree = colsample_bytree,
            )

        reg.fit(X_trn, y_trn,
                eval_set=eval_set, 
                eval_metric="rmse",
                early_stopping_rounds=100,
                verbose= 500)

        del X_trn, y_trn
        y_val_pred = reg.predict(X_val)

        cv_preds[val_index] = np.exp(y_val_pred)-1
        val_rmse = rmse(np.exp(y_val)-1, np.exp(y_val_pred)-1)
        cv_rmse[fold_n+1] = val_rmse

        del X_val, y_val_pred

        logger.info('{} fold RMSE:{}'.format(fold_n+1, val_rmse))
        
     #   logger.info('Test predict START')
      #  y_pred = reg.predict(X_test_fold)
      #  y_preds += (np.exp(y_pred)-1)/FOLD
      #  logger.info('Test predict END')

        #del y_pred

        gc.collect()
        
        cv_fold_end_time = time()
        logger.info('{} fold completed in {}s'.format(fold_n+1 ,cv_fold_end_time - cv_fold_start_time))
        logger.debug('{} fold completed in {}s'.format(fold_n+1 ,cv_fold_end_time - cv_fold_start_time))

    cv_rmse = pd.DataFrame(cv_rmse,index=['RMSE',])
    logger.info('CV RMSE:{}'.format(cv_rmse.mean(axis=1)[0]))
    logger.debug('CV RMSE mean:{}'.format(cv_rmse.mean(axis=1)[0]))

    
    return cv_rmse.mean(axis=1)[0]


def rmse(y_true,y_pred):
    return np.sqrt(mean_squared_error(y_true,y_pred))

if __name__=='__main__':

    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')
    
    # ハンドラはログ記録の適切な送り先等を決める
    handler = StreamHandler()

    # level よりも深刻でないログメッセージは無視される
    handler.setLevel('INFO')
    logger = getLogger(__name__)
    logger.addHandler(handler)

    # 保存先ファイル名
    log_path=os.path.basename(__file__)
    # ログの保存先
    handler = FileHandler(log_path+'.log', 'a')

    # ログレベルをDEBUGに設定することで、コマンドラインにログが出力される
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)

    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    main()