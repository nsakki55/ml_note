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
from logging import basicConfig, getLogger, DEBUG, INFO


FOLD = 5

def rmse(y_true,y_pred):
    return np.sqrt(mean_squared_error(y_true,y_pred))

def rmse_lgb(preds,dtrain):
    y=dtrain
    score=rmse(y,preds)
    return 'rmse',score,True


def main():



    logger.info('read datasets')
    train=pd.read_feather('../data/train_feat6_all.ftr')
    test=pd.read_feather('../data/test_feat6_all.ftr')

    y_train=train.rent_log

    X_train=train.drop(['id','rent_log'],axis=1)
    X_test=test.drop(['id','rent_log'],axis=1)

    kfold=KFold(n_splits=FOLD,shuffle=True,random_state=0)

    f=open('../data/feat6_category.pickle','rb')
    categorical_features=pickle.load(f)

    feature_importances = pd.DataFrame()

    cv_preds=np.zeros(X_train.shape[0])
    y_preds = np.zeros(X_test.shape[0])

    cv={}

    for col in X_train.select_dtypes(include='category').columns:
        X_train[col]=X_train[col].astype(int)
        X_test[col]=X_test[col].astype(int)

    # クロスバリデーションのfoldごとにtarget encodingをやり直す
    for fold_n,(train_index,val_index) in enumerate(kfold.split(X_train)):
        cv_fold_start_time = time()
        logger.info('** Training fold {} **'.format(fold_n + 1))

        X_trn,X_val=X_train.iloc[train_index],X_train.iloc[val_index]
        y_trn,y_val=y_train[train_index],y_train[val_index]
        eval_set  = [(X_trn,y_trn), (X_val, y_val)]
        
        for c in categorical_features:

            # 訓練データから、一時的に目的変数をもったDFを作成
            data_tmp_test=pd.DataFrame({c:X_train[c],'target':y_train})

            # 訓練データの各カテゴリの目的変数の平均値をとる
            targe_median_test=data_tmp_test.groupby(c)['target'].median()

            # テストデータに訓練データ全体での中央値を入れる
            X_test[c+'_target']=X_test[c].map(targe_median_test)


            # 訓練データから、一時的に目的変数をもったDFを作成
            data_tmp=pd.DataFrame({c : X_trn[c],'target' : y_trn})

            # 訓練データの各カテゴリの目的変数の平均値をとる
            targe_median=data_tmp.groupby(c)['target'].median()

            # テストデータに訓練データ全体での平均を入れる
            X_val.loc[:,c+'_target']=X_val[c].map(targe_median)

            tmp=np.repeat(np.nan,X_trn.shape[0])
            
            # クロスバリデーションごとに訓練データのTarget Encoding 用のfold
            kfold_enc=KFold(n_splits=4,shuffle=True,random_state=0)
            
            for idx_1,idx_2 in kfold_enc.split(X_trn):
                
                # out-of-foldで各カテゴリにおける目的変数の平均を計算
                target_mean=data_tmp.iloc[idx_1].groupby(c)['target'].mean()
        
                # 変換後の値を一時配列に保存
                tmp[idx_2]=X_train[c].iloc[idx_2].map(target_mean)
            
            X_trn.loc[:,c+'_target']=tmp

        params = {
            'max_bin' : 63,
            'n_estimators' : 10000,
            'learning_rate': 0.01,
            'min_data_in_leaf' : 50,
            'num_leaves' : 100,
            'sparse_threshold' : 1.0,
            'device' : 'cpu',
            'save_binary': True,
            'seed' : 42,
            'feature_fraction_seed': 42,
            'bagging_seed' : 42,
            'drop_seed' : 42,
            'data_random_seed' : 42,
            'objective' : 'regression',
            'boosting_type' : 'gbdt',
            'verbose' : 0,
            'metric' : 'RMSE',
            'is_unbalance' : True,
            'boost_from_average' : False,
                }
        
        
        reg = lgb.LGBMRegressor(**params)
        reg.fit(X_trn, y_trn,
                eval_set=eval_set, 
                eval_metric='rmse',
                early_stopping_rounds=100,
                #categorical_feature = categorical_features,
                verbose= 500)
        
        
        val_pred=reg.predict(X_val)

        cv_preds[val_index]=np.exp(val_pred)-1

        feature_importances['feature'] = X_trn.columns
        feature_importances['fold_{}'.format(fold_n + 1)] = reg.feature_importances_
        
        del X_trn, y_trn
        
        y_preds+=(np.exp(reg.predict(X_test))-1)/FOLD
        del reg, X_val
        val_rmse=rmse(np.exp(y_val)-1,np.exp(val_pred)-1)
        logger.info('RMSE accuracy: {}'.format(val_rmse))
        logger.debug('{} fold, RMSE accuracy: {}'.format(fold_n+1,val_rmse))
        cv[fold_n+1]=val_rmse
        del val_pred,y_val,val_rmse

        gc.collect()
        
        cv_fold_end_time = time()
        logger.info('fold completed in {}s'.format(cv_fold_end_time - cv_fold_start_time))

    logger.info(cv_preds)
    cv=pd.DataFrame(cv,index=['cv',])
    logger.info('CV RMSE:{}'.format(cv.mean(axis=1)[0]))
    logger.debug('CV RMSE mean:{}'.format(cv.mean(axis=1)[0]))

    feature_importances['average'] = feature_importances.mean(axis=1)
    feature_importances.to_feather('feat6_lighgbm_importance.ftr')
    plt.figure(figsize=(16, 16))
    sns.barplot(data=feature_importances.sort_values(by='average', ascending=False).head(50), x='average', y='feature')
    plt.title('50 TOP feature importance over cv folds average')
    plt.savefig('feat6_lighgbm_importance.png')

    sub=pd.read_csv('../../../input/sample_submit.csv',header=None)
    sub[1]=y_preds
    sub.to_csv('feature6_lightgbm_CV={:.4f}.csv'.format(cv.mean(axis=1)[0]),index=False,header=False)

    f = open('train_feat6_lightgbm_train_CV.pickle', 'wb')
    pickle.dump(cv_preds, f)
    f.close()


if __name__=='__main__':

    from logging import StreamHandler, DEBUG, Formatter, FileHandler

    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')
    
    # ハンドラはログ記録の適切な送り先等を決める
    handler = StreamHandler()

    # level よりも深刻でないログメッセージは無視される
    handler.setLevel('INFO')
    logger = getLogger(__name__)
    logger.addHandler(handler)

    # ログの保存先
    handler = FileHandler('feat6_lightgbm_target_median.py.log', 'a')

    # ログレベルをDEBUGに設定することで、コマンドラインにログが出力される
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)

    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    main()