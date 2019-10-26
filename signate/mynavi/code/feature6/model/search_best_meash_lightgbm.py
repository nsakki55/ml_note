from make_mesh_feat6 import make_mesh
from IPython.display import display
from sklearn.model_selection import ParameterGrid, KFold
from sklearn.metrics import mean_squared_error
from logging import basicConfig, getLogger, DEBUG, INFO, Formatter, StreamHandler,FileHandler
import os, pickle, gc,warnings
warnings.filterwarnings('ignore')
from time import time 
import numpy as np
import pandas as pd

import lightgbm as lgb 



FOLD = 5

def rmse(y_true,y_pred):
    return np.sqrt(mean_squared_error(y_true,y_pred))

def rmse_lgb(preds,dtrain):
    y=dtrain
    score=rmse(y,preds)
    return 'rmse',score,True


def main():
    logger.info('read datasets')

    params_grid={'lat_bin':[i for i in range(5,26,5)],'long_bin':[i for i in range(5,26,5)]}

    mm=make_mesh()

    grid_rmse=dict()
    min_rmse=1e+10
    best_params=0

    for params in ParameterGrid(params_grid):
        logger.info('START {}'.format(params))
        train,test=mm.get_data(**params)
        
        

        y_train=train.rent_log

        X_train=train.drop(['id','rent_log'],axis=1)
        X_test=test.drop(['id','rent_log'],axis=1)

        kfold=KFold(n_splits=FOLD,shuffle=True,random_state=0)

        f=open('../data/feat6_category.pickle','rb')
        categorical_features=pickle.load(f)

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

            X_trn,X_val=X_train.loc[train_index,:],X_train.loc[val_index,:]
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

            params_lgb = {
                'n_estimators' : 10000,
                'max_depth' : 10,
                'min_child_weight' : 10,
                'subsample' : 0.5,
                'colsample_bytree': 0.7
            }


            reg = lgb.LGBMRegressor(**params_lgb)
            reg.fit(X_trn, y_trn,
                    eval_set=eval_set, 
                    eval_metric='rmse',
                    early_stopping_rounds=100,
                    #categorical_feature = categorical_features,
                    verbose= 500)
            
            
            val_pred=reg.predict(X_val)

            cv_preds[val_index]=np.exp(val_pred)-1

            del X_trn, y_trn
            
            y_preds+=(np.exp(reg.predict(X_test))-1)/FOLD
            del reg, X_val
            val_rmse=rmse(np.exp(y_val)-1,np.exp(val_pred)-1)
            logger.info('params: {}, {} fold, RMSE accuracy: {}'.format(params,fold_n+1,val_rmse))
            logger.debug('params:{}, {} fold, RMSE accuracy: {}'.format(params,fold_n+1,val_rmse))

            cv[fold_n+1]=val_rmse
            del val_pred,y_val,val_rmse

            gc.collect()
            
            cv_fold_end_time = time()
            logger.info('fold completed in {}s'.format(cv_fold_end_time - cv_fold_start_time))

        cv=pd.DataFrame(cv,index=['RMSE',])
        rmse_value=cv.mean(axis=1)[0]
        grid_rmse[str(params)]=rmse_value

        logger.info('params:{}, CV RMSE:{}'.format(params,rmse_value))
        logger.debug('parmas:{}, CV RMSE mean:{}'.format(params,rmse_value))

        if rmse_value<min_rmse:
            min_rmse=rmse_value
            best_params=params
        logger.info('END {}'.format(params))

        logger.debug('best parameter:{}'.format(best_params))   
        logger.info('best parameter:{}'.format(best_params))
        
    print(grid_rmse)
    pd.DataFrame(grid_rmse,index=['parmas',]).to_csv('grid_params_rmse2.csv')



if __name__=='__main__':

    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')
    
    # ハンドラはログ記録の適切な送り先等を決める
    handler = StreamHandler()

    # level よりも深刻でないログメッセージは無視される
    handler.setLevel('INFO')
    logger = getLogger(__name__)
    logger.addHandler(handler)

    log_file=os.path.basename(__file__)
    # ログの保存先
    handler = FileHandler(log_file+'.log', 'a')

    # ログレベルをDEBUGに設定することで、コマンドラインにログが出力される
    handler.setLevel('DEBUG')
    handler.setFormatter(log_fmt)

    logger.setLevel('DEBUG')
    logger.addHandler(handler)

    main()
