import pandas as pd
import numpy as np
from IPython.display import display
from sklearn.linear_model import Ridge,Lasso,ElasticNet
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from time import time 
import gc
from argparse import ArgumentParser


def main(args):
    train=pd.read_feather('../data/train_feat5_scalling_continuous.ftr')
    test=pd.read_feather('../data/test_feat5_scalling_continuous.ftr')

    Y_train=train.rent
    X_train=train.drop(['id','rent'],axis=1)
    X_test=test.drop(['id','rent'],axis=1)

    splits = 5
    folds = KFold(n_splits=splits,shuffle=True)

    training_start_time = time()

    cv={}
    y_preds = np.zeros(X_test.shape[0])

    for fold_n, (trn_idx, val_idx) in enumerate(folds.split(X_train,Y_train)):
        cv_fold_start_time = time()
        print ('** Training fold {}'.format(fold_n + 1))
        X_trn, X_val = X_train.iloc[trn_idx], X_train.iloc[val_idx]
        y_trn, y_val = Y_train[trn_idx], Y_train[val_idx]
        eval_set  = [(X_trn,y_trn), (X_val, y_val)]
        
        reg=Ridge(alpha=0.9831531431676297)
        reg.fit(X_trn,y_trn)
        
        del X_trn, y_trn
        
        val_pred=reg.predict(X_val)
        
        y_preds+=reg.predict(X_test)/splits
        del reg, X_val
        val_rmse=rmse(y_val,val_pred)
        print('RMSE accuracy: {}'.format(val_rmse))
        cv[fold_n+1]=val_rmse
        del val_pred,y_val,val_rmse

        gc.collect()
        
        cv_fold_end_time = time()
        print ('fold completed in {}s'.format(cv_fold_end_time - cv_fold_start_time))
    cv=pd.DataFrame(cv,index=['cv',])
    print('CV RMSE:{}'.format(cv.mean(axis=1)[0]))

    sub=pd.read_csv('../../../input/sample_submit.csv',header=None)
    sub[1]=y_preds
    save_path='feat5_scalling_default_params_{}_CV=={}'.format(args.model,cv.mean(axis=1)[0])
    sub.to_csv(save_path,index=False,header=False)

def rmse(y_true,y_pred):
    return np.sqrt(mean_squared_error(y_true,y_pred))

if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('model', type=str,help='Lasso Ridge ElasticNet')

    args = parser.parse_args()  
    main(args)