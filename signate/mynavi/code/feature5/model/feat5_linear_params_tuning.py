import pandas as pd
import numpy as np
from IPython.display import display
from sklearn.linear_model import Ridge,Lasso,ElasticNet
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from time import time 
import gc
from argparse import ArgumentParser
from sklearn.model_selection import GridSearchCV
import optuna 

# Ridge alpha:.9831531431676297


def main(args):
    
    # optuna
    study=optuna.create_study()
    study.optimize(objective, n_trials=30)

    # 最適解
    print(study.best_params)
    print(study.best_value)
    print(study.best_trial)


def objective(trial):
    train=pd.read_feather('../data/train_feat5_scalling.ftr')
    test=pd.read_feather('../data/test_feat5_scalling.ftr')

    Y_train=train.rent
    X_train=train.drop(['id','rent'],axis=1)
    X_test=test.drop(['id','rent'],axis=1)

    if args.model=='ridge' or args.model=='lasso':
        # alpha
        param_alpha = trial.suggest_loguniform("alpha",0.01,0.99)

    elif args.model=='elasticnet':
        param_alpha=trial.suggest_loguniform("alpha",0.01,0.99)
        param_l1ratio=trial.suggest_loguniform("alpha",0.,1.)

    folds=KFold(n_splits=5)
    
    cv={}
    # CrossvalidationのMSEで比較（最大化がまだサポートされていない）
    for fold_n, (trn_idx, val_idx) in enumerate(folds.split(X_train,Y_train)):
        cv_fold_start_time = time()
        print ('** Training fold {}'.format(fold_n + 1))
        X_trn, X_val = X_train.iloc[trn_idx], X_train.iloc[val_idx]
        y_trn, y_val = Y_train[trn_idx], Y_train[val_idx]

        if args.model=='ridge':
            reg = Ridge(alpha=param_alpha)
        
        elif args.model=='lasso':
            reg=Lasso(alpha=param_alpha,tol=0.01)
        
        elif args.model=='elasticnet':
            reg=ElasticNet(alpha=param_alpha,l1_ratio=param_l1ratio)
        
        else:
            print('Unknown model')

        reg.fit(X_trn,y_trn)
        
        del X_trn, y_trn
        
        val_pred=reg.predict(X_val)
        
        del reg, X_val

        val_rmse=rmse(y_val,val_pred)
        print('RMSE accuracy: {}'.format(val_rmse))
        cv[fold_n+1]=val_rmse
        del val_pred,y_val,val_rmse

        gc.collect()
        
        cv_fold_end_time = time()
    print ('fold completed in {}s'.format(cv_fold_end_time - cv_fold_start_time))
    cv=pd.DataFrame(cv,index=['cv',])
    return cv.mean(axis=1)[0]


def rmse(y_true,y_pred):
    return np.sqrt(mean_squared_error(y_true,y_pred))

if __name__=='__main__':
    parser=ArgumentParser()
    parser.add_argument('model',type=str)
    args=parser.parse_args()

    main(args)