from sklearn.preprocessing import StandardScaler,RobustScaler, MinMaxScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers.core import Dense, Layer, Dropout, Activation
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.layers.advanced_activations import ELU, PReLU
from tensorflow.python.keras.optimizers import SGD
from tensorflow.python.keras.utils import np_utils, generic_utils
from tensorflow.keras.callbacks import EarlyStopping

import tensorflow as tf
from tensorflow.keras.models import load_model

import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from fillna_feat6 import make_data
import pickle, gc, os 
from time import time
from logging import StreamHandler, DEBUG, Formatter, FileHandler, getLogger
import numpy as np
from tensorflow.python import debug as tf_debug

from tensorflow.python.debug.lib.debug_data import has_inf_or_nan

from keras import backend as K



FOLD=5
SEED=0
SCALER='standard'


class KerasDNNRegressor:
    def __init__(self, input_dropout=0.2, hidden_layers=2, hidden_units=64, 
                hidden_activation="relu", hidden_dropout=0.5, batch_norm=None, 
                optimizer="adadelta", epochs=50, batch_size=64):
        self.input_dropout = input_dropout
        self.hidden_layers = hidden_layers
        self.hidden_units = hidden_units
        self.hidden_activation = hidden_activation
        self.hidden_dropout = hidden_dropout
        self.batch_norm = batch_norm
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.scaler = None
        self.model = None

    def __str__(self):
            return self.__repr__()

    def __repr__(self):
        return ("%s(input_dropout=%f, hidden_layers=%d, hidden_units=%d, \n"
                    "hidden_activation=\'%s\', hidden_dropout=%f, batch_norm=\'%s\', \n"
                    "optimizer=\'%s\', epochs=%d, batch_size=%d)" % (
                    self.__class__.__name__,
                    self.input_dropout,
                    self.hidden_layers,
                    self.hidden_units,
                    self.hidden_activation,
                    self.hidden_dropout,
                    str(self.batch_norm),
                    self.optimizer,
                    self.epochs,
                    self.batch_size,
                ))


    def fit(self, X, y, X_val, y_val):
        ## scaler
#        self.scaler = StandardScaler()
#        X = self.scaler.fit_transform(X)

        #### build model
        self.model = Sequential()
        ## input layer
        self.model.add(Dropout(self.input_dropout, input_shape=(X.shape[1],)))
        ## hidden layers
        first = True
        hidden_layers = self.hidden_layers
        while hidden_layers > 0:
            self.model.add(Dense(self.hidden_units))
            if self.batch_norm == "before_act":
                self.model.add(BatchNormalization())
            if self.hidden_activation == "prelu":
                self.model.add(PReLU())
            elif self.hidden_activation == "elu":
                self.model.add(ELU())
            else:
                self.model.add(Activation(self.hidden_activation))
            if self.batch_norm == "after_act":
                self.model.add(BatchNormalization())
            self.model.add(Dropout(self.hidden_dropout))
            hidden_layers -= 1

        ## output layer
        output_dim = 1
        output_act = "linear"
        self.model.add(Dense(output_dim))
        self.model.add(Activation(output_act))
        
        ## loss
        if self.optimizer == "sgd":
            sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
            self.model.compile(loss="mse", optimizer=sgd)
        else:
            self.model.compile(loss="mse", optimizer=self.optimizer)

        ## callback
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')
        
        ## fit
        self.model.fit(X, y,
                    epochs=self.epochs, 
                    batch_size=self.batch_size,
                    validation_data=[X_val,y_val], 
                    callbacks=[early_stopping],
                    verbose=1)
        return self

    def predict(self, X):
        X = self.scaler.transform(X)
        y_pred = self.model.predict(X)
        y_pred = y_pred.flatten()
        return y_pred


def set_debugger_session():
    sess = K.get_session()
    sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    sess.add_tensor_filter('has_inf_or_nan', has_inf_or_nan)
    K.set_session(sess)



def main():
    
#    set_debugger_session()

    md = make_data()
    train, test = md.load_data()

    y_train = train.rent_log
    X_train = train.drop(['id','rent_log'],axis=1)
    X_test = test.drop(['id','rent_log'],axis=1)

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
            kfold_enc=KFold(n_splits=4,shuffle=True,random_state=0)
            
            for idx_1,idx_2 in kfold_enc.split(X_trn):
                
                # out-of-foldで各カテゴリにおける目的変数の平均を計算
                target_mean=data_tmp.iloc[idx_1].groupby(c)['target'].mean()
        
                # 変換後の値を一時配列に保存
                tmp[idx_2]=X_train[c].iloc[idx_2].map(target_mean)
            
            X_trn.loc[:,c+'_target']=tmp
        
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
        
        pd.DataFrame(X_trn).to_csv('train.csv')

        nn=KerasDNNRegressor()

        logger.info('Training START')
        nn.fit(X_trn ,y_trn.values, X_val, y_val.values)


if __name__=='__main__':

    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')
    
    # ハンドラはログ記録の適切な送り先等を決める
    handler = StreamHandler()

    # level よりも深刻でないログメッセージは無視される
    handler.setLevel('INFO')
    logger = getLogger(__name__)
    logger.addHandler(handler)

    log_path=os.path.basename(__file__)
    # ログの保存先
    handler = FileHandler(log_path+'.log', 'a')

    # ログレベルをDEBUGに設定することで、コマンドラインにログが出力される
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)

    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    main()