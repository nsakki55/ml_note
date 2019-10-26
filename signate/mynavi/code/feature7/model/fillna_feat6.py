import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder,StandardScaler,MinMaxScaler,RobustScaler

TRAIN_PATH =  '/Users/satsuki/kaggle/signate/mynavi/code/feature6/data/train_feat6_all.ftr'
TEST_PATH = '/Users/satsuki/kaggle/signate/mynavi/code/feature6/data/test_feat6_all.ftr'

class make_data:
    def __init__(self):

        self.train_feat6 = pd.read_feather(TRAIN_PATH)
        self.test_feat6 = pd.read_feather(TEST_PATH)
        self.train_index=len(self.train_feat6)
        self.test_index=len(self.test_feat6)
        self.df_all_feat6=pd.concat([self.train_feat6,self.test_feat6], axis=0, sort=False)


    def load_data(self):
        df_all_feat7 = self.modify_null(self.df_all_feat6)
        train_feat7 = df_all_feat7[:self.train_index]
        test_feat7 = df_all_feat7[self.train_index:]

        return train_feat7, test_feat7

    def modify_null(self,df):

        # 部屋の階を最頻値で補完
        df['room_floor']=df['room_floor'].replace(-999,df.query('room_floor !=-999')['room_floor'].mode()[0])

        # 建物の階を最頻値で補完
        df['building_floor']=df['building_floor'].replace(-999,df.query('building_floor !=-999')['building_floor'].mode()[0])

        # 建物の全ての階を最頻値で補完
        df['total_floor']=df['total_floor'].replace(-999,df.query('total_floor !=-999')['total_floor'].mode()[0])

        # 駐車場料金を0で補完
        df['parking_price_car']=df['parking_price_car'].replace(-999,0)

        # 駐車場料金を0で補完
        df['parking_price_bicycle']=df['parking_price_bicycle'].replace(-999,0)

        # 駐車場料金を0で補完
        df['parking_price_bike']=df['parking_price_bike'].replace(-999,0)

        def is_null(x):
            if x==-999:
                return 1
            else:
                return 0

        # 駐車台数を０で補完、欠損かどうかの特徴を追加
        for col in ['parking_number_car','parking_number_bicycle','parking_number_bike']:
            df[col+'_isnull']=df[col].apply(lambda x:is_null(x))
            df[col]=df[col].replace(-999,0)

        #  駐車場の有無をワンホットエンコ
        dummy_cols=['is_parking_car','is_house_parking_car','is_other_parking_car',
                    'is_parking_bicycle','is_house_parking_bicycle','is_other_parking_bicycle',
                'is_parking_bike','is_house_parking_bike','is_other_parking_bike']

        for col in dummy_cols:
            df[col]=df[col].astype(str)
            df=pd.get_dummies(df,columns=[col])

        # コンビニの数を最頻値で補完
        df['convenience_count']=df['convenience_count'].replace(-999,df.query('convenience_count !=-999')['convenience_count'].mode()[0])

        # スーパーの数を最頻値で補完
        df['supermarket_count']=df['supermarket_count'].replace(-999,df.query('supermarket_count !=-999')['supermarket_count'].mode()[0])

        # スーパーの数を最頻値で補完
        df['neighbor_count']=df['neighbor_count'].replace(-999,df.query('neighbor_count !=-999')['neighbor_count'].mode()[0])

        # 部屋の階と建物の階の積を平均で補完
        df['room_building_mul']=df['room_building_mul'].replace(-999,df.query('room_building_mul !=-999')['room_building_mul'].mean())

        # 面積のログと階の積を平均で補完
        df['square_log_building_mul']=df['square_log_building_mul'].replace(-999,df.query('square_log_building_mul !=-999')['square_log_building_mul'].mean())
        
        return df

