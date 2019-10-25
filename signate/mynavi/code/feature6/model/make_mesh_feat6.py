import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from IPython.display import display

class make_mesh:
    def __init__(self):

        self.train=pd.read_feather('../data/train_feat6_all.ftr')
        self.test=pd.read_feather('../data/test_feat6_all.ftr')
        self.df_all=pd.concat([self.train,self.test],axis=0,sort=False)

        self.train_index=len(self.train)
        self.test_index=len(self.test)


    def get_data(self,lat_bin,long_bin):
        self.df_all_feat6=self.df_all.copy()
        
        self.df_all_feat6['cat_latitude']=self.add_bin_latitude(self.df_all,lat_bin)
        self.df_all_feat6['cat_longitude']=self.add_bin_longitude(self.df_all,long_bin)

        self.df_all_feat6['mesh_category']=self.add_lat_long_category(self.df_all_feat6)
        self.df_all_feat6['mesh_category_enc']=self.mesh_encoder(self.df_all_feat6)
        self.df_all_feat6['mesh_category_enc']=self.df_all_feat6['mesh_category_enc'].astype('category')

        self.df_all_feat6.drop(['mesh_category','cat_latitude','cat_longitude'],axis=1,inplace=True)

        train_feat6=self.df_all_feat6[:self.train_index]
        test_feat6=self.df_all_feat6[self.train_index:]

        return train_feat6, test_feat6


    def add_bin_latitude(self,df,bins):
        return pd.cut(df['latitude'],bins).astype(str)


    def add_bin_longitude(self,df,bins):
        return pd.cut(df['longitude'],bins).astype(str)


    def add_lat_long_category(self,df):
        le_lat=LabelEncoder()
        df['cat_latitude'].fillna('nan',inplace=True)
        le_lat.fit(list(df['cat_latitude'].astype(str).values))
        lat_str=le_lat.transform(list(df['cat_latitude'].astype(str).values))

        le_long=LabelEncoder()
        df['cat_longitude'].fillna('nan',inplace=True)
        le_long.fit(list(df['cat_longitude'].astype(str).values))
        long_str=le_long.transform(list(df['cat_longitude'].astype(str).values))
        
        lat_long_category=[]
        for lat,long in zip(lat_str,long_str):
            lat_long_category.append(str(lat)+'_'+str(long))
            
        return lat_long_category

    def mesh_encoder(self,df):
        mesh_le=LabelEncoder()
        mesh_le.fit(df['mesh_category'])
        mesh_encode=mesh_le.transform(df['mesh_category'])
        return mesh_encode

