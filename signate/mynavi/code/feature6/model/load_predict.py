import pickle 
import pandas as pd


f = open('train_feat6_lightgbm_train_CV.pickle','rb')
pred=pickle.load(f)
print(len(pred))