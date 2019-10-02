# kaggle
## 機械学習で役立つTips
LightGBMと他のブースティングのまとめ（PyDataでのスライド）  
XGBoostと比べ、並列化による高速化、大規模データセットへの対応が可能となった。  
ヒストグラムによる学習→連続値をbinにまとめることで計算量を減らしている。  
深さベースではなく、葉ベースでの学習  
短所：カテゴリカル特徴量が多すぎると役に立たない。→CatBoostの利用を考える。  
重要なパラメータたち:  
num_leaves:ツリーの葉の数。2^(max_depth)より小さくする必要あり。  
max_depth：ツリーの深さの最大値7がちょうどいいことが多いらしい。  
min_child_samples (min_data_in_leaf):葉を作るのに必要な最低サンプル数。データが少なく厳しいときは、これを小さくする。  
min_child_weight (min_sum_hessian_in_leaf):葉を分割するのに必要な（ロスの）hessianの合計値。小さければ小さいほどロスを小さくしようと葉を分割するが、それはオーバーフィッティングを引き起こす。  
subsample (bagging_fraction):バギングの割合（訓練データの何パーセントを利用するか）  
subsample_freq:バギングを行う間隔  
colsample_bytree (feature_fraction):特徴量サンプリングの割合（何パーセントの特徴量を利用するか）  
min_split_gain:葉を分割する条件として設定するロス改善度の最小値。この値以上の改善が無ければ葉を分割しない。  
reg_alpha:L1正則化。過学習していそうなら調整する。  
reg_lambda:L2正則化。過学習していそうなら調整する。。


learning_rateは0,01の固定でよい。チューニング


