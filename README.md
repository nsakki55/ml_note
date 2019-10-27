# kaggle
## 機械学習で役立つTips
### LightGBMと他のブースティングのまとめ（PyDataでのスライド）  
https://alphaimpact.jp/downloads/pydata20190927.pdf
XGBoostと比べ、並列化による高速化、大規模データセットへの対応が可能となった。  
ヒストグラムによる学習→連続値をbinにまとめることで計算量を減らしている。  
深さベースではなく、葉ベースでの学習  
短所：カテゴリカル特徴量が多すぎると役に立たない。→CatBoostの利用を考える。  
重要なパラメータたち:  
num_leaves:ツリーの葉の数。2^(max_depth)より小さくする必要あり。  
max_depth：ツリーの深さの最大値7がちょうどいいことが多いらしい。深いほど、特徴量間の相互作用を捉えられるらしい。  
min_child_samples (min_data_in_leaf):葉を作るのに必要な最低サンプル数。データが少なく厳しいときは、これを小さくする。  
min_child_weight (min_sum_hessian_in_leaf):葉を分割するのに必要な（ロスの）hessianの合計値。小さければ小さいほどロスを小さくしようと葉を分割するが、それはオーバーフィッティングを引き起こす。分割するのに最低限必要な葉の構成データ数という認識がわかりやすい。  
gamma:決定木を分岐させるために最低限減らさなくてはいけない目的関数の値。gammaを大きくすると分岐が起こりにくくなる。  
subsample (bagging_fraction):バギングの割合（訓練データの何パーセントを利用するか）行のサンプリング  
subsample_freq:バギングを行う間隔  
colsample_bytree (feature_fraction):特徴量サンプリングの割合（何パーセントの特徴量を利用するか）列のサンプリング  
min_split_gain:葉を分割する条件として設定するロス改善度の最小値。この値以上の改善が無ければ葉を分割しない。  
reg_alpha:L1正則化。過学習していそうなら調整する。  
reg_lambda:L2正則化。過学習していそうなら調整する。。
### 各パラメタの性質
・max_depth,min_child_weight,gamma：分岐の深さ、分岐を行うかどうかを制御しモデルの複雑さの調節ができる。  
・alpha,lambda:決定木の葉のウェイトの正則化によりモデルの複雑さを調整できる。  
・subsample,colsample_bytree:ランダム性を加えることで過学習を抑えることができる。  

### 効率的な探索の流れ
１；max_depthの最適化、５〜８ぐらいを試す。  
2:col_sample_levelの最適化0.5〜０．１を０．１刻みで試す。  
3:min_child_weightの最適化。１、２、４、８、１６、３２と２倍ごとに試す。  
4:lambda,alphaの最適化。  
early_stoppin_roundは10/eta程度

num_round:作成する決定木の本数。１０００などの十分大きな値とし、early stoppingで自動的に決めるのがよい  
learning_rate:0,01の固定でよい。チューニングする必要はない。値が小さいと学習の時間がかかるので、初めは大きな値で、後半に小さな値にするのがベスト  
objective: 回帰、二値分類、他クラス分類化を指定する。  
勾配ブースティングの重要度は  
ゲイン：特徴量の分岐により得た目的関数の現象  
カバー：特徴量によって分岐させられたデータの数  
頻度：特徴量が分岐に現れた回数  
の三種類。もっとも重要度を表現しているのはゲイン。デフォルトだと頻度でだされるので、total_geinを指定する　　
https://sites.google.com/view/lauraepp/parameters  
公式ドキュメント大事：https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html

## 特徴量選択
・単変量統計を用いる。あくまで特徴量と目的変数の1対１の関係をみるので、特徴の相互作用は考慮されない。  
相関係数：絶対値の大きい方から特徴を選択する。ピアソンの積率相関係数は線形以外の関係を捉えられない。スピアマンの順位相関係数は値の大きさの順位関係のみを捉える。 
カイ二乗検定：カイ二乗検定の統計量から特徴量を選択する。クラス分類、非負値、スケーリングが必要と、手間と制約が大きいのでまぁ使う機会は少ない  
相互情報量：片方を知ることでもう一方をより推測できるようになる場合に値が大きくなる。  
学習データ全体で特徴量選択をすると、たまたま学習データで偏りが出ている特徴が選択されてしまうことがある。学習データ全体で特徴量を選択しているので、一種のリークとも言える。


## NN参考
テーブルデータでのNNモデル  
https://github.com/ChenglongChen/Kaggle_HomeDepot  
https://github.com/dkivaranovic/kaggledays-recruit  
https://github.com/puyokw/kaggle_Otto  


## 不均衡データの処理
・アンダーサンプリング：多いデータを減らす。データの選択は重複選択なしがよい。事前にデータをクラスタリングし、クラスごとにサンプリングを行う方法がある。アンダーサンプリングは情報量を減らしてしまうため、なるべくオーバーサンプリングが好まれる。
・オーバーサンプリング：少ないデータを増やす。代表的な手法はSMOTE。データ間の直線間からサンプリングする手法ため次元数が大きくなる場合は偏りが大きくなる。その場合はバギング（多数のモデルをアンサンブルすること）することで解消できる。

## 欠損値補完
・定数による補完：欠損が多いとデータの分散が本来のものと大きくかけ離れやすい  
・集計値による補完：平均、中央、最大、最小値をいれる  
・カテゴリごとに集計値を求めるのが有効な場合もある。
・他のデータに基づく予測値：すでにあるデータから機械学習モデルを作成し、欠損部分の予測を行う  
・時系列の関係から補完：時間に対して連続している値はMCAR、MARが有効  
・多重代入法：補完したデータを複数作成し、結果を統合する。PMM.fancyimputeライブラリで実装可能。
・最尤法：潜在変数を導入し、EMアルゴリズムを用いて尤度を最大化することで補完する。  
・購買ブースティングを用いるときは、欠損値の補完は行わないのが基本。  
・欠損値を予測するモデルを作成する  
・欠損している変数の数をカウントする。  

## カテゴリー変数
変換を行う前に、テストデータにのみ存在するカテゴリカル変数があるかを確認する。
・テストデータにのみ存在するカテゴリのレコードが少ない場合は、スコアに与える影響が少ないためそのままでもよいことがある。  
・最頻値や予測で補完、補完はTargetエンコードなら学習データ全体の平均を入れるなどする。  
GBDTなど、決定木ベースの場合はLabel EncodingかTarget Encodingが最も効く。その他はone hotがよい。  
ニューラルネットワークの場合はEmbeddingも有効  
・Frequency Encoding  
→目的変数と各レコードの出現頻度に関連性がある場合に有効。各ラベルの出現回数、頻度で置換える。   
・Target Encoding  
カテゴリーごとに目的変数の平均値を入れる。リークが起きやすいのでクロスバリデーションを行う。  
目的変数の平均をとると、外れ値の影響を大きく受けるので、中央値を用いると良い場合がある。  
文字列の場合は、学習済みのNNのembeding層を用いる方法がる。  

## 次元削減
PCA、SVD、NMF,LDAが特に有名。  
・次元削減後の特徴量を元の特徴量に追加する場合  
t-SNE、UMAPは元の特徴量空間上で近い点が圧縮後の平面でも近くなるように圧縮される。非線形な関係を捉えることができる

## クラスタリング
どのクラスに分類するかをカテゴリ変数とし、クラスタ中心からの距離などを特徴量とすることができる。  
k-Mean,DBSCAN,Agglomerative Clusteringがよく用いられる。

## 文字列処理
文章を扱うとき、語順を考慮に入れると複雑になるため、語順を考慮に入れないBag of Words を試すのが早い。  
MeCabを利用すると、日本語文章を形態素解析してくれるので便利。  
TF-IDF：TF(Term Frequency)文章内の出現割合（[対象の単語の数]/[文章に含まれる単語の合計数]）とIDF(Inverse Document Frequency)単語の出現割合
のスコア(log[全文書数]/[対象の単語が出現している文書数]＋1)を用いる。

## バリデーションの分け方
常にテストデータの分布に近くなるようにバリデーションデータは分けるようにする。  
訓練データとテストデータの共通のユーザーIDがあるなど、グルーピングできる場合もある。  
時系列データの場合は、素直に時間で区切るのが良い。  
訓練データと、テストデータで分布が異なる場合、adversarial validationという、訓練データとテストデータを合わせて、テストデータかどうかの
二値分類問題モデルを作成し、テストデータと予測した訓練データをバリデーション用に使う。

## ハイパーパラメーターチューニング
パラメタの調整の流れは、  
１、ベースラインのパラメタでの学習、  
２、1〜３種類のパラメタと２〜５個程度でグリッドサーチ  
３、本格的にチューニングを行う段階でベイズ最適化を用いる  
モデルの複雑性を増すパラメータ、モデルを単純化させるパラメータがあり、学習が上手くいかないときの考察に有効  


## アンサンブル
アンサンブルにおいては精度よりも多様性が重要で、単体での精度が低いモデルだからといって捨てない方がよいことがある。  
ハイパラや特徴量が同じモデルでも乱数シードを変えて平均をとるだけでも精度が上がることがある。  
アンサンブルに使われるモデル  
・２〜３つのGBDT（決定木の深さが浅いもの、中くらいのもの、深いもの）  
・１〜２のランダムフォレスト（深さが浅いもの、深いもの）  
・１〜２のニューラルネット（１つは層の数が多いもの、２つは少ないもの）  
・１つの線形モデル  
ハイパーパラメータを変える  
・交互相互作用の効き具合を変える（決定木の深さを変えるなど）  
・正則化の強さを変える  
・モデルの表現力を変える。  
特徴量を変える  
・特定の特徴量の組みを使う、使わない  
・特徴量をスケーリングする/しない  
・特徴選択を強く行う、行わない  
・外れ値を除く、除かない  
・データの前処理や変換の方法を変える  
別の値を目的変数としたモデルの予測値を特徴量とする  
・回帰タスクで、ある値以上かどうかの２値分類タスクのモデルを作る  
・他クラス分類の場合は一部のクラスのみを予測するモデルを作る。　　
・重要だが欠損が多い特徴量がある場合に、その特徴量を予測するモデルを作る。  
・あるモデルによる予測値の残渣（＝目的変数ー予測値）に対して予測をするモデルを作る  



## スタッキング
元の学習データで学習したモデルを１層目のモデルとし、１層目のモデルでの予測値という特徴量を用いて学習したモデルを２層目のモデルとする。  
1:学習データをクロスバリデーションのfoldに分ける  
2:バリデーションデータへの予測値を作成し、学習データのモデルの予測値を作成する。  
3:各foldで学習したモデルでテストデータを予測し、平均をとったものをテストデータの特徴量とする  
4:2-3をスタッキングしたいモデルの数だけ繰り返す。(1層目のモデル)  
5:２〜４で作成した特徴量を使ってモデルの学習と予測を行う(2層目のモデル)

スタッキングが有効なのは学習データとテストデータが同じ分布でデータ量が多い場合。  
時系列データや、学習データとテストデータの分布が異なる場合は、モデルの加重平均のアンサンブルが有効な場合が多い  
評価指標により違いがあり、accuracyよりもloglossの方が細かく予測値をチューニングすることによるスコアの向上がみられる。

参考：
https://github.com/ChenglongChen/Kaggle_HomeDepot

## 知識
モデルが複雑→平均的な予測値と真値との乖離（バイアス）が小さくなる一方、予測の不安定性（バリアンス）は大きくなる。  
モデルが単純→バイアスが大きく、バリアンスが小さい  
http://ibisforest.org/index.php?%E3%83%90%E3%82%A4%E3%82%A2%E3%82%B9-%E3%83%90%E3%83%AA%E3%82%A2%E3%83%B3%E3%82%B9  
bestfitting氏の言葉  
http://blog.kaggle.com/2018/05/07/profiling-top-kagglers-bestfitting-currently-1-in-the-world/

###　　
