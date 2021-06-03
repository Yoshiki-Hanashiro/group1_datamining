import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

#読み込んだ元データを、データとラベルに分割する。今回はデータ数が多い白ワインを使用する。
wine = pd.read_csv('data_mining/Experiment_datamining/G1/group1_datamining/dataset/honhyo.csv',sep=',')
label = wine['quality']
data = wine.drop(columns = 'quality')

# 用意したデータ各列の平均と標準偏差を確認
print('元データの平均：'+ str(np.mean(data.values[:, :])))
print('元データの標準偏差：'+ str(np.std(data.values[:, :])))
 
# 標準化を行う
sc = StandardScaler()
std = sc.fit_transform(data)

print('標準化後の平均：'+ str(np.mean(std[:, :])))
print('標準化後の標準偏差：'+ str(np.std(std[:, :])))

#データを学習用とテスト用に分割する。今回はデータの半数を学習用とする。
data_train,data_test,label_train,label_test = train_test_split(data,label,test_size = 0.5)
data_std_train,data_std_test = train_test_split(std,test_size = 0.5)

#データをLinearSVCを使用して学習する。
clf = svm.LinearSVC()
clf.fit(data_train,label_train)

#学習したモデルを使用して精度を確認する。
test_pred = clf.predict(data_test)
print('正答率は'+ str(accuracy_score(label_test,test_pred) * 100) + '%です。(LinearSVC)')
print('分類結果(評価段階は0~10ですが実際には3~9までの評価しかないため7*7行列となります。)')
print(confusion_matrix(label_test,test_pred))