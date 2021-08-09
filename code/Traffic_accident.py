'''SVRで学習し、予測された回帰曲面をプロットし、事故平均と誤差の平均を出力するプログラム。
回帰曲面は三次元でプロットされており、実際のデータセットとの比較もできる。
'''
from numpy import testing
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


#読み込んだ元データを、データとラベルに分割する。
accident = pd.read_csv('group1_datamining/dataset/accident_okinawa.csv', sep=',')
accident = accident.dropna()
label = accident['count']
data = accident.iloc[:,5:7]
print(accident.isnull().sum())

#データを学習用とテスト用に分割する。今回はデータの半数を学習用とする。
data_train,data_test,label_train,label_test = train_test_split(data,label,test_size = 0.5)

# サポートベクターマシン(SVR)による学習
clf = svm.SVR(C=1.0, kernel='sigmoid', gamma='auto', epsilon=0.5)
clf.fit(data_train, label_train)

# 学習済モデルを使って予測
grid_line = np.arange(0, 50, 3) # 回帰式の軸を作成
X2, Y2 = np.meshgrid(grid_line, grid_line) # グリッドを作成
Z2 = clf.predict(np.array([X2.ravel(), Y2.ravel()]).T) # 予測
Z2 = Z2.reshape(X2.shape)                                         # プロット用にデータshapeを変換
r2 = clf.score(data_train, label_train)                                            # 決定係数算出

def plot_regressionplane():
    '''回帰曲面とデータセットをプロットする。
    '''
    # フォントの種類とサイズを設定する。
    plt.rcParams['font.size'] = 14
    plt.rcParams['font.family'] = 'Times New Roman'

    #  グラフの入れ物を用意する。
    fig = plt.figure()
    ax1 = Axes3D(fig)

    # 軸のラベルを設定する。
    ax1.set_xlabel('temperature')
    ax1.set_ylabel('rain')
    ax1.set_zlabel('Traffic_Accident')

    # データプロットする。
    ax1.scatter3D(data_test['temperature'], data_test['rain'], label_test, label='Dataset')
    ax1.plot_wireframe(X2, Y2, Z2, label='Regression plane K=sigmoid', color = 'red')
    plt.legend()

    # グラフ内に決定係数を記入
    ax1.text(0.0, 0.0, 300, zdir=(1,1,0), s='$\ R^{2}=$' + str(round(r2, 2)), fontsize=20)
    
    # グラフを表示する。
    plt.show()
    fig.savefig("group1_datamining/result/img3.png")# グラフをimg3.pngとして保存
    plt.close()

#プロット関数を実行
plot_regressionplane()

#学習したモデルを使用して精度を確認する。
test_pred = clf.predict(data_test)
difference = np.mean(test_pred - label_test)
print("誤差の平均:" + str(difference))
ave = np.mean(label_test)
print("一時間当たりの事故の平均は" + str(ave) + "件です。")