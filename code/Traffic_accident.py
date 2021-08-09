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

"""
#データをLinearSVCを使用して学習する。
clf = SGDRegressor(max_iter=1000)
clf.fit(data_train,label_train)
"""

# サポートベクターマシン(SVR)による学習
clf = svm.SVR(C=1.0, kernel='sigmoid', gamma='auto', epsilon=0.5)
clf.fit(data_train, label_train)

# 学習済モデルを使って予測
grid_line = np.arange(0, 50, 3) # 回帰式の軸を作成
X2, Y2 = np.meshgrid(grid_line, grid_line) # グリッドを作成
Z2 = clf.predict(np.array([X2.ravel(), Y2.ravel()]).T) # 予測
Z2 = Z2.reshape(X2.shape)                                         # プロット用にデータshapeを変換
r2 = clf.score(data_train, label_train)                                            # 決定係数算出

# ここからグラフ描画----------------------------------------------------------------
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
fig.savefig("group1_datamining/result/img3.png")
plt.close()
# ---------------------------------------------------------------------------------

#学習したモデルを使用して精度を確認する。
test_pred = clf.predict(data_test)
difference = np.mean(test_pred - label_test)
print("誤差の平均:" + str(difference))
ave = np.mean(label_test)
print("一時間当たりの事故の平均は" + str(ave) + "件です。")

image = accident.iloc[:,5:]
image = image[image['count'] != 0]

fig = plt.figure(figsize=(10,10))
plt.scatter(image['temperature'], image['rain'], marker = '.')
plt.title("Traffic_Accident")
plt.xlabel("temperature")
plt.ylabel("rain")
plt.grid(True)
fig.savefig("group1_datamining/result/img2.png")

"""
#結果をプロットする(回帰平面)------------------------------------------------
# パラメータ算出
reg_wn = clf.coef_ # 偏回帰係数
reg_w0 = clf.intercept_ # 切片
r2 = clf.score(data_test, label_test) # 決定係数
print(reg_wn)
print(reg_w0)
print(r2)
 
# パラメータからモデルを可視化するために3次元データを作成する
X1 = np.arange(0, 35, 5.0) # x軸を作成
X2 = np.arange(0, 50, 5.0) # y軸を作成
X, Y = np.meshgrid(X1, X2) # x軸とy軸からグリッドデータを作成
Z = reg_w0 + (reg_wn[0] * X) + (reg_wn[1] * Y) # 回帰平面のz値を作成
 
# ここからグラフ描画
# フォントの種類とサイズを設定する。
plt.rcParams['font.size'] = 14
plt.rcParams['font.family'] = 'Times New Roman'
 
# グラフの入れ物を用意する。
fig = plt.figure()
ax1 = Axes3D(fig)
 
# 軸のラベルを設定する。
ax1.set_xlabel('temperature')
ax1.set_ylabel('rain')
ax1.set_zlabel('Traffic_Accident')
 
# データプロットする。
ax1.scatter3D(data_test['temperature'], data_test['rain'], label_test, label='Dataset')
ax1.plot_wireframe(X, Y, Z, label='Regression plane', color = 'red')
plt.legend()
 
# グラフを表示する。
plt.show()
fig.savefig("group1_datamining/result/img3.png")
"""

"""
#解説 6：結果をプロットする------------------------------------------------

line_X=np.arange(-4, 4, 0.1) #3から10まで1刻み
line_Y=clf.predict(line_X[:, np.newaxis])
fig = plt.figure(figsize=(10,10))
plt.subplot(2, 1, 1)
plt.scatter(data_train, label_train, c='b', marker='s')
plt.plot(line_X, line_Y, c='r')
plt.show
fig.savefig("group1_datamining/result/img1.png")

#解説 7：誤差をプロットする-------------------------------------------------
Y_rm_pred=clf.predict(data_test)
plt.subplot(2, 1, 2)
plt.scatter(label_test, Y_rm_pred-label_test, c='b', marker='s', label="RM_only")

plt.legend()
plt.hlines(y=0, xmin=0, xmax=50, colors='black')
plt.show
fig.savefig("group1_datamining/result/img2.png")

print("\n平均2乗誤差")
RMS=np.mean((Y_pred - label_test) ** 2)
print(RMS)
"""