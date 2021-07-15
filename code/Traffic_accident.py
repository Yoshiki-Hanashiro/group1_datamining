from numpy import testing
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import mean_absolute_error

#読み込んだ元データを、データとラベルに分割する。
#沖縄のみ、かつ事故件数ゼロありデータセット
#accident = pd.read_csv('data-mining/Experiment_datamining/G1/group1_datamining/dataset/accident_okinawa.csv', sep=',')

#全国、事故件数ゼロなしデータセット
accident = pd.read_csv('../dataset/all.csv', sep=',')
accident = accident.dropna()
accident = accident.set_axis(['noname', 'year', 'month', 'day', 'hour', 'rain', 'temperature', 'count'], axis='columns')
print(accident)
label = accident['count']
#rainとtemperatureのみ
data = accident.iloc[:,5:7]
#hourを追加
data = accident.iloc[:,4:7]
print(accident.isnull().sum())

#標準化

scaler = StandardScaler()
data = scaler.fit_transform(data)
#print(np.mean(data[:, 0]), np.mean(data[:, 1]))
#print(np.std(data[:, 0]), np.std(data[:, 1]))

#データを学習用とテスト用に分割する。今回はデータの半数を学習用とする。
data_train,data_test,label_train,label_test = train_test_split(data,label,test_size = 0.5)

#SGDRegressorを使用して学習する

model = SGDRegressor(learning_rate='adaptive')
model.fit(data_train,label_train)


#データをSVRを使用して学習する。
'''
model = svm.SVR(C=1.0, kernel='linear', epsilon=0.1)
model.fit(data_train,label_train)
'''

#学習したモデルを使用して精度を確認する。
test_pred = model.predict(data_test)
difference = np.mean(test_pred - label_test)
print("誤差の平均:" + str(difference))

#MSE:二乗誤差の平均
mse = mean_squared_error(label_test, test_pred)
print("MSE:" + str(mse))

#RMSE:MSEの平方根をとった指標
rmse = sqrt(mean_squared_error(label_test, test_pred))
print("RMSE:" + str(rmse))

#MAE:誤差の絶対値の平均，外れ値の影響無視したい時
mae = mean_absolute_error(label_test, test_pred)
print("MAE:" + str(mae))


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
fig.savefig("../../group1_datamining/result/img2.png")



'''
#結果をプロットする(回帰平面)------------------------------------------------
# パラメータ算出
reg_wn = model.coef_ # 偏回帰係数
reg_w0 = model.intercept_ # 切片
r2 = model.score(data_test, label_test) # 決定係数
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
fig.savefig("data-mining/Experiment_datamining/G1/group1_datamining/result/img3.png")
'''