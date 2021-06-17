from numpy import testing
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


#読み込んだ元データを、データとラベルに分割する。
accident = pd.read_csv('data-mining/Experiment_datamining/G1/group1_datamining/dataset/accident_okinawa.csv', sep=',')
accident = accident.dropna()
label = accident['count']
data = accident.iloc[:,5:7]
print(accident.isnull().sum())

#データを学習用とテスト用に分割する。今回はデータの半数を学習用とする。
data_train,data_test,label_train,label_test = train_test_split(data,label,test_size = 0.5)

#データをLinearSVCを使用して学習する。
clf = SGDRegressor(max_iter=1000)
clf.fit(data_train,label_train)

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
fig.savefig("data-mining/Experiment_datamining/G1/group1_datamining/result/img2.png")


"""
#解説 6：結果をプロットする------------------------------------------------

line_X=np.arange(-4, 4, 0.1) #3から10まで1刻み
line_Y=clf.predict(line_X[:, np.newaxis])
fig = plt.figure(figsize=(10,10))
plt.subplot(2, 1, 1)
plt.scatter(data_train, label_train, c='b', marker='s')
plt.plot(line_X, line_Y, c='r')
plt.show
fig.savefig("data-mining/Experiment_datamining/G1/group1_datamining/result/img1.png")

#解説 7：誤差をプロットする-------------------------------------------------
Y_rm_pred=clf.predict(data_test)
plt.subplot(2, 1, 2)
plt.scatter(label_test, Y_rm_pred-label_test, c='b', marker='s', label="RM_only")

plt.legend()
plt.hlines(y=0, xmin=0, xmax=50, colors='black')
plt.show
fig.savefig("data-mining/Experiment_datamining/G1/group1_datamining/result/img2.png")

print("\n平均2乗誤差")
RMS=np.mean((Y_pred - label_test) ** 2)
print(RMS)
"""