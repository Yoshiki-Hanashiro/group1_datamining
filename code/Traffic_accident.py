import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

#関数化
def plotter(timei,timei_str):
    #読み込んだ元データを、データとラベルに分割する。
    accident = pd.read_csv('../dataset/accident_'+ timei_str+'.csv', sep=',')
    label = accident['count']
    data = accident.iloc[:,4:5]


    #データを学習用とテスト用に分割する。今回はデータの半数を学習用とする。
    data_train,data_test,label_train,label_test = train_test_split(data,label,test_size = 0.5)

    #データをLinearSVCを使用して学習する。
    clf = SGDRegressor(max_iter=1000)
    clf.fit(data_train,label_train)

    #学習したモデルを使用して精度を確認する。

    #解説 6：結果をプロットする------------------------------------------------

    line_X=np.arange(-4, 4, 0.1) #3から10まで1刻み
    line_Y=clf.predict(line_X[:, np.newaxis])
    fig = plt.figure(figsize=(10,10))
    plt.subplot(2, 1, 1)
    plt.scatter(data_train, label_train, c='b', marker='s')
    plt.plot(line_X, line_Y, c='r')
    plt.show
    fig.savefig("../result/"+ timei_str +"_img1.png")

    #解説 7：誤差をプロットする-------------------------------------------------
    Y_rm_pred=clf.predict(data_test)
    plt.subplot(2, 1, 2)
    plt.scatter(label_test, Y_rm_pred-label_test, c='b', marker='s', label="RM_only")

    Y_pred=clf.predict(data_test)
    plt.scatter(label_test, Y_pred-label_test, c='r', marker='s',label="ALL")
    plt.legend()
    plt.hlines(y=0, xmin=0, xmax=50, colors='black')
    plt.show
    fig.savefig("../result/"+ timei_str +"_img2.png")

    print("\n平均2乗誤差")
    RMS=np.mean((Y_pred - label_test) ** 2)
    print(RMS)