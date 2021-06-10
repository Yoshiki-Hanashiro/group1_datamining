#97 101 沖縄県那覇
# #読み込んだ元データを、データとラベルに分割する。

import pandas as pd
import numpy as np

accident = pd.read_csv('data-mining/Experiment_datamining/G1/group1_datamining/dataset/honhyo.csv', sep=',', encoding='shift_jis')
rain = pd.read_csv('data-mining/Experiment_datamining/G1/group1_datamining/dataset/naha_rain.csv', sep=',', encoding='shift_jis',names=('year', 'month', 'day', 'hour','rain', 'not', 'quality', 'homogeneous'))
temperature = pd.read_csv('data-mining/Experiment_datamining/G1/group1_datamining/dataset/naha_temperature.csv', sep=',', encoding='shift_jis',names=('year', 'month', 'day', 'hour','temperature', 'puality', 'homogeneous'))
print(accident.iloc[:5,10:17])
print(len(rain))
print(temperature.shape)
data_rain = rain.iloc[:,:5]
data_temperature = temperature['temperature']
data = pd.concat([data_rain, data_temperature], axis=1)

sum = []

for i in range(len(rain)):
    number_accident = ((accident['発生日時　　年'] == data.iloc[i,0]) & (accident['発生日時　　月'] == data.iloc[i,1]) & (accident['発生日時　　日'] == data.iloc[i,2]) & (accident['発生日時　　時'] == data.iloc[i,3]) & (accident['都道府県コード'] == 97))
    count = number_accident.sum()
    sum.append(count)
    print(i,count)

data['count'] = sum

print(data)
print(data.shape)

data.to_csv('data-mining/Experiment_datamining/G1/group1_datamining/dataset/accident_okinawa.csv')