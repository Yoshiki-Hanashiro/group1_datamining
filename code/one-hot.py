import pandas as pd

df = pd.read_csv("../dataset/honhyo_2019.csv", encoding="shift-jis")#データセットを読み込む

#pd.set_option('display.max_columns', 50)#表示する列数を設定、今回は11列
print(df.head())#上から5つ表示

one_hot_df_w = pd.get_dummies(df['天候'])#ダミー変数作成、ここでone-hotに？
print(one_hot_df_w.shape)#one_hot_df_wの行列の形を調べる
print(df['天候'].head())#one-hot適応前出力

for i in range(5):
    print(one_hot_df_w.values[i])#one-hot適応後出力


one_hot_df_r = pd.get_dummies(df['路面状態'])#ダミー変数作成、ここでone-hotに？
print(one_hot_df_r.shape)#one_hot_df_rの行列の形を調べる
print(df['路面状態'].head())#one-hot適応前出力

for i in range(5):
    print(one_hot_df_r.values[i])#one-hot適応後出力