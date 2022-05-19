import pandas as pd
from xgboost import XGBClassifier

data_train = pd.read_csv('classify/train.csv', engine='python', encoding='utf8')
data_test = pd.read_csv('classify/test.csv', engine='python', encoding='utf8')

x_data_train = data_train.drop(columns=['flag'])
y_data_train = data_train['flag']
# 去掉id 因为只是编号 和营业情况无关
x_data_train_noid = x_data_train.drop(columns=['ID'])
data_test_noid = data_test.drop(columns=['ID'])


model = XGBClassifier()
model.fit(x_data_train_noid, y_data_train)
y_data_pre = model.predict(data_test_noid)
data_test['flag'] = y_data_pre

data_test.to_csv('classify/data_pre.csv', columns=['ID', 'flag'], encoding='utf8')
