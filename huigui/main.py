from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

train_data = pd.read_csv('regression/train.csv', encoding='utf8')
test_data = pd.read_csv('regression/test.csv', encoding='utf8')

x_data_train = train_data.drop(columns=['净利润_2017', 'ID'])
y_data_train = train_data['净利润_2017']
x_test = test_data.drop(columns=['ID'])

# 线性回归
model = linear_model.LinearRegression()
model.fit(x_data_train, y_data_train)
y_pre = model.predict(x_test)
test_data['净利润_2017'] = y_pre

test_data['净利润'] = (test_data['净利润_2017'] - test_data['净利润_2017'].min()) / (test_data['净利润_2017'].max()
                                                                            - test_data['净利润_2017'].min())
test_data.to_csv('regression/pre_linear.csv', columns=['ID', '净利润'], encoding='utf8')

# 随机森林
forest = RandomForestRegressor()
forest.fit(x_data_train, y_data_train)
y_pre_forest = forest.predict(x_test)
test_data['净利润_2017_f'] = y_pre_forest
test_data['净利润2'] = (test_data['净利润_2017_f'] - test_data['净利润_2017_f'].min()) / (test_data['净利润_2017_f'].max()
                                                                                 - test_data['净利润_2017_f'].min())
test_data.to_csv('regression/pre_forest.csv', columns=['ID', '净利润2'], encoding='utf8')
