import pandas as pd
from collections import defaultdict
import numpy as np
base_test = pd.read_csv("classify/base-test.csv", encoding='ANSI')
knowledge_test = pd.read_csv("classify/knowledge-test.csv", encoding='ANSI')
money_test = pd.read_csv("classify/money-test.csv", encoding='ANSI')
year_test = pd.read_csv("classify/year-test.csv", encoding='ANSI')

# 将属性值由中文字符串换为数字
change_map_career = {
    '服务业': 0,
    '工业': 1,
    '交通运输业': 2,
    '零售业': 3,
    '商业服务业': 4,
    '社区服务': 5
}
change_map_business_type = {
    '有限责任公司': 0,
    '农民专业合作社': 1,
    '集体所有制企业': 2,
    '合伙企业': 3,
    '股份有限公司': 4
}
change_map_control = {
    '自然人': 0,
    '企业法人': 1
}
base_test['行业'] = base_test['行业'].map(change_map_career)
base_test['企业类型'] = base_test['企业类型'].map(change_map_business_type)
base_test['控制人类型'] = base_test['控制人类型'].map(change_map_control)

# 是否是僵尸企业似乎与区域无关
base_test = base_test.drop(columns=["区域"])

# 处理年份缺失
lost = set()
loc = year_test['year'][year_test['year'].isnull()].index.tolist()
for i in range(len(loc)):
    lost.add(loc[i])
years = {2015, 2016, 2017}
lost_id_years = defaultdict(set)
for index, row in year_test.iterrows():
    if index in lost:
        continue
    else:
        lost_id_years[row['ID']].add(row['year'])
delete_list = []
for key, val in lost_id_years.items():
    lost_id_years[key] = years-val
    if len(lost_id_years[key]) == 0:
        delete_list.append(key)
for key in delete_list:
    del lost_id_years[key]
for ind in loc:
    id = year_test.iloc[ind]['ID']
    year_test.iloc[ind, 1] = lost_id_years[id].pop()

lost = set()
loc = money_test['year'][money_test['year'].isnull()].index.tolist()
for i in range(len(loc)):
    lost.add(loc[i])
years = {2015, 2016, 2017}
lost_id_years = defaultdict(set)
for index, row in money_test.iterrows():
    if index in lost:
        continue
    else:
        lost_id_years[row['ID']].add(row['year'])
delete_list = []
for key, val in lost_id_years.items():
    lost_id_years[key] = years-val
    if len(lost_id_years[key]) == 0:
        delete_list.append(key)
for key in delete_list:
    del lost_id_years[key]
for ind in loc:
    id = money_test.iloc[ind]['ID']
    money_test.iloc[ind, 1] = lost_id_years[id].pop()

# 利用均值处理数据缺失
for col in list(base_test.columns[base_test.isnull().sum() > 0]):
    fill = base_test[col].mean()
    base_test[col].fillna(fill, inplace=True)
for col in list(knowledge_test.columns[knowledge_test.isnull().sum() > 0]):
    fill = round(knowledge_test[col].mean())
    knowledge_test[col].fillna(fill, inplace=True)
for col in list(year_test.columns[year_test.isnull().sum() > 0]):
    fill = round(year_test[col].mean())
    year_test[col].fillna(fill, inplace=True)
for col in list(money_test.columns[money_test.isnull().sum() > 0]):
    fill = round(money_test[col].mean())
    money_test[col].fillna(fill, inplace=True)
# print('year', len(year_test))
# print('money', len(money_test))

# 合并base和knowledge 以及 money和year
base_knowledge_test = pd.merge(base_test, knowledge_test, on='ID', how='inner')
money_year_test = pd.merge(money_test, year_test, on=['ID', 'year'], how='inner')
# print(money_year_test)
# 将三年的数据分别进行提取 并合并到同一个表
money_year_test_15 = money_year_test.loc[money_year_test['year'] == 2015].add_suffix('_2015')
money_year_test_15.rename(columns={'ID_2015': 'ID', 'year_2015': 'year'}, inplace=True)
# print(len(money_year_test_15))
money_year_test_16 = money_year_test.loc[money_year_test['year'] == 2016].add_suffix('_2016')
money_year_test_16.rename(columns={'ID_2016': 'ID', 'year_2016': 'year'}, inplace=True)
# print(len(money_year_test_16))
money_year_test_17 = money_year_test.loc[money_year_test['year'] == 2017].add_suffix('_2017')
money_year_test_17.rename(columns={'ID_2017': 'ID', 'year_2017': 'year'}, inplace=True)
# print(len(money_year_test_17))

money_year_test_15_16 = pd.merge(money_year_test_15, money_year_test_16, on='ID')
# print(money_year_test_15_16)
money_year_test_15_16_17 = pd.merge(money_year_test_15_16, money_year_test_17, on='ID')
test_total_data = pd.merge(money_year_test_15_16_17, base_knowledge_test, on='ID')

test_total_data = test_total_data.drop(columns=['year_x', 'year_y', 'year'])
for columns in list(test_total_data.columns[test_total_data.isnull().sum() > 0]):
    fill = test_total_data[columns].mean()
    test_total_data[columns].fillna(fill, inplace=True)


# 增加一些特征以便更好分类
test_df = pd.DataFrame(test_total_data)
test_df['利润营收比_2015'] = test_df['利润总额_2015']/(test_df['营业总收入_2015']+1)
test_df['利润营收比_2016'] = test_df['利润总额_2016']/(test_df['营业总收入_2016']+1)
test_df['利润营收比_2017'] = test_df['利润总额_2017']/(test_df['营业总收入_2017']+1)
test_df['政府依赖度_2015'] = test_df['项目融资和政策融资额度_2015']/(test_df['营业总收入_2015']+1)
test_df['政府依赖度_2016'] = test_df['项目融资和政策融资额度_2016']/(test_df['营业总收入_2016']+1)
test_df['政府依赖度_2017'] = test_df['项目融资和政策融资额度_2017']/(test_df['营业总收入_2017']+1)

test_df.to_csv("classify/test.csv", encoding='utf8')


# test_total_data.to_csv("classify/test.csv", encoding='utf8')
