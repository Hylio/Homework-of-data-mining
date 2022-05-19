import pandas as pd

base_test = pd.read_csv("regression/base-test.csv", encoding='ANSI')
knowledge_test = pd.read_csv("regression/knowledge-test.csv", encoding='ANSI')
money_test = pd.read_csv("regression/money-test.csv", encoding='ANSI')
year_test = pd.read_csv("regression/year-test.csv", encoding='ANSI')

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

# 是否是僵尸企业似乎与区域无关，故去除此维度
base_test = base_test.drop(columns=["区域"])

# 处理数据缺失
for col in list(base_test.columns[base_test.isnull().sum() > 0]):
    fill = base_test[col].mean()
    base_test[col].fillna(fill, inplace=True)

for col in list(knowledge_test.columns[knowledge_test.isnull().sum() > 0]):
    fill = round(knowledge_test[col].mean())
    knowledge_test[col].fillna(fill, inplace=True)

# 合并base和knowledge 以及 money和year
base_knowledge_test = pd.merge(base_test, knowledge_test, on='ID', how='inner')
money_year_test = pd.merge(money_test, year_test, on=['ID', 'year'], how='inner')

# 将三年的数据分别进行提取，并合并到同一个表
money_year_test_15 = money_year_test.loc[money_year_test['year'] == 2015].add_suffix('_2015')
money_year_test_15.rename(columns={'ID_2015': 'ID', 'year_2015': 'year'}, inplace=True)
money_year_test_16 = money_year_test.loc[money_year_test['year'] == 2016].add_suffix('_2016')
money_year_test_16.rename(columns={'ID_2016': 'ID', 'year_2016': 'year'}, inplace=True)

money_year_test_15_16 = pd.merge(money_year_test_15, money_year_test_16, on='ID')
test_total_data = pd.merge(money_year_test_15_16, base_knowledge_test, on='ID')

# 因为具体年份已经可以通过加后缀得以区分了
test_total_data = test_total_data.drop(columns=['year_x', 'year_y'])

for columns in list(test_total_data.columns[test_total_data.isnull().sum() > 0]):
    fill = test_total_data[columns].mean()
    test_total_data[columns].fillna(fill, inplace=True)

test_total_data.to_csv("regression/test.csv", encoding='utf8')
