import pandas as pd

base_train = pd.read_csv("regression/base-train.csv", encoding='ANSI')
knowledge_train = pd.read_csv("regression/knowledge-train.csv", encoding='ANSI')
money_train = pd.read_csv("regression/money-train.csv", encoding='ANSI')
year_train = pd.read_csv("regression/year-train.csv", encoding='ANSI')

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

base_train['行业'] = base_train['行业'].map(change_map_career)
base_train['企业类型'] = base_train['企业类型'].map(change_map_business_type)
base_train['控制人类型'] = base_train['控制人类型'].map(change_map_control)

# 是否是僵尸企业似乎与区域无关，故去除此维度
base_train = base_train.drop(columns=["区域"])

# 处理数据缺失
for col in list(base_train.columns[base_train.isnull().sum() > 0]):
    fill = base_train[col].mean()
    base_train[col].fillna(fill, inplace=True)

for col in list(knowledge_train.columns[knowledge_train.isnull().sum() > 0]):
    fill = round(knowledge_train[col].mean())
    knowledge_train[col].fillna(fill, inplace=True)

# 合并base和knowledge 以及 money和year
base_knowledge_train = pd.merge(base_train, knowledge_train, on='ID', how='inner')
money_year_train = pd.merge(money_train, year_train, on=['ID', 'year'], how='inner')

# 将三年的数据分别进行提取，并合并到同一个表
money_year_train_15 = money_year_train.loc[money_year_train['year'] == 2015].add_suffix('_2015')
money_year_train_15.rename(columns={'ID_2015': 'ID', 'year_2015': 'year'}, inplace=True)
money_year_train_16 = money_year_train.loc[money_year_train['year'] == 2016].add_suffix('_2016')
money_year_train_16.rename(columns={'ID_2016': 'ID', 'year_2016': 'year'}, inplace=True)
money_year_train_17 = money_year_train.loc[money_year_train['year'] == 2017].add_suffix('_2017')
money_year_train_17.rename(columns={'ID_2017': 'ID', 'year_2017': 'year'}, inplace=True)
target = pd.DataFrame(columns=['ID', '净利润_2017'])
target['净利润_2017'] = money_year_train_17['净利润_2017']
target['ID'] = money_year_train_17['ID']
money_year_train_15_16 = pd.merge(money_year_train_15, money_year_train_16, on='ID')
money_year_train_15_16_17 = pd.merge(money_year_train_15_16, target, on='ID')
train_total_data = pd.merge(money_year_train_15_16_17, base_knowledge_train, on='ID')

# train_total_data['净利润_2017'] = money_year_train_17['净利润_2017']

# 因为具体年份已经可以通过加后缀得以区分了
train_total_data = train_total_data.drop(columns=['year_x', 'year_y'])

for columns in list(train_total_data.columns[train_total_data.isnull().sum() > 0]):
    fill = train_total_data[columns].mean()
    train_total_data[columns].fillna(fill, inplace=True)


train_total_data.to_csv("regression/train.csv", encoding='utf8')
