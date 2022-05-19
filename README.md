# 电子科技大学信软学院数据挖掘课程大作业

## 问题1：僵尸企业分类问题
问题描述：僵尸企业是指缺乏盈利能力却能够以低于市场最优利率成本获得信贷资源，依靠外界输血而缺乏自生能力的企业。僵尸企业的存在破坏了市场机制，加剧了信贷资源的错配，带来了严重的产能过剩问题，还对其他非僵尸企业产生了投资挤出效应。 因此需要对正常企业和僵尸企业进行分类，现给出一批有标签的企业数据作为训练集，标签为0表示正常企业，标签为1表示僵尸企业；同时给出无标签数据作为测试集，请对无标签数据进行分类。

数据集介绍：本作业提供的数据集包括训练集和测试集两部分，每个部分又包括企业基本数据、企业知识产权数据、企业金融数据、企业年报数据四个子集。其中企业基本数据包含企业的一些基本属性以及企业的标签（即flag）；企业知识产权数据包含企业的知识产权相关信息，根据id可与基本数据一一对应；企业金融数据包含企业2015至2017三年的金融相关信息，根据id可与基本数据相对应；企业年报数据包含企业2015至2017三年的年报数据，根据id可与基本数据相对应。

实验结果提交形式：实验结果为测试集的分类结果，要求以TXT文本格式提交，使用UTF-8编码，文本中共两列数据，第一列为企业id，保持基本数据中的id不变，第二列为分类结果，取值为0或 1，0表示正常企业，1表示僵尸企业，两列数据以英文逗号隔开，请勿添加空格或其他符号，第一行为列名，从第二行起每行为一条数据。

## 问题2：企业净利润预测问题
问题描述：已知企业的基本信息、知识产权信息和2015-2016两年的金融和年报信息，请预测2017年企业的净利润。

数据集介绍：本作业提供的数据集包括训练集和测试集两部分，每个部分又包括企业基本数据、企业知识产权数据、企业金融数据、企业年报数据四个子集。其中企业基本数据包含企业的一些基本属性；企业知识产权数据包含企业的知识产权相关信息，根据id可与基本数据一一对应；训练集的企业金融数据包含企业2015至2017三年的金融相关信息，测试集的企业金融数据包含企业2015至2016两年的金融相关信息，根据id可与基本数据相对应；训练集的企业年报数据包含企业2015至2017三年的年报数据，测试集的企业年报数据包含企业2015至2016两年的年报数据，根据id可与基本数据相对应。

提交形式：实验结果；实验结果为测试集的预测结果，要求以名称为姓名+学号.txt的TXT文本格式提交，如惠国强+201852090625.txt，文件内容使用UTF-8编码，文本中共两列数据，第一列为企业id，保持基本数据中的id不变，第二列为企业2017年净利润预测值，请将预测结果进行线性函数归一化，取值为0~ 1.0，两列数据以英文逗号隔开，请勿添加空格或其他符号，第一行为列名，从第二行起每行为一条数据。
