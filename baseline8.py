#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from lightgbm.sklearn import LGBMClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, log_loss, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse
from scipy.stats import kurtosis
from collections import Counter
import time
import warnings
import datetime
import matplotlib.pylab as plt
import math
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
plt.rcParams['font.sans-serif'] = ['SimHei']


# In[2]:


import pandarallel as pdl
pdl.pandarallel.initialize(10000)


# In[3]:


from tqdm import tqdm, tqdm_notebook
tqdm_notebook().pandas()


# In[4]:


train_df = pd.read_csv('dataset/train.csv', parse_dates=['auditing_date', 'due_date', 'repay_date'])


# In[5]:


train_df.head()


# In[6]:


def plus_1_day(s):
    return s + datetime.timedelta(days=1)


# In[7]:


# 如果违约,还款日期为due_date的后一天
train_df['repay_date'] = train_df[['due_date', 'repay_date']].apply(
    lambda x: x['repay_date'] if x['repay_date'] != '\\N' else plus_1_day(x['due_date']), axis=1
)


# In[8]:


# 如果违约,还款金额为0
train_df['repay_amt'] = train_df['repay_amt'].apply(lambda x: x if x != '\\N' else 0).astype('float32')


# In[9]:


# 设定label
train_df['label'] = (train_df['due_date'] - train_df['repay_date']).dt.days


# In[10]:


train_df.head()


# In[11]:


train_df['label'].value_counts(sort=False)


# In[12]:


train_df['label'].nunique()


# In[13]:


# 为什么要把-1换作32:方便分类编号和日期对应
# train_df.loc[train_df['repay_amt'] == -1, 'label'] = 32 错误写法
train_df['label'].replace(-1, 32, inplace=True)


# In[14]:


clf_labels = train_df['label'].values


# In[15]:


clf_labels.shape


# In[16]:


amt_labels = train_df['repay_amt'].values


# In[17]:


train_due_amt_df = train_df['due_amt'].values


# In[18]:


train_num = train_df.shape[0]


# In[19]:


test_df = pd.read_csv('dataset/test.csv', parse_dates=['auditing_date', 'due_date'])


# In[20]:


test_df.head()


# In[21]:


sub = test_df[['user_id', 'listing_id', 'auditing_date', 'due_amt', 'due_date']]


# In[22]:


sub.head()


# In[23]:


test_df.shape


# In[24]:


df = pd.concat([train_df.drop(columns=['label','repay_amt','repay_date']), test_df], axis=0, ignore_index=True)


# In[25]:


df.shape


# In[26]:


df_listing_id = df['listing_id']


# In[27]:


# 时间对比控制不用未来的数据
def merge_before_auditing(df, df2, df2_time='info_insert_date', on='user_id'):
    df3 = df.merge(df2, on=on, how='left')
    df4 = df3[(df3['auditing_date']>df3[df2_time])]
    df5 = df.merge(df4, on=df.columns.tolist(), how='left')
    return df5


# # *listing_info*

# In[28]:


listing_info_df = pd.read_csv('dataset/listing_info.csv', parse_dates=['auditing_date'])


# In[29]:


listing_info_df['principal_per_term'] = listing_info_df['principal'] / listing_info_df['term']


# In[30]:


listing_info_df.head()


# ## 历史贷款信息的统计特征

# In[31]:


listing_hist_df = listing_info_df.drop(columns=['listing_id'])
listing_hist_df.rename({'auditing_date': 'hist_auditing_date'},axis=1,inplace=True)


# In[32]:


df = merge_before_auditing(df, listing_hist_df, df2_time='hist_auditing_date', on="user_id")


# In[33]:


df.head()


# In[34]:


# 总估算利息
df['interest']=(1+df['term'])*df['principal']*df['rate']/24


# In[35]:


groups = df.groupby('listing_id', as_index=False)


# In[36]:


# 历史借款数
df = df.merge(groups['principal'].agg({'hist_loans': len}))


# In[37]:


# 'pricipal'为空时历史借款数为0
df.loc[df['principal'].isnull(),'hist_loans']=0


# In[38]:


df.head()


# In[39]:


# 历史总估算利息统计
df = df.merge(groups['interest'].agg({
    'hist_interest_mean':'mean', 'hist_interest_median':'median', 'hist_interest_std':'std', 'hist_interest_max':'max', 
    'hist_interest_min':'min', 'hist_interest_skew':'skew', 'hist_interest_sum':'sum'
}), on = 'listing_id', how='left')


# In[40]:


df.head()


# In[41]:


# 历史借款额统计
df = df.merge(groups['principal'].agg({
    'hist_principal_mean':'mean', 'hist_principal_median':'median', 'hist_principal_std':'std', 'hist_principal_max':'max', 
    'hist_principal_min':'min', 'hist_principal_skew':'skew', 'hist_principal_sum':'sum'
}), on = 'listing_id', how='left')


# In[42]:


# 历史期数统计
df = df.merge(groups['term'].agg({
    'hist_term_mean':'mean', 'hist_term_median':'median', 'hist_term_std':'std', 'hist_term_max':'max', 
    'hist_term_min':'min', 'hist_term_skew':'skew', 'hist_term_sum':'sum'
}), on = 'listing_id', how='left')


# In[43]:


# 历史期均借款额统计
df = df.merge(groups['principal_per_term'].agg({
    'principal_per_term_mean':'mean', 'principal_per_term_median':'median', 'principal_per_term_std':'std', 
    'principal_per_term_max':'max', 'principal_per_term_min':'min', 'principal_per_term_skew':'skew', 
    'principal_per_term_sum':'sum'
}), on = 'listing_id', how='left')


# In[44]:


# 留下最新的'hist_auditing_date’数据，计算距离最近一次贷款的天数
df = df.sort_values(by='hist_auditing_date', ascending=False).drop_duplicates('listing_id').sort_index().reset_index(drop=True)


# In[45]:


# df['days_after_last_loan'] = (df['auditing_date']-df['hist_auditing_date']).dt.days


# In[46]:


del df['hist_auditing_date'], df['term'], df['rate'], df['principal'], df['interest'], df['principal_per_term']


# ## 当前贷款信息的特征

# In[47]:


del listing_info_df['user_id'], listing_info_df['auditing_date']


# In[48]:


listing_info_df.head()


# In[49]:


listing_info_df.head()


# In[50]:


df = df.merge(listing_info_df, on='listing_id', how='left')


# In[51]:


df.head()


# In[52]:


cate_cols2 = []


# In[53]:


#df['principal_per_term'] = df['principal'] / df['term']
df['ttl_due_amt'] = df['term'] * df['due_amt']
df['ttl_interest'] = df['ttl_due_amt'] - df['principal']
df['interest_per_term'] = df['ttl_interest'] / df['term']
df['ttl_interest/ttl_due_amt'] = df['ttl_interest'] / df['ttl_due_amt']
df['principal_per_term/due_amt'] = df['principal_per_term'] / df['due_amt']
df['ttl_interest/principal'] = df['ttl_interest'] / df['principal']
df['ttl_interest/ttl_due_amt'] = df['ttl_interest'] / df['ttl_due_amt']
df['due_period'] = (df['due_date'] - df['auditing_date']).dt.days
df['due_amt_per_days'] = df['due_amt'] / df['due_period']
df['due_date星期几'] = df['due_date'].dt.dayofweek
df['auditing_date星期几'] = df['auditing_date'].dt.dayofweek
# 这2个特征下面有处理
# df['due_date是当月第几日'] = df['due_date'].dt.day
# df['auditing_date是当月第几日'] = df['auditing_date'].dt.day


# In[54]:


cate_cols2.append('due_date星期几')
cate_cols2.append('auditing_date星期几')
cate_cols2.append('due_period')


# In[55]:


df.head()


# # *user_info*

# In[56]:


user_info_df = pd.read_csv('dataset/user_info.csv', parse_dates=['reg_mon', 'insertdate'])


# In[57]:


user_info_df.head()


# In[58]:


user_info_df.rename(columns={'insertdate': 'info_insert_date'}, inplace=True)


# In[59]:


df = merge_before_auditing(df, user_info_df, df2_time='info_insert_date', on='user_id')


# In[60]:


df = df.sort_values(by='info_insert_date', ascending=False).drop_duplicates('listing_id').sort_index().reset_index(drop=True)


# In[61]:


df['time_bt_aud&reg'] = (df['auditing_date'] - df['reg_mon']).dt.days


# In[62]:


df['time_bt_info&reg'] = (df['info_insert_date'] - df['reg_mon']).dt.days


# In[63]:


df['cell&id_province_is_same'] = (df['cell_province'] == df['id_province']).apply(lambda s: 1 if s is True else 0)


# In[64]:


df.shape


# In[65]:


np.where((df['listing_id'] == df_listing_id)==False)


# # *user_taglist*

# In[66]:


user_tag_df = pd.read_csv('dataset/user_taglist.csv', parse_dates=['insertdate'])


# In[67]:


user_tag_df.head()


# In[68]:


user_tag_df.rename(columns={'insertdate': 'tag_insert_date'}, inplace=True)


# In[69]:


df = merge_before_auditing(df, user_tag_df, df2_time='tag_insert_date', on='user_id')


# In[70]:


df = df.sort_values(by='tag_insert_date', ascending=False).drop_duplicates('listing_id').sort_index().reset_index(drop=True)


# In[71]:


df.shape


# In[72]:


np.where((df['listing_id'] == df_listing_id)==False)


# In[73]:


df['taglist'] = df['taglist'].astype('str').apply(lambda x: x.strip().replace('|', ' ').strip())


# In[74]:


sentences = df['taglist'].apply(lambda x:x.split())


# In[75]:


# get unigram probability
counter = Counter(np.hstack(sentences))
total_num = sum(counter.values())
for i in counter.keys():
    counter[i] = counter[i] / total_num


# In[76]:


model = Word2Vec(sentences, size=100, sg=1, workers=87)


# In[77]:


model.save("./taglist_word2vec.model")


# In[78]:


# sentence embedding: step1 - weighted averaging
sentence_vec = []
for sentence in sentences:
    sentence_vec.append(np.sum([model.wv[word] * 0.001/(0.001 + counter[word]) for word in sentence],axis=0)/len(sentence))


# In[79]:


# step2 - substract the projection of sentence_vec to their first principal component
pca = PCA(n_components = 1)
pca.fit(sentence_vec)
sentence_vec = [vs - np.dot(pca.components_[0], vs) * pca.components_[0] for vs in sentence_vec]


# In[80]:


sentence_vec = np.array(sentence_vec)
tag_cols = ['tag_vec{}'.format(i) for i in range(100)]
for i, tag_col in enumerate(tag_cols):
    df[tag_col] = np.array(sentence_vec)[:,i]


# In[81]:


del df['taglist']


# # *user_behavior_logs*

# In[82]:


user_behavior_df = pd.read_csv('dataset/user_behavior_logs.csv', parse_dates=['behavior_time'])


# In[83]:


user_behavior_df.head()


# In[84]:


user_bh_df = user_behavior_df.set_index('user_id')


# In[85]:


df2 = df[['user_id', 'listing_id', 'auditing_date']].copy()


# In[86]:


df3 = merge_before_auditing(df2, user_behavior_df, df2_time='behavior_time', on = 'user_id' )


# In[87]:


df3.head()


# In[88]:


df.set_index('listing_id', inplace = True)


# In[89]:


def length(a):
    return len(a)


# In[90]:


get_ipython().run_cell_magic('time', '', "df['behavior的个数'] = df3.groupby('listing_id').progress_apply(length)\n\n# df5['behavior为1的个数'] = df3.groupby('listing_id').progress_apply(lambda s: len(s[s['behavior_type']==1]))\ndf['behavior为1的个数'] = df3[df3['behavior_type']==1].groupby('listing_id').progress_apply(length)\ndf['behavior为2的个数'] = df3[df3['behavior_type']==2].groupby('listing_id').progress_apply(length)\ndf['behavior为3的个数'] = df3[df3['behavior_type']==3].groupby('listing_id').progress_apply(length)\n\n\n# df5['behavior为2的个数'] = df3.groupby('listing_id').progress_apply(lambda s: len(s[s['behavior_type']==2]))\n# df5['behavior为3的个数'] = df3.groupby('listing_id').progress_apply(lambda s: len(s[s['behavior_type']==3]))\ndf['behavior的个数'].fillna(0, inplace=True)\ndf['behavior为1的个数'].fillna(0, inplace=True)\ndf['behavior为2的个数'].fillna(0, inplace=True)                                                                 \ndf['behavior为3的个数'].fillna(0, inplace=True)")


# In[91]:


df.head()


# In[92]:


df3['dayofmonth'] = df3['behavior_time'].dt.day


# In[93]:


df3.head()


# In[94]:


df3['days'] = (df3['auditing_date']-df3['behavior_time']).dt.days


# In[95]:


df7 = df3[(df3['days']>=0)&(df3['days']<30)]


# In[96]:


df7.shape


# In[97]:


get_ipython().run_cell_magic('time', '', "df['1月内behavior的个数'] = df7.groupby('listing_id').progress_apply(length)\n# df5['behavior为1的个数'] = df3.groupby('listing_id').progress_apply(lambda s: len(s[s['behavior_type']==1]))\ndf['1月内behavior为1的个数'] = df7[df7['behavior_type']==1].groupby('listing_id').progress_apply(length)                                                    \ndf['1月内behavior为2的个数'] = df7[df7['behavior_type']==2].groupby('listing_id').progress_apply(length)                                                 \ndf['1月内behavior为3的个数'] = df7[df7['behavior_type']==3].groupby('listing_id').progress_apply(length)\ndf['1月内behavior的个数'].fillna(0, inplace=True)\ndf['1月内behavior为1的个数'].fillna(0, inplace=True)\ndf['1月内behavior为2的个数'].fillna(0, inplace=True)                                                                 \ndf['1月内behavior为3的个数'].fillna(0, inplace=True)")


# In[98]:


df.reset_index(inplace=True)


# In[99]:


df.head()


# In[100]:


df.shape


# In[101]:


np.where((df['listing_id'] == df_listing_id)==False)


# # *user_repay_logs*

# In[102]:


repay_log_df = pd.read_csv('dataset/user_repay_logs.csv', parse_dates=['due_date', 'repay_date'])


# In[103]:


repay_log_df = repay_log_df[repay_log_df['order_id'] == 1].reset_index(drop=True)


# In[104]:


repay_log_df.sort_values(by='due_date',ascending=False).head()


# In[105]:


def getRepay(date):
    if date!=datetime.datetime(2200,1,1):
        return 1
    else:
        return 0

# repay: 0[expired] 1[on time]
repay_log_df['repay'] = repay_log_df['repay_date'].parallel_apply(getRepay)


# In[106]:


repay_log_df['early_repay_days'] = (repay_log_df['due_date'] - repay_log_df['repay_date']).dt.days


# In[107]:


repay_log_df['early_repay_days'] = repay_log_df['early_repay_days'].apply(lambda x: x if x >= 0 else -1)


# In[108]:


def adjustDate(df):
    if df['repay_date']!=datetime.datetime(2200,1,1):
        return df['repay_date']
    else:
        return df['due_date']

repay_log_df['repay_date'] = repay_log_df[['repay_date','due_date']].parallel_apply(adjustDate, axis=1)


# In[109]:


def divide(df):
    if df['early_repay_days'] < 0:
        return df['due_amt']/df['early_repay_days']
    else:
        return df['due_amt']/(df['early_repay_days']+1)

repay_log_df['due_amt/early_repay_date'] = repay_log_df[['due_amt','early_repay_days']].parallel_apply(divide, axis=1)


# In[110]:


repay_log_df.head(10)


# In[111]:


# 删除'listing_id', 'order_id', 'due_date','repay_amt'，保留'repay_date'以便之后的时间对比
for f in ['listing_id', 'order_id', 'due_date','repay_amt']:
    del repay_log_df[f]


# In[112]:


repay_log_df = repay_log_df.rename(columns={'due_amt':'log_due_amt', 'repay_date':'log_repay_date'})


# In[113]:


repay_log_df.head()


# In[114]:


df.shape


# In[115]:


df = merge_before_auditing(df, repay_log_df, df2_time='log_repay_date', on='user_id')


# In[116]:


group = df.groupby('listing_id', as_index=False)


# In[117]:


group.ngroups


# In[118]:


df = df.merge(
    group['repay'].agg({'repay_mean': 'mean'}), on='listing_id', how='left'
)


# In[119]:


df = df.merge(group['due_amt/early_repay_date'].agg({
    '(due_amt/early_repay_date)_mean': 'mean', '(due_amt/early_repay_date)_std': 'std', 
    '(due_amt/early_repay_date)_median': 'median', '(due_amt/early_repay_date)_median': 'skew',
    '(due_amt/early_repay_date)_max':'max', '(due_amt/early_repay_date)_min': 'min'
}), on='listing_id', how='left')


# In[120]:


df = df.merge(group['early_repay_days'].agg({
    'early_repay_days_max': 'max', 'early_repay_days_min': 'min', 'early_repay_days_median': 'median', 
    'early_repay_days_sum': 'sum', 'early_repay_days_mean': 'mean', 'early_repay_days_std': 'std',
    'early_repay_days_skew': 'skew'
}), on='listing_id', how='left')


# In[121]:


df = df.merge(group['log_due_amt'].agg({
    'due_amt_max': 'max', 'due_amt_min': 'min', 'due_amt_median': 'median',
    'due_amt_mean': 'mean', 'due_amt_sum': 'sum', 'due_amt_std': 'std',
    'due_amt_skew': 'skew', 'due_amt_kurt': kurtosis, 'due_amt_ptp': np.ptp
}), on='listing_id', how='left')


# In[122]:


# 最近一次'early_repay_days'和'latest_(due_amt/early_repay_date)'
# df = df.drop_duplicates('listing_id').reset_index(drop=True)
df = df.sort_values(by='log_repay_date', ascending=False).drop_duplicates('listing_id').sort_index().reset_index(drop=True)
df.rename(columns={'early_repay_days':'latest_early_repay_days', 'due_amt/early_repay_date': 'latest_(due_amt/early_repay_date)'}, inplace=True)


# In[123]:


df.head()


# In[124]:


del df['repay'], df['log_due_amt'], df['log_repay_date']


# In[125]:


df.shape


# In[126]:


np.where((df['listing_id'] == df_listing_id)==False)


# # 根据age聚合

# In[127]:


df['age'].value_counts().sort_index().plot(kind='bar',use_index=True)


# In[128]:


age_kmeans = KMeans(n_clusters=8, random_state=2019).fit(df[['age']])


# In[129]:


df['age_label'] = age_kmeans.labels_


# In[130]:


df.head()


# In[131]:


age_groups = df.groupby('age_label',as_index=False)


# In[132]:


# 各组的'early_repay_days_mean'统计信息
df = df.merge(age_groups['early_repay_days_mean'].agg({
    'early_repay_days_mean_age_mean':'mean', 'early_repay_days_mean_age_std':'std'
}), on='age_label', how='left')


# In[133]:


# 各组的'early_repay_days_median'统计信息
df = df.merge(age_groups['early_repay_days_median'].agg({
    'early_repay_days_median_age_mean':'mean', 'early_repay_days_median_age_std':'std'
}), on='age_label', how='left')


# In[134]:


# 各组的'hist_principal_mean'统计信息
df = df.merge(age_groups['hist_principal_mean'].agg({
    'hist_principal_mean_age_mean':'mean', 'hist_principal_mean_age_std':'std'
}), on='age_label', how='left')


# In[135]:


# 各组的'due_amt_mean'统计信息
df = df.merge(age_groups['due_amt_mean'].agg({
    'due_amt_mean_age_mean':'mean', 'due_amt_mean_age_std':'std'
}), on='age_label', how='left')


# In[136]:


# 各组的'hist_loans'统计信息
df = df.merge(age_groups['hist_loans'].agg({
    'hist_loans_age_mean':'mean', 'hist_loans_age_std':'std'
}), on='age_label', how='left')


# In[137]:


# 计算个体值和组统计均值的差值
df['early_repay_days_mean_age_diff'] = df['early_repay_days_mean'] - df['early_repay_days_mean_age_mean']
df['early_repay_days_median_age_diff'] = df['early_repay_days_median'] - df['early_repay_days_median_age_mean']
df['hist_principal_mean_age_diff'] = df['hist_principal_mean'] - df['hist_principal_mean_age_mean']
df['due_amt_mean_age_diff'] = df['due_amt_mean'] -df['due_amt_mean_age_mean']
df['hist_loans_age_diff'] = df['hist_loans'] -df['hist_loans_age_mean']


# In[138]:


del df['age_label']


# # 根据hist_loans聚合

# In[139]:


df[df['hist_loans']>50]['hist_loans'].value_counts().sort_index().plot(kind='bar',use_index=True)


# In[140]:


hist_loans_kmeans = KMeans(n_clusters=8, random_state=2019).fit(df[['hist_loans']])


# In[141]:


df['hist_loans_label'] = hist_loans_kmeans.labels_


# In[142]:


hist_loans_groups = df.groupby('hist_loans_label',as_index=False)


# In[143]:


# 各组的'early_repay_days_mean'统计信息
df = df.merge(hist_loans_groups['early_repay_days_mean'].agg({
    'early_repay_days_mean_hist_mean':'mean', 'early_repay_days_mean_hist_std':'std'
}), on='hist_loans_label', how='left')


# In[144]:


# 各组的'early_repay_days_median'统计信息
df = df.merge(hist_loans_groups['early_repay_days_median'].agg({
    'early_repay_days_median_hist_mean':'mean', 'early_repay_days_median_hist_std':'std'
}), on='hist_loans_label', how='left')


# In[145]:


# 计算个体值和组统计均值的差值
df['early_repay_days_mean_hist_diff'] = df['early_repay_days_mean'] - df['early_repay_days_mean_hist_mean']
df['early_repay_days_median_hist_diff'] = df['early_repay_days_median'] - df['early_repay_days_median_hist_mean']


# In[146]:


del df['hist_loans_label']


# # 处理类别特征和日期

# In[147]:


cate_cols = ['gender', 'cell_province', 'id_province', 'id_city']


# In[148]:


# 这个lgb应该有参数可以直接传 lgb.train(categorical_feature=cate_cols)
for f in cate_cols:
    df[f] = df[f].map(dict(zip(df[f].unique(), range(df[f].nunique())))).astype('int32')


# In[149]:


date_cols = ['auditing_date', 'due_date', 'reg_mon', 'info_insert_date', 'tag_insert_date']


# In[150]:


for f in date_cols:
    if f in ['reg_mon', 'info_insert_date', 'tag_insert_date']:
        df[f + '_year'] = df[f].dt.year
    df[f + '_month'] = df[f].dt.month
    if f in ['auditing_date', 'due_date', 'info_insert_date', 'tag_insert_date']:
        df[f + '_day'] = df[f].dt.day
        df[f + '_dayofweek'] = df[f].dt.dayofweek


# In[151]:


df.drop(columns=date_cols, axis=1, inplace=True)


# In[152]:


df['big_month'] = df['auditing_date_month'].apply(lambda x: 1 if x in [1,3,5,7,8,10,12] else 0)
df['February'] = df['auditing_date_month'].apply(lambda x: 1 if x==2 else 0)


# In[153]:


# one-hot encoding for tags
df['taglist'] = df['taglist'].astype('str').apply(lambda x: x.strip().replace('|', ' ').strip())


# In[ ]:


# vectorizer = CountVectorizer(min_df=10, max_df=0.9)
# tag_cv = vectorizer.fit_transform(df['taglist'])


# In[ ]:


del df['user_id'], df['listing_id']


# In[ ]:


# 也可以用lgb.train(categorical_features=cate_cols)
df = pd.get_dummies(df, columns=cate_cols)
df = pd.get_dummies(df, columns=cate_cols2)


# # 训练模型

# In[ ]:


def add_1_month(s):
    s = s.strftime('%F')
    y, m, d = str(s).split('-')
    y = int(y)
    m = int(m)
    d = int(d)
    m = m + 1
    if m == 13:
        m = 1
        y = y + 1
    if m in [4,6,9,11]:
        if d == 31:
            d = 30
    if m == 2:
        if d in [29, 30, 31]:
            if y in [2012, 2016]:
                d = 29
            else:
                d = 28
    return datetime.datetime.strptime(str(y)+'-'+str(m)+'-'+str(d), '%Y-%m-%d')


# In[ ]:


# 把整个验证集看作一个资产组合计算rmse
def new_rmse(val_df, prob_oof):
    val_df2=val_df[['listing_id']].copy()
    # 制作一个类似submission的表
    val_df['pre_repay_date'] = val_df['auditing_date']
    val_df_temp = val_df.copy()
    for i in range(31):
        val_df_temp['pre_repay_date'] = plus_1_day(val_df_temp['pre_repay_date'])
        val_df= pd.concat([val_df, val_df_temp],axis=0, ignore_index=True)
    val_df = val_df[val_df['pre_repay_date']<=val_df['due_date']]
    
    prob_cols = ['prob_{}'.format(i) for i in range(33)]
    for i, f in enumerate(prob_cols):
        val_df2[f] = prob_oof[:, i]
    val_df = val_df.merge(val_df2, on='listing_id', how='left')
    val_df['days'] = (val_df['due_date'] - val_df['pre_repay_date']).dt.days
    val_prob = val_df[prob_cols].values
    val_labels = val_df['days'].values
    val_prob = [val_prob[i][val_labels[i]] for i in range(val_prob.shape[0])]
    
    val_df['pre_repay_amt'] = val_df['due_amt'] * val_prob
    val_df['repay_amt'] = val_df[val_df['pre_repay_date']==val_df['repay_date']]['due_amt']
    groups_date = val_df.groupby('pre_repay_date')
    repay_amt = groups_date.repay_amt.sum()
    pre_repay_amt = groups_date.pre_repay_amt.sum()
    days = groups_date.ngroups
    
    return np.sqrt(mean_squared_error(repay_amt, pre_repay_amt))


# In[ ]:


feature_name = np.concatenate((df.keys(),vectorizer.get_feature_names()))


# In[ ]:


df_sp = sparse.hstack((df.values, tag_cv), format='csr', dtype='float32')


# In[ ]:


train_values, test_values = df_sp[:train_num], df_sp[train_num:]


# In[ ]:


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2019)
clf = LGBMClassifier(
    boosting_type='gbdt',
    objective='multiclass',
    num_leaves=70,
    reg_alpha=3,
    reg_lambda=5,
    max_depth=7,
    n_jobs=87,
    learning_rate=0.1,
    n_estimators=5000,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.77,
    random_state=2019,
    min_child_weight=4,
    min_child_samples=5,
    min_split_gain=0
)
amt_oof = np.zeros(train_num)
prob_oof = np.zeros((train_num, 33))
test_pred_prob = np.zeros((test_values.shape[0], 33))
for i, (trn_idx, val_idx) in enumerate(skf.split(train_values, clf_labels)):
    print(i, 'fold...')
    t = time.time()

    trn_x, trn_y = train_values[trn_idx], clf_labels[trn_idx]
    val_x, val_y = train_values[val_idx], clf_labels[val_idx]
    val_repay_amt = amt_labels[val_idx]
    val_due_amt = train_due_amt_df[val_idx]
    val_df = train_df[['listing_id','auditing_date','due_date','due_amt','repay_date']].iloc[val_idx]

    clf.fit(
        trn_x, trn_y,
        eval_set=[(trn_x, trn_y), (val_x, val_y)],
        early_stopping_rounds=50, verbose=5, feature_name=list(feature_name)
    )
    # shape = (-1, 33)
    val_pred_prob_everyday = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)
    prob_oof[val_idx] = val_pred_prob_everyday
    val_pred_prob_today = [val_pred_prob_everyday[i][val_y[i]] for i in range(val_pred_prob_everyday.shape[0])]
    val_pred_repay_amt = val_due_amt * val_pred_prob_today
    print('val rmse:', np.sqrt(mean_squared_error(val_repay_amt, val_pred_repay_amt)))
    print('val mae:', mean_absolute_error(val_repay_amt, val_pred_repay_amt))
    print('val new rmse:', new_rmse(val_df, val_pred_prob_everyday))
    amt_oof[val_idx] = val_pred_repay_amt
    test_pred_prob += clf.predict_proba(test_values, num_iteration=clf.best_iteration_) / skf.n_splits

    print('runtime: {}\n'.format(time.time() - t))

print('\ncv rmse:', np.sqrt(mean_squared_error(amt_labels, amt_oof)))
print('cv mae:', mean_absolute_error(amt_labels, amt_oof))
print('cv logloss:', log_loss(clf_labels, prob_oof))
print('cv acc:', accuracy_score(clf_labels, np.argmax(prob_oof, axis=1)))
print('cv new_rmse:', new_rmse(train_df[['listing_id','auditing_date','due_date','due_amt','repay_date']], prob_oof))


# In[ ]:


# # 原本的输出
# prob_cols = ['prob_{}'.format(i) for i in range(33)]
# for i, f in enumerate(prob_cols):
#     sub[f] = test_pred_prob[:, i]
# sub_example = pd.read_csv('dataset/submission.csv', parse_dates=['repay_date'])
# sub_example = sub_example.merge(sub, on='listing_id', how='left')
# sub_example['days'] = (sub_example['repay_date'] - sub_example['auditing_date']).dt.days
# # shape = (-1, 33)
# test_prob = sub_example[prob_cols].values
# test_labels = sub_example['days'].values
# test_prob = [test_prob[i][test_labels[i]] for i in range(test_prob.shape[0])]
# sub_example['repay_amt'] = sub_example['due_amt'] * test_prob
# sub_example[['listing_id', 'repay_date', 'repay_amt']].to_csv('sub.csv', index=False)


# In[ ]:


import pickle
with open("test_pred_prob8.pkl", 'wb') as f:
    pickle.dump(test_pred_prob, f)


# In[ ]:


prob_cols = ['prob_{}'.format(i) for i in range(33)]


# In[ ]:


for i, f in enumerate(prob_cols):
    sub[f] = test_pred_prob[:, i]


# In[ ]:


sub_example = pd.read_csv('dataset/submission.csv', parse_dates=['repay_date'])


# In[ ]:


sub_example = sub_example.merge(sub, on='listing_id', how='left')


# In[ ]:


sub_example['due_date'] = sub_example['auditing_date'].parallel_apply(add_1_month)


# In[ ]:


sub_example['days'] = (sub_example['due_date'] - sub_example['repay_date']).dt.days


# In[ ]:


test_prob = sub_example[prob_cols].values


# In[ ]:


test_labels = sub_example['days'].values


# In[ ]:


test_prob = [test_prob[i][test_labels[i]] for i in range(test_prob.shape[0])]


# In[ ]:


sub_example['repay_amt'] = sub_example['due_amt'] * test_prob


# In[ ]:


sub_example[['listing_id', 'repay_date', 'repay_amt']].to_csv(f"sub_{datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')}.csv", index=False)


# In[ ]:


lgb.plot_importance(clf, max_num_features=100,figsize=(10,30))


# In[ ]:


sub_example2 = sub_example.copy()


# In[ ]:


sub_example2['prob'] = test_prob


# In[ ]:


threshold_up = 0.57
id_with_highpro = sub_example[sub_example2['prob']>=threshold_up]['listing_id']


# In[ ]:


sub_example2.loc[sub_example2['listing_id'].isin(id_with_highpro.values),'repay_amt']=0


# In[ ]:


sub_example2['repay_amt']= sub_example2.apply(lambda x:x['repay_amt'] if x['prob']<threshold_up else x['due_amt'],axis=1)


# In[ ]:


sub_example2[['listing_id', 'repay_date', 'repay_amt']].to_csv(f"sub_{datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')}_0_{threshold_up}.csv", index=False)


# In[ ]:


sub_example.groupby('auditing_date').apply(lambda x:x['repay_amt'].sum()).plot(kind='line')


# In[ ]:


sub_example.groupby('repay_date').apply(lambda x:x['repay_amt'].sum()).plot(kind='line')


# def alter(df, date, a):
#     """
#     data就是要增加的日期
#     a就是这天增加的比例
#     """
#     if date not in df['repay_date'].values:
#         return df
#     else:
#         dates = df['repay_date'].tolist()
#         dates.remove(date)
#         df = df.set_index('repay_date')
#         b = df.loc[date, 'repay_amt'] * a
#         df.loc[date, 'repay_amt'] += b
#         df.loc[dates, 'repay_amt'] -= b / len(dates)
#         df.reset_index(inplace=True)
#         return df

# sub_example_1 = sub_example.copy()

# sub_example_1['repay_date']=sub_example_1['repay_date'].astype(str)

# sub_example_1 = sub_example_1.groupby('listing_id', as_index=False).apply(alter, date="2019-03-29", a=0.1).reset_index(drop=True)[['listing_id', 'repay_date', 'repay_amt']]
# sub_example_1 = sub_example_1.groupby('listing_id', as_index=False).apply(alter, date="2019-03-30", a=0.1).reset_index(drop=True)[['listing_id', 'repay_date', 'repay_amt']]
# sub_example_1 = sub_example_1.groupby('listing_id', as_index=False).apply(alter, date="2019-03-31", a=0.1).reset_index(drop=True)[['listing_id', 'repay_date', 'repay_amt']]

# sub_example_1.to_csv("sub_ex.csv",index=False)
