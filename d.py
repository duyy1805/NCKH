import warnings

import pandas as pd  # importing the necessary packages

warnings.filterwarnings("ignore")
import shutil

from sklearn.metrics import roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

get_ipython().run_line_magic('matplotlib', 'inline')
import itertools
import os
from time import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics, svm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.feature_selection import (SelectFromModel, SelectKBest, chi2,
                                       mutual_info_classif)
#from mlxtend.regressor import StackingCVRegressor
from sklearn.linear_model import (Lasso, PassiveAggressiveClassifier,
                                  Perceptron, Ridge, RidgeClassifier,
                                  SGDClassifier)
#from mlxtend.classifier import StackingCVClassifier
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.utils.extmath import density

matplotlib.use(u'nbAgg')
import codecs  # this is used for file operations
import multiprocessing
import pickle
import random as r
from collections import Counter
from multiprocessing import Process  # this is used for multithreading

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.sparse import hstack
from sklearn import preprocessing
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, log_loss,
                             plot_confusion_matrix, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     RepeatedStratifiedKFold, StratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import (FunctionTransformer, LabelEncoder,
                                   OneHotEncoder, PowerTransformer,
                                   StandardScaler)
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

#Đọc dữ liệu
df_train=pd.read_csv("NSL_KDD_Dataset\KDDTrain+.txt")
df_test=pd.read_csv("NSL_KDD_Dataset\KDDTest+.txt")

# In[ ]:

df_train.head() #Check 5 hàng đầu

# In[ ]:


df_train.tail() #check 5 hàng cuối

# In[ ]:


print("Number of data points in train data", df_train.shape)  
print('-'*50)
print("The attributes of data :", df_train.columns.values) #check thuộc tính

# In[ ]:


print("Number of data points in test data", df_test.shape)
print(df_test.columns.values)
df_test.head(2)

# In[ ]:


df_train.dtypes  #check kiểu dữ liệu

# In[3]:

#Đổi tên sang 43 feature

df_train = df_train.rename(columns={"0":"Duration","tcp":"protocol_type","ftp_data":"service","SF":"flag","491":"src_bytes",
                                    "0.1":"dest_bytes","0.2":"Land","0.3":"wrong_fragment","0.4":"Urgent packets","0.5":"hot",
                                    "0.6":"num_failed_logins","0.7":"logged_in","0.8":"num_compromised","0.9":"root_shell",
                                    "0.10":"su_attempted","0.11":"num_root","0.12":"num_file_creations","0.13":"num_shells",
                                    "0.14":"num_access_files","0.15":"num_outbound_cmds","0.16":"is_host_login","0.17":"is_guest_login",
                                    "2":"count","2.1":"srv_count","0.00":"serror_rate","0.00.1":"srv_serror_rate","0.00.2":"rerror_rate",
                                    "0.00.3":"srv_rerror_rate","1.00":"same_srv_rate","0.00.4":"diff_srv_rate","0.00.5":"srv_diff_host_rate",
                                    "150":"dst_host_count","25":"dst_host_srv_count","0.17.1":"dst_host_same_srv_rate",
                                    "0.03":"dst_host_diff_srv_rate","0.17.2":"dst_host_same_src_port_rate",
                                    "0.00.6":"dst_host_srv_diff_host_rate","0.00.7":"dst_host_serror_rate",
                                    "0.00.8":"dst_host_srv_serror_rate","0.05":"dst_host_rerror_rate","0.00.9":"dst_host_srv_rerror_rate",
                                    "normal":"attack_type","20":"Score"})

# In[ ]:
print(df_train['protocol_type'].value_counts())
print(df_train['flag'].value_counts())

df_train.head()  #check dataframe sau khi đổi tên

# In[4]:


#Tương tự với df_test 

df_test = df_test.rename(columns={"0":"Duration","tcp":"protocol_type","private":"service","REJ":"flag","0.1":"src_bytes",
                                    "0.2":"dest_bytes","0.3":"Land","0.4":"wrong_fragment","0.5":"Urgent packets","0.6":"hot",
                                    "0.7":"num_failed_logins","0.8":"logged_in","0.9":"num_compromised","0.10":"root_shell",
                                    "0.11":"su_attempted","0.12":"num_root","0.13":"num_file_creations","0.14":"num_shells",
                                    "0.15":"num_access_files","0.16":"num_outbound_cmds","0.17":"is_host_login","0.18":"is_guest_login",
                                    "229":"count","10":"srv_count","0.00":"serror_rate","0.00.1":"srv_serror_rate","1.00":"rerror_rate",
                                    "1.00.1":"srv_rerror_rate","0.04":"same_srv_rate","0.06":"diff_srv_rate","0.00.2":"srv_diff_host_rate",
                                    "255":"dst_host_count","10.1":"dst_host_srv_count","0.04.1":"dst_host_same_srv_rate",
                                    "0.06.1":"dst_host_diff_srv_rate","0.00.3":"dst_host_same_src_port_rate",
                                    "0.00.4":"dst_host_srv_diff_host_rate","0.00.5":"dst_host_serror_rate",
                                    "0.00.6":"dst_host_srv_serror_rate","1.00.2":"dst_host_rerror_rate","1.00.3":"dst_host_srv_rerror_rate",
                                    "neptune":"attack_type","21":"Score"})

# In[ ]:


df_train.head()

# In[ ]:


df_test.head()

# In[ ]:


# hàng trùng lặp
duplicate_rows_df = df_train[df_train.duplicated()]
print("number of duplicate rows: ", duplicate_rows_df.shape)

# In[ ]:


# Giá trị null
print(df_train.isnull().sum())

# In[5]:


label_encoder1 = preprocessing.LabelEncoder() 
df_train['protocol_type']= label_encoder1.fit_transform(df_train['protocol_type']) 
a=label_encoder1.classes_ 
label_encoder1.classes_

# In[8]:


int_features=['tcp','private','REJ']
int_features

# In[9]:


for i in range(len(a)):
        if a[i]==int_features[0]:
            int_features[0]=i


int_features

# In[11]:


import pickle

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# # Exploratory Data Analysis

# In[ ]:


y_value_counts = df_train['attack_type'].value_counts()  #kiểm tra sự phân phối của các lớp khác nhau của từng nhãn
y_value_counts

# Ploting the bar plot of attack type variable to check the distribution of different class in the dataset-Train

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
y_value_counts.plot(kind="bar", fontsize=10)

# Tương tự với test-data

# In[ ]:


y_test_value_counts = df_test['attack_type'].value_counts() 
y_test_value_counts

get_ipython().run_line_magic('matplotlib', 'inline')
y_test_value_counts.plot(kind="bar", fontsize=10)

# Observation: The above plot clearly shows that the attack type "normal" has the highest distribution in the data followed by "neptune" and then the other classes whose value count is very less compared to these two classes. The distribution is almost same for both test dataset and train dataset.

# In[ ]:


# counter = Counter(df_train['attack_type'])
# a=dict(counter)
# per=[]
# for k,v in counter.items():
# 	per.append(v / len(df_train['attack_type']) * 100) #calculating the percentage distribution of my class label

# ## Plotting the pie chart of attack type with the percentage distribution of each attack type 

# In[ ]:


# patches, texts = plt.pie(per, startangle=90, radius=2)  #https://stackoverflow.com/questions/23577505/how-to-avoid-overlapping-of-labels-autopct-in-a-matplotlib-pie-chart
# labels = ['{0} - {1:1.2f} %'.format(i,j) for i,j in zip(a.keys(), per)]
# patches, labels, dummy =  zip(*sorted(zip(patches, labels, per),
#                                           key=lambda x: x[2],
#                                           reverse=True))

# plt.legend(patches,labels , loc='center left', bbox_to_anchor=(-0.1, 1.), fontsize=8)

# plt.savefig('piechart.png', bbox_inches='tight')

# The above plot gives an idea of the percentage value of each class. The normal class covers almost 53% of the data followed by neptune class which covers 32% and then the rest of the classes each covering less than 3% of the entire dataset. From the above plot we can conclude that our dataset is an imbalanced dataset with huge difference in the distribution of different class labels

# Lets have a look at the distribution of each feature of the dataframe.
# 
# ---
# 
# ---
# 
# 
# 
# 

# In[ ]:


# df_train.hist(figsize=(35,35)) 
# plt.tight_layout()
# plt.show()

# ## Now lets view the correlation between features and target variable.

# In[ ]:

#Ma trận tương quan


# In[ ]:


import phik
from phik import report, resources

corr_matrix=df_train.phik_matrix()
corr_matrix

# In[ ]:


print(corr_matrix["attack_type"].sort_values(ascending=False)[1:])


# In[ ]:


corr = corr_matrix["attack_type"].sort_values(ascending=False)

attack_sep={'normal':"Normal",'neptune':"DOS",
            'satan':"Probe",'ipsweep':"Probe",'named':"R2L",
            'ps':"U2R",'sendmail':"R2L",'xterm':"U2R",'xlock':"R2L",
            'xsnoop':"R2L",'udpstorm':"DOS",'sqlattack':"U2R",'worm':"DOS",'portsweep':"Probe",
            'smurf':"DOS",'nmap':"Probe",'back':"DOS",'mscan':"Probe",'apache2':"DOS",'processtable':"DOS",
            'snmpguess':"R2L",'saint':"Probe",'mailbomb':"DOS",'snmpgetattack':"R2L",'httptunnel':"R2L",'teardrop':"DOS",
            'warezclient':"R2L",'pod':"DOS",'guess_passwd':"R2L",'buffer_overflow':"U2R",'warezmaster':"R2L",'land':"DOS",'imap':"R2L",
            'rootkit':"U2R",'loadmodule':"U2R",'ftp_write':"R2L",'multihop':"R2L",'phf':"R2L",'perl':"U2R",'spy':"R2L"}


# In[7]:


df_train.replace({'attack_type':attack_sep},inplace=True)


# In[8]:


df_test.replace({'attack_type':attack_sep},inplace=True)


# # Lets train a base model on the entire dataset and evaluate the performance.

# In[9]:


from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder() 

df_train['protocol_type']= label_encoder.fit_transform(df_train['protocol_type']) 
df_test['protocol_type']= label_encoder.fit_transform(df_test['protocol_type']) 

label_encoder = preprocessing.LabelEncoder() 
  
df_train['service']= label_encoder.fit_transform(df_train['service']) 
df_test['service']= label_encoder.transform(df_test['service']) 

label_encoder = preprocessing.LabelEncoder() 

df_train['flag']= label_encoder.fit_transform(df_train['flag']) 
df_test['flag']= label_encoder.transform(df_test['flag'])

# In[10]:

#tách feature và labels

y=df_train['attack_type']   #labels
X=df_train.drop(['attack_type'],axis=1) #feature

y_test=df_test['attack_type']
X_test=df_test.drop(['attack_type'],axis=1)

# In[11]:


sc = StandardScaler()  #Chuẩn hóa dữ liệu
X_train = sc.fit_transform(X)
X_test = sc.transform(X_test)

# # Lets build a base model on our dataset

# In[18]:


def falseposrate(conf_matrix,y_test,pred):
  FP = conf_matrix.sum(axis=0) - np.diag(conf_matrix) 
  FN = conf_matrix.sum(axis=1) - np.diag(conf_matrix)
  TP = np.diag(conf_matrix)
  TN = conf_matrix.sum() - (FP + FN + TP)
  FP = FP.astype(float)
  FN = FN.astype(float)
  TP = TP.astype(float)
  TN = TN.astype(float)
  FPR = FP/(FP+TN)
  recall = recall_score(y_test, pred,average='micro')
  precision = precision_score(y_test, pred,average='micro')
  return FPR,recall,precision

# In[ ]:

#sử dụng Support Vector Machine
# get_ipython().run_line_magic('matplotlib', 'inline')
# clf= svm.SVC(kernel='linear',probability=True)
# clf.fit(X_train,y) #learning
# pred = clf.predict(X_test) #kết quả sau khi test
# recall = recall_score(y_test, pred,average='micro')
# precision = precision_score(y_test, pred,average='micro')
# score = metrics.accuracy_score(y_test, pred)
# f1score= f1_score(y_test, pred, average='micro')
# print("Accuracy :",score)
# print('=' * 50)
# print("F1 score :",f1score)

# cnf_matrix = confusion_matrix(y_test, pred) #confusion matrix của nhẫn tấn công (test data) và kết quả sau khi test
# #sns.heatmap(cnf_matrix)
# fig, ax = plt.subplots(figsize=(15, 8))
# disp = plot_confusion_matrix(clf, X_test, y_test,ax=ax,cmap=plt.cm.Blues)
# plt.show()

# print('_' * 50)
# print(cnf_matrix)

# FPR= falseposrate(cnf_matrix, y_test, pred)
# print('=' * 50)
# print("|False positive Rate :|")
# print('=' * 50)
# print(FPR)
# print('=' * 50)
# print("|Precision:|")
# print('=' * 50)
# print(precision)
# print('=' * 50)
# print("|recall:|")
# print('=' * 50)
# print( recall)
# print('=' * 50)
# print("|Classification report|")
# print('=' * 50)
# print(metrics.classification_report(y_test,pred))

# In[ ]:

#sử dụng DecisionTreeClassifier
get_ipython().run_line_magic('matplotlib', 'inline')
clf= DecisionTreeClassifier()
clf.fit(X_train,y) #learning
pred = clf.predict(X_test) #kết quả sau khi test
recall = recall_score(y_test, pred,average='micro')
precision = precision_score(y_test, pred,average='micro')
score = metrics.accuracy_score(y_test, pred)
f1score= f1_score(y_test, pred, average='micro')
print("Accuracy :",score)
print('=' * 50)
print("F1 score :",f1score)

cnf_matrix = confusion_matrix(y_test, pred) #confusion matrix của nhẫn tấn công (test data) và kết quả sau khi test
#sns.heatmap(cnf_matrix)
fig, ax = plt.subplots(figsize=(15, 8))
disp = plot_confusion_matrix(clf, X_test, y_test,ax=ax,cmap=plt.cm.Blues)
plt.show()

print('_' * 50)
print(cnf_matrix)

FPR= falseposrate(cnf_matrix, y_test, pred)
print('=' * 50)
print("|False positive Rate :|")
print('=' * 50)
print(FPR)
print('=' * 50)
print("|Precision:|")
print('=' * 50)
print(precision)
print('=' * 50)
print("|recall:|")
print('=' * 50)
print( recall)
print('=' * 50)
print("|Classification report|")
print('=' * 50)
print(metrics.classification_report(y_test,pred))

# In[ ]:

#sử dụng Naive Bayes
get_ipython().run_line_magic('matplotlib', 'inline')
clf= GaussianNB()
clf.fit(X_train,y) #learning
pred = clf.predict(X_test) #kết quả sau khi test
recall = recall_score(y_test, pred,average='micro')
precision = precision_score(y_test, pred,average='micro')
score = metrics.accuracy_score(y_test, pred)
f1score= f1_score(y_test, pred, average='micro')
print("Accuracy :",score)
print('=' * 50)
print("F1 score :",f1score)

cnf_matrix = confusion_matrix(y_test, pred) #confusion matrix của nhẫn tấn công (test data) và kết quả sau khi test
#sns.heatmap(cnf_matrix)
fig, ax = plt.subplots(figsize=(15, 8))
disp = plot_confusion_matrix(clf, X_test, y_test,ax=ax,cmap=plt.cm.Blues)
plt.show()

print('_' * 50)
print(cnf_matrix)

FPR= falseposrate(cnf_matrix, y_test, pred)
print('=' * 50)
print("|False positive Rate :|")
print('=' * 50)
print(FPR)
print('=' * 50)
print("|Precision:|")
print('=' * 50)
print(precision)
print('=' * 50)
print("|recall:|")
print('=' * 50)
print( recall)
print('=' * 50)
print("|Classification report|")
print('=' * 50)
print(metrics.classification_report(y_test,pred))

#phần này đang thử chưa hoàn thiện
#Cải thiện hiệu suất bằng việc lựa chọn thuộc tính (feature selection)

abs_corr = abs(corr)
relevant_features = abs_corr[abs_corr>0.5]

new_df= df_train[relevant_features.index]
new_df_test= df_test[relevant_features.index]

y_cfs=new_df['attack_type']
X_cfs=new_df.drop(['attack_type'],axis=1)

y_test_cfs=new_df_test['attack_type']
X_test_cfs=new_df_test.drop(['attack_type'],axis=1)

sc = StandardScaler()
X_train_cfs = sc.fit_transform(X_cfs)
X_test_cfs = sc.transform(X_test_cfs)

from catboost import CatBoostClassifier

# In[ ]:


def training(clf,xtrain,xtest,ytrain,ytest,attack_type):
    print('\n')
    print('=' * 50)
    print("Training ",attack_type)
    print(clf)
    clf.fit(xtrain, ytrain)
    print('_' * 50)
    pred = clf.predict(xtest)
    print('_' * 50)
    roc = roc_auc_score(ytest, clf.predict_proba(xtest), multi_class='ovo', average='weighted')
    score = metrics.accuracy_score(ytest, pred)
    f1score= f1_score(ytest, pred, average='micro')
    print("accuracy:   %0.3f" % score)
    print()
    print('_' * 50)
    print("|classification report|")
    print('_' * 50)
    print(metrics.classification_report(ytest, pred))
    print('_' * 50)
    print("confusion matrix:")
    print(metrics.confusion_matrix(ytest, pred))
    cm= metrics.confusion_matrix(ytest, pred)
    print()
    print('_' * 50)
    print("ROC AUC Score :",roc)
    FPR,precision,recall= falseposrate(cm,ytest,pred)
    print('_' * 50)
    print("False Positive Rate is :",FPR)
    clf_descr = str(clf).split('(')[0]
    return clf_descr,score, f1score,roc,FPR,precision,recall

results_CFS= []

for clf, name in (
        (GaussianNB() ,"Naive Bayes"),
        # (KNeighborsClassifier(n_neighbors = 7),"KNN"),
        #(OneVsRestClassifier(svm.SVC(probability=True)),"One vs Rest SVM "),
        (RandomForestClassifier(), "Random forest"),(DecisionTreeClassifier(random_state=0),"Decision Tree"),
        #(XGBClassifier(),"XGBOOST"),(svm.SVC(kernel='linear',probability=True),"SVM Linear"),
        #(CatBoostClassifier(iterations=5,learning_rate=0.1),"CAT Boost")
        ):
    print('=' * 80)
    print(name)

    results_CFS.append(training(clf,X_train_cfs,X_test_cfs,y_cfs,y_test_cfs,"CFS"))

import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.models import Sequential

df_train=pd.read_csv("NSL_KDD_Dataset/KDDTrain+.txt")
df_test=pd.read_csv("NSL_KDD_Dataset/KDDTest+.txt")

df_train.head()

df_train.dropna(inplace=True,axis=1) # For now, just drop NA's 
# (rows with missing values)

# The CSV file has no column heads, so add them
df_train.columns = [
    'duration',
    'protocol_type',
    'service',
    'flag',
    'src_bytes',
    'dst_bytes',
    'land',
    'wrong_fragment',
    'urgent',
    'hot',
    'num_failed_logins',
    'logged_in',
    'num_compromised',
    'root_shell',
    'su_attempted',
    'num_root',
    'num_file_creations',
    'num_shells',
    'num_access_files',
    'num_outbound_cmds',
    'is_host_login',
    'is_guest_login',
    'count',
    'srv_count',
    'serror_rate',
    'srv_serror_rate',
    'rerror_rate',
    'srv_rerror_rate',
    'same_srv_rate',
    'diff_srv_rate',
    'srv_diff_host_rate',
    'dst_host_count',
    'dst_host_srv_count',
    'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate',
    'dst_host_srv_serror_rate',
    'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate',
    'attack_type',
    'Score'
]

df_train.drop('Score',axis=1, inplace=True)

df_train.head()

# Encode a numeric column as zscores
def encode_numeric_zscore(df, name, mean=None, sd=None):
    if mean is None:
        mean = df[name].mean()

    if sd is None:
        sd = df[name].std()

    df[name] = (df[name] - mean) / sd
    
# Encode text values to dummy variables(i.e. [1,0,0],
# [0,1,0],[0,0,1] for red,green,blue)
def encode_text_dummy(df, name):
    dummies = pd.get_dummies(df[name])
    for x in dummies.columns:
        dummy_name = f"{name}-{x}"
        df[dummy_name] = dummies[x]
    df.drop(name, axis=1, inplace=True)

encode_numeric_zscore(df_train, 'duration')
encode_text_dummy(df_train, 'protocol_type')
encode_text_dummy(df_train, 'service')
encode_text_dummy(df_train, 'flag')
encode_numeric_zscore(df_train, 'src_bytes')
encode_numeric_zscore(df_train, 'dst_bytes')
encode_text_dummy(df_train, 'land')
encode_numeric_zscore(df_train, 'wrong_fragment')
encode_numeric_zscore(df_train, 'urgent')
encode_numeric_zscore(df_train, 'hot')
encode_numeric_zscore(df_train, 'num_failed_logins')
encode_text_dummy(df_train, 'logged_in')
encode_numeric_zscore(df_train, 'num_compromised')
encode_numeric_zscore(df_train, 'root_shell')
encode_numeric_zscore(df_train, 'su_attempted')
encode_numeric_zscore(df_train, 'num_root')
encode_numeric_zscore(df_train, 'num_file_creations')
encode_numeric_zscore(df_train, 'num_shells')
encode_numeric_zscore(df_train, 'num_access_files')
encode_numeric_zscore(df_train, 'num_outbound_cmds')
encode_text_dummy(df_train, 'is_host_login')
encode_text_dummy(df_train, 'is_guest_login')
encode_numeric_zscore(df_train, 'count')
encode_numeric_zscore(df_train, 'srv_count')
encode_numeric_zscore(df_train, 'serror_rate')
encode_numeric_zscore(df_train, 'srv_serror_rate')
encode_numeric_zscore(df_train, 'rerror_rate')
encode_numeric_zscore(df_train, 'srv_rerror_rate')
encode_numeric_zscore(df_train, 'same_srv_rate')
encode_numeric_zscore(df_train, 'diff_srv_rate')
encode_numeric_zscore(df_train, 'srv_diff_host_rate')
encode_numeric_zscore(df_train, 'dst_host_count')
encode_numeric_zscore(df_train, 'dst_host_srv_count')
encode_numeric_zscore(df_train, 'dst_host_same_srv_rate')
encode_numeric_zscore(df_train, 'dst_host_diff_srv_rate')
encode_numeric_zscore(df_train, 'dst_host_same_src_port_rate')
encode_numeric_zscore(df_train, 'dst_host_srv_diff_host_rate')
encode_numeric_zscore(df_train, 'dst_host_serror_rate')
encode_numeric_zscore(df_train, 'dst_host_srv_serror_rate')
encode_numeric_zscore(df_train, 'dst_host_rerror_rate')
encode_numeric_zscore(df_train, 'dst_host_srv_rerror_rate')

df_train.dropna(inplace=True,axis=1)
df_train[0:5]
# This is the numeric feature vector, as it goes to the neural net

import phik
from phik import report, resources

corr_matrix=df_train.phik_matrix()
corr_matrix

# In[ ]:


print(corr_matrix["attack_type"].sort_values(ascending=False)[1:])


# In[ ]:


corr = corr_matrix["attack_type"].sort_values(ascending=False)

attack_sep={'normal':"Normal",'neptune':"DOS",
            'satan':"Probe",'ipsweep':"Probe",'named':"R2L",
            'ps':"U2R",'sendmail':"R2L",'xterm':"U2R",'xlock':"R2L",
            'xsnoop':"R2L",'udpstorm':"DOS",'sqlattack':"U2R",'worm':"DOS",'portsweep':"Probe",
            'smurf':"DOS",'nmap':"Probe",'back':"DOS",'mscan':"Probe",'apache2':"DOS",'processtable':"DOS",
            'snmpguess':"R2L",'saint':"Probe",'mailbomb':"DOS",'snmpgetattack':"R2L",'httptunnel':"R2L",'teardrop':"DOS",
            'warezclient':"R2L",'pod':"DOS",'guess_passwd':"R2L",'buffer_overflow':"U2R",'warezmaster':"R2L",'land':"DOS",'imap':"R2L",
            'rootkit':"U2R",'loadmodule':"U2R",'ftp_write':"R2L",'multihop':"R2L",'phf':"R2L",'perl':"U2R",'spy':"R2L"}


# In[7]:


df_train.replace({'attack_type':attack_sep},inplace=True)


# In[8]:




# # Lets train a base model on the entire dataset and evaluate the performance.

# Convert to numpy - Classification

x_columns = df_train.columns.drop('attack_type')
x = df_train[x_columns].values
dummies = pd.get_dummies(df_train['attack_type']) # Classification
outcomes = dummies.columns
num_classes = len(outcomes)
y = dummies.values

df_train.head()

df_train.groupby('attack_type')['attack_type'].count()

from sklearn import metrics
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.models import Sequential

# Create a test/train split.  25% test
# Split into train/test
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=42)

# Create neural net
model = Sequential()
model.add(Dense(10, input_dim=x.shape[1], activation='relu'))
model.add(Dense(50, input_dim=x.shape[1], activation='relu'))
model.add(Dense(10, input_dim=x.shape[1], activation='relu'))
model.add(Dense(1, kernel_initializer='normal'))
model.add(Dense(y.shape[1],activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, 
                        patience=5, verbose=1, mode='auto',
                           restore_best_weights=True)
model.fit(x_train,y_train,validation_data=(x_test,y_test),
          callbacks=[monitor],verbose=2,epochs=1000)

import numpy as np
import pandas as pd
import seaborn as sn

pred = model.predict(x_test)
pred = np.argmax(pred,axis=1)

y_eval = np.argmax(y_test,axis=1)
score = metrics.accuracy_score(y_eval, pred)
f1score= f1_score(y_eval, pred, average='micro')
print("Validation score: {}".format(score))
print("f1 score: {}".format(f1score))
print(metrics.confusion_matrix(y_eval, pred))

print('\n')
print('=' * 50)
print("Training ","CFS")
print("Deep learning")
print('_' * 50)
pred
print('_' * 50)
roc = roc_auc_score(y_eval, model.predict_proba(x_test), multi_class='ovo', average='weighted')
score = metrics.accuracy_score(y_eval, pred)
f1score= f1_score(y_eval, pred, average='micro')
print("accuracy:   %0.3f" % score)
print()
print('_' * 50)
print("|classification report|")
print('_' * 50)
print(metrics.classification_report(y_eval, pred))
print('_' * 50)
print("confusion matrix:")
print(metrics.confusion_matrix(y_eval, pred))
cm= metrics.confusion_matrix(y_eval, pred)
print()
print('_' * 50)
print("ROC AUC Score :",roc)
FPR,precision,recall= falseposrate(cm,y_eval,pred)
print('_' * 50)
print("False Positive Rate is :",FPR)
clf_descr = str(clf).split('(')[0]
