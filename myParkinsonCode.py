"""
Parkinson disease classification

Dataset by: Erdogdu Sakar, B., Isenkul, M., Sakar, C.O., Sertbas, A., Gurgen, F., Delil, S., Apaydin, H., Kursun, O.,
'Collection and Analysis of a Parkinson Speech Dataset with Multiple Types of Sound Recordings',
IEEE Journal of Biomedical and Health Informatics, vol. 17(4), pp. 828-834, 2013.

Author: Jagath

"""
from sklearn.model_selection import LeaveOneOut
import pandas as pd
import numpy as np
from sklearn.svm import SVC,NuSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import scipy
import pickle as pkl
from sklearn.metrics import confusion_matrix

#change name to wherever dataset is in your system.
data_location = r'C:\Desktop\docs\7th sem\ml\MyParkinson\\'

#to seperate the data of each subject
def split_data(data,n,split_size,column = None):
    if column != None:
        d = np.array(data[column])
    else:
        d = np.array(data)
    data_split = []
    group = []
    index = 0
    while(index < n):
        group.append(d[index])
        if index % split_size == split_size - 1:
            data_split.append(group)
            group = []
        index += 1
    return data_split
    
def summarize_data(data, rowc, columns, split_size):
    calculate = ['mean','median','trim10','trim25','std','iqr','mad']#mad - mean absolute deviation, iqr - interquartile range
    df = pd.DataFrame()
    for i in columns:
        data_split = split_data(data,rowc,split_size,i)
        df['mean_'+i] = np.mean(data_split,axis = 1)
        df['median_'+i] = np.median(data_split,axis = 1)
        df['trim10_'+i] = scipy.stats.trim_mean(data_split, 0.1, axis = 1)
        df['trim25_'+i] = scipy.stats.trim_mean(data_split, 0.25, axis = 1)
        df['std_'+i] = np.std(data_split,axis = 1)
        df['iqr_'+i] = scipy.stats.iqr(data_split, axis = 1)
        df['mad_'+i] = scipy.stats.median_abs_deviation(data_split,axis = 1)
    return df
    
#read train data
df = pd.read_csv(data_location+"train_data.txt")
column_list = list(df.iloc[:,1:-2].columns)
X_train,Y_train = np.array(df.iloc[:,1:-2]),np.array(df.iloc[:,-1])
train_data_len = len(X_train) 
print('feature list:')
print(column_list)
print('\n')
print('sample train data(one subject)(one subject and 4 features only:')
print(df.iloc[:26,:5])
print()

#read test data
df1 = pd.read_csv(data_location+"test_data.txt")
X_test,Y_test = np.array(df1.iloc[:,1:-1]),np.array(df1.iloc[:,-1])
test_data_len = len(X_test)
print('sample test data:(one subject and 4 features only)')
print(df1.iloc[:6,:5])
print()

#create and train clasifier
clf = make_pipeline(StandardScaler(), NuSVC(gamma='auto'))#standard scaler makes mean = 0 and variance = 1
#clf = NuSVC(gamma='auto') #causes overfitting, if done without scaling
clf.fit(X_train,Y_train)

#predict
test_pred_Y = clf.predict(X_test)

#accuracies of individual audio 
print("individually test accuracy without summarizing(individual samples):"+str(clf.score(X_test,Y_test)))
print("individually train accuracy without summarizing(individual samples):"+str(clf.score(X_train,Y_train)))


#accuracy subject-wise
grouped_test_pred_Y = split_data(test_pred_Y,test_data_len,6,None)
test_pred_Y_ForEachSubject = []
for i in grouped_test_pred_Y:
    count = 0
    for j in i:
        if j == 1:
            count += 1
    if count >= 3:
        test_pred_Y_ForEachSubject.append(1)
    else:
        test_pred_Y_ForEachSubject.append(0)


print('\n')
print('accuracy using non linear svm without summarizing is:(considering all classified audio clips of a subject, by maximum voting)')
count = 0
for i in test_pred_Y_ForEachSubject:
    if i == 1:
        count += 1
print(count/len(test_pred_Y_ForEachSubject))
print('\n')

#train with summarizing
df_summ =  summarize_data(df, train_data_len, column_list, 26)
print('data after summarizing:(only 3 features shown)')
print(df_summ.iloc[:,:3])

#add class label
label = [1]*20
label.extend([0]*20)
df_summ['class'] = label
df_summ.to_csv(data_location+'summ_train_data.csv',index=False)
X_train,Y_train = np.array(df_summ.iloc[:,:-1]),np.array(df_summ.iloc[:,-1])


clf1 = make_pipeline(StandardScaler(), NuSVC(gamma='auto'))
clf1.fit(X_train,Y_train)
print('\n\n')
print('train set accuracy after summarizing:(By using maximum voting) '+str(clf1.score(X_train,Y_train)))

#read and summarize test data:
df_summ_test = summarize_data(df1,test_data_len,column_list,6)
df_summ_test['class'] = [1]*28
df_summ_test.to_csv(data_location+'summ_test_data.csv',index=False)
X_test,Y_test = np.array(df_summ_test.iloc[:,:-1]),np.array(df_summ_test.iloc[:,-1])
print('test set accuracy after summarizing: '+str(clf1.score(X_test,Y_test)))
print('actual classes of test data: ')
print(Y_test)
print('predicted classes of test data: ')
print(clf1.predict(X_test))

#save model in pickle file
pkl.dump(clf1, open(data_location+'summ_model.sav', 'wb'))
pkl.dump(clf, open(data_location+'no_summ_model.sav', 'wb'))