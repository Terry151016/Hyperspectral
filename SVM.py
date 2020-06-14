#refer:https://github.com/KGPML/Hyperspectral/blob/master/Decoder_Spatial_CNN.ipynb
# 自行装spectral包，专门为光谱图像设计
import matplotlib.pyplot as plt  
import numpy as np
from scipy.io import loadmat
import spectral
import pandas as pd
import joblib
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
from sklearn import preprocessing 
from functools import reduce


# 获取mat格式的数据，loadmat输出的是dict，所以需要进行定位
input_image = loadmat('data/PaviaU.mat')['paviaU']
output_image = loadmat('data/PaviaU_gt.mat')['paviaU_gt']

a = input_image.shape#:(610, 340, 103)
b= output_image.shape#:(610, 340)
c= np.unique(output_image)  # array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8)
# 统计每类样本所含个数
dict_k = {}
for i in range(output_image.shape[0]):
    for j in range(output_image.shape[1]):
        #if output_image[i][j] in [m for m in range(1,17)]:
        if output_image[i][j] in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]:
            if output_image[i][j] not in dict_k:
                dict_k[output_image[i][j]]=0
            dict_k[output_image[i][j]] +=1

print (dict_k)
print (reduce(lambda x,y:x+y,dict_k.values()))

# {1: 6631, 2: 18649, 3: 2099, 4: 3064, 5: 1345, 6: 5029, 7: 1330, 8: 3682, 9: 947}
# 42776
# 展示地物
ground_truth = spectral.imshow(classes = output_image.astype(int),figsize =(9,9))

ksc_color =np.array([[255,255,255],
     [102,102,102],
     [255,0,255],
     [0,128,0],
     [128,128,128],
     [0,255,255],
     [0,255,0],
     [128,0,0],
     [255,0,0],
     [192,192,192],
     [0,0,128],
     [0,128,128],
     [128,0,128],
	 [0,0,255],
	 [128,128,0],
	 [255,255,0],
	 [0,0,0],
        ])
    
# 展示地物
#ground_truth = spectral.imshow(classes = input_image.astype(int),figsize =(9,9),colors=ksc_color)
ground_truth = spectral.imshow(classes = output_image.astype(int),figsize =(9,9),colors=ksc_color)

# 除掉 0 这个非分类的类，把所有需要分类的元素提取出来
need_label = np.zeros([output_image.shape[0],output_image.shape[1]])
for i in range(output_image.shape[0]):
    for j in range(output_image.shape[1]):
        if output_image[i][j] != 0:
        #if output_image[i][j] in [1,2,3,4,5,6,7,8,9]:
            need_label[i][j] = output_image[i][j]


new_datawithlabel_list = []
for i in range(output_image.shape[0]):
    for j in range(output_image.shape[1]):
        if need_label[i][j] != 0:
            c2l = list(input_image[i][j])
            c2l.append(need_label[i][j])
            new_datawithlabel_list.append(c2l)
new_datawithlabel_array = np.array(new_datawithlabel_list)  # new_datawithlabel_array.shape (5211,177),包含了数据维度和标签维度，数据176维度，也就是176个波段，最后177列是标签维
print(new_datawithlabel_array.shape)

data_D = preprocessing.StandardScaler().fit_transform(new_datawithlabel_array[:,:-1])
#data_D = preprocessing.MinMaxScaler().fit_transform(new_datawithlabel_array[:,:-1])
data_L = new_datawithlabel_array[:,-1]

## 将结果存档后续处理
#
new = np.column_stack((data_D,data_L))
new_ = pd.DataFrame(new)
print(new.shape)
print(new_.shape)
new_.to_csv('data/PaviaU.csv',header=False,index=False)
## 验证高光谱数据的分类结果，并在图中进行分类结果的标记
#
## 导入数据集切割训练与测试数据
#data = pd.read_csv('data/PaviaU.csv',header=None)
#data = data.as_matrix()
#data_D = data[:,:-1]
#data_L = data[:,-1]
#data_train, data_test, label_train, label_test = train_test_split(data_D,data_L,test_size=0.5)
#print(data_L)
#
## 模型训练与拟合
#clf = SVC(kernel='rbf',gamma=0.125,C=16)
#clf.fit(data_train,label_train)
#pred = clf.predict(data_test)
#accuracy = metrics.accuracy_score(label_test, pred)*100
#print (accuracy)
#
#
## 存储结果学习模型，方便之后的调用
#joblib.dump(clf, "PaviaU_MODEL.m")
## mat文件的导入
#
#
## KSC
#input_image = loadmat('data/PaviaU.mat')['paviaU']
#output_image = loadmat('data/PaviaU_gt.mat')['paviaU_gt']
#
#
#testdata = np.genfromtxt('data/PaviaU.csv',delimiter=',')
#data_test = testdata[:,:-1]
#label_test = testdata[:,-1]
#
## /Users/mrlevo/Desktop/CBD_HC_MCLU_MODEL.m
#clf = joblib.load("PaviaU_MODEL.m")
#
#predict_label = clf.predict(data_test)
#accuracy = metrics.accuracy_score(label_test, predict_label)*100
#
#print (accuracy) # 97.1022836308
#
#
## 将预测的结果匹配到图像中
#new_show = np.zeros((output_image.shape[0],output_image.shape[1]))
#k = 0
#for i in range(output_image.shape[0]):
#    for j in range(output_image.shape[1]):
#        if output_image[i][j] != 0 :
#            new_show[i][j] = predict_label[k]
#            k +=1 
#
## print new_show.shape
#
## 展示地物
#ground_truth = spectral.imshow(classes = output_image.astype(int),figsize =(9,9),colors=ksc_color)
#ground_predict = spectral.imshow(classes = new_show.astype(int), figsize =(9,9),colors=ksc_color)
