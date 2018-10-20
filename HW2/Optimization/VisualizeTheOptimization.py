import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math 
from sklearn import decomposition 


## 数据源
x_data = np.linspace(-10, 10, 1000)[:, np.newaxis]
y_data1 = np.sin(5*math.pi*x_data)/(5*math.pi*x_data)
y_data2 = np.cos(5*math.pi*x_data)/(3*math.pi*x_data)
y_data3 = np.cos(3*math.pi*x_data)/(6*math.pi*x_data)
y_data4 = np.sin(4*math.pi*x_data)/(1*math.pi*x_data)
y_data5 = np.cos(5*math.pi*x_data)/(2*math.pi*x_data)
y_data6 = np.cos(5*math.pi*x_data)/(4*math.pi*x_data)

#输入
X = tf.placeholder(tf.float32,[None,1])
Y = tf.placeholder(tf.float32,[None,1])

# 每一层的权重、偏移和激活函数计算后的矩阵
W1 = tf.Variable(tf.random_normal([1, 4]))
b1 = tf.Variable(tf.random_normal([4]))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)

W2 = tf.Variable(tf.random_normal([4, 3]))
b2 = tf.Variable(tf.random_normal([3]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)

W3 = tf.Variable(tf.random_normal([3, 1]))
b3 = tf.Variable(tf.random_normal([1]))
hypothesis = tf.matmul(L2, W3) + b3

#损失函数
cost = tf.reduce_mean(tf.square(hypothesis-Y))
#优化器
optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
sess = tf.Session()


# 开始训练模型
# weight_matrix = []
# costOfEpoch = []
# for step in range(24): 
#     feed_dict = {X:x_data,Y:y_data1}
#     cost_new,_ = sess.run([cost,optimizer],feed_dict=feed_dict)


#     # if(cost_new >= 0.015):
#     #     print("step:","%4d"%(step+1),"cost=",cost_new)
#     if step % 3 == 0:
#         weight = np.append(sess.run(W1[0]),np.append(sess.run(W2[0]),sess.run(W3[0])))
#         weight_matrix.append(weight)
#         print("step:","%4d"%(step+1),"cost=",cost_new,weight)
#         costOfEpoch.append(cost_new)

def trainTheWeight(sess,feed_dict):
    sess.run(tf.global_variables_initializer())
    weight_matrix = []
    costOfEpoch = []
    for step in range(24): 
        cost_new,_ = sess.run([cost,optimizer],feed_dict=feed_dict)
        if step % 3 == 0:
            weight = np.append(sess.run(W1[0]),np.append(sess.run(W2[0]),sess.run(W3[0])))
            weight_matrix.append(weight)
            print("step:","%4d"%(step+1),"cost=",cost_new)
            costOfEpoch.append(cost_new)
    return (weight_matrix,costOfEpoch)


def getPrincipalComponents(dimension,weight_matrix):
    pca = decomposition.PCA(n_components=dimension)
    return pca.fit_transform(weight_matrix)

feed_dict = {X:x_data,Y:y_data1}
feed_dict1 = {X:x_data,Y:y_data1}
feed_dict2 = {X:x_data,Y:y_data1}
feed_dict3 = {X:x_data,Y:y_data1}
feed_dict4 = {X:x_data,Y:y_data1}
feed_dict5 = {X:x_data,Y:y_data1}
weight_matrix,costOfEpoch = trainTheWeight(sess,feed_dict)
weight_matrix1,costOfEpoch1 = trainTheWeight(sess,feed_dict1)
weight_matrix2,costOfEpoch2 = trainTheWeight(sess,feed_dict2)
weight_matrix3,costOfEpoch3 = trainTheWeight(sess,feed_dict3)
weight_matrix4,costOfEpoch4 = trainTheWeight(sess,feed_dict4)
weight_matrix5,costOfEpoch5 = trainTheWeight(sess,feed_dict5)

# print('Learning Finished!')
# print(weight_matrix)
new_W = getPrincipalComponents(2,weight_matrix)
new_W1 = getPrincipalComponents(2,weight_matrix1)
new_W2 = getPrincipalComponents(2,weight_matrix2)
new_W3 = getPrincipalComponents(2,weight_matrix3)
new_W4 = getPrincipalComponents(2,weight_matrix4)
new_W5 = getPrincipalComponents(2,weight_matrix5)

com11 = np.array(new_W)[:,0]
com12 = np.array(new_W)[:,1]
com21 = np.array(new_W1)[:,0]
com22 = np.array(new_W1)[:,1]
com31 = np.array(new_W2)[:,0]
com32 = np.array(new_W2)[:,1]
com41 = np.array(new_W3)[:,0]
com42 = np.array(new_W3)[:,1]
com51 = np.array(new_W4)[:,0]
com52 = np.array(new_W4)[:,1]
com61 = np.array(new_W5)[:,0]
com62 = np.array(new_W5)[:,1]


fig, ax = plt.subplots()
ax.scatter(com11, com12,c='orange')
ax.scatter(com21, com22,c='yellow')
ax.scatter(com31, com32,c='blue')
ax.scatter(com41, com42,c='purple')
ax.scatter(com51, com52,c='green')
ax.scatter(com61, com62,c='red')

for i, txt in enumerate(costOfEpoch):
    ax.annotate(txt, (com11[i],com12[i]))

for i, txt in enumerate(costOfEpoch1):
    ax.annotate(txt, (com21[i],com22[i]))

for i, txt in enumerate(costOfEpoch2):
    ax.annotate(txt, (com31[i],com32[i]))

for i, txt in enumerate(costOfEpoch3):
    ax.annotate(txt, (com41[i],com42[i]))

for i, txt in enumerate(costOfEpoch4):
    ax.annotate(txt, (com51[i],com52[i]))

for i, txt in enumerate(costOfEpoch5):
    ax.annotate(txt, (com61[i],com62[i]))

plt.show()


