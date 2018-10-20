import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math 
from sklearn import decomposition 


## 数据源
x_data = np.linspace(-10, 10, 1000)[:, np.newaxis]
y_data1 = np.sin(5*math.pi*x_data)/(5*math.pi*x_data)


#输入
X = tf.placeholder(tf.float32,[None,1])
Y = tf.placeholder(tf.float32,[None,1])

# 每一层的权重、偏移和激活函数计算后的矩阵
W1 = tf.Variable(tf.random_normal([1, 2]))
b1 = tf.Variable(tf.random_normal([2]))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)

W2 = tf.Variable(tf.random_normal([2, 2]))
b2 = tf.Variable(tf.random_normal([2]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)

W3 = tf.Variable(tf.random_normal([2, 1]))
b3 = tf.Variable(tf.random_normal([1]))
hypothesis = tf.matmul(L2, W3) + b3

#损失函数
cost = tf.reduce_mean(tf.square(hypothesis-Y))
#优化器
optimizer = tf.train.GradientDescentOptimizer(0.01)
gradAndVar = optimizer.compute_gradients(cost)
trainStep = optimizer.apply_gradients(gradAndVar)
sess = tf.Session()

gradientNormList = []
lossList = []
stepList = []

def getGraientNorm(gv):
    grad_all = 0
    for gradVar in gv:
        grad = 0
        if gradVar[0] is not None:
            grad = np.array(gradVar[0] ** 2).sum()
        grad_all += grad
    
    return grad_all ** 0.5



def trainTheWeight(sess,feed_dict):
    sess.run(tf.global_variables_initializer())
    for step in range(701): 
        cost_new,gv,tp = sess.run([cost,gradAndVar,trainStep],feed_dict=feed_dict)
        gradientNormList.append(getGraientNorm(gv))
        lossList.append(cost_new)
        stepList.append(step)
        if step % 100 == 0:
            print("step:","%4d"%(step+1),"cost=",cost_new)      
    return (gradientNormList,lossList,step)


feed_dict = {X:x_data,Y:y_data1}
gradientNormList,lossList,step = trainTheWeight(sess,feed_dict)

print('Learning Finished!')

plt.xlabel('step')
plt.ylabel('loss')
plt.title('1*4*3*1 network')
plt.plot(stepList,lossList)
plt.show()

plt.xlabel('step')
plt.ylabel('gradientNorm')
plt.title('1*4*3*1 network')
plt.plot(stepList,gradientNormList)
plt.show()







