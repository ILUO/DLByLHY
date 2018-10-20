import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


## 数据源
x_data = np.linspace(-1, 1, 1000)[:, np.newaxis]
y_data = x_data*x_data + 5 * x_data + np.random.randn(*x_data.shape) * 0.3 

#输入
X = tf.placeholder(tf.float32,[None,1])
Y = tf.placeholder(tf.float32,[None,1])

# 每一层的权重、偏移和激活函数计算后的矩阵
W1 = tf.Variable(tf.random_normal([1, 9]))
b1 = tf.Variable(tf.random_normal([9]))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)

W2 = tf.Variable(tf.random_normal([9, 1]))
b2 = tf.Variable(tf.random_normal([1]))
hypothesis = tf.nn.relu(tf.matmul(L1, W2) + b2)

# W3 = tf.Variable(tf.random_normal([4, 1]))
# b3 = tf.Variable(tf.random_normal([1]))
# hypothesis = tf.matmul(L2, W3) + b3

#损失函数
cost = tf.reduce_mean(tf.square(hypothesis-Y))
#优化器
optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
x = []
y = []

# 开始训练模型
for step in range(1001):
    feed_dict = {X:x_data,Y:y_data}
    cost_new,_ = sess.run([cost,optimizer],feed_dict=feed_dict)
    x.append(step)
    y.append(cost_new)
    # print("step:","%4d"%(step+1),"cost=",cost_new)
    if step % 100 == 0:
        print("step:","%4d"%(step+1),"cost=",cost_new)
print('Learning Finished!')

plt.xlabel('step')
plt.ylabel('loss')
plt.title('1*9*1 network')
plt.plot(x,y)
plt.show()

