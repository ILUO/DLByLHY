# Lab 10 MNIST and NN
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777)  # reproducibility

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# Check out https://www.tensorflow.org/get_started/mnist/beginners for
# more information about the mnist dataset



# parameterrandom_shuffle
learning_rate = 0.001
training_epochs = 15
batch_size = 100
 
# input place holders
X = tf.placeholder(tf.float32, [None, 784]) #每个字符是28*28的灰度矩阵
Y = tf.placeholder(tf.float32, [None, 10]) #输出为十个类

# weights & bias for nn layers，每一层的权重、偏移和激活函数计算后的矩阵
W1 = tf.Variable(tf.random_normal([784, 256]))
b1 = tf.Variable(tf.random_normal([256]))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)

W2 = tf.Variable(tf.random_normal([256, 256]))
b2 = tf.Variable(tf.random_normal([256]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)

W3 = tf.Variable(tf.random_normal([256, 10]))
b3 = tf.Variable(tf.random_normal([10]))
hypothesis = tf.matmul(L2, W3) + b3

# define cost/loss & optimizer 代价函数（交叉熵）
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate) #优化代价函数
gradAndVar = optimizer.compute_gradients(cost)
trainStep = optimizer.apply_gradients(gradAndVar)

# initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())

batchList = []
batchTestList =[]
lossList = []
lossTestList = []


gradientNormList = []

def getGraientNorm(gv):
    grad_all = 0
    for gradVar in gv:
        grad = 0
        if gradVar[0] is not None:
            grad = np.array(gradVar[0] ** 2).sum()
        grad_all += grad
    
    return grad_all ** 0.5
countBatchTrain = 0
countBatchTest = 0
# train my model
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size) #计算每次优化使用多少训练集


    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)


        # 随机打乱标签
        # np.random.shuffle(batch_xs) 
        # if(i == 0):
        #     print(batch_ys[0])
        #     np.random.shuffle(batch_ys) 
        #     print(batch_ys[0])
        #     print("2333333")

        # np.random.shuffle(batch_ys)

        feed_dict = {X: batch_xs, Y: batch_ys}
        c, gv,tp = sess.run([cost, gradAndVar,trainStep], feed_dict=feed_dict)
        avg_cost += c / total_batch
        batchList.append(countBatchTrain)
        gradientNormList.append(getGraientNorm(gv))
        lossList.append(c)
        countBatchTrain = countBatchTrain + 1

        

        # 每个batch都记录一下在测试集上的cost
        # correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
        # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels})
        # accuracyList.append(accuracy)


    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))


    #记录在测试集上的cost
    test_batch = int(mnist.test.num_examples / batch_size)
    for i in range(test_batch):
        batch_xtest,batch_ytest = mnist.test.next_batch(batch_size)
        feed_dict = {X: batch_xtest, Y: batch_ytest}
        cost_on_test = sess.run([cost],feed_dict=feed_dict)

        batchTestList.append(countBatchTest)
        lossTestList.append(cost_on_test)
        countBatchTest = countBatchTest + 1
    


print('Learning Finished!')

# Test model and check accuracy
correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy:', sess.run(accuracy, feed_dict={
      X: mnist.test.images, Y: mnist.test.labels}))

# Get one and predict
# r = random.randint(0, mnist.test.num_examples - 1)
# print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
# print("Prediction: ", sess.run(
#     tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r:r + 1]}))

# plt.imshow(mnist.test.images[r:r + 1].
#           reshape(28, 28), cmap='Greys', interpolation='nearest')
# plt.show()





plt.figure()

#创建小图1
##将小图分成2行1列,第三个参数表示第n个图
plt.subplot(2,1,1)
#设置小图的x,y坐标
plt.xlabel('step')
plt.ylabel('loss')
plt.title('loss On Train')
plt.plot(batchList,lossList)

plt.subplot(2,1,2)
#设置小图的x,y坐标
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('loss On Test')
plt.plot(batchTestList,lossTestList)


plt.show()