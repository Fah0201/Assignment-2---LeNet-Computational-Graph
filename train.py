import numpy as np
import time
from net import LeNet5
from tools import normalization
import matplotlib.pyplot as plt
import os
from PIL import Image
from tqdm import tqdm
import cv2

def read_path(directory):
    paths = []
    labels = []
    with open(directory, 'r') as f:
        for line in f:
            paths.append(line.split(' ')[0])
            labels.append(line.split(' ')[1])
    labels = np.array(labels, dtype=np.uint8)
    return paths, labels

class InputLayer:
    def __init__(self, input_shape):
        self.input_shape = input_shape
    
    def forward(self, X):
        self.X = np.zeros((len(X), self.input_shape[0], self.input_shape[1]))
        for i, path in enumerate(X):
            with Image.open(path) as img:
                img = img.resize(self.input_shape[:2])
                img_arr = np.asarray(img)
                if len(img_arr.shape) == 3:
                    img_arr = img_arr[:,:,0] + img_arr[:,:,1] + img_arr[:,:,2]
                    #img_arr = img_arr[:, :, :self.input_shape[-1]]
                    self.X[i] = img_arr
        return self.X

def output_tensor(input_layer, output_file, img_paths):

    # check if output file exists
    if os.path.exists(output_file):
        # load output file and print its shape
        output = np.load(output_file)
        print('Output loaded from', output_file)
        print('Output shape:', output.shape)
        #output = output.reshape(output.shape[0]*3, 28*28)
        output = (output.astype(np.uint8)).reshape(output.shape[0], 28*28)
        return output
    else:
        # pass input image paths to the input layer
        output = input_layer.forward(img_paths)

        # save output to file
        np.save(output_file, output)
        print('Output saved to', output_file)
        print('Output shape:', output.shape)
        output = (output.astype(np.uint8)).reshape(output.shape[0], 28*28)
        return output

trainPaths, train_labels = read_path('train.txt')
testPaths, test_labels = read_path('test.txt')
valPaths, val_labels = read_path('val.txt')

train_images = output_tensor(InputLayer(input_shape=(28, 28, 3)), "trainOutput.npy", trainPaths)
test_images = output_tensor(InputLayer(input_shape=(28, 28, 3)), "testOutput.npy", testPaths)

batch_size = 100  # 训练时的batch size
test_batch = 50  # 测试时的batch size
epoch = 20
learning_rate = 1e-3

ax = []  # 保存训练过程中x轴的数据（训练次数）用于画图
ay_loss = []  # 保存训练过程中y轴的数据（loss）用于画图
ay_acc = []
testx = [] # 保存测试过程中x轴的数据（训练次数）用于画图
testy_acc = []  # 保存测试过程中y轴的数据（loss）用于画图
plt.ion()   # 打开交互模式
iterations_num = 0 # 记录训练的迭代次数
plt.rcParams['font.sans-serif']=['SimHei']   #防止中文标签乱码，还有通过导入字体文件的方法
plt.rcParams['axes.unicode_minus'] = False

net = LeNet5.LeNet5()

for E in range(epoch):
    batch_loss = 0
    batch_acc = 0

    epoch_loss = 0
    epoch_acc = 0

    for i in range(train_images.shape[0] // batch_size):
        img = train_images[i*batch_size:(i+1)*batch_size].reshape(batch_size, 1, 28, 28)
        img = normalization.normalization(img)
        label = train_labels[i*batch_size:(i+1)*batch_size]
        loss, prediction = net.forward(img, label, is_train=True)   # 训练阶段

        epoch_loss += loss
        batch_loss += loss
        for j in range(prediction.shape[0]):
            if np.argmax(prediction[j]) == label[j]:
                epoch_acc += 1
                batch_acc += 1

        net.backward(learning_rate)

        if (i+1)%50 == 0:
            print(time.strftime("%Y-%m-%d %H:%M:%S") +
                  "   epoch:%5d , batch:%5d , avg_batch_acc:%.4f , avg_batch_loss:%.4f , lr:%f "
                  % (E+1, i+1, batch_acc/(batch_size*50), batch_loss/(batch_size*50), learning_rate))
            # 绘制loss和acc变化曲线
            plt.figure(1)
            iterations_num += 1
            plt.clf()
            ax.append(iterations_num)
            ay_loss.append(batch_loss/(batch_size*50))
            ay_acc.append(batch_acc/(batch_size*50))
            plt.subplot(1, 2, 1)
            plt.title('Train_Loss')
            plt.xlabel('Iteration', fontsize=10)
            plt.ylabel('loss', fontsize=10)
            plt.plot(ax, ay_loss, 'g-')

            plt.subplot(1, 2, 2)
            plt.title('Train_Acc')
            plt.xlabel('Iteration', fontsize=10)
            plt.ylabel('Acc', fontsize=10)
            plt.plot(ax, ay_acc, 'g-')
            plt.pause(0.4)

            batch_loss = 0
            batch_acc = 0



    print(time.strftime("%Y-%m-%d %H:%M:%S") +
          "    **********epoch:%5d , avg_epoch_acc:%.4f , avg_epoch_loss:%.4f *************"
          % (E+1, epoch_acc/train_images.shape[0], epoch_loss/train_images.shape[0]))
    # 在test set上进行测试
    test_acc = 0
    for k in range(test_images.shape[0] // test_batch):
        img = test_images[k*test_batch:(k+1)*test_batch].reshape(test_batch, 1 ,28, 28)
        img = normalization.normalization(img)
        label = test_labels[k*test_batch:(k+1)*test_batch]
        _, prediction = net.forward(img, label, is_train=False)   # 测试阶段

        for j in range(prediction.shape[0]):
            if np.argmax(prediction[j]) == label[j]:
                test_acc += 1

    print("------------test_set_acc:%.4f---------------" % (test_acc / test_images.shape[0]))
    plt.figure(2)
    plt.clf()
    testx.append(E)
    testy_acc.append(test_acc / test_images.shape[0])
    plt.subplot()
    plt.title('Test_Acc')
    plt.xlabel('Iteration', fontsize=10)
    plt.ylabel('Acc', fontsize=10)
    plt.plot(testx, testy_acc, 'g-')
    plt.pause(0.4)  # 设置暂停时间，太快图表无法正常显示


plt.ioff()       # 关闭画图的窗口，即关闭交互模式
plt.show()       # 显示图片，防止闪退
