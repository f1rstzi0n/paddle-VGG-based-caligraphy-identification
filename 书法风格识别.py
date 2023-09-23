#!/usr/bin/env python
# coding: utf-8
'''
 1 数据集的准备与介绍
  [书法风格数据集](https://aistudio.baidu.com/aistudio/datasetdetail/86197/0)
  来源于AIstudio的书法风格公开数据集，是一个以20位中国著名书法家的风格编写的汉字作品的数据集。每个子集中有1000-7000张jpg图像（平均5251张图像）。每张图像的尺寸为64 * 64，代表一个汉字。
  训练数据集文件名称：Chinese Calligraphy Styles.zip
  包含3个子文件，分别为：训练数据集（原始图片）文件、训练数据集（标注图片）文件和Summary.csv文件，详细介绍如下：
  是一个以20位中国著名书法家的风格编写的汉字作品的数据集。每个子集中有1000-7000张jpg图像（平均5251张图像）。每张图像的尺寸为64 * 64，代表一个汉字。数据集分为训练集（80％）和测试集（20％）。书法家的姓名缩写用作标签。更多详细信息，请参见Summary.csv文件。
'''
# In[1]:


# 创建data/shufa
# !mkdir data/shufa
# 解压所挂载的书法风格数据集在同级目录下
get_ipython().system(
    "unzip -oq '/home/aistudio/data/data86197/Chinese Calligraphy Styles.zip' -d /home/aistudio/data/shufa")
get_ipython().system('cd /home/aistudio/data/shufa')
get_ipython().system('mv data/shufa/Summary.csv /home/aistudio/data')

# In[2]:


# 查看数据集的目录结构
get_ipython().system('tree data/shufa/ -d')

# In[3]:


import cv2
import matplotlib.pyplot as plt

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

import matplotlib.font_manager as fm

myfont = fm.FontProperties(fname=r'work/font/simhei.ttf')  # 设置字体

get_ipython().run_line_magic('matplotlib', 'inline')

plt.title("八大山人书法示例", FontProperties=myfont)
plt.imshow(cv2.imread("data/shufa/train/bdsr/0037.jpg"))

plt.show()

# In[4]:


from PIL import Image
import numpy as np

image = Image.open("data/shufa/train/bdsr/0037.jpg")
image = np.array(image)
print('image.shape', image.shape)

# # 2 图像数据的统计分析

# In[5]:


# 计算图像数据整体均值和方差

# glob是python自己带的一个文件操作相关模块，用它可以查找符合自己目的的文件，类似于Windows下的文件搜索，支持通配符操作
import glob


def get_mean_std(image_path_list):
    print('Total images:', len(image_path_list))
    max_val, min_val = np.zeros(3), np.ones(3) * 255
    mean, std = np.zeros(3), np.zeros(3)
    for image_path in image_path_list:
        image = cv2.imread(image_path)
        for c in range(3):
            mean[c] += image[:, :, c].mean()
            std[c] += image[:, :, c].std()
            max_val[c] = max(max_val[c], image[:, :, c].max())
            min_val[c] = min(min_val[c], image[:, :, c].min())

    mean /= len(image_path_list)
    std /= len(image_path_list)

    mean /= max_val - min_val
    std /= max_val - min_val

    return mean, std


# 单个书法家字体图片均值与方差
mean, std = get_mean_std(glob.glob('data/shufa/train/bdsr/*.jpg'))
print('mean:', mean)
print('std:', std)

mean, std = get_mean_std(glob.glob('data/shufa/test/*/*.jpg'))
print('mean:', mean)
print('std:', std)

mean, std = get_mean_std(glob.glob('data/shufa/train/*/*.jpg'))
print('mean:', mean)
print('std:', std)

# 数据集中训练集与测试集书法字体图片的均值与方差
mean, std = get_mean_std(glob.glob('data/shufa/*/*/*.jpg'))
print('mean:', mean)
print('std:', std)

# # 3 数据集类的定义

# In[6]:


# 引入需要的模块
import os
import random
import json
import paddle
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from paddle.io import Dataset

random.seed(200)


# In[7]:
'''
遍历数据集中的所有类别文件夹，统计每个类别中图片的数量，并将每个图片的路径和类别标签存储到train.txt和eval.txt中。
同时，用一个列表class_detail记录每个类别的信息，包括类别名称、类别标签、训练集数量和测试集数量。
最后，将所有类别的信息存储到train_parameters字典中。

'''

def get_data_list(target_path, train_list_path, eval_list_path):
    '''
    生成数据列表
    '''
    # 存放所有类别的信息
    class_detail = []
    # 获取所有类别保存的文件夹名称
    data_list_path = target_path + "shufa/"
    class_dirs = os.listdir(data_list_path)
    # 总的图像数量
    all_class_images = 0
    # 存放类别标签
    class_label = 0
    # 存放类别数目
    class_dim = 0
    # 存储要写进eval.txt和train.txt中的内容
    trainer_list = []
    eval_list = []
    class_flag = 0
    class_dim_count = 0
    # 读取每个类别，['baihe', 'gouqi','jinyinhua','huaihua','dangshen']
    for class_dir in class_dirs:
        if class_dir != ".DS_Store":
            class_dim_count += 1
            # 每个类别的信息

            eval_sum = 0
            trainer_sum = 0
            # 统计每个类别有多少张图片
            class_sum = 0
            # 获取类别路径
            path = data_list_path + class_dir
            Tpath = os.listdir(path)
            cls_name = ''
            for paths in Tpath:

                class_dim += 1
                class_detail_list = {}
                # 获取所有图片
                paths = path + '/' + paths
                cls_name = paths
                img_paths = os.listdir(paths)
                for img_path in img_paths:  # 遍历文件夹下的每个图片
                    name_path = paths + '/' + img_path  # 每张图片的路径
                    if 'test' in paths:
                        eval_sum += 1  # test_sum为测试数据的数目
                        eval_list.append(name_path + "\t%d" % class_label + "\n")
                    elif 'train' in paths:
                        trainer_sum += 1
                        trainer_list.append(name_path + "\t%d" % class_label + "\n")  # trainer_sum测试数据的数目
                    class_sum += 1  # 每类图片的数目
                    all_class_images += 1  # 所有类图片的数目

                if class_label <= 20:
                    if class_flag < 1:
                        # 说明的json文件的class_detail数据
                        class_detail_list['class_name'] = cls_name  # 类别名称
                        class_detail_list['class_label'] = class_label  # 类别标签
                        class_detail_list['class_eval_images'] = eval_sum  # 该类数据的测试集数目
                        class_detail_list['class_trainer_images'] = trainer_sum  # 该类数据的训练集数目
                        class_detail.append(class_detail_list)

                        # 初始化标签列表
                        train_parameters['label_dict'][str(class_label)] = cls_name
                    else:
                        if eval_sum != 0:
                            class_detail[class_label]['class_eval_images'] = eval_sum;
                        if trainer_sum != 0:
                            class_detail[class_label]['class_trainer_images'] = trainer_sum;
                    if class_label == 19:
                        class_flag = 2

                class_label += 1
                if class_label == 20:
                    class_label = 0

    # 初始化分类数
    train_parameters['class_dim'] = int(class_dim / class_dim_count)

    # #乱序  
    random.shuffle(eval_list)

    if os.path.exists(eval_list_path):  # 如果文件存在
        os.remove(eval_list_path)  # 则删除
    if os.path.exists(train_list_path):  # 如果文件存在
        os.remove(train_list_path)  # 则删除

    with open(eval_list_path, 'a') as f:
        for eval_image in eval_list:
            f.write(eval_image)

    random.shuffle(trainer_list)
    with open(train_list_path, 'a') as f2:
        for train_image in trainer_list:
            f2.write(train_image)

            # 说明的json文件信息
    readjson = {}
    readjson['all_class_name'] = data_list_path  # 文件父目录
    readjson['all_class_images'] = all_class_images
    readjson['class_detail'] = class_detail
    jsons = json.dumps(readjson, sort_keys=True, indent=4, separators=(',', ': '))

    if os.path.exists(train_parameters['readme_path']):  # 如果文件存在
        os.remove(train_parameters['readme_path'])  # 则删除
    with open(train_parameters['readme_path'], 'w') as f:
        f.write(jsons)
    print('生成数据列表完成！')


# In[8]:


train_parameters = {
    "target_path": "/home/aistudio/data/",  # 图像相关路径
    "train_list_path": "/home/aistudio/data/train.txt",  # train.txt路径
    "test_list_path": "/home/aistudio/data/test.txt",  # eval.txt路径
    "label_dict": {},  # 标签字典
    "readme_path": "/home/aistudio/data/readme.json",  # readme.json路径
    "class_dim": -1,  # 分类数
}

target_path = train_parameters['target_path']
train_list_path = train_parameters['train_list_path']
test_list_path = train_parameters['test_list_path']
get_data_list(target_path, train_list_path, test_list_path)
print(train_parameters)


# In[9]:


# 定义数据读取器
class dataset(Dataset):
    def __init__(self, data_path, mode='train'):
        """
        数据读取器
        :param data_path: 数据集所在路径
        :param mode: train or eval
        """
        super().__init__()
        self.data_path = data_path
        self.img_paths = []
        self.labels = []

        if mode == 'train':
            with open(os.path.join(self.data_path, "train.txt"), "r", encoding="utf-8") as f:
                self.info = f.readlines()
            for img_info in self.info:
                img_path, label = img_info.strip().split('\t')
                self.img_paths.append(img_path)
                self.labels.append(int(label))

        else:
            with open(os.path.join(self.data_path, "test.txt"), "r", encoding="utf-8") as f:
                self.info = f.readlines()
            for img_info in self.info:
                img_path, label = img_info.strip().split('\t')
                self.img_paths.append(img_path)
                self.labels.append(int(label))

    def __getitem__(self, index):
        """
        获取一组数据
        :param index: 文件索引号
        :return:
        """
        # 第一步打开图像文件并获取label值
        img_path = self.img_paths[index]
        img = Image.open(img_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize((224, 224), Image.BILINEAR)
        # img = rand_flip_image(img)
        img = np.array(img).astype('float32')
        img = img.transpose((2, 0, 1)) / 255
        label = self.labels[index]
        label = np.array([label], dtype="int64")
        return img, label

    def print_sample(self, index: int = 0):
        print("文件名", self.img_paths[index], "\t标签值", self.labels[index])

    def __len__(self):
        return len(self.img_paths)


# In[10]:


# 训练数据加载
train_dataset = dataset('data/', mode='train')
train_loader = paddle.io.DataLoader(train_dataset, batch_size=32, shuffle=True)
# 评估数据加载
eval_dataset = dataset('data/', mode='eval')
eval_loader = paddle.io.DataLoader(eval_dataset, batch_size=8, shuffle=False)

# In[11]:


print(train_dataset.data_path, len(train_dataset.img_paths), len(train_dataset.labels))
img, label = train_dataset.__getitem__(0)
print(img.shape, label)


# In[12]:


# 定义卷积池化网络
class ConvPool(paddle.nn.Layer):
    '''卷积+池化'''

    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 pool_size,
                 pool_stride,
                 groups,
                 conv_stride=1,
                 conv_padding=1,
                 ):
        super(ConvPool, self).__init__()

        # groups代表卷积层的数量
        for i in range(groups):
            self.add_sublayer(  # 添加子层实例
                'bb_%d' % i,
                paddle.nn.Conv2D(  # layer
                    in_channels=num_channels,  # 通道数
                    out_channels=num_filters,  # 卷积核个数
                    kernel_size=filter_size,  # 卷积核大小
                    stride=conv_stride,  # 步长
                    padding=conv_padding,  # padding
                )
            )
            self.add_sublayer(
                'relu%d' % i,
                paddle.nn.ReLU()
            )
            num_channels = num_filters

        self.add_sublayer(
            'Maxpool',
            paddle.nn.MaxPool2D(
                kernel_size=pool_size,  # 池化核大小
                stride=pool_stride  # 池化步长
            )
        )

    def forward(self, inputs):
        x = inputs
        for prefix, sub_layer in self.named_children():
            # print(prefix,sub_layer)
            x = sub_layer(x)
        return x


# In[13]:


# VGG网络
class VGGNet(paddle.nn.Layer):
    def __init__(self):
        super(VGGNet, self).__init__()
        # 5个卷积池化操作
        self.convpool01 = ConvPool(
            3, 64, 3, 2, 2, 2)  # 3:通道数，64：卷积核个数，3:卷积核大小，2:池化核大小，2:池化步长，2:连续卷积个数
        self.convpool02 = ConvPool(
            64, 128, 3, 2, 2, 2)
        self.convpool03 = ConvPool(
            128, 256, 3, 2, 2, 3)
        self.convpool04 = ConvPool(
            256, 512, 3, 2, 2, 3)
        self.convpool05 = ConvPool(
            512, 512, 3, 2, 2, 3)
        self.pool_5_shape = 512 * 7 * 7
        # 三个全连接层
        self.fc01 = paddle.nn.Linear(self.pool_5_shape, 4096)
        self.drop1 = paddle.nn.Dropout(p=0.5)
        self.fc02 = paddle.nn.Linear(4096, 4096)
        self.drop2 = paddle.nn.Dropout(p=0.5)
        self.fc03 = paddle.nn.Linear(4096, train_parameters['class_dim'])

    def forward(self, inputs, label=None):
        # print('input_shape:', inputs.shape) #[8, 3, 224, 224]
        """前向计算"""
        out = self.convpool01(inputs)
        # print('convpool01_shape:', out.shape)           #[8, 64, 112, 112]
        out = self.convpool02(out)
        # print('convpool02_shape:', out.shape)           #[8, 128, 56, 56]
        out = self.convpool03(out)
        # print('convpool03_shape:', out.shape)           #[32, 256, 28, 28]
        out = self.convpool04(out)
        # print('convpool04_shape:', out.shape)           #[32, 512, 14, 14]
        out = self.convpool05(out)
        # print('convpool05_shape:', out.shape)           #[32, 512, 7, 7]         

        out = paddle.reshape(out, shape=[-1, 512 * 7 * 7])
        out = self.fc01(out)
        out = self.drop1(out)
        out = self.fc02(out)
        out = self.drop2(out)
        out = self.fc03(out)

        if label is not None:
            acc = paddle.metric.accuracy(input=out, label=label)
            return out, acc
        else:
            return out


# In[14]:


# 折线图，用于观察训练过程中loss和acc的走势
def draw_process(title, color, iters, data, label):
    plt.title(title, fontsize=24)
    plt.xlabel("iter", fontsize=20)
    plt.ylabel(label, fontsize=20)
    plt.plot(iters, data, color=color, label=label)
    plt.legend()
    plt.grid()
    plt.show()


# In[15]:


# 参数配置，要保留之前数据集准备阶段配置的参数，所以使用update更新字典
train_parameters.update({
    "input_size": [3, 64, 64],  # 输入图片的shape
    "num_epochs": 5,  # 训练轮数
    "skip_steps": 10,  # 训练时输出日志的间隔
    "save_steps": 100,  # 训练时保存模型参数的间隔
    "learning_strategy": {  # 优化函数相关的配置
        "lr": 0.0001  # 超参数学习率
    },
    "checkpoints": "/home/aistudio/work/checkpoints"  # 保存的路径
})

# In[16]:


print(train_parameters['class_dim'])

# In[17]:


model = VGGNet()
model.train()
# 配置loss函数
cross_entropy = paddle.nn.CrossEntropyLoss()
# 配置参数优化器
optimizer = paddle.optimizer.Adam(learning_rate=train_parameters['learning_strategy']['lr'],
                                  parameters=model.parameters())

steps = 0
Iters, total_loss, total_acc = [], [], []

for epo in range(train_parameters['num_epochs']):
    for _, data in enumerate(train_loader()):
        steps += 1
        x_data = data[0]
        y_data = data[1]
        predicts, acc = model(x_data, y_data)
        loss = cross_entropy(predicts, y_data)
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        if steps % train_parameters["skip_steps"] == 0:
            Iters.append(steps)
            total_loss.append(loss.numpy()[0])
            total_acc.append(acc.numpy()[0])
            # 打印中间过程
            print('epo: {}, step: {}, loss is: {}, acc is: {}'.format(epo, steps, loss.numpy(), acc.numpy()))
        # 保存模型参数
        if steps % train_parameters["save_steps"] == 0:
            save_path = train_parameters["checkpoints"] + "/" + "save_dir_" + str(steps) + '.pdparams'
            print('save model to: ' + save_path)
            paddle.save(model.state_dict(), save_path)
paddle.save(model.state_dict(), train_parameters["checkpoints"] + "/" + "save_dir_final.pdparams")
draw_process("trainning loss", "red", Iters, total_loss, "trainning loss")
draw_process("trainning acc", "green", Iters, total_acc, "trainning acc")

# In[18]:


# 模型评估
# 加载训练过程保存的最后一个模型
model__state_dict = paddle.load('work/checkpoints/save_dir_final.pdparams')
model_eval = VGGNet()
model_eval.set_state_dict(model__state_dict)
model_eval.eval()
accs = []
# 开始评估
for _, data in enumerate(eval_loader()):
    x_data = data[0]
    y_data = data[1]
    predicts = model_eval(x_data)
    acc = paddle.metric.accuracy(predicts, y_data)
    accs.append(acc.numpy()[0])
print('模型在验证集上的准确率为：', np.mean(accs))


# In[19]:


def load_image(img_path):
    '''
    预测图片预处理
    '''
    img = Image.open(img_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize((224, 224), Image.BILINEAR)
    img = np.array(img).astype('float32')
    img = img.transpose((2, 0, 1)) / 255  # HWC to CHW 及归一化
    return img


label_dic = train_parameters['label_dict']

# # 4 数据集类的测试

# In[20]:


import time

# 加载训练过程保存的最后一个模型
model__state_dict = paddle.load('work/checkpoints/save_dir_final.pdparams')
model_predict = VGGNet()
model_predict.set_state_dict(model__state_dict)
model_predict.eval()
infer_imgs_path = os.listdir("infer")
# print(infer_imgs_path)

# 预测所有图片
for infer_img_path in infer_imgs_path:
    if '.jpg' in infer_img_path:
        infer_img = load_image("infer/" + infer_img_path)
        infer_img = infer_img[np.newaxis, :, :, :]  # reshape(-1,3,224,224)
        infer_img = paddle.to_tensor(infer_img)
        result = model_predict(infer_img)
        lab = np.argmax(result.numpy())
        print("样本: {},被预测为:{}".format(infer_img_path, label_dic[str(lab)]))
        img = Image.open("infer/" + infer_img_path)
        plt.imshow(img)
        plt.axis('off')
        plt.show()
        sys.stdout.flush()
        time.sleep(0.5)
