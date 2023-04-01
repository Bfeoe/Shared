import os #控制台命令

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from transformers import get_cosine_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
#训练集共60000张图
EPOCHS = 3
BATCH_SIZE = 64
#采用cpu还是gpu进行计算
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#============================1、设定训练集和测试集============================
 # 数据预处理
transform = transforms.Compose([
    #因为MNIST所有图像都已经处理成28×28的灰度图，所以省去了Resize、RandomCrop等一系列步骤
    #transforms.Grayscale(num_output_channels=3), #如果要使用预训练模型，记得转一下三通道
    transforms.ToTensor(),  #接受一个PIL图像或numpy类型，转成Tensor
    transforms.Normalize([0.5], [0.5]), #归一化，参数为[全局平均值][方差];这里因为MNIST是灰度图，如果是三通道图，就是[0.5,0.5,0.5][0.5,0.5,0.5]
])

train_dataset = datasets.MNIST('./', train=True, transform=transform, download=True)  #如果当前路径没有MNIST数据集，则下载；不会重复下载
test_dataset = datasets.MNIST('./', train=False, transform=transform, download=True)  #train=True则下载训练集，=False则下载测试集
print(train_dataset.class_to_idx)  #可以查看各个分类对应的标签

train_loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True) 
test_loader = Data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)

#============================2、确定模型============================
'''
ResNet = models.resnet50(pretrained = True)
ResNet.fc = nn.Linear(in_features=2048, out_features=10, bias=True)
'''
'''
可替换为AlexNet或其他(记得transforms.Resize([224,224]),否则训练集图片会由于像素太小而在传播过程中逐渐消失[Error:Given input size: (512x1x1). Calculated output size: (512x0x0). Output size is too small])
不要用VGG,显存不够,Resnet占用显存反而小,一方面是resnet不需要Resize,另一方面resnet网络中每个卷积层输出大小都是vgg的一半
每个模型最终准确率均为99%
'''
'''
老师说要自己设计一个网络
该网络最终准确率98%左右
目前这个参数是准确率最高的了(照抄LeNet可以使准确率上升到99
'''
class myNet(nn.Module):
    def __init__(self):
        super(myNet,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=4,kernel_size=3) #26*26
        self.pool1 = nn.MaxPool2d(2,2) #13*13
        self.conv2 = nn.Conv2d(in_channels=4,out_channels=8,kernel_size=5) #9*9
        self.pool2 = nn.MaxPool2d(3,3) #3*3
        self.fc1 = nn.Linear(in_features=8*3*3,out_features=256,bias=True) #拉伸成一维向量
        self.fc2 = nn.Linear(in_features=256,out_features=128,bias=True)
        self.fc3 = nn.Linear(in_features=128,out_features=10,bias=True) #经测试两层fc准确率并不高 ，所以再加一层
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1,8*3*3)  #拉伸成一维向量，-1表示拉伸后的列数由上一层的行列数+该层的行数决定
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 实例化模型并移动到GPU
model = myNet().to(DEVICE)
#model = ResNet.to(DEVICE)
#============================3、指定损失函数、优化器、学习率衰减============================
cross_entropy_loss = nn.CrossEntropyLoss()
learning_rate = 1e-3
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#scheduler = optim.lr_scheduler.ExponentialLR(optimizer,gamma = 0.01)
scheduler = get_cosine_schedule_with_warmup(optimizer,num_warmup_steps=0,num_training_steps=EPOCHS*len(test_loader)) #在采用自定义网络的情况下提高了1%的准确率，果然没让我失望,请认准唯一指定学习率衰减函数(bushi)
#============================4、定义训练函数============================
def train(model, device, train_loader, optimizer, epoch):
    model.train() 
    for batch_idx, (data, label) in enumerate(train_loader): 
        data, label = data.to(device), label.to(device)
        #————————以下例行公事————————
        output = model(data)
        loss = cross_entropy_loss(output, label)
        optimizer.zero_grad() #必须写在loss.backward()前面
        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % 20 == 0: 
            writer.add_scalar(tag="loss", scalar_value=loss, global_step=batch_idx)
        if (batch_idx + 1) % 100 == 0:
            print('正在训练的Epoch: {} [{}/{} ({:.0f}%)]\t损失: {:.4f}'.format(epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),100. * (batch_idx + 1) / len(train_loader), loss.item()))

#============================5、定义测试函数============================ 
def test(model, device, test_loader): 
    model.eval() #切换至评估模式（训练模式：model.train()）
    test_loss = 0
    correct = 0
    with torch.no_grad(): #配合model.eval()使用；no_grad()的作用：①使新增的tensor可以没有梯度；②使原先带梯度的tensor可以进行原地运算
        for data, label in test_loader:
            data, label = data.to(device), label.to(device)
            output = model(data)  
            test_loss += cross_entropy_loss(output, label).item()  #nn.CrossEntropyLoss()的时候会自对输入进行一次softmax转换，所以不要事先再写一遍sigmoid或softmax，不然相当于用了两次softmax
            pred = output.data.max(1)[1].to(device) 
            correct += pred.eq(label.data).sum()   #【两个向量逐元素比较】torch.eq(tensor1, tensor2, out=None) || tensor1.eq(tensor2,out=None)：tensor1对应的元素等于tensor2的元素会返回True，否则返回False。参数out表示为一个数或者是与第一个参数相同形状和类型的tensor。
        print('\n测试集:损失: {:.4f}\t准确率: {}/{} ({:.2f}%)\n'.format(test_loss, correct, len(test_loader.dataset),100. * correct / len(test_loader.dataset)))

#————————————————————————————————————————————————————————————————开始———————————————————————————————————————————————————————————————
#初始化tensorboard
writer = SummaryWriter()
#os.system('tensorboard --logdir_spec ep1:D:\Python\Thur_1_2\logs\log1,ep2:D:\Python\Thur_1_2\logs\log2,ep3:D:\Python\Thur_1_2\logs\log3') #指定日志目录
for epoch in range(1, EPOCHS+1):
    writer = SummaryWriter(log_dir='./logs/log'+str(epoch))
    train(model, DEVICE, train_loader, optimizer, epoch) 
    test(model, DEVICE, test_loader)
    scheduler.step()
writer.close()

