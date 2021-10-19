import torch
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self,num_classes=100,init_weights=False):
        super().__init__()
        self.features=nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),  # input[3, 224, 224]  output[48, 55, 55]
            nn.ReLU(inplace=True),                                  #inplace=true:增加计算量但是能降低内存使用
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[48, 27, 27]
            nn.Conv2d(48, 128, kernel_size=5, padding=2),           # output[128, 27, 27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 13, 13]
            nn.Conv2d(128, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),          # output[128, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 6, 6]
        )
        self.classifier=nn.Sequential(
            nn.Dropout(p=0.5),#默认就是0.5
            nn.Linear(128*6*6,2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048,2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048,num_classes)
        )
        if init_weights:
            self._initialize_weights()
    def forward(self,x):
        x=self.features(x)
        x=torch.flatten(x,start_dim=1)#start_dim=1    展平的时候从索引=1开始
        x=self.classifier(x)
        return x

    def _initialize_weights(self):
        #这是一个model的父类，返回一个迭代器，用来遍历所有的层
        for m in self.modules():
            if isinstance(m,nn.Conv2d):#如果是二维卷积层
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')#对卷积权重进行初始化
                if m.bias is not None:#如果偏执不为空    用0进行初始化
                    nn.init.constant_(m.bias,0)
            elif isinstance(m,nn.Linear):#r如果是全连接层，用正态分布初始化
                nn.init.normal_(m.weight,0,0.01)
                nn.init.constant_(m.bias,0)
