'''
对现有的模型进行修改

加载现有的训练好的模型
加载没有训练好的模型
'''
import torchvision

# train_data = torchvision.datasets.ImageNet("../data_image_net", split='train', download=True,
#                                            transform=torchvision.transforms.ToTensor())
from torch import nn

#加载没有训练好的模型
vgg16_false = torchvision.models.vgg16(pretrained=False)
# 加载现有的训练好的模型
vgg16_true = torchvision.models.vgg16(pretrained=True)

print(vgg16_true)

train_data = torchvision.datasets.CIFAR10('../datasets/CIFAR10', train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)

#在训练好的模型最后加上全连接层    用来输出结果
vgg16_true.classifier.add_module('add_linear', nn.Linear(1000, 10))
print(vgg16_true)



print(vgg16_false)
vgg16_false.classifier[6] = nn.Linear(4096, 10)
print(vgg16_false)


