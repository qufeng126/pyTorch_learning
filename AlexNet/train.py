import os
import json
import time

import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
# from tqdm import tqdm
from model import AlexNet
import torch.utils.data

device=torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
print(device)
#数据预处理，对数据进行归一化处理
data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

# os.getcwd()获取当前文件所在的目录
data_root=os.path.abspath(os.path.join(os.getcwd(),".."))
image_path=data_root+"/datasets/flower_data/"
train_dataset=datasets.ImageFolder(root=image_path+"/train",transform=data_transform["train"])
train_num=len(train_dataset)#训练集的个数

#将训练标签生成字典形式，并转为jeson格式
# {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
flower_list=train_dataset.class_to_idx
cla_dict=dict((val,key) for key,val in flower_list.items())#将字典的key和value反过来

# write dict into json file
json_str = json.dumps(cla_dict, indent=4)
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)


batch_size=32
train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True,
                                         num_workers=0)#num_workers=0加载数据时候的线程个数

# nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
# print('Using {} dataloader workers every process'.format(nw))

#测试集
validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                        transform=data_transform["val"])
val_num = len(validate_dataset)
validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                              batch_size=4, shuffle=False,
                                              num_workers=0)

print("using {} images for training, {} images for validation.".format(train_num,
                                                                       val_num))
# #数据集展示
# test_data_iter = iter(validate_loader)
# test_image, test_label = test_data_iter.next()
#
# def imshow(img):
#     img = img / 2 + 0.5  # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()
#
# print(' '.join('%5s' % cla_dict[test_label[j].item()] for j in range(4)))
# imshow(utils.make_grid(test_image))

net=AlexNet(num_classes=5,init_weights=True)
net.to(device)

loss_function=nn.CrossEntropyLoss()
loss_function.to(device)
optimzer=optim.Adam(net.parameters(),lr=0.0002)


save_path="AlexNet.pth"
best_acc=0.0#最佳准确率
train_steps = len(train_loader)
for epoch in range(10):
    #train
    net.train()#加上这个之后，droup只会在训练的时候生效
    running_loss=0.0
    t1=time.perf_counter()
    for step,data in enumerate(train_loader,start=0):
        images,labels=data
        optimzer.zero_grad()
        outputs=net(images.to(device))
        loss=loss_function(outputs,labels.to(device))
        loss.backward()
        optimzer.step()

        # 统计输出
        running_loss+=loss.item()

        # train process
        rate=(step+1)/len(train_loader)
        a="*"*int(rate*50)
        b="."*int((1-rate)*50)
        print("\rtrain loss :{:^3.0f}%[{}->{}]{:.3f}".format(int(rate*100),a,b,loss),end="")
    print()
    print(time.perf_counter()-t1)

    #validate
    net.eval()
    acc=0.0
    with torch.no_grad():
        for data_test in validate_loader:
            test_images,test_labels=data_test
            outputs=net(test_images.to(device))
            predict_y=torch.max(outputs,dim=1)[1]#获取最大值的坐标
            acc+=(predict_y==test_labels.to(device)).sum().item()

    accurate_tes=acc/val_num
    if accurate_tes>best_acc:
        best_acc=accurate_tes
        torch.save(net.state_dict(),save_path)
    print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
        (epoch + 1, running_loss / train_steps, accurate_tes))



