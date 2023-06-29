'''
author: cbb
email: 2833844911@qq.com
date: 2023-04-18
'''

import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

import torch
import cv2
import random
import os
import torch.optim as optim
import torch.nn as nn

import torchvision.models as models
from tqdm import tqdm
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# 将模型移动到GPU上（如果有可用的话）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


num_classes = 360


 # 导入图片大小
daorimg = (480, 480)

class CustomResNet50(nn.Module):
    def __init__(self, num_classes=num_classes):
        super(CustomResNet50, self).__init__()
        # 加载ResNet50模型
        resnet50 = models.resnet50(pretrained=True)
        self.resnet_layers = nn.Sequential(*list(resnet50.children())[:-1])
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(2048, num_classes)
        # self.fc = nn.Linear(8192, num_classes)
        self.sfm = nn.Sigmoid()

    def forward(self, x):
        # print(x.shape)
        x = self.resnet_layers(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.sfm(x)
        return x

resnet = CustomResNet50(num_classes=num_classes)
# resnet = torch.load('./modelyzm.pth')
resnet.to(device)

def get_jpg_paths(folder_path):
    jpg_paths = []
    for file in os.listdir(folder_path):
        jpg_paths.append(folder_path+'/'+file)

    return jpg_paths
def crop_max_square(img):
    size = min(img.shape[:2])
    x = (img.shape[1] - size) // 2
    y = (img.shape[0] - size) // 2
    return img[y:y+size, x:x+size]
folder_path = './imgs'
allfile = get_jpg_paths(folder_path)
random.shuffle(allfile)
huaf = int(len(allfile)*0.9)
tranfiles = allfile[:huaf]
textfiles = allfile[huaf:]
def add_gaussian_noise(image, mean=0, sigma=25):
    """
    添加高斯噪声到图像
    :param image: 输入图像
    :param mean: 噪声均值
    :param sigma: 噪声方差
    :return: 带有高斯噪声的图像
    """
    # 将图像转换为浮点数，以防止整数溢出
    # noisy_image = np.array(image, dtype=np.float64)

    # 生成与图像尺寸相同的高斯噪声
    # noise = np.random.normal(mean, sigma, image.shape)
    #
    # # 将噪声添加到图像
    # noisy_image = image + noise

    # 确保结果在0到255范围内
    # noisy_image = np.clip(noisy_image, 0, 255)

    # 将图像类型转换回原始类型
    # noisy_image = noisy_image.astype(np.uint8)

    return image
class data_Dataset(Dataset):
    def __init__(self, data) -> None:
        super().__init__()
        self.data = data

    def __getitem__(self, index):
        tmfile = self.data[index]
        data = cv2.imread(tmfile)



        data =cv2.resize(data, daorimg, interpolation=cv2.INTER_AREA)
        
        # 计算旋转矩阵
        center = (data.shape[1] // 2, data.shape[0] // 2)
        angle = random.randint(0, num_classes-1)
        scale = 1
        M = cv2.getRotationMatrix2D(center, angle, scale)

        # 应用旋转矩阵
        croppedcj = cv2.warpAffine(data, M, (data.shape[1], data.shape[0]),borderValue=(255,255,255))
        


        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
        img_tensor = transform(croppedcj)
        kk = img_tensor.shape[0]
        if kk != 3:

            img_tensor = torch.permute(img_tensor, (2, 0, 1))
        # print(img_tensor.shape)
        return img_tensor.to(device), torch.tensor( angle).to(device)

    def __len__(self):
        return len(self.data) -1


train_dataset = data_Dataset(tranfiles)

test_dataset = data_Dataset(textfiles)
batch_size = 4
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class CustomClassificationLoss(nn.Module):
    def __init__(self, num_classes, alpha=18):
        super(CustomClassificationLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha


    def forward(self, logits, targets):

        ce_loss = 0
        for i in range(logits.shape[0]):
            wz = targets[i].item()
            kj = wz
            lossOne = (1-logits[i][wz])
            for r in range(1, 180):
                wz -= 1
                if wz <0:
                    wz = 359
                lossOne += torch.abs((0.98 ** r) - logits[i][wz])

            for r in range(1, 180):
                kj += 1
                if kj >359:
                    kj = 0
                lossOne += torch.abs((0.98 ** r) - logits[i][kj])
            ce_loss += lossOne


        # 组合损失
        loss = ce_loss/targets.shape[0]
        return loss, 0, loss.item()

# model = LeNet5()
criterion = CustomClassificationLoss(360)
optimizer = optim.Adam(resnet.parameters(), lr=0.0001)

max_xl = 1000

# 训练轮数
epochs = 100
for epoch in range(epochs):
    resnet.train()
    running_loss = 0
    xiao_loss = 0
    mmmm_loss = 0
    max_xlk = 0
    train_loader = tqdm(train_loader)
    for i, (inputs, labels) in enumerate(train_loader, 0):
        bashz = inputs.shape[0]
        if bashz != batch_size:
            continue
        optimizer.zero_grad()
        outputs = resnet(inputs)
        loss, xiao, mmmm = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        xiao_loss += xiao
        mmmm_loss += mmmm
        max_xlk =  running_loss/((i+1))
        train_loader.set_description(desc='Epoch %d loss: %.3f xiao_loss %.3f mmmm_loss: %.3f' % (epoch+1, running_loss/((i+1)), xiao_loss/((i+1)) , mmmm_loss/((i+1))))
    if max_xl > max_xlk:
        torch.save(resnet, './modelyzm.pth')
        max_xl = max_xlk



    lossall = 0
    alllen = 0
    
    with torch.no_grad():
        test_loader = tqdm(test_loader)
        resnet.eval()
        for data in test_loader:
            images, labels = data
            bashz = images.shape[0]
            if bashz != batch_size:
                continue
            alllen+=1
            outputs = resnet(images)
            loss, xiao, mmmm = criterion(outputs, labels)
            lossall += loss.item()

            test_loader.set_description(desc='test loss: {}'.format(lossall / alllen))

