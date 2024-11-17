import os
import time
import random
from tqdm import tqdm
from tempfile import TemporaryDirectory

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import timm

from unet import UNet
from segformer import SegFormer
from torchvision.models.segmentation import fcn_resnet50
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from torchvision.models.segmentation import lraspp_mobilenet_v3_large

from script.preprocessing import collect_segmentation_dataset
import SimpleITK as sitk

import numpy as np
try:
    from cv2 import cv2 
except:
    import cv2


class PR_SEG_DATASET:

    def __init__(self, root, split='train', ratio=0.8, eval=False) -> None:
        assert split in ['train', 'val']

        data = collect_segmentation_dataset(root, 'PR')
        # random.seed(42)
        # random.shuffle(data)
        num = len(data)
        data = data[:int(num*ratio)] if split=='train' else data[int(num*ratio):]
        # data = data[:10]

        self.images, self.labels, self.sizes, self.origin_images = [], [], [], []
        for i, item in enumerate(tqdm(data)):
            # image
            image = sitk.GetArrayFromImage(sitk.ReadImage(item['path']))[0]
            self.sizes.append(image.shape)
            max_value = image.max()
            image = image.astype(np.float32) / max_value
            image = image * 2. - 1.
            self.origin_images.append(image[None])
            image = cv2.resize(image, (224,224))
            self.images.append(image[None])
            # label
            label = sitk.GetArrayFromImage(sitk.ReadImage(item['label']))[0]
            label = cv2.resize(label, (224,224), interpolation=cv2.INTER_NEAREST)
            label[label!=0] = 1
            self.labels.append(label)

            # if i > 10: break

        self.eval = eval

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        if self.eval:
            return self.images[index], self.labels[index], self.sizes[index], self.origin_images[index]
        else:
            return self.images[index], self.labels[index]


def dice_loss(pred, target, smooth=1.0):  
    """  
    计算 Dice Loss  
    Args:  
        pred (Tensor): 预测值，形状为 [B, C, H, W]，其中 C=1  
        target (Tensor): 真实标签，形状为 [B, C, H, W]，其中 C=1  
        smooth (float): 平滑项，避免除以零的错误  
  
    Returns:  
        Tensor: Dice Loss 值  
    """  
    # 确保预测值和真实标签的通道数为 1  
    assert pred.shape[1] == 1  
    assert target.shape[1] == 1  
  
    # 将预测值和真实标签展平到 [B, H*W]  
    pred    = pred.squeeze(1).contiguous().view(-1)  
    target  = target.squeeze(1).contiguous().view(-1)  
  
    # 计算交集和并集  
    intersection = (pred * target).sum()  
    union = (pred + target).sum()  
  
    # 计算 Dice 系数并转化为 Dice Loss  
    dice = (2. * intersection + smooth) / (union + smooth)
    # print('dice ', dice,it)
    loss = 1. - dice  
  
    return loss


if __name__=='__main__':
    
    # 1. load dataset
    print('Load dataset ... ')
    train_data          = PR_SEG_DATASET(r'G:\txt+dcm', 'train')
    val_data            = PR_SEG_DATASET(r'G:\txt+dcm', 'val')

    train_dataloader    = DataLoader(train_data, batch_size=2, shuffle=True)
    val_dataloader      = DataLoader(val_data, batch_size=1, shuffle=False)

    # dataloaders         = {'train': train_dataloader, 'val': val_dataloader}
    # dataset_sizes       = {'train': len(train_data), 'val': len(val_data)}

    # for image, label in train_data:
    #     # print(image.shape, image.min(), image.max(), image.dtype, label)
    #     image = (image + 1.) * 0.5 * 255.
    #     image = image.astype(np.uint8)
    #     image = image.transpose((1,2,0))
    #     cv2.imshow('PR', image)

    #     # print(label.shape, label.dtype, label.min(), label.max(), '<-----')
    #     label = label.astype(np.uint8)
    #     label[label!=0] = 255
    #     cv2.imshow('Label', label)
    #     cv2.waitKey()
    #     # break

    # 2. build model
    print('Build model ... ')
    # model = UNet(num_class=1).cuda()
    # model = SegFormer(num_classes=1).cuda()
    # model = fcn_resnet50(num_classes=1).cuda()
    # model = deeplabv3_mobilenet_v3_large(num_classes=1).cuda()
    model = lraspp_mobilenet_v3_large(num_classes=1).cuda()

    # 3. loss
    criterion = nn.CrossEntropyLoss()

    # 4. optimizer & scheduler
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    # optimizer = optim.SGD(model.parameters(), lr=0.01)

    # 5. train
    n_epochs = 100 # 40
    n_interval = 10
    for i_epoch in range(n_epochs):

        for i_step, (images, labels) in enumerate(train_dataloader):
            images, labels = images.cuda(), labels.float().cuda()

            images = torch.cat((images, images, images), dim=1)

            # print('------------------------------------------->')
            # print(images.shape, images.dtype, images.min(), images.max())
            # print(labels.shape, labels.dtype, labels.min(), labels.max())
            
            # prediction
            # logits  = model(images)
            logits  = model(images)['out']
            probs   = logits.sigmoid()

            # loss
            # loss = criterion(logits, labels)
            gts = labels[:,None]
            loss_dice   = dice_loss(probs, gts)
            # loss_ce     = F.binary_cross_entropy(probs, gts)
            # loss = loss_dice + loss_ce
            loss = loss_dice

            # # update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if i_step % n_interval == n_interval - 1:
                print(f"[Epoch: {i_epoch}|{n_epochs}] Step: [{i_step}|{len(train_dataloader)}] Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), 'ckpt.pt')
