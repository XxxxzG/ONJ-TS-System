import torch

from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from torchvision.models.segmentation import lraspp_mobilenet_v3_large

if __name__=='__main__':

    # 1. data
    images = torch.rand(8,3,320,320).cuda()

    # 2. model

    # model = fcn_resnet50(
    #     num_classes=2,
    # ).cuda()
    # model = deeplabv3_mobilenet_v3_large(
    #     num_classes=2,
    # ).cuda()
    model = lraspp_mobilenet_v3_large(
        num_classes=2,
    ).cuda()

    # 3. forward
    logits = model(images)
    print(logits.keys())
    print(logits['out'].shape)