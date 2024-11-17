
import os
import os.path as osp
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from unet import UNet
from segformer import SegFormer
from torchvision.models.segmentation import fcn_resnet50
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from torchvision.models.segmentation import lraspp_mobilenet_v3_large
import cv2
import numpy as np
from tqdm import tqdm

from train import PR_SEG_DATASET


def tensor_to_array(t, label=False):
    a = t.cpu().numpy()
    if not label:
        a = (a + 1) * 0.5
    a = (a * 255).astype(np.uint8)
    return a


def compute_confmat(pred, gt):
    """only for binary classification"""
    tp = torch.sum(pred * gt)
    tn = torch.sum((1-pred) * (1-gt))
    fp = torch.sum(pred * (1-gt))
    fn = torch.sum((1-pred) * gt)
    return tp, tn, fp, fn


if __name__=="__main__":
    # 0. Configuration
    model_pretrained_path = "segformer.pt"

    # 1. Dataset
    dataset_train    = PR_SEG_DATASET(r'G:\txt+dcm', 'train', eval=True)
    dataloader_train = DataLoader(dataset_train, 1, shuffle=False)

    # 2. Model
    # model = UNet(num_class=1).cuda()
    model = SegFormer(num_classes=1).cuda()
    # model = fcn_resnet50(num_classes=1).cuda()
    # model = deeplabv3_mobilenet_v3_large(num_classes=1).cuda()
    # model = lraspp_mobilenet_v3_large(num_classes=1).cuda()
    model.load_state_dict(torch.load(model_pretrained_path))

    # 3. Visualization
    for i, (images, labels, sizes, origin_images) in enumerate(dataloader_train):
        images, labels, origin_images = images.cuda(), labels.cuda(), origin_images.cuda()

        if model.__class__.__name__ not in ['UNet', 'SegFormer']:
            images = torch.cat((images, images, images), dim=1)

        with torch.no_grad():
            if model.__class__.__name__ not in ['UNet', 'SegFormer']:
                logits = model(images)['out']
            else:
                logits = model(images)
        # prob = F.softmax(logits, dim=1)
        prob = F.sigmoid(logits)
        # print(logits.shape)

        # print(sizes)

        opacity = 0.4

        # image = images[0,0]
        image = origin_images[0,0]
        image = ((image.cpu().numpy() + 1.) * 0.5 * 255).astype(np.uint8)

        image_label = np.stack([image,image,image], axis=-1)
        label = labels[0].cpu().numpy()
        label = cv2.resize(label, (sizes[1].item(), sizes[0].item()))
        # image_label[label!=0,2] = 255
        image_label[label!=0,0] = opacity*image_label[label!=0,0] + (1-opacity)*127
        image_label[label!=0,1] = opacity*image_label[label!=0,1] + (1-opacity)*174
        image_label[label!=0,2] = opacity*image_label[label!=0,2] + (1-opacity)*127

        image_pred = np.stack([image,image,image], axis=-1)
        pred = prob[0,0].cpu().numpy()
        pred = cv2.resize(pred, (sizes[1].item(), sizes[0].item()))
        # image_pred[pred>0.5,1] = 255
        image_pred[pred>0.5,0] = opacity*image_pred[pred>0.5,0] + (1-opacity)*79
        image_pred[pred>0.5,1] = opacity*image_pred[pred>0.5,1] + (1-opacity)*101
        image_pred[pred>0.5,2] = opacity*image_pred[pred>0.5,2] + (1-opacity)*216

        image_overlap = np.stack([image,image,image], axis=-1)
        overlap = (pred>0.5).astype(np.int32)*2 + (label!=0).astype(np.int32)
        # image_overlap[pred>0.5,1] = 255
        image_overlap[overlap==1,0] = opacity*image_overlap[overlap==1,0] + (1-opacity)*127
        image_overlap[overlap==1,1] = opacity*image_overlap[overlap==1,1] + (1-opacity)*174
        image_overlap[overlap==1,2] = opacity*image_overlap[overlap==1,2] + (1-opacity)*127
        image_overlap[overlap==2,0] = opacity*image_overlap[overlap==2,0] + (1-opacity)*79
        image_overlap[overlap==2,1] = opacity*image_overlap[overlap==2,1] + (1-opacity)*101
        image_overlap[overlap==2,2] = opacity*image_overlap[overlap==2,2] + (1-opacity)*216
        image_overlap[overlap==3,0] = opacity*image_overlap[overlap==3,0] + (1-opacity)*70
        image_overlap[overlap==3,1] = opacity*image_overlap[overlap==3,1] + (1-opacity)*220
        image_overlap[overlap==3,2] = opacity*image_overlap[overlap==3,2] + (1-opacity)*230


        if not osp.exists(model.__class__.__name__): os.makedirs(model.__class__.__name__)
        # cv2.imshow("image", tensor_to_array(images[0,0]))
        # cv2.imshow("label", tensor_to_array(labels[0], True))
        # cv2.imshow("prediction", tensor_to_array(prob[0,0], True))
        cv2.imwrite(f"{model.__class__.__name__}/{i:04d}_original.jpg", image)
        cv2.imwrite(f"{model.__class__.__name__}/{i:04d}_label.jpg", image_label)
        cv2.imwrite(f"{model.__class__.__name__}/{i:04d}_pred.jpg", image_pred)
        cv2.imwrite(f"{model.__class__.__name__}/{i:04d}_overlap.jpg", image_overlap)

        # cv2.imshow("image", image)
        # cv2.imshow("label", image_label)
        # cv2.imshow("pred", image_pred)
        # cv2.waitKey()

    # 4. Metric
    # tot_tp, tot_tn, tot_fp, tot_fn = 0, 0, 0, 0
    # for i, (images, labels, _, _) in enumerate(tqdm(dataloader_train)):
    #     images, labels = images.cuda(), labels.cuda()
 
    #     if model.__class__.__name__ not in ['UNet', 'SegFormer']:
    #         images = torch.cat((images, images, images), dim=1)

    #     with torch.no_grad():
    #         print(images.shape)
    #         if model.__class__.__name__ not in ['UNet', 'SegFormer']:
    #             logits = model(images)['out']
    #         else:
    #             logits = model(images)

    #     prob = F.sigmoid(logits)
    #     # prob = F.softmax(logits, dim=1)[:,1]
    #     pred = torch.zeros_like(prob)
    #     pred[prob > 0.5] = 1.0

    #     tp, tn, fp, fn = compute_confmat(pred, labels)
    #     tot_tp += tp
    #     tot_tn += tn
    #     tot_fp += fp
    #     tot_fn += fn

    # pa = (tot_tp + tot_tn) / (tot_tp + tot_fp + tot_fn + tot_tn)
    # iou = tot_tp / (tot_tp + tot_fp + tot_fn)
    # dice = 2*tot_tp / (2*tot_tp + tot_fp + tot_fn)
    # precision = tot_tp / (tot_tp + tot_fp)
    # recall = tot_tp / (tot_tp + tot_fn)
    # specificity = tot_tn / (tot_tn + tot_fp)
    # f1 = 2 * precision * recall / (precision + recall)
    # print(f"PA: {pa:.4f} | IOU: {iou:.4f} | Dice: {dice:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | Specificity {specificity:.4f} | F1 {f1:.4f}")