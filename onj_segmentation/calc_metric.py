


import cv2 
import numpy as np

import SimpleITK as sitk


def compute_confmat(pred, gt):
    """only for binary classification"""
    tp = np.sum(pred * gt)
    tn = np.sum((1-pred) * (1-gt))
    fp = np.sum(pred * (1-gt))
    fn = np.sum((1-pred) * gt)
    return tp, tn, fp, fn


def load_dcm(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(gt_path))[0].astype(np.float32)


if __name__=='__main__':

    gt_path = r'C:\Users\XinzheGao\Desktop\pic\ScalarVolume_33\IMG0001.dcm'
    hm_path = r'C:\Users\XinzheGao\Desktop\pic\ScalarVolume_80\IMG0001.dcm'

    gt = load_dcm(gt_path)
    hm = load_dcm(hm_path)

    gt = cv2.resize(gt, dsize=None, fx=0.25, fy=0.25)
    hm = cv2.resize(hm, dsize=None, fx=0.25, fy=0.25)
    cv2.imshow('gt', gt)
    cv2.imshow('hm', hm)
    cv2.waitKey()

    p = np.random.rand(*gt.shape)
    hm[p>0.90] = 1
    p = np.random.rand(*gt.shape)
    hm[p>0.90] = 0 
    
    print(type(gt), gt.min(), gt.max(), gt.shape)
    print(type(hm), hm.min(), hm.max(), hm.shape)

    tot_tp, tot_tn, tot_fp, tot_fn = 0, 0, 0, 0
    tp, tn, fp, fn = compute_confmat(hm, gt)

    tot_tp += tp
    tot_tn += tn
    tot_fp += fp
    tot_fn += fn

    pa = (tot_tp + tot_tn) / (tot_tp + tot_fp + tot_fn + tot_tn)
    iou = tot_tp / (tot_tp + tot_fp + tot_fn)
    dice = 2*tot_tp / (2*tot_tp + tot_fp + tot_fn)
    precision = tot_tp / (tot_tp + tot_fp)
    recall = tot_tp / (tot_tp + tot_fn)
    specificity = tot_tn / (tot_tn + tot_fp)
    f1 = 2 * precision * recall / (precision + recall)
    print(f"PA: {pa:.4f} | IOU: {iou:.4f} | Dice: {dice:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | Specificity {specificity:.4f} | F1 {f1:.4f}")


    # gt = sitk.GetArrayFromImage(sitk.ReadImage(gt_path))[0]
    # print(len(gt), type(gt), gt.dtype, gt.min(), gt.max())
    # gt = gt.astype(np.uint8)
    # gt[gt!=0] = 255
    # # gt = cv2.imread(r'C:\Users\XinzheGao\Desktop\pic\Segmentation_6-gt-label.tiff')
    # # print(gt.min(), gt.max(), gt.shape)
    # gt = cv2.resize(gt, dsize=None, fx=0.25, fy=0.25)
    # cv2.imshow('gt', gt)
    # cv2.waitKey()


