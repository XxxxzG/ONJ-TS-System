import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics


def calc_metric(root, name, threshold=0.5, verbose=True):
    data = np.load(osp.join(root, 'testset_results.npz'))
    gt, prob = data['gt'], data['prob']

    # ROC & AUC
    fpr, tpr, thresholds = metrics.roc_curve(gt, prob)
    auc = metrics.roc_auc_score(gt, prob)

    gt, prob = np.array(gt), np.array(prob) 

    TP = np.sum(gt * (prob > threshold))
    TN = np.sum((1-gt) * (prob <= threshold))
    FP = np.sum((1-gt) * (prob > threshold))
    FN = np.sum(gt * (prob <= threshold))

    # accuracy
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    specificity = TN / (TN + FP)
    f1 = 2 * precision * recall / (precision + recall)

    if verbose:
        print(f'{name} :')
        print(f'accuracy: {accuracy:.4f}')
        print(f'precision: {precision:.4f}')
        print(f'recall: {recall:.4f}')
        print(f'specificity: {specificity:.4f}')
        print(f'f1: {f1:.4f}')
    
    return TP, TN, FP, FN, fpr, tpr, auc, f1


def find_best(fn, root, name):

    best_f1, best_t = None, None
    for t in np.linspace(0, 1, 100):
        f1 = fn(root, name, threshold=t, verbose=False)[-1]
        if best_f1 is None:
            best_f1, best_t = f1, t
        else:
            if best_f1 < f1:
                best_f1, best_t = f1, t

    TP, TN, FP, FN, fpr, tpr, auc, f1 = fn(root, name, threshold=best_t)

    return TP, TN, FP, FN, fpr, tpr, auc


def plot_confusion_matrix():
    # tp fp tn fn
    # confusion_matrix = np.array([[166, 16],[112, 7]])
    # o_tp, o_fp, o_tn, o_fn = 166, 16, 112, 7
    # o_tp, o_fp, o_tn, o_fn = 164, 36, 92, 9
    # o_tp, o_fp, o_tn, o_fn = 140, 15, 113, 33
    # o_tp, o_fp, o_tn, o_fn = 160, 88, 40, 13
    o_tp, o_fp, o_tn, o_fn = 99, 59, 69, 74
    # o_tp, o_fp, o_tn, o_fn = 87, 59, 69, 86
    tp = int(o_tp / 301 * 74)
    fp = int(o_fp / 301 * 74)
    tn = int(o_tn / 301 * 74)
    fn = 74 - tp - fp - tn
    confusion_matrix = np.array([[tp, fp],[fn, tn]])
    plt.figure()
    plt.rcParams.update({'font.size': 20})
    # plt.title('SwinTransformer')
    # plt.title('CrossViT')
    # plt.title('ViT')
    # plt.title('VGG')
    plt.title('ResNet')
    # plt.title('Inception')
    sns.heatmap(confusion_matrix, annot=True, fmt='g', cmap='Blues')
    plt.xticks([0.5, 1.5], ['Positive', 'Negative'])
    plt.xlabel('Prediction')
    plt.yticks([0.5, 1.5], ['Positive', 'Negative'])
    plt.ylabel('Ground-Truth')
    plt.tight_layout()
    plt.show()


def plot_roc():

    _, _, _, _, swin_transformer_fpr, swin_transformer_tpr, swin_transformer_auc, _ = \
        calc_metric('results/swin_transformer', 'swin_transformer')

    _, _, _, _, crossvit_fpr, crossvit_tpr, crossvit_auc, _ = \
        calc_metric('results/crossvit', 'crossvit')

    _, _, _, _, vit_fpr, vit_tpr, vit_auc, _ = \
        calc_metric('results/vit_small_patch16_224', 'vit_small_patch16_224')
    
    _, _, _, _, vgg_fpr, vgg_tpr, vgg_auc, _ = \
        calc_metric('results/vgg16_bn', 'vgg16_bn')
    
    _, _, _, _, resnet_fpr, resnet_tpr, resnet_auc, _ = \
        calc_metric('results/resnet50', 'resnet50')
    
    _, _, _, _, inception_fpr, inception_tpr, inception_auc, _ = \
        calc_metric('results/inception_v3', 'inception_v3')


    plt.figure()
    plt.rcParams.update({'font.size': 11})

    plt.plot(swin_transformer_fpr, swin_transformer_tpr, color='crimson', lw=2, label=f'SwinTransformer (AUC = {swin_transformer_auc:0.4f})')
    plt.plot(crossvit_fpr, crossvit_tpr, color='lightseagreen', lw=2, label=f'CrossViT (AUC = {crossvit_auc:0.4f})')
    plt.plot(vit_fpr, vit_tpr, color='gold', lw=2, label=f'ViT (AUC = {vit_auc:0.4f})')
    plt.plot(vgg_fpr, vgg_tpr, color='skyblue', lw=2, label=f'VGG (AUC = {vgg_auc:0.4f})')
    plt.plot(resnet_fpr, resnet_tpr, color='lightpink', lw=2, label=f'ResNet (AUC = {resnet_auc:0.4f})')
    plt.plot(inception_fpr, inception_tpr, color='plum', lw=2, label=f'Inception (AUC = {inception_auc:0.4f})')

    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')  
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.01])  
    plt.xlabel('False Positive Rate')  
    plt.ylabel('True Positive Rate')  
    plt.title('Receiver Operating Characteristic')  
    plt.legend(loc="lower right") 
    plt.tight_layout()
    plt.show()


if __name__=='__main__':

    plot_roc()
    
    # vit_tp, vit_tn, vit_fp, vit_fn, vit_fpr, vit_tpr, vit_auc = \
    #     calc_metric('results/vit_small_patch16_224_f1', 'vit_small_patch16_224_f1', threshold=0.60)

    # vgg_tp, vgg_tn, vgg_fp, vgg_fn, vgg_fpr, vgg_tpr, vgg_auc = \
    #     calc_metric('results/vit_small_patch16_224_f2', 'vit_small_patch16_224_f2', threshold=0.60)

    # resnet_tp, resnet_tn, resnet_fp, resnet_fn, resnet_fpr, resnet_tpr, resnet_auc = \
    #     calc_metric('results/vit_small_patch16_224_f3', 'vit_small_patch16_224_f3', threshold=0.60)
    
    # inception_tp, inception_tn, inception_fp, inception_fn, inception_fpr, inception_tpr, inception_auc = \
    #     calc_metric('results/vit_small_patch16_224_f4', 'vit_small_patch16_224_f4', threshold=0.60)

    # swim_transformer_tp, swim_transformer_tn, swim_transformer_fp, swim_transformer_fn, swim_transformer_fpr, swim_transformer_tpr, swim_transformer_auc = \
    #     calc_metric('results/vit_small_patch16_224_f5', 'vit_small_patch16_224_f5', threshold=0.60)


    # vit_tp, vit_tn, vit_fp, vit_fn, vit_fpr, vit_tpr, vit_auc = \
    #     calc_metric('results/crossvit_tiny_240_0', 'crossvit_tiny_240_0', threshold=0.60)

    # vgg_tp, vgg_tn, vgg_fp, vgg_fn, vgg_fpr, vgg_tpr, vgg_auc = \
    #     calc_metric('results/crossvit_tiny_240_1', 'crossvit_tiny_240_1', threshold=0.60)

    # resnet_tp, resnet_tn, resnet_fp, resnet_fn, resnet_fpr, resnet_tpr, resnet_auc = \
    #     calc_metric('results/crossvit_tiny_240_2', 'crossvit_tiny_240_2', threshold=0.60)
    
    # inception_tp, inception_tn, inception_fp, inception_fn, inception_fpr, inception_tpr, inception_auc = \
    #     calc_metric('results/crossvit_tiny_240_3', 'crossvit_tiny_240_3', threshold=0.60)

    # swim_transformer_tp, swim_transformer_tn, swim_transformer_fp, swim_transformer_fn, swim_transformer_fpr, swim_transformer_tpr, swim_transformer_auc = \
    #     calc_metric('results/crossvit_tiny_240_4', 'crossvit_tiny_240_4', threshold=0.60)


    # vit_tp, vit_tn, vit_fp, vit_fn, vit_fpr, vit_tpr, vit_auc = \
    #     calc_metric('results/inception_v3_0', 'inception_v3_0', threshold=0.60)

    # vgg_tp, vgg_tn, vgg_fp, vgg_fn, vgg_fpr, vgg_tpr, vgg_auc = \
    #     calc_metric('results/inception_v3_1', 'inception_v3_1', threshold=0.60)

    # resnet_tp, resnet_tn, resnet_fp, resnet_fn, resnet_fpr, resnet_tpr, resnet_auc = \
    #     calc_metric('results/inception_v3_2', 'inception_v3_2', threshold=0.60)
    
    # inception_tp, inception_tn, inception_fp, inception_fn, inception_fpr, inception_tpr, inception_auc = \
    #     calc_metric('results/inception_v3_3', 'inception_v3_3', threshold=0.60)

    # swim_transformer_tp, swim_transformer_tn, swim_transformer_fp, swim_transformer_fn, swim_transformer_fpr, swim_transformer_tpr, swim_transformer_auc = \
    #     calc_metric('results/inception_v3_4', 'inception_v3_4', threshold=0.60)


    # vit_tp, vit_tn, vit_fp, vit_fn, vit_fpr, vit_tpr, vit_auc = \
    #     calc_metric('results/resnet50_0', 'resnet50_0', threshold=0.60)

    # vgg_tp, vgg_tn, vgg_fp, vgg_fn, vgg_fpr, vgg_tpr, vgg_auc = \
    #     calc_metric('results/resnet50_1', 'resnet50_1', threshold=0.60)

    # resnet_tp, resnet_tn, resnet_fp, resnet_fn, resnet_fpr, resnet_tpr, resnet_auc = \
    #     calc_metric('results/resnet50_2', 'resnet50_2', threshold=0.60)
    
    # inception_tp, inception_tn, inception_fp, inception_fn, inception_fpr, inception_tpr, inception_auc = \
    #     calc_metric('results/resnet50_3', 'resnet50_3', threshold=0.60)

    # swim_transformer_tp, swim_transformer_tn, swim_transformer_fp, swim_transformer_fn, swim_transformer_fpr, swim_transformer_tpr, swim_transformer_auc = \
    #     calc_metric('results/resnet50_4', 'resnet50_4', threshold=0.60)
    
    # swim_tp, swin_tn, swin_fp, swin_fn, swin_fpr, swin_tpr, swin_auc = \
    #     find_best(calc_metric, 'results/swin_tiny_patch4_window7_224_0', 'swin_tiny_patch4_window7_224_0')
    
    # swin_tp, swin_tn, swin_fp, swin_fn, swin_fpr, swin_tpr, swin_auc = \
    #     find_best(calc_metric, 'results/swin_tiny_patch4_window7_224_1', 'swin_tiny_patch4_window7_224_1')
    
    # swin_tp, swin_tn, swin_fp, swin_fn, swin_fpr, swin_tpr, swin_auc = \
    #     find_best(calc_metric, 'results/swin_tiny_patch4_window7_224_2', 'swin_tiny_patch4_window7_224_2')

    # swin_tp, swin_tn, swin_fp, swin_fn, swin_fpr, swin_tpr, swin_auc = \
    #     find_best(calc_metric, 'results/swin_tiny_patch4_window7_224_3', 'swin_tiny_patch4_window7_224_3')
    
    # swin_tp, swin_tn, swin_fp, swin_fn, swin_fpr, swin_tpr, swin_auc = \
    #     find_best(calc_metric, 'results/swin_tiny_patch4_window7_224_4', 'swin_tiny_patch4_window7_224_4')
    

    # vgg16_bn_tp, vgg16_bn_tn, vgg16_bn_fp, vgg16_bn_fn, vgg16_bn_fpr, vgg16_bn_tpr, vgg16_bn_auc = \
    #     find_best(calc_metric, 'results/swin_tiny_patch4_window7_224_0', 'swin_tiny_patch4_window7_224_0')
    
    # vgg16_bn_tp, vgg16_bn_tn, vgg16_bn_fp, vgg16_bn_fn, vgg16_bn_fpr, vgg16_bn_tpr, vgg16_bn_auc = \
    #     find_best(calc_metric, 'results/vgg16_bn_1', 'vgg16_bn_1')
    
    # vgg16_bn_tp, vgg16_bn_tn, vgg16_bn_fp, vgg16_bn_fn, vgg16_bn_fpr, vgg16_bn_tpr, vgg16_bn_auc = \
    #     find_best(calc_metric, 'results/vgg16_bn_2', 'vgg16_bn_2')

    # vgg16_bn_tp, vgg16_bn_tn, vgg16_bn_fp, vgg16_bn_fn, vgg16_bn_fpr, vgg16_bn_tpr, vgg16_bn_auc = \
    #     find_best(calc_metric, 'results/vgg16_bn_3', 'vgg16_bn_3')
    
    # vgg16_bn_tp, vgg16_bn_tn, vgg16_bn_fp, vgg16_bn_fn, vgg16_bn_fpr, vgg16_bn_tpr, vgg16_bn_auc = \
    #     find_best(calc_metric, 'results/vgg16_bn_4', 'vgg16_bn_4')
    
    # vit_tp, vit_tn, vit_fp, vit_fn, vit_fpr, vit_tpr, vit_auc = \
    #     calc_metric('results/vgg16_bn_0', 'vgg16_bn_0', threshold=0.60)

    # vgg_tp, vgg_tn, vgg_fp, vgg_fn, vgg_fpr, vgg_tpr, vgg_auc = \
    #     calc_metric('results/vgg16_bn_1', 'vgg16_bn_1', threshold=0.60)

    # resnet_tp, resnet_tn, resnet_fp, resnet_fn, resnet_fpr, resnet_tpr, resnet_auc = \
    #     calc_metric('results/vgg16_bn_2', 'vgg16_bn_2', threshold=0.60)
    
    # inception_tp, inception_tn, inception_fp, inception_fn, inception_fpr, inception_tpr, inception_auc = \
    #     calc_metric('results/vgg16_bn_3', 'vgg16_bn_3', threshold=0.60)

    # swim_transformer_tp, swim_transformer_tn, swim_transformer_fp, swim_transformer_fn, swim_transformer_fpr, swim_transformer_tpr, swim_transformer_auc = \
    #     calc_metric('results/vgg16_bn_4', 'vgg16_bn_4', threshold=0.60)


    # vit_tp, vit_tn, vit_fp, vit_fn, vit_fpr, vit_tpr, vit_auc = \
    #     calc_metric('results/vit_small_patch16_224', 'vit', threshold=0.60)

    # vgg_tp, vgg_tn, vgg_fp, vgg_fn, vgg_fpr, vgg_tpr, vgg_auc = \
    #     calc_metric('results/vgg16_bn', 'vgg', threshold=0.50)

    # resnet_tp, resnet_tn, resnet_fp, resnet_fn, resnet_fpr, resnet_tpr, resnet_auc = \
    #     calc_metric('results/resnet50', 'resnet', threshold=0.50)
    
    # inception_tp, inception_tn, inception_fp, inception_fn, inception_fpr, inception_tpr, inception_auc = \
    #     calc_metric('results/inception_v3', 'inception', threshold=0.8)

    # swim_transformer_tp, swim_transformer_tn, swim_transformer_fp, swim_transformer_fn, swim_transformer_fpr, swim_transformer_tpr, swim_transformer_auc = \
    #     calc_metric('results/swim_transformer', 'swim_transformer', threshold=0.5)

    # crossvit_tp, crossvit_tn, crossvit_fp, crossvit_fn, crossvit_fpr, crossvit_tpr, crossvit_auc = \
    #     calc_metric('results/crossvit', 'crossvit', threshold=0.5)


    # confusion_matrix = np.array([[crossvit_tp, crossvit_fp],[crossvit_tn, crossvit_fn]])
    # plt.figure()
    # plt.title('Confusion Matrix')
    # sns.heatmap(confusion_matrix, annot=True, fmt='g', cmap='Blues')
    # plt.show()


    # swim_transformer_tp, swim_transformer_tn, swim_transformer_fp, swim_transformer_fn, swim_transformer_fpr, swim_transformer_tpr, swim_transformer_auc, _ = \
    #     calc_metric('results/swin_tiny_patch4_window7_224_0_frozen', 'swim_transformer', threshold=0.5)

    # crossvit_tp, crossvit_tn, crossvit_fp, crossvit_fn, crossvit_fpr, crossvit_tpr, crossvit_auc, _ = \
    #     calc_metric('results/crossvit_tiny_240_0', 'crossvit', threshold=0.5)

    # vit_tp, vit_tn, vit_fp, vit_fn, vit_fpr, vit_tpr, vit_auc, _ = \
    #     calc_metric('results/vit_small_patch16_224_f1_unfozen', 'vit', threshold=0.60)
    
    
    # vgg_tp, vgg_tn, vgg_fp, vgg_fn, vgg_fpr, vgg_tpr, vgg_auc, _ = \
    #     calc_metric('results/vgg16_bn_0', 'vgg', threshold=0.50)
    
    # resnet_tp, resnet_tn, resnet_fp, resnet_fn, resnet_fpr, resnet_tpr, resnet_auc, _ = \
    #     calc_metric('results/resnet50_0', 'resnet', threshold=0.50)
    
    # inception_tp, inception_tn, inception_fp, inception_fn, inception_fpr, inception_tpr, inception_auc, _ = \
    #     calc_metric('results/inception_v3_0', 'inception', threshold=0.8)

    pass