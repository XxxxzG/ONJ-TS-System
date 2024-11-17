import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import timm

from sklearn.manifold import TSNE

from train import PR_CLS_DATASET


@torch.no_grad()
def extract_feature(dataloaders, model, device='cuda:0'):
    model.eval()

    train_feature_list, train_label_list = [], []
    for inputs, labels in dataloaders['train']:
        inputs = inputs.to(device)
        labels = labels.to(device)

        features = model.forward_features(inputs)
        x = features
        # if model.attn_pool is not None:
        #     x = model.attn_pool(x)
        # elif model.global_pool == 'avg':
        #     x = x[:, model.num_prefix_tokens:].mean(dim=1)
        # elif model.global_pool:
        #     x = x[:, 0]  # class token
        # x = model.fc_norm(x)
        # x = model.head_drop(x)
        features = x
        # print(features.shape, 'feature shape')

        features    = features[0].cpu().numpy()
        labels      = labels.cpu().numpy()

        train_feature_list.append(features)
        train_label_list.append(labels)

    train_feature_list = np.stack(train_feature_list)
    train_label_list = np.stack(train_label_list)

    # gt_lists, prob_lists = [], []
    # for inputs, labels in dataloaders['train']:
    #     inputs = inputs.to(device)
    #     labels = labels.to(device)

    #     outputs = model(inputs)
    #     preds   = F.softmax(outputs, dim=1)

    #     gts     = labels.cpu().numpy().tolist()
    #     probs   = preds.cpu().numpy()[:,1].tolist()

    #     gt_lists    += gts
    #     prob_lists  += probs

    # np.savez_compressed('testset_results.npz', gt=gt_lists, prob=prob_lists)

    return train_feature_list, train_label_list


if __name__=='__main__':
    
    # 1. load dataset
    print('Load dataset ... ')
    train_data          = PR_CLS_DATASET(r'G:\txt+dcm', 'train')
    val_data            = PR_CLS_DATASET(r'G:\txt+dcm', 'val')

    train_dataloader    = DataLoader(train_data, batch_size=1, shuffle=True)
    val_dataloader      = DataLoader(val_data, batch_size=1, shuffle=False)

    dataloaders         = {'train': train_dataloader, 'val': val_dataloader}
    dataset_sizes       = {'train': len(train_data), 'val': len(val_data)}

    # 2. build model
    print('Build model ... ')
    model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True)
    model.reset_classifier(2)

    model.load_state_dict(torch.load('results/swin_tiny_patch4_window7_224_0/ckpt.pt'))

    model = model.cuda()
    
    # 3. test
    train_feature_list, train_label_list = extract_feature(dataloaders, model)
    
    # np.savez_compressed('trainset_features.npz', feature=gt_lists, prob=prob_lists)

    # 4. t-SNE
    tsne = TSNE(n_components=2, random_state=0)

    N = train_feature_list.shape[0]
    feature = train_feature_list.reshape((N,-1))
    label = train_label_list.reshape(-1)

    feature_2d = tsne.fit_transform(feature)

    plt.figure(figsize=(6, 5))  
    plt.scatter(feature_2d[label == 0, 0], feature_2d[label == 0, 1], c='r', label='other')
    plt.scatter(feature_2d[label == 1, 0], feature_2d[label == 1, 1], c='g', label='sick')  
    # colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'  
    # for i, c, label in zip(range(10), colors, digits.target_names):  
    #     plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1], c=c, label=label)  
    plt.legend()  
    plt.show()