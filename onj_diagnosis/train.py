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

from script.preprocessing import collect_classification_dataset
import SimpleITK as sitk

import numpy as np
try:
    from cv2 import cv2
except:
    import cv2

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import argparse
parser = argparse.ArgumentParser()  
parser.add_argument("--fold", type=int, help="fold id")  
parser.add_argument("--model_name", type=str, help="model name")  
args = parser.parse_args()


class PR_CLS_DATASET:

    def __init__(self, root, split='train', ratio=0.2, fold_id=0, eval=False) -> None:
        assert split in ['train', 'val']

        data = collect_classification_dataset(root, 'PR')
        random.seed(42)
        random.shuffle(data)
        num = len(data)
        # data = data[:int(num*ratio)] if split=='train' else data[int(num*ratio):]
        block_num = int(num*ratio)
        data_blocks = [
            data[block_num*0 : block_num*1],
            data[block_num*1 : block_num*2],
            data[block_num*2 : block_num*3],
            data[block_num*3 : block_num*4],
            data[block_num*4 : block_num*5]]
        
        print('fold_id ', fold_id)
        if split=='val':
            data = data_blocks[fold_id]
            # data = data_blocks[1]
            # data = data_blocks[2]
            # data = data_blocks[3]
            # data = data_blocks[4]
        else:
            if fold_id == 0:
                data = data_blocks[1] + data_blocks[2] + data_blocks[3] + data_blocks[4]
            elif fold_id == 1:
                data = data_blocks[0] + data_blocks[2] + data_blocks[3] + data_blocks[4]
            elif fold_id == 2:
                data = data_blocks[0] + data_blocks[1] + data_blocks[3] + data_blocks[4]
            elif fold_id == 3:
                data = data_blocks[0] + data_blocks[1] + data_blocks[2] + data_blocks[4]
            elif fold_id == 4:
                data = data_blocks[0] + data_blocks[1] + data_blocks[2] + data_blocks[3]

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
            image = np.stack((image, image, image), axis=0)
            self.images.append(image)

            # label
            label = item['label']
            label = 0 if label==0 else 1
            self.labels.append(label)

        self.eval = eval

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        if self.eval:
            return self.images[index], self.labels[index], self.sizes[index], self.origin_images[index]
        else:
            return self.images[index], self.labels[index]


global_feature_list = []
def forward_hook(module, input, output):  
    # print(f"Inside {module.__class__.__name__}.forward")  
    # print(f"Input: {input[0]}")  # 输入是一个元组，对于nn.Linear，通常只包含一个tensor  
    # print(f"Output: {output}") 
    features = input[0].cpu().numpy()

    global_feature_list.append(features)
    

@torch.no_grad()
def extract_feature(dataloaders, model, device='cuda:0'):
    model.eval()

    # model.head.fc.register_forward_hook(forward_hook)
    model.head.register_forward_hook(forward_hook)

    # train_feature_list, train_label_list = [], []
    train_label_list = []
    for inputs, labels in dataloaders['train']:
        inputs = inputs.to(device)
        labels = labels.to(device)

        model(inputs)
        # features = model.forward_features(inputs)
        # x = features
        # if model.attn_pool is not None:
        #     x = model.attn_pool(x)
        # elif model.global_pool == 'avg':
        #     x = x[:, model.num_prefix_tokens:].mean(dim=1)
        # elif model.global_pool:
        #     x = x[:, 0]  # class token
        # x = model.fc_norm(x)
        # x = model.head_drop(x)
        # features = x
        # print(features.shape, '<-------')

        # features    = features[0].cpu().numpy()
        labels      = labels.cpu().numpy()

        # train_feature_list.append(features)
        train_label_list.append(labels)

    # train_feature_list = np.stack(train_feature_list)
    train_feature_list = np.stack(global_feature_list)
    train_label_list = np.stack(train_label_list)

    return train_feature_list, train_label_list


def train_model(dataloaders, dataset_sizes, model, criterion, optimizer, scheduler, num_epochs=25, device='cuda:0'):
    since = time.time()

    output_dir = os.path.join('results', f'{args.model_name}_{args.fold}')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        # best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')
        best_model_params_path = 'ckpt.pt'

        torch.save(model.state_dict(), os.path.join(output_dir, best_model_params_path))
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

           
                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), os.path.join(output_dir, best_model_params_path))

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        model.load_state_dict(torch.load(os.path.join(output_dir, best_model_params_path)))
    return model


@torch.no_grad()
def test_model(dataloaders, model, device='cuda:0'):
    model.eval()

    output_dir = os.path.join('results', f'{args.model_name}_{args.fold}')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    gt_lists, prob_lists = [], []
    for inputs, labels in dataloaders['train']:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        preds   = F.softmax(outputs, dim=1)

        gts     = labels.cpu().numpy().tolist()
        probs   = preds.cpu().numpy()[:,1].tolist()

        gt_lists    += gts
        prob_lists  += probs

    np.savez_compressed(os.path.join(output_dir, 'trainset_results.npz'), gt=gt_lists, prob=prob_lists)

    gt_lists, prob_lists = [], []
    for inputs, labels in dataloaders['val']:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        preds   = F.softmax(outputs, dim=1)

        gts     = labels.cpu().numpy().tolist()
        probs   = preds.cpu().numpy()[:,1].tolist()

        gt_lists    += gts
        prob_lists  += probs

    np.savez_compressed(os.path.join(output_dir, 'testset_results.npz'), gt=gt_lists, prob=prob_lists)


if __name__=='__main__':
    
    # 1. load dataset
    print('Load dataset ... ')
    train_data          = PR_CLS_DATASET(r'G:\txt+dcm', 'train', fold_id=args.fold)
    val_data            = PR_CLS_DATASET(r'G:\txt+dcm', 'val', fold_id=args.fold)

    train_dataloader    = DataLoader(train_data, batch_size=1, shuffle=True)
    val_dataloader      = DataLoader(val_data, batch_size=1, shuffle=False)

    dataloaders         = {'train': train_dataloader, 'val': val_dataloader}
    dataset_sizes       = {'train': len(train_data), 'val': len(val_data)}

    # for image, label in train_data:
    #     print(image.shape, image.min(), image.max(), image.dtype, label)
    #     image = (image + 1.) * 0.5 * 255.
    #     image = image.astype(np.uint8)
    #     cv2.imshow('PR', image)
    #     cv2.waitKey()
    #     # break

    # 2. build model
    print('Build model ... ')

    model = timm.create_model(args.model_name, pretrained=True)
    model.reset_classifier(2)

    # model = timm.create_model('vit_small_patch16_224', pretrained=True)
    # num_firs = model.head.in_features
    # model.head = nn.Linear(num_firs, 2)

    # model = timm.create_model('vgg16_bn', pretrained=True)
    # model.reset_classifier(2)

    # model = timm.create_model('resnet50', pretrained=True)
    # model.reset_classifier(2)

    # model = timm.create_model('inception_v3', pretrained=True)
    # model.reset_classifier(2)

    # model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True)
    # model.reset_classifier(2)

    # model = timm.create_model('crossvit_tiny_240', pretrained=True)
    # model.reset_classifier(2)

    model = model.cuda()

    # 3. loss
    criterion = nn.CrossEntropyLoss()

    # 4. optimizer & scheduler
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    try:
        optimizer = optim.SGD(model.head.parameters(), lr=0.001, momentum=0.9)
    except:
        optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)
    # optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # 5. train
    model = train_model(
                dataloaders,
                dataset_sizes,
                model, 
                criterion, 
                optimizer, 
                scheduler, 
                num_epochs=40)
    
    # 6. test
    test_model(dataloaders, model)

    # 4. t-SNE
    # train_feature_list, train_label_list = extract_feature(dataloaders, model)
    # tsne = TSNE(n_components=2, random_state=0)

    # N = train_feature_list.shape[0]
    # feature = train_feature_list.reshape((N,-1))
    # label = train_label_list.reshape(-1)

    # feature_2d = tsne.fit_transform(feature)

    # plt.figure(figsize=(6, 5))  
    # plt.scatter(feature_2d[label == 1, 0], feature_2d[label == 1, 1], c='c', label='Osteomyelitis') 
    # plt.scatter(feature_2d[label == 0, 0], feature_2d[label == 0, 1], c='orange', label='Other') 
    # # colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'  
    # # for i, c, label in zip(range(10), colors, digits.target_names):  
    # #     plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1], c=c, label=label)  
    # plt.legend()  
    # plt.show()