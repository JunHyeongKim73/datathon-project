import numpy as np
from matplotlib import pyplot as plt
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch_optimizer as optimizer
from torch.optim import lr_scheduler

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torchvision.transforms as transforms

from torch.utils.data import DataLoader, Subset

from tqdm.auto import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report

from foodDataset import FoodDataset
from model import EffNetModel  
from dataTransformer import DataTransformer

if __name__ == '__main__':
    # 데이터 어그멘테이션 (학습용)
    train_transform = A.Compose([
        A.Resize(256, 256),
        A.CenterCrop(224, 224),
        A.OneOf([
            A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=1),
            A.HorizontalFlip(p=1),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, p=1),
        ], p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    # 데이터 변환 (검증용)
    test_transform = A.Compose([
        A.Resize(256, 256),
        A.CenterCrop(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    # CustomDataset으로부터 데이터를 불러온다
    dataset = FoodDataset(label='../images_update.csv', root='../kfoods')
    batch_size = 64
    epochs = 40

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    # K-Fold 교차 검증
    best_models = [] # 폴드별로 가장 acc가 높은 모델 저장
    
    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(dataset), 1):
        # Train & Test Dataset 정의
        train_subset = Subset(dataset, train_idx)
        test_subset = Subset(dataset, test_idx)
        
        # Train & Test Dataset에 각각 Transform을 적용
        train_dataset = DataTransformer(train_subset, train_transform);
        test_dataset = DataTransformer(test_subset, test_transform);
        
        # Train & Test DataLoader 정의
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        
        # 모델 로드
        model_name = 'efficientnet-b0'

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = EffNetModel(model_name)
        model.to(device)
        
        # 하이퍼 파라미터
        learning_rate = 0.001
        optimizer = optimizer.RAdam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        scheduler = lr_scheduler.StepLR(optimizer=optimizer, step_size=7, gamma=0.1)
        
        # 학습에 필요한 변수를 초기화한다
        best_acc = 0.0
        best_loss = 0.0
        best_epoch = 0
        best_model = None
        
        # Early Stopping에 필요한 변수
        early_stop_cnt = 0
        early_stop = 7
        min_loss = 10^3

        since = time.time()
        
        for epoch in range(epochs):
            epoch += 1
            print('Epoch {}/{}'.format(epoch, epochs))
            print('-' * 10)

            train_len = 0    
            running_loss = 0.0
            running_corrects = 0
            
            # 모델 학습
            model.train()
            
            for inputs, labels in tqdm(train_loader, desc='Train'):
                inputs = inputs.to(device)
                labels = labels.to(device)
                # forward
                with torch.set_grad_enabled(True):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    # CrossEntropyLoss는 정수형 label 데이터가 들어오면 
                    # 원핫 인코딩으로 데이터를 변환한다
                    loss = criterion(outputs, labels)
                    # backward + optimize
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                # statistics
                batch_size = inputs.size(0)
                running_loss += loss.item() * batch_size
                running_corrects += torch.sum(preds == labels.data)
                
                train_len += batch_size
            
            epoch_loss = running_loss / train_len
            epoch_acc = running_corrects.double() / train_len

            print('Train Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
            
            # learning rate를 조절한다
            scheduler.step()

            # 모델 검증
            model.eval()

            test_len = 0
            running_loss = 0.0
            running_corrects = 0
            # hi
            for inputs, labels in tqdm(test_loader, desc='Test'):
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(False):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                # statistics
                batch_size = inputs.size(0)
                running_loss += loss.item() * batch_size
                running_corrects += torch.sum(preds == labels.data)

                test_len += batch_size

            epoch_loss = running_loss / test_len
            epoch_acc = running_corrects.double() / test_len

            print('Test Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
            
            # Save the Best Model Information
            if epoch_acc > best_acc:
                early_stop_cnt = 0
                best_acc = epoch_acc
                best_loss = epoch_loss
                best_epoch = epoch
                best_model = model

            elif epoch_loss < min_loss:
                early_stop_cnt = 0
                min_loss = epoch_loss

            else:
                early_stop_cnt += 1

            if early_stop_cnt > early_stop:
                print('Early Stopped!!')
                break
                
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))
        # Save the Best Model
        path = f'../models/{fold_idx}_{model_name}_{best_acc:.4f}_{best_loss:.4f}_epoch_{best_epoch}.pth'
        torch.save(best_model.state_dict(), path)
        # 폴드별로 가장 좋은 모델 저장
        best_models.append(best_model)
        # 현재 1 Fold로만 진행
        break