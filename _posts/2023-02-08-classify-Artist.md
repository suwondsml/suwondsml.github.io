---
layout: post
title: Classify Artist
authors: [Seongmin Cho]
categories: [1기 AI/SW developers(개인 프로젝트)]
---


# 프로젝트 설명
- 이번 프로젝트는 Dacon의 예술 작품 화가 분류 대회의 데이터를 활용하여 진행하였습니다.
- 본 프로젝트에서 풀어야할 문제는 크게 2가지입니다.
	1. 어떤 모델을 사용하여야 하는가?
	2. 데이터 불균형 문제를 어떻게 해결해야하는가?

- 데이터를 분석하고 예측할 때 가장 흔히 접할 수 있는 문제가 위와 같은 문제이며, 2가지 문제를 풀기 위해 다양한 기법들을 시도하였습니다.
- [월간 데이콘 예술 작품 화가 분류 AI 경진대회](https://dacon.io/competitions/official/236006/overview/description)

## Data
train 데이터로는 총 5911개의 데이터가 주어집니다.
- 사진1 : Diego Velazquez
![img1](https://github.com/suwondsml/suwondsml.github.io/blob/main/data/2023-02-08-Classify-Artist/image1.jpg?raw=true)
- 사진2 : Vincent van Gogh
-![img1](https://github.com/suwondsml/suwondsml.github.io/blob/main/data/2023-02-08-Classify-Artist/image2.jpg?raw=true)
- 사진3 : Claude Monet
![img1](https://github.com/suwondsml/suwondsml.github.io/blob/main/data/2023-02-08-Classify-Artist/image3.jpg?raw=true)

다음으로 test 데이터로는 12670개의 데이터가 있다고 합니다. 하지만 test 데이터는 후에 평가를 위해 있는 것이기 때문에 따로 공개가 되지 않습니다.

먼저 이미지의 특징을 보고 분류를 해야하기 때문에 비전 딥러닝 모델이 필수적이라고 판단됩니다.  하지만 학습 전에 데이터가 어떤 식으로 존재하고 또 각 데이터의 수가 어느정도로 분포되어있는지 판단하는 작업을 먼저 시행하였습니다.

먼저 데이터를 불러와 살펴보겠습니다.


# 코드 실습

## Module Import
먼저 사용한 모듈들을 먼저 import 해주고 시작합니다.
```python
import os
import sys
import time
import random

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import pandas as pd
import glob
from IPython.display import clear_output
import cv2
import imageio
import scipy.ndimage
import pandas as pd

import torchvision.transforms as transforms

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


import torchsummary
from tqdm import notebook

import matplotlib.style as style
import seaborn as sns

sys.path.insert(0, '..')


from sklearn.model_selection import train_test_split
from sklearn import preprocessing
```
## 랜덤성을 배제한 환경 고정
다음으로 랜덤성을 배제하기 위해 seed를 통일 시켜주도록 하겠습니다.
```python
random_seed= 42
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # select device for training, i.e. gpu or cpu
print(DEVICE)
```

이제 환경설정이 모두 끝났으므로 데이터를 불러와 살펴보도록 하겠습니다.
## Data Load
```python
df = pd.read_csv("./data/open/train.csv")
df.head(5)
```
![img1](https://github.com/suwondsml/suwondsml.github.io/blob/main/data/2023-02-08-Classify-Artist/image4.png?raw=true)

train.csv파일에 화가의 이름 즉, 정답이 적혀있습니다.

총 화가의 종류는 어느정도 일지 확인해봅시다.

```python
artist_name = list(df['artist'].value_counts().index)
artist_name
```
![img1](https://github.com/suwondsml/suwondsml.github.io/blob/main/data/2023-02-08-Classify-Artist/image5.png?raw=true)

모델이 화가의 이름을 라벨로 인지하게 하기 위해 encoding을 진행하였습니다. 진행 방식은 총 화가의 종류가 50가지 이므로 0 ~ 49 로 labeling을 하였습니다.

다음으로 데이터의 불균형을 확인하기 위해 데이터의 수를 비교해보도록 하겠습니다.

## 데이터 수 비교
```python
# Get label frequencies in descending order
label_freq = df['artist'].apply(lambda s: str(s).split('|')).explode().value_counts().sort_values(ascending=False)

# Bar plot
# style.use("fivethirtyeight")
plt.figure(figsize=(12,10))
sns.barplot(y=label_freq.index.values, x=label_freq, order=label_freq.index)
plt.title("Label frequency", fontsize=14)
plt.xlabel("")
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()
```

![img1](https://github.com/suwondsml/suwondsml.github.io/blob/main/data/2023-02-08-Classify-Artist/image6.png?raw=true)

데이터의 불균형 문제가 많이 심각합니다. 보면 Vincent van Gogh 의 작품은 600개가 넘어가지만 jackson Pollock의 작품은 약 30개 정도로 차이가 많이 있습니다. 만약 이대로 학습을 진행한다면 성능이 안좋게 나오는 것은 뻔하기 때문에 데이터의 불균형 문제를 먼저 해결해 줄 필요가 있어 보입니다.

## Data Split
활용할 수 있는 데이터는 오직 train 데이터 뿐입니다. 이 train 데이터를 train과 validation으로 쪼개어 후에 검증의 목적으로 사용하도록 하겠습니다.

```python
X_train, X_val, y_train, y_val = train_test_split(df, df['label'].values, test_size=0.2)
```
0.2만큼 validation 데이터를 나누었으며, 각 데이터의 수는 다음과 같습니다.
- train data : 4728
- validation data : 1183

![img1](https://github.com/suwondsml/suwondsml.github.io/blob/main/data/2023-02-08-Classify-Artist/image7.png?raw=true)

다음으로 데이터 프레임 안에 사진을 불러오기 위해 path를 재설정 해줍니다.

```python
X_train = X_train.sort_values(by=['id']) # id 기준 정렬
X_train['img_path'] = [base_path + '/data/open/' + path[2:] for path in X_train['img_path']]
```
![img1](https://github.com/suwondsml/suwondsml.github.io/blob/main/data/2023-02-08-Classify-Artist/image8.png?raw=true)


## 데이터 확인
이제 위에서 만든 데이터가 제대로 불러와지는지 확인해봅시다.

```python
for i in range(0, 5):
    path = list(X_train['img_path'])[i]
    artist = list(X_train['artist'])[i]
    label = list(X_train['label'])[i]
    ndarray = img.imread(path)

    plt.imshow(ndarray)
    print(artist)
    print(label)
    plt.axis('off')
    plt.show()
    print(ndarray.shape)
```
![img1](https://github.com/suwondsml/suwondsml.github.io/blob/main/data/2023-02-08-Classify-Artist/image9.png?raw=true)

**X_train에는 경로, 화가이름, 라벨 총 3가지가 포함되어 있습니다.**
- 첫 번째, 이미지 경로를 통해 image를 출력할 수 있으며, shape을 확인해보니 (1300 x 1024 x 3) 의 형태로 구성되어 있습니다. 
- 두 번째,  해당 이미지의 화가 이름을 출력해보니 Vincent Van Gogh 가 일치하게 출력되는 것을 확인할 수 있습니다.
-  세 번째, 화가 이름을 인코딩한 라벨 역시 0으로 잘 출력됩니다.




## transform
데이터를 전처리 하기 위해 torchvision 라이브러리에서 제공하는 transforms 함수를 사용하였습니다.

```python
import torchvision.transforms as transforms
resize = 224, 224

transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(((resize))),
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5)
])

transform_val = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(((resize))),
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5)
])
```

## Custom Dataset
다음으로 커스텀 데이터셋 클래스를 생성하였습니다.

```python
import cv2
class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, transform=None):
        self.data = dataframe
        self.transform = transform   # 데이터 전처리
        

        self.lst_input = list(self.data['img_path'])
        self.lst_label = list(self.data['label'])

    def __len__(self):
        return len(self.lst_label)

    def __getitem__(self, index):

        label = self.lst_label[index] 
        img_input = cv2.imread(self.lst_input[index]) # H W C


        if self.transform:
            img_input = self.transform(img_input)

        return img_input, label


# 데이터셋 클래스 적용
custom_dataset_train = Dataset(X_train, transform=transform_train)
print("My custom training-dataset has {} elements".format(len(custom_dataset_train)))

custom_dataset_val = Dataset(X_val, transform=transform_val)
print("My custom valing-dataset has {} elements".format(len(custom_dataset_val)))
```


## 데이터 불균형 문제 해결
위에서 말했던 것처럼 사용할 데이터는 불균형 문제가 심각합니다. 이를 해결할 수 있는 방법은 다양하지만 이번에는 Weighted Random Sampling 기법과 Weighted Loss Function 기법을 사용하여 성능을 비교해보도록 하겠습니다.

먼저 Weighted Random Sampling 기법을 구현해보도록 하겠습니다.

```python
def make_weights(labels, nclasses):
    labels = np.array(labels) 
    weight_arr = np.zeros_like(labels) 
    
    _, counts = np.unique(labels, return_counts=True) 
    for cls in range(nclasses):
        weight_arr = np.where(labels == cls, 1/counts[cls], weight_arr) 
        # 각 클래스의의 인덱스를 산출하여 해당 클래스 개수의 역수를 확률로 할당한다.
        # 이를 통해 각 클래스의 전체 가중치를 동일하게 한다.
 
    return weight_arr


weights_trian = make_weights(custom_dataset_train.lst_label, 50)
weights_trian = torch.DoubleTensor(weights_trian)

weights_val = make_weights(custom_dataset_val.lst_label, 50)
weights_val = torch.DoubleTensor(weights_val)

sampler_train = torch.utils.data.sampler.WeightedRandomSampler(weights_trian, len(weights_trian))
sampler_val = torch.utils.data.sampler.WeightedRandomSampler(weights_val, len(weights_val))

# Define variables for the training
BATCH_SIZE = 10

# 가중 무작위 샘플링 사용 여부
use = False

if use:
    # 가중 무작위 샘플링 사용
    dataloader_train = torch.utils.data.DataLoader(custom_dataset_train, batch_size=BATCH_SIZE, sampler = sampler_train, num_workers=2)
    dataloader_val = torch.utils.data.DataLoader(custom_dataset_val, batch_size=BATCH_SIZE, sampler = sampler_val, num_workers=2)
else:
    # 가중 무작위 샘플링 사용 X
    dataloader_train = torch.utils.data.DataLoader(custom_dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    dataloader_val = torch.utils.data.DataLoader(custom_dataset_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
```

use 변수에 True와 False를 할당하여 sampling을 사용할지 말지를 선택할 수 있도록 구현하였습니다. 샘플링 기법은 사용한다고 항상 성능이 올라가지 않습니다. 오히려 떨어지는 경우도 빈번하기 때문에 사용했을 때와 사용하지 않았을 때를 비교하는 것은 필수적입니다.


다음으로 불균형 문제를 해결하기 위해 Weighted Loss Function 기법을 구현하였습니다.

```python
# 가중치 손실 함수 사용 여부
Use = False

if Use:
    num_artist # 각 artist당 그림 개수
    weights = [1 - (x/sum(num_artist)) for x in num_artist]
    class_weights = torch.FloatTensor(weights).to(DEVICE)
    class_weights = (class_weights*0.25)**2
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
else:
    criterion = nn.CrossEntropyLoss()

class FocalLoss(nn.Module):
    def __init__(self, num_classes=20):
        super().__init__()
        self.num_classes = num_classes # VOC dataset 20

    # alternative focal loss
    def focal_loss_alt(self, x, y):
        alpha = 0.25

        t = one_hot_embedding(y.data.cpu(), 1+self.num_classes)
        t = t[:,1:] # 배경 제외
        t = t.cuda()

        xt = x*(2*t-1) # xt = x if t > 0 else -x
        pt = (2*xt+1).sigmoid()

        w = alpha*t + (1-alpha)*(1-t)
        loss = -w*pt.log() / 2
        return loss.sum()

    def forward(self, loc_preds, loc_targets, cls_preds, cls_targets):
        # (loc_preds, loc_targets)와 (cls_preds, cls_targets) 사이의 loss 계산
        # loc_preds: [batch_size, #anchors, 4]
        # loc_targets: [batch_size, #anchors, 4]
        # cls_preds: [batch_size, #anchors, #classes]
        # cls_targets: [batch_size, #anchors]

        # loss = SmoothL1Loss(loc_preds, loc_targets) + FocalLoss(cls_preds, cls_targets)

        batch_size, num_boxes = cls_targets.size()
        pos = cls_targets > 0
        num_pos = pos.data.long().sum()

        # loc_loss = SmoothL1Loss(pos_loc_preds, pos_loc_targets)
        mask = pos.unsqueeze(2).expand_as(loc_preds) # [N, #anchors, 4], 객체가 존재하는 앵커박스 추출
        masked_loc_preds = loc_preds[mask].view(-1,4)      # [#pos, 4]
        masked_loc_targets = loc_targets[mask].view(-1, 4) # [#pos, 4]
        loc_loss = F.smooth_l1_loss(masked_loc_preds, masked_loc_targets, reduction='sum')

        # cls_loss = FocalLoss(loc_preds, loc_targets)
        pos_neg = cls_targets > -1 # ground truth가 할당되지 않은 anchor 삭제
        mask = pos_neg.unsqueeze(2).expand_as(cls_preds)
        masked_cls_preds = cls_preds[mask].view(-1, self.num_classes)
        cls_loss = self.focal_loss_alt(masked_cls_preds, cls_targets[pos_neg])

        # print('loc_loss: %.3f | cls_loss: %.3f' % (loc_loss.item(), cls_loss))
        loss = (loc_loss+cls_loss)/num_pos
        return loss

Use = True

if Use:
    criterion = FocalLoss()
else:
    criterion = nn.CrossEntropyLoss()
```

이 기법역시 사용하였을 때와 사용하지 않았을 때를 비교하기 위해 Use 변수를 사용하였습니다. 손실함수에 가중치를 두어 적은 데이터의 중요도를 높이는 방법이라 할 수 있겠습니다.

## Train
이제 학습을 시켜봅시다. 그 전에 과적합이 와 성능이 저하되는 것을 막기 위해 Early Stopping 기능을 구현하겠습니다.

```python
class EarlyStopping:
    """주어진 patience 이후로 validation loss가 개선되지 않으면 학습을 조기 중지"""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): validation loss가 개선된 후 기다리는 기간
                            Default: 7
            verbose (bool): True일 경우 각 validation loss의 개선 사항 메세지 출력
                            Default: False
            delta (float): 개선되었다고 인정되는 monitered quantity의 최소 변화
                            Default: 0
            path (str): checkpoint저장 경로
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = f'./checkpoints/final/ckpt_{model_name}.pth'

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''validation loss가 감소하면 모델을 저장한다.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
```

지금 까지 구현하였던 함수들을 활용하여 훈련을 시작하겠습니다.

```python
def train(epochs=30, patience=7, Early_Stopping = True):
    
    if Early_Stopping:
        early_stopping = EarlyStopping(patience = patience, verbose = True)
    
    for epoch in range(1, epochs + 1):
        print('\n[ Train epoch: %d ]' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0

        for batch_idx, (inputs, labels) in enumerate(dataloader_train):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)

            total += labels.size(0)
            current_correct = (predicted == labels).sum().item()
            correct += current_correct

            if batch_idx % 100 == 0:
                print('\nCurrent batch:', str(batch_idx))
                print('Current batch average train accuracy:', current_correct / labels.size(0))
                print('Current batch average train loss:', loss.item() / labels.size(0))            
            

        # 훈련이 모두 끝난 후 정확도 / 솔실함수 값을 출력  
        print('\nTotal average train accuarcy:', correct / total)
        print('Total average train loss:', train_loss / total)


        # validation
        print('\n[ Test epoch: %d ]' % epoch)
        net.eval()
        loss = 0
        correct = 0
        total = 0
        valid_losses = []
        
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(dataloader_val):
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                total += labels.size(0)

                outputs = net(inputs)
                loss += criterion(outputs, labels).item()

                valid_losses.append(loss)

                _, predicted = outputs.max(1)
                correct += (predicted == labels).sum().item()

            print('\nTotal average test accuarcy:', correct / total)
            print('Total average test loss:', loss / total)

            valid_loss = np.average(valid_losses)
            
            
            if Early_Stopping:
                early_stopping(valid_loss, net)

                if early_stopping.early_stop:

                    print("Early stopping")

                    break
            
            
            else:
                state = {
                    'net' : net.state_dict()
                }
                if not os.path.isdir('checkpoints/final'):
                    os.mkdir('checkpoints/final')
                torch.save(state, f'./checkpoints/final/ckpt_{model_name}.pth')
                print('Model Saved!')

models = ['resnet18', 'wide_resnet50_2', 'EfficientNet_b4', 
          'visformer_small', 'vit_base_patch8_224', 'vit_small_patch8_224_dino']
models = ['wide_resnet50_2']

for model_name in models:
    net = mk_model(model_name)
    optimizer = torch.optim.Adam(net.parameters(), lr = 1e-4, weight_decay = 1e-8)
    
    print(f'##### {model_name} ##### {model_name} ##### {model_name} ##### {model_name} ##### {model_name} ##### {model_name} #####')
    train(epochs = 30, patience = 7, Early_Stopping = True)
    print("#######################################################################################################################")
    print("#######################################################################################################################")
    print("#######################################################################################################################")
    print("#######################################################################################################################")
    print("#######################################################################################################################")
    print(f'#END# {model_name} #END# {model_name} #END# {model_name} #END# {model_name} #END# {model_name} #END# {model_name} #END#')
```

이 함수를 실행시키면 다음과 같이 결과가 나오게 됩니다.


 _##### wide_resnet50_2 ##### wide_resnet50_2 ##### wide_resnet50_2 ##### wide_resnet50_2 ##### wide_resnet50_2 ##### wide_resnet50_2 #####_

[ Train epoch: 1 ]

Current batch: 0
Current batch average train accuracy: 0.1
Current batch average train loss: 0.3922434329986572

Current batch: 100
Current batch average train accuracy: 0.2
Current batch average train loss: 0.23646013736724852

Current batch: 200
Current batch average train accuracy: 0.5
Current batch average train loss: 0.22266743183135987

Current batch: 300
Current batch average train accuracy: 0.6
Current batch average train loss: 0.18211838006973266

Current batch: 400
Current batch average train accuracy: 0.4
Current batch average train loss: 0.22381784915924072

Total average train accuarcy: 0.3945945945945946
Total average train loss: 0.24340428129346334

[ Test epoch: 1 ]

Total average test accuarcy: 0.5942519019442096
Total average test loss: 0.1547493291475565
Validation loss decreased (inf --> 92.566729).  Saving model ...

[ Train epoch: 2 ]

Current batch: 0
Current batch average train accuracy: 0.5
Current batch average train loss: 0.14742729663848878

Current batch: 100
Current batch average train accuracy: 0.7
Current batch average train loss: 0.13615922927856444

Current batch: 200
Current batch average train accuracy: 0.7
Current batch average train loss: 0.11106669902801514

Current batch: 300
Current batch average train accuracy: 0.5
Current batch average train loss: 0.2168853759765625

Current batch: 400
Current batch average train accuracy: 0.7
Current batch average train loss: 0.1413419246673584

Total average train accuarcy: 0.6606345475910693
Total average train loss: 0.13263236651269306

[ Test epoch: 2 ]

Total average test accuarcy: 0.6449704142011834
Total average test loss: 0.13444265058155494
Validation loss decreased (92.566729 --> 79.540437).  Saving model ...

... ... ... ...

먼저 위에서 말했던 것처럼 사용한 데이터의 문제는 불균형입니다. 
이를 해결하기 위해 Weighted Random Sampling 기법과 Weighted Loss Function 기법을 사용하였습니다. 하지만 이 기법들은 사용한다고 항상 성능이 올라가지 않고 오히려 저하될 수 있다는 문제가 있었죠.

즉, 어떤 기법을 사용해야할지 실험을 해야만 했습니다. 먼저 어떤 기법을 사용하였을 때 성능이 올라가는지 각각 비교를 하여 실험을 진행하였습니다.

<실험결과>

![img1](https://github.com/suwondsml/suwondsml.github.io/blob/main/data/2023-02-08-Classify-Artist/image11.png?raw=true)

데이터의 불균형 문제를 해결하였는지 설명할 수 있는 지표인 F1-score를 사용하였습니다. F1-score에 대한 간단한 설명은 다음과 같습니다.

먼저 왜 단순 Accuracy만으로는 데이터 불균형이 해결되었는지 확인이 어려운지 간단한 예시를 들어보겠습니다.

만약 일주일에 평균적으로 1번 비가 온다고 했을 때, 기상청에서 **항상 비가 내리지 않습니다.** 라고 한다면 이는 좋은 예측이라 할 수 없을 것입니다. 하지만 평균적으로 6/7의 정확도를 보여주기에 정확도로만 본다면 좋은 예측으로 결과가 나오게 될 것입니다.

하지만 F1-score를 사용하게 되면 모델이 True라고 분류한 것들 중에서 실제로 True인 것의 비율을 나타내는 Precision과 실제 True인 것들 중에서 모델이 True라고 예측한 것의 비율을 나타내는 Recall을 모두 고려하기 때문에 단순 정확도보다 불균형에 대해 훨씬 민감하게 평가를 할 수 있게 됩니다.

---
이제 각 모델들에 대해 총 3가지에 대해 실험을 진행하였습니다.
1. 가중 무작위 샘플링과 가중 손실 함수 둘 모두 적용
2. 가중 손실 함수만 적용
3. 두 기법 모두 적용 X


여러 모델을 대상으로 실험을 하였지만 보여주는 인사이트는 모두 비슷했습니다. 위에 보이는 것은 resnet18 모델을 대상으로 실험을 진행한 것이고 보이는 것처럼 가중 손실 함수 만을 사용하였을 때 성능이 좋은 것을 확인할 수 있습니다.


![img1](https://github.com/suwondsml/suwondsml.github.io/blob/main/data/2023-02-08-Classify-Artist/image10.png?raw=true)

다음으로 각 모델들을 모두 돌려보고 각 정확도와 손실함수를 적어 비교해보았습니다. 

위에서 알 수 있는 결과는 EfficientNet_b4 모델을 사용하였을 때 가장 성능이 좋았습니다. 아무래도 19년도에 SOTA 를 차지하였던 EfficientNet 모델을 기반으로 한 모델이니만큼 다른 모델들에 비해 성능이 좋았던 것 같습니다.

또한 데이터의 수가 상당히 많은 양도 아니었기에 단순히 모델의 깊이를 늘려가며 성능을 높여가는 VGGNet과 모델을 넓게 즉, width를 넓여 성능을 넓히려 하였던 InceptionNet과 달리 EfficientNet은 성능을 높일 수 있는 모든 조건들을 모두 사용하여 최적의 성능을 도출합니다.

적은 데이터를 사용하였을 때는 과접합이 오기 상당히 쉽습니다. 하지만 모델의 width와 depth를 모두 고려하여 가장 최적의 모델을 도출해내기 때문에 적당한 데이터로 최상의 결과를 만들 수 있었다고 생각합니다. 

---




- EfficientNet 에 대한 설명은 제 개인 블로그에 자세히 설명해놓았으니 참고하시면 될 것 같습니다.
- [머신러닝 이야기 - [논문 리뷰] EfficientNet(2019), 파이토치 구현](https://smcho1201.tistory.com/56)
