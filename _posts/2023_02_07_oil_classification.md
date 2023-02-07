---
layout: post
title: Oil Classification
authors: [seoyeon Gang, nojung Kim, sieun Kim, subin Sung]
categories: [1기 AI/SW developers(팀 프로젝트)]
---



<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#패키지-불러오기." data-toc-modified-id="패키지-불러오기.-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>패키지 불러오기.</a></span></li><li><span><a href="#하이퍼-파라미터-생성하기." data-toc-modified-id="하이퍼-파라미터-생성하기.-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>하이퍼 파라미터 생성하기.</a></span></li><li><span><a href="#seed를-고정하는-코드" data-toc-modified-id="seed를-고정하는-코드-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>seed를 고정하는 코드</a></span></li><li><span><a href="#훈련-데이터와-테스트-데이터-불러오기." data-toc-modified-id="훈련-데이터와-테스트-데이터-불러오기.-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>훈련 데이터와 테스트 데이터 불러오기.</a></span></li><li><span><a href="#데이터-전처리하기." data-toc-modified-id="데이터-전처리하기.-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>데이터 전처리하기.</a></span><ul class="toc-item"><li><span><a href="#범주형과-수치형-특징-분리하기." data-toc-modified-id="범주형과-수치형-특징-분리하기.-5.1"><span class="toc-item-num">5.1&nbsp;&nbsp;</span>범주형과 수치형 특징 분리하기.</a></span></li><li><span><a href="#결측치-처리하기." data-toc-modified-id="결측치-처리하기.-5.2"><span class="toc-item-num">5.2&nbsp;&nbsp;</span>결측치 처리하기.</a></span></li><li><span><a href="#데이터-분포를-시각화를-통해-알아보기." data-toc-modified-id="데이터-분포를-시각화를-통해-알아보기.-5.3"><span class="toc-item-num">5.3&nbsp;&nbsp;</span>데이터 분포를 시각화를 통해 알아보기.</a></span></li><li><span><a href="#훈련-데이터와-검증-데이터-분할하기." data-toc-modified-id="훈련-데이터와-검증-데이터-분할하기.-5.4"><span class="toc-item-num">5.4&nbsp;&nbsp;</span>훈련 데이터와 검증 데이터 분할하기.</a></span></li><li><span><a href="#knn" data-toc-modified-id="knn-5.5"><span class="toc-item-num">5.5&nbsp;&nbsp;</span>knn</a></span></li><li><span><a href="#스케일링-(표준화)-진행하기." data-toc-modified-id="스케일링-(표준화)-진행하기.-5.6"><span class="toc-item-num">5.6&nbsp;&nbsp;</span>스케일링 (표준화) 진행하기.</a></span></li></ul></li><li><span><a href="#사용자-정의-데이터셋-만들기." data-toc-modified-id="사용자-정의-데이터셋-만들기.-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>사용자 정의 데이터셋 만들기.</a></span><ul class="toc-item"><li><span><a href="#사용자-정의-데이터셋-만드는-과정" data-toc-modified-id="사용자-정의-데이터셋-만드는-과정-6.1"><span class="toc-item-num">6.1&nbsp;&nbsp;</span>사용자 정의 데이터셋 만드는 과정</a></span></li><li><span><a href="#사용자-정의-데이터셋-만드는-코드" data-toc-modified-id="사용자-정의-데이터셋-만드는-코드-6.2"><span class="toc-item-num">6.2&nbsp;&nbsp;</span>사용자 정의 데이터셋 만드는 코드</a></span></li><li><span><a href="#간편한-API-DataLoader" data-toc-modified-id="간편한-API-DataLoader-6.3"><span class="toc-item-num">6.3&nbsp;&nbsp;</span>간편한 API DataLoader</a></span></li></ul></li><li><span><a href="#딥러닝-모델-지식의-증류-기법-:-교사---학생-지식-증류-기법" data-toc-modified-id="딥러닝-모델-지식의-증류-기법-:-교사---학생-지식-증류-기법-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>딥러닝 모델 지식의 증류 기법 : 교사 - 학생 지식 증류 기법</a></span></li><li><span><a href="#이미-학습된-무거운-모델,-교사-모델-정의하기." data-toc-modified-id="이미-학습된-무거운-모델,-교사-모델-정의하기.-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>이미 학습된 무거운 모델, 교사 모델 정의하기.</a></span><ul class="toc-item"><li><span><a href="#Teacher-Model-신경망-구현하기.-(순전파를-이용하여-예측값-구하기.)" data-toc-modified-id="Teacher-Model-신경망-구현하기.-(순전파를-이용하여-예측값-구하기.)-8.1"><span class="toc-item-num">8.1&nbsp;&nbsp;</span>Teacher Model 신경망 구현하기. (순전파를 이용하여 예측값 구하기.)</a></span></li><li><span><a href="#훈련-데이터-교사-모델과-검증-데이터-교사-모델-이용하여-loss를-최소로-하는-가중치-구하기." data-toc-modified-id="훈련-데이터-교사-모델과-검증-데이터-교사-모델-이용하여-loss를-최소로-하는-가중치-구하기.-8.2"><span class="toc-item-num">8.2&nbsp;&nbsp;</span>훈련 데이터 교사 모델과 검증 데이터 교사 모델 이용하여 loss를 최소로 하는 가중치 구하기.</a></span></li><li><span><a href="#교사-모델-돌리기." data-toc-modified-id="교사-모델-돌리기.-8.3"><span class="toc-item-num">8.3&nbsp;&nbsp;</span>교사 모델 돌리기.</a></span></li></ul></li><li><span><a href="#학생-모델-정의하기." data-toc-modified-id="학생-모델-정의하기.-9"><span class="toc-item-num">9&nbsp;&nbsp;</span>학생 모델 정의하기.</a></span><ul class="toc-item"><li><span><a href="#학생-모델-신경망-구현하기.-(순전파를-이용하여-예측값-구하기.)" data-toc-modified-id="학생-모델-신경망-구현하기.-(순전파를-이용하여-예측값-구하기.)-9.1"><span class="toc-item-num">9.1&nbsp;&nbsp;</span>학생 모델 신경망 구현하기. (순전파를 이용하여 예측값 구하기.)</a></span></li><li><span><a href="#교사---학생-지식-증류-손실-함수-값-정의하기." data-toc-modified-id="교사---학생-지식-증류-손실-함수-값-정의하기.-9.2"><span class="toc-item-num">9.2&nbsp;&nbsp;</span>교사 - 학생 지식 증류 손실 함수 값 정의하기.</a></span></li><li><span><a href="#훈련-데이터-학생-모델과-검증-데이터-학생-모델-이용하여-loss를-최소로-하는-가중치-구하기." data-toc-modified-id="훈련-데이터-학생-모델과-검증-데이터-학생-모델-이용하여-loss를-최소로-하는-가중치-구하기.-9.3"><span class="toc-item-num">9.3&nbsp;&nbsp;</span>훈련 데이터 학생 모델과 검증 데이터 학생 모델 이용하여 loss를 최소로 하는 가중치 구하기.</a></span></li><li><span><a href="#학생-모델-돌리기." data-toc-modified-id="학생-모델-돌리기.-9.4"><span class="toc-item-num">9.4&nbsp;&nbsp;</span>학생 모델 돌리기.</a></span></li></ul></li><li><span><a href="#임계값-추론하기." data-toc-modified-id="임계값-추론하기.-10"><span class="toc-item-num">10&nbsp;&nbsp;</span>임계값 추론하기.</a></span><ul class="toc-item"><li><span><a href="#훈련용-데이터셋으로-최적의-임계값과-최적의-f1-score-구하기." data-toc-modified-id="훈련용-데이터셋으로-최적의-임계값과-최적의-f1-score-구하기.-10.1"><span class="toc-item-num">10.1&nbsp;&nbsp;</span>훈련용 데이터셋으로 최적의 임계값과 최적의 f1-score 구하기.</a></span></li><li><span><a href="#위에서-구한-최적의-임계값을-이용하여-테스트-데이터셋의-확률을-0-또는-1로-변경하기." data-toc-modified-id="위에서-구한-최적의-임계값을-이용하여-테스트-데이터셋의-확률을-0-또는-1로-변경하기.-10.2"><span class="toc-item-num">10.2&nbsp;&nbsp;</span>위에서 구한 최적의 임계값을 이용하여 테스트 데이터셋의 확률을 0 또는 1로 변경하기.</a></span></li></ul></li><li><span><a href="#이진-분류-결과-제출하기." data-toc-modified-id="이진-분류-결과-제출하기.-11"><span class="toc-item-num">11&nbsp;&nbsp;</span>이진 분류 결과 제출하기.</a></span></li></ul></div>

## 패키지 불러오기.

목적 : 일단 Y_LABEL (오일 정상 여부)를 분류하는 것이기 때문에, (0: 정상, 1: 이상) 이진 분류이다.


```python
import torch # pytorch 불러오기.
import torch.nn as nn # 신경망 만들기.
import torch.nn.functional as F # 신경망 만들기.
import torch.optim as optim # optimizer
from torch.utils.data import DataLoader, Dataset # 방대한 데이터를 불러올 때 사용하는 기능이다. 하나씩 불러와서 사용한다.

from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler # 평균 0, 분산 1로 지정한다.
from sklearn.preprocessing import LabelEncoder # 문자를 숫자로 매핑하기.
from sklearn.model_selection import train_test_split # 훈련 데이터와 테스트 데이터 나누기.

import os
import pandas as pd
import numpy as np
from tqdm.auto import tqdm # print를 많이 사용하면, 메모리가 터질 수 있기 때문에 꼭 사용해야한다.
import random

import warnings # 경고창 무시하는 코드
warnings.filterwarnings(action='ignore') 

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') # pytorch에서 GPU 사용 설정하기.
```

## 하이퍼 파라미터 생성하기.


```python
CFG = {
    'EPOCHS': 20, # 전체 훈련 데이터가 학습에 한 번 사용된 주기가 30회이다.
    'LEARNING_RATE':1e-2, # gradient의 보폭을 의미한다.
    'BATCH_SIZE':256, # 전체 데이터를 256개로 나누어서 학습시키기.
    'SEED':30 # 숫자가 중요한 부분이 아니고, 서로 다른 시드를 사용하면 서로 다른 난수를 생성한다는 점 기억하기.
}
```

## seed를 고정하는 코드


```python
# seed를 고정하는 코드이다.
def seed_everything(seed):
    random.seed(seed) # python 자체의 random seed를 고정시키기.
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed) # numpy library의 random seed를 고정시키기.
    # pytorch의 random seed를 고정시키기.
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True # 고정시키면 학습 속도가 느려짐을 알 수 있다. 정확한 모델 성능 재현이 필요하지 않은 경우에는 꺼두기.

seed_everything(CFG['SEED'])
```

## 훈련 데이터와 테스트 데이터 불러오기.


```python
train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')
```

## 데이터 전처리하기.

### 범주형과 수치형 특징 분리하기.


```python
categorical_features = ['COMPONENT_ARBITRARY', 'YEAR']
# Inference(실제 진단 환경)에 사용하는 컬럼
test_stage_features = ['COMPONENT_ARBITRARY', 'ANONYMOUS_1', 'YEAR' , 'ANONYMOUS_2', 'AG', 'CO', 'CR', 'CU', 'FE', 'H2O', 'MN', 'MO', 'NI', 'PQINDEX', 'TI', 'V', 'V40', 'ZN']
```

### 결측치 처리하기.

- 결측치 처리 : 결측치 비율이 20%가 넘는 Column 제거 후 0으로 채워 넣음


```python
train.isna().sum()/len(train)
```




    ID                     0.000000
    COMPONENT_ARBITRARY    0.000000
    ANONYMOUS_1            0.000000
    YEAR                   0.000000
    SAMPLE_TRANSFER_DAY    0.000000
    ANONYMOUS_2            0.000000
    AG                     0.000000
    AL                     0.000000
    B                      0.000000
    BA                     0.000000
    BE                     0.000000
    CA                     0.000000
    CD                     0.098900
    CO                     0.000000
    CR                     0.000000
    CU                     0.000000
    FH2O                   0.724016
    FNOX                   0.724016
    FOPTIMETHGLY           0.724016
    FOXID                  0.724016
    FSO4                   0.724016
    FTBN                   0.724016
    FE                     0.000000
    FUEL                   0.724016
    H2O                    0.000000
    K                      0.163107
    LI                     0.000000
    MG                     0.000000
    MN                     0.000000
    MO                     0.000000
    NA                     0.000000
    NI                     0.000000
    P                      0.000000
    PB                     0.000000
    PQINDEX                0.000000
    S                      0.000000
    SB                     0.000000
    SI                     0.000000
    SN                     0.000000
    SOOTPERCENTAGE         0.724016
    TI                     0.000000
    U100                   0.835686
    U75                    0.835686
    U50                    0.835686
    U25                    0.835686
    U20                    0.835686
    U14                    0.849734
    U6                     0.849734
    U4                     0.849734
    V                      0.000000
    V100                   0.735793
    V40                    0.000000
    ZN                     0.000000
    Y_LABEL                0.000000
    dtype: float64




```python
[train.isna().sum()/len(train) > 0.2]
```




    [ID                     False
     COMPONENT_ARBITRARY    False
     ANONYMOUS_1            False
     YEAR                   False
     SAMPLE_TRANSFER_DAY    False
     ANONYMOUS_2            False
     AG                     False
     AL                     False
     B                      False
     BA                     False
     BE                     False
     CA                     False
     CD                     False
     CO                     False
     CR                     False
     CU                     False
     FH2O                    True
     FNOX                    True
     FOPTIMETHGLY            True
     FOXID                   True
     FSO4                    True
     FTBN                    True
     FE                     False
     FUEL                    True
     H2O                    False
     K                      False
     LI                     False
     MG                     False
     MN                     False
     MO                     False
     NA                     False
     NI                     False
     P                      False
     PB                     False
     PQINDEX                False
     S                      False
     SB                     False
     SI                     False
     SN                     False
     SOOTPERCENTAGE          True
     TI                     False
     U100                    True
     U75                     True
     U50                     True
     U25                     True
     U20                     True
     U14                     True
     U6                      True
     U4                      True
     V                      False
     V100                    True
     V40                    False
     ZN                     False
     Y_LABEL                False
     dtype: bool]




```python
train=train.drop(columns = train.columns[train.isna().sum()/len(train) > 0.2], axis = 1)
train.shape
```




    (14095, 37)




```python
train = train.fillna(0) # 결측치에 0 대입하기.
test = test.fillna(0)
```

### 데이터 분포를 시각화를 통해 알아보기.


```python
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import seaborn as sns
```


```python
train2 = train.drop(train[categorical_features], axis=1)
train2['Y_LABEL'] = train['Y_LABEL']

corr = train2.corr()
```


```python
plt.figure(figsize=(50,50))
sns.heatmap(data=corr, annot=True, fmt='.2f', linewidths=.5, cmap='Blues')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f5c0f5b47c0>




    
![png](output_22_1.png)
    



```python
## 상관계수 순위 

train_corr2 = corr.unstack()
train_corr_rank = pd.DataFrame(train_corr2['Y_LABEL'].sort_values(ascending=False), columns=['Y_LABEL'])
train_corr_rank.style.background_gradient(cmap='viridis')

## Y_LABEL과 가장 연관성이 큰 feature는 AL
```




<style type="text/css">
#T_b39a1_row0_col0 {
  background-color: #fde725;
  color: #000000;
}
#T_b39a1_row1_col0 {
  background-color: #25848e;
  color: #f1f1f1;
}
#T_b39a1_row2_col0 {
  background-color: #3e4989;
  color: #f1f1f1;
}
#T_b39a1_row3_col0 {
  background-color: #443b84;
  color: #f1f1f1;
}
#T_b39a1_row4_col0, #T_b39a1_row5_col0 {
  background-color: #443a83;
  color: #f1f1f1;
}
#T_b39a1_row6_col0 {
  background-color: #453882;
  color: #f1f1f1;
}
#T_b39a1_row7_col0, #T_b39a1_row8_col0, #T_b39a1_row9_col0, #T_b39a1_row10_col0 {
  background-color: #453581;
  color: #f1f1f1;
}
#T_b39a1_row11_col0, #T_b39a1_row12_col0, #T_b39a1_row13_col0, #T_b39a1_row14_col0 {
  background-color: #463480;
  color: #f1f1f1;
}
#T_b39a1_row15_col0, #T_b39a1_row16_col0 {
  background-color: #46327e;
  color: #f1f1f1;
}
#T_b39a1_row17_col0, #T_b39a1_row18_col0, #T_b39a1_row19_col0 {
  background-color: #46307e;
  color: #f1f1f1;
}
#T_b39a1_row20_col0, #T_b39a1_row21_col0 {
  background-color: #472f7d;
  color: #f1f1f1;
}
#T_b39a1_row22_col0, #T_b39a1_row23_col0, #T_b39a1_row24_col0 {
  background-color: #472e7c;
  color: #f1f1f1;
}
#T_b39a1_row25_col0, #T_b39a1_row26_col0, #T_b39a1_row27_col0 {
  background-color: #472d7b;
  color: #f1f1f1;
}
#T_b39a1_row28_col0, #T_b39a1_row29_col0 {
  background-color: #472c7a;
  color: #f1f1f1;
}
#T_b39a1_row30_col0 {
  background-color: #482677;
  color: #f1f1f1;
}
#T_b39a1_row31_col0 {
  background-color: #482576;
  color: #f1f1f1;
}
#T_b39a1_row32_col0 {
  background-color: #482475;
  color: #f1f1f1;
}
#T_b39a1_row33_col0 {
  background-color: #440154;
  color: #f1f1f1;
}
</style>
<table id="T_b39a1_" class="dataframe">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th class="col_heading level0 col0" >Y_LABEL</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_b39a1_level0_row0" class="row_heading level0 row0" >Y_LABEL</th>
      <td id="T_b39a1_row0_col0" class="data row0 col0" >1.000000</td>
    </tr>
    <tr>
      <th id="T_b39a1_level0_row1" class="row_heading level0 row1" >AL</th>
      <td id="T_b39a1_row1_col0" class="data row1 col0" >0.370512</td>
    </tr>
    <tr>
      <th id="T_b39a1_level0_row2" class="row_heading level0 row2" >BA</th>
      <td id="T_b39a1_row2_col0" class="data row2 col0" >0.104840</td>
    </tr>
    <tr>
      <th id="T_b39a1_level0_row3" class="row_heading level0 row3" >FE</th>
      <td id="T_b39a1_row3_col0" class="data row3 col0" >0.047992</td>
    </tr>
    <tr>
      <th id="T_b39a1_level0_row4" class="row_heading level0 row4" >NI</th>
      <td id="T_b39a1_row4_col0" class="data row4 col0" >0.046806</td>
    </tr>
    <tr>
      <th id="T_b39a1_level0_row5" class="row_heading level0 row5" >ANONYMOUS_1</th>
      <td id="T_b39a1_row5_col0" class="data row5 col0" >0.044197</td>
    </tr>
    <tr>
      <th id="T_b39a1_level0_row6" class="row_heading level0 row6" >SI</th>
      <td id="T_b39a1_row6_col0" class="data row6 col0" >0.036731</td>
    </tr>
    <tr>
      <th id="T_b39a1_level0_row7" class="row_heading level0 row7" >PQINDEX</th>
      <td id="T_b39a1_row7_col0" class="data row7 col0" >0.028966</td>
    </tr>
    <tr>
      <th id="T_b39a1_level0_row8" class="row_heading level0 row8" >S</th>
      <td id="T_b39a1_row8_col0" class="data row8 col0" >0.027923</td>
    </tr>
    <tr>
      <th id="T_b39a1_level0_row9" class="row_heading level0 row9" >TI</th>
      <td id="T_b39a1_row9_col0" class="data row9 col0" >0.025637</td>
    </tr>
    <tr>
      <th id="T_b39a1_level0_row10" class="row_heading level0 row10" >CU</th>
      <td id="T_b39a1_row10_col0" class="data row10 col0" >0.024975</td>
    </tr>
    <tr>
      <th id="T_b39a1_level0_row11" class="row_heading level0 row11" >MN</th>
      <td id="T_b39a1_row11_col0" class="data row11 col0" >0.024274</td>
    </tr>
    <tr>
      <th id="T_b39a1_level0_row12" class="row_heading level0 row12" >V40</th>
      <td id="T_b39a1_row12_col0" class="data row12 col0" >0.023195</td>
    </tr>
    <tr>
      <th id="T_b39a1_level0_row13" class="row_heading level0 row13" >K</th>
      <td id="T_b39a1_row13_col0" class="data row13 col0" >0.020959</td>
    </tr>
    <tr>
      <th id="T_b39a1_level0_row14" class="row_heading level0 row14" >V</th>
      <td id="T_b39a1_row14_col0" class="data row14 col0" >0.020862</td>
    </tr>
    <tr>
      <th id="T_b39a1_level0_row15" class="row_heading level0 row15" >AG</th>
      <td id="T_b39a1_row15_col0" class="data row15 col0" >0.014671</td>
    </tr>
    <tr>
      <th id="T_b39a1_level0_row16" class="row_heading level0 row16" >CR</th>
      <td id="T_b39a1_row16_col0" class="data row16 col0" >0.014233</td>
    </tr>
    <tr>
      <th id="T_b39a1_level0_row17" class="row_heading level0 row17" >BE</th>
      <td id="T_b39a1_row17_col0" class="data row17 col0" >0.010685</td>
    </tr>
    <tr>
      <th id="T_b39a1_level0_row18" class="row_heading level0 row18" >CO</th>
      <td id="T_b39a1_row18_col0" class="data row18 col0" >0.008175</td>
    </tr>
    <tr>
      <th id="T_b39a1_level0_row19" class="row_heading level0 row19" >P</th>
      <td id="T_b39a1_row19_col0" class="data row19 col0" >0.007602</td>
    </tr>
    <tr>
      <th id="T_b39a1_level0_row20" class="row_heading level0 row20" >CD</th>
      <td id="T_b39a1_row20_col0" class="data row20 col0" >0.003960</td>
    </tr>
    <tr>
      <th id="T_b39a1_level0_row21" class="row_heading level0 row21" >LI</th>
      <td id="T_b39a1_row21_col0" class="data row21 col0" >0.002921</td>
    </tr>
    <tr>
      <th id="T_b39a1_level0_row22" class="row_heading level0 row22" >SN</th>
      <td id="T_b39a1_row22_col0" class="data row22 col0" >0.002359</td>
    </tr>
    <tr>
      <th id="T_b39a1_level0_row23" class="row_heading level0 row23" >MO</th>
      <td id="T_b39a1_row23_col0" class="data row23 col0" >0.001206</td>
    </tr>
    <tr>
      <th id="T_b39a1_level0_row24" class="row_heading level0 row24" >SB</th>
      <td id="T_b39a1_row24_col0" class="data row24 col0" >-0.002028</td>
    </tr>
    <tr>
      <th id="T_b39a1_level0_row25" class="row_heading level0 row25" >PB</th>
      <td id="T_b39a1_row25_col0" class="data row25 col0" >-0.003549</td>
    </tr>
    <tr>
      <th id="T_b39a1_level0_row26" class="row_heading level0 row26" >H2O</th>
      <td id="T_b39a1_row26_col0" class="data row26 col0" >-0.004262</td>
    </tr>
    <tr>
      <th id="T_b39a1_level0_row27" class="row_heading level0 row27" >SAMPLE_TRANSFER_DAY</th>
      <td id="T_b39a1_row27_col0" class="data row27 col0" >-0.004315</td>
    </tr>
    <tr>
      <th id="T_b39a1_level0_row28" class="row_heading level0 row28" >MG</th>
      <td id="T_b39a1_row28_col0" class="data row28 col0" >-0.008807</td>
    </tr>
    <tr>
      <th id="T_b39a1_level0_row29" class="row_heading level0 row29" >NA</th>
      <td id="T_b39a1_row29_col0" class="data row29 col0" >-0.010820</td>
    </tr>
    <tr>
      <th id="T_b39a1_level0_row30" class="row_heading level0 row30" >ZN</th>
      <td id="T_b39a1_row30_col0" class="data row30 col0" >-0.027551</td>
    </tr>
    <tr>
      <th id="T_b39a1_level0_row31" class="row_heading level0 row31" >B</th>
      <td id="T_b39a1_row31_col0" class="data row31 col0" >-0.029787</td>
    </tr>
    <tr>
      <th id="T_b39a1_level0_row32" class="row_heading level0 row32" >ANONYMOUS_2</th>
      <td id="T_b39a1_row32_col0" class="data row32 col0" >-0.033641</td>
    </tr>
    <tr>
      <th id="T_b39a1_level0_row33" class="row_heading level0 row33" >CA</th>
      <td id="T_b39a1_row33_col0" class="data row33 col0" >-0.150379</td>
    </tr>
  </tbody>
</table>





```python
#연관성이 가장 높은 6가지 열의 boxplot

corr_features = [ "AL", "BA","FE","NI","ANONYMOUS_1","SI"]

for col in corr_features :
    plt.figure(figsize=(12, 8))
    plt.boxplot(train[col], sym='b')
    plt.title(col)
    print(" ")

    plt.show()

#이상치가 많이 존재해서 제거하지 않고 그대로 시각화
```

     
    


    
![png](output_24_1.png)
    


     
    


    
![png](output_24_3.png)
    


     
    


    
![png](output_24_5.png)
    


     
    


    
![png](output_24_7.png)
    


     
    


    
![png](output_24_9.png)
    


     
    


    
![png](output_24_11.png)
    



```python
# 카테고리형 변수 시각화
plt.figure(figsize=(12, 8))
sns.countplot(x = 'YEAR',data = train)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f5b3ddcb760>




    
![png](output_25_1.png)
    



```python
plt.figure(figsize=(12, 8))
sns.countplot(x = 'COMPONENT_ARBITRARY',data = train)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f5b3dd92ca0>




    
![png](output_26_1.png)
    



```python
# 수치형은 이렇게 나옴
plt.figure(figsize=(12, 8))
sns.countplot(x = 'ANONYMOUS_1',data = train)
plt.ylim(0, 65)
```




    (0.0, 65.0)




    
![png](output_27_1.png)
    


### 훈련 데이터와 검증 데이터 분할하기.

- train_X, train_y : 훈련 데이터
- val_X, val_y : 검증 데이터
- test : 테스트 데이터 (성능 평가에 이용할 것이다.)


```python
all_X = train.drop(['ID', 'Y_LABEL'], axis = 1) # 모든 x 데이터에 ID, Y_LABEL 제거시키기.
all_y = train['Y_LABEL'] # 모든 y 데이터에 Y_LABEL만 남겨놓기.

test = test.drop(['ID'], axis = 1) # test 데이터에 ID 제거시키기.

train_X, val_X, train_y, val_y = train_test_split(all_X, all_y, test_size=0.2, random_state=CFG['SEED'], stratify=all_y)
# train : test = 8 : 2로 지정하고, 아까 고정시킨 seed를 random_state로 지정하고, y 데이터에서 계층적 데이터 추출 옵션 (분류 모델에서 추천!)
```

### knn

- train, test에는 문자형 데이터가 있다. 이를 수치화 하는 코드가 필요하다.


```python
# 2. 인코딩 : 문자를 수치화하기.
le = LabelEncoder()
for col in categorical_features: # 카테고리 특징 불러오기.  
    train_X[col] = le.fit_transform(train_X[col]) # 훈련 데이터 문자에서 수치화하기.
    val_X[col] = le.transform(val_X[col]) # 검증 데이터 문자에서 수치화하기.
    if col in test.columns: # 테스트 칼럼 불러오기.
        test[col] = le.transform(test[col]) # test data에서는 fit_transform을 이용하지 않는다.
```


```python
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5)
```


```python
classifier.fit(train_X, train_y)
```




    KNeighborsClassifier()




```python
print(classifier.score(val_X, val_y))
```

    0.908478183753104
    


```python
import matplotlib.pyplot as plt
k_list = range(1,101)
accuracies = []
for k in k_list:
    classifier = KNeighborsClassifier(n_neighbors = k)
    classifier.fit(train_X, train_y)
    accuracies.append(classifier.score(val_X, val_y))
plt.plot(k_list, accuracies)
plt.xlabel("k")
plt.ylabel("Validation Accuracy")
plt.title("Breast Cancer Classifier Accuracy")
plt.show()
```


    
![png](output_36_0.png)
    



```python
from sklearn.impute import KNNImputer

#임퓨터 선언(5개의 평균으로 계산하겠다)
imputer=KNNImputer(n_neighbors=5)

#임퓨터를 사용하여 filled_train으로 저장 이후 같은 임퓨터를 사용할때는 imputer.transform()으로 사용하면됨
filled_train=imputer.fit_transform(train_X)

#사용하면 array값으로 나오기때문에 dataframe으로 바꿔주고 컬럼을가져옴
filled_train=pd.DataFrame(filled_train, columns=train_X.columns)
# print(filled_train)

#임퓨터 선언(5개의 평균으로 계산하겠다)
imputer=KNNImputer(n_neighbors=5)

#임퓨터를 사용하여 filled_train으로 저장 이후 같은 임퓨터를 사용할때는 imputer.transform()으로 사용하면됨
filled_val_X=imputer.fit_transform(val_X)

#사용하면 array값으로 나오기때문에 dataframe으로 바꿔주고 컬럼을가져옴
filled_val_X=pd.DataFrame(filled_val_X, columns=val_X.columns)
# print(filled_val_X)
```

### 스케일링 (표준화) 진행하기.


```python
def get_values(value):
    return value.values.reshape(-1, 1) # 열을 1개로 만들어주기.

#데이터 전처리 : 평균 0, 분산 1로 만들어주기.
for col in train_X.columns: # 훈련용 X 데이터 칼럼 불러오기.
    if col not in categorical_features: # 'COMPONENT_ARBITRARY', 'YEAR'가 없으면
        scaler = StandardScaler() # 평균 0, 분산 1로 만들어주기. (데이터 전처리하기.)
        train_X[col] = scaler.fit_transform(get_values(train_X[col])) # train_X[col]의 value 값 가져와서 fit()을 통해서 설정한 뒤에 이를 기반으로 학습 데이터의 transform()을 수행하되 학습 데이터에서 설정된 변환을 위한 기반 설정을 그대로 테스트 데이터에도 적용하기 위해서입니다.
        val_X[col] = scaler.transform(get_values(val_X[col])) # 검증 데이터도 마찬가지로 처리시켜주기.
        if col in test.columns: # 테스트 데이터 칼럼 불러오기.
            test[col] = scaler.transform(get_values(test[col])) # test data에서는 fit_transform을 이용하지 않는다. 

```

## 사용자 정의 데이터셋 만들기.

### 사용자 정의 데이터셋 만드는 과정

1. 아까 만든 train_X, train_y, val_X, val_y를 이용하여 CustomDataset (사용자 정의 데이터셋) 을 만들어주고, (RAM 터지지 않게 하기 위함이다.)
2. DataLoader를 이용하여 샘플에 쉽게 접근할 수 있도록 순회 가능한 객체(iterable)로 감싸기. (목표는 데이터에 더 쉽게 접근할 수 있도록 하기.)

### 사용자 정의 데이터셋 만드는 코드

사용자 정의 데이터셋 클래스는 3개의 함수로 반드시 구현해야 한다.

- __ init __
- __ len __
- __ getitem __

사용자 정의 데이터셋 만드는 이유 : 모든 데이터를 한번에 불러와서 학습시키면 RAM 터질 확률이 크다. 그래서 조금씩 가져오는 사용자 정의 데이터셋이 필요하다.


```python
class CustomDataset(Dataset): # 사용자 정의 데이터셋 만드는 클래스 (붕어빵 틀)
    # RAM 터지지 않게 하기 위하여 사용자 정의 데이터셋 만드는 과정이다.
    def __init__(self, data_X, data_y, distillation=False): # 객체 생성 시 1번 이용한다.
        super(CustomDataset, self).__init__()
        self.data_X = data_X # input feature
        self.data_y = data_y # label 혹은 정답
        self.distillation = distillation # 증류
        
    def __len__(self): # 데이터 샘플 개수를 반환한다.
        return len(self.data_X)
    
    def __getitem__(self, index): # 주어진 인덱스 idx 에 해당하는 샘플을 데이터셋에서 불러오고 반환한다.
        if self.distillation:
            # 지식 증류 학습 시
            teacher_X = torch.Tensor(self.data_X.iloc[index])
            student_X = torch.Tensor(self.data_X[test_stage_features].iloc[index])
            y = self.data_y.values[index]
            return teacher_X, student_X, y
        else: # 지식 증류 학습 아닐 시
            if self.data_y is None:
                test_X = torch.Tensor(self.data_X.iloc[index])
                return test_X
            else:
                teacher_X = torch.Tensor(self.data_X.iloc[index])
                y = self.data_y.values[index]
                return teacher_X, y
```

훈련용 사용자 데이터셋 객체와 검증용 사용자 데이터셋 객체를 생성하였다.


```python
train_dataset = CustomDataset(train_X, train_y, False) # 훈련 사용자 데이터셋 객체
val_dataset = CustomDataset(val_X, val_y, False) # 검증 사용자 데이터셋 객체
```

### 간편한 API DataLoader

DataLoader로 학습용 데이터를 준비할 수 있다.

Dataset은 데이터셋의 특징 가져오고, 하나의 샘플에 정답을 지정하는 일을 한번에 한다. 모델을 학습할 때 일반적으로 샘플들을 미니배치로 전달하고, 매 에폭마다 데이터를 다시 섞어서 과적합을 막고, multiprocessing을 사용하여 데이터 검색 속도를 높인다. 복잡한 과정을 간단한 API로 만든 것이다.


```python
train_loader = DataLoader(train_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=False)
```

## 딥러닝 모델 지식의 증류 기법 : 교사 - 학생 지식 증류 기법

* 딥러닝 모델 지식의 증류 기법 중에서 교사 - 학생 지식 증류 기법을 이용할 것이다.
* 지식 증류 (모델 경량화) : 잘 학습된 모델 경향성(soft label)을 학습하는 것이다. 두 가지 모델을 불러오고, 두 가지 loss를 합치는 것 외에는 모든 과정이 동일하다.
* Teacher Model : 이미 학습된 무거운 모델
* Student Model : 학습할 가벼운 모델

## 이미 학습된 무거운 모델, 교사 모델 정의하기.

### Teacher Model 신경망 구현하기. (순전파를 이용하여 예측값 구하기.)


```python
class Teacher(nn.Module): # 교사 모델 신경망 만드는 코드이다. (학습용 feature 데이터를 넣어서 예측값 구하기.)
    def __init__(self): # 생성자에서 사용하려는 모든 레이어를 선언하기.
        super(Teacher, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features=35, out_features=256), # 단순 선형 회귀, input_size=35, output_size=256
            nn.BatchNorm1d(256), # 1 Dim batch normalization
            nn.LeakyReLU(), # dying Relu 현상 해결하기 위함이다. (0보다 작은 구간은 기울기가 0이기 때문이다.)
            nn.Linear(in_features=256, out_features=1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Linear(in_features=1024, out_features=256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(in_features=256, out_features=1),
            nn.Sigmoid()
        )
        
    def forward(self, x): # 입력부터 출력까지 모델이 실행되는 방식을 선언하기.
        output = self.classifier(x)
        return output
```

이 코드는 단순 선형 회귀와 1차원 배치 정규화 (데이터 분포를 맞추기 위한 해결책), 그리고 활성화 함수는 leaky Relu와 Sigmoid 함수를 이용하였다.

- epoch : 학습 데이터를 전체적으로 한번 학습하는 것을 의미한다.
- batch : 학습 데이터를 일정한 개수로 나눈 횟수를 의미한다.

하지만 batch 단위로 학습하는 경우 문제점이 생긴다. (데이터의 분포에 대한 문제) 그래서 batch-normalization을 이용하여 데이터의 분포를 맞춰준다.

###  훈련 데이터 교사 모델과 검증 데이터 교사 모델 이용하여 loss를 최소로 하는 가중치 구하기.

**딥러닝 학습 순서**
1. 학습용 feature data를 이용하여 예측값 구하기. (순전파)
2. 예측값과 실제값 사이의 오차 구하기. (loss, 손실 함수 값)
3. loss를 줄일 수 있는 가중치 값을 업데이트하기. (역전파)
4. 1 ~ 3번을 계속 반복하여, loss를 최소로 하는 가중치 값을 구하기.


```python
def train(model, optimizer, train_loader, val_loader, scheduler, device): # 학습 데이터에 대한 교사 모델 학습하기.
    model.to(device)

    best_score = 0
    best_model = None
    criterion = nn.BCELoss().to(device)

    for epoch in range(CFG["EPOCHS"]):
        train_loss = [] # 학습 데이터 손실함수 구하기.
  
        model.train() # 모델을 학습시키기. 
        for X, y in tqdm(train_loader): # tqdm : enumerate 대신 사용하기. DataLoader로 학습 데이터 준비하기.
            X = X.float().to(device)
            y = y.float().to(device)
            
            optimizer.zero_grad()
            
            y_pred = model(X) # 모델을 이용한 예측값 구하기.
            
            loss = criterion(y_pred, y.reshape(-1, 1))
            loss.backward() # 역전파를 이용하여 가중치를 업데이트하기.
            
            optimizer.step()

            train_loss.append(loss.item()) # 손실 함수 정의하기.

        val_loss, val_score = validation_teacher(model, val_loader, criterion, device)
        # 검증 데이터를 이용한 교사 모델 이용하여 손실함수 값과 점수 구하기.
        print(f'Epoch [{epoch}], Train Loss : [{np.mean(train_loss) :.5f}] Val Loss : [{np.mean(val_loss) :.5f}] Val F1 Score : [{val_score:.5f}]')

        if scheduler is not None:
            scheduler.step(val_score)
            
        if best_score < val_score: # score가 작으면 작을수록 좋은 모델이다.
            best_model = model
            best_score = val_score
        
    return best_model 
```

- f1-score : 모델 성능을 평가하는 지표
- loss (손실 함수 값) : 실제값과 예측값 사이의 오차


```python
def competition_metric(true, pred):
    return f1_score(true, pred, average="macro")

def validation_teacher(model, val_loader, criterion, device):
    model.eval()

    val_loss = []
    pred_labels = []
    true_labels = []
    threshold = 0.35
    
    with torch.no_grad():
        for X, y in tqdm(val_loader):
            X = X.float().to(device)
            y = y.float().to(device)
            
            model_pred = model(X.to(device)) # 모델을 이용한 예측값 구하기.
            
            loss = criterion(model_pred, y.reshape(-1, 1))
            val_loss.append(loss.item()) # 검증 데이터의 손실 함수 값 구하기.     
            
            model_pred = model_pred.squeeze(1).to('cpu')  
            pred_labels += model_pred.tolist() # 예측 라벨 구하기.
            true_labels += y.tolist() # 실제 라벨 구하기.
        
        pred_labels = np.where(np.array(pred_labels) > threshold, 1, 0) # 예측 라벨 구하기. 
        val_f1 = competition_metric(true_labels, pred_labels) # 실제 라벨과 예측 라벨 비교하기.
    return val_loss, val_f1 # 검증 데이터 손실함수 값, f1-score 값
```

###  교사 모델 돌리기.


```python
model = Teacher() # 교사 모델 객체 만들기.
model.eval()
optimizer = torch.optim.Adam(model.parameters(), lr=CFG['LEARNING_RATE']) # optimizer를 Adam 이용하기.
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1, threshold_mode='abs',min_lr=1e-8, verbose=True)

teacher_model = train(model, optimizer, train_loader, val_loader, scheduler, device)
```


      0%|          | 0/45 [00:00<?, ?it/s]



      0%|          | 0/12 [00:00<?, ?it/s]


    Epoch [0], Train Loss : [0.25404] Val Loss : [0.19905] Val F1 Score : [0.76579]
    


      0%|          | 0/45 [00:00<?, ?it/s]



      0%|          | 0/12 [00:00<?, ?it/s]


    Epoch [1], Train Loss : [0.19513] Val Loss : [0.17333] Val F1 Score : [0.80909]
    


      0%|          | 0/45 [00:00<?, ?it/s]



      0%|          | 0/12 [00:00<?, ?it/s]


    Epoch [2], Train Loss : [0.17876] Val Loss : [0.16549] Val F1 Score : [0.80042]
    


      0%|          | 0/45 [00:00<?, ?it/s]



      0%|          | 0/12 [00:00<?, ?it/s]


    Epoch [3], Train Loss : [0.16885] Val Loss : [0.19041] Val F1 Score : [0.78102]
    Epoch 00004: reducing learning rate of group 0 to 5.0000e-03.
    


      0%|          | 0/45 [00:00<?, ?it/s]



      0%|          | 0/12 [00:00<?, ?it/s]


    Epoch [4], Train Loss : [0.16466] Val Loss : [0.16072] Val F1 Score : [0.80911]
    


      0%|          | 0/45 [00:00<?, ?it/s]



      0%|          | 0/12 [00:00<?, ?it/s]


    Epoch [5], Train Loss : [0.15973] Val Loss : [0.15999] Val F1 Score : [0.80530]
    Epoch 00006: reducing learning rate of group 0 to 2.5000e-03.
    


      0%|          | 0/45 [00:00<?, ?it/s]



      0%|          | 0/12 [00:00<?, ?it/s]


    Epoch [6], Train Loss : [0.15312] Val Loss : [0.16422] Val F1 Score : [0.81488]
    


      0%|          | 0/45 [00:00<?, ?it/s]



      0%|          | 0/12 [00:00<?, ?it/s]


    Epoch [7], Train Loss : [0.14002] Val Loss : [0.16881] Val F1 Score : [0.80809]
    


      0%|          | 0/45 [00:00<?, ?it/s]



      0%|          | 0/12 [00:00<?, ?it/s]


    Epoch [8], Train Loss : [0.13696] Val Loss : [0.16817] Val F1 Score : [0.80717]
    Epoch 00009: reducing learning rate of group 0 to 1.2500e-03.
    


      0%|          | 0/45 [00:00<?, ?it/s]



      0%|          | 0/12 [00:00<?, ?it/s]


    Epoch [9], Train Loss : [0.13463] Val Loss : [0.16843] Val F1 Score : [0.80355]
    


      0%|          | 0/45 [00:00<?, ?it/s]



      0%|          | 0/12 [00:00<?, ?it/s]


    Epoch [10], Train Loss : [0.12960] Val Loss : [0.16746] Val F1 Score : [0.80536]
    Epoch 00011: reducing learning rate of group 0 to 6.2500e-04.
    


      0%|          | 0/45 [00:00<?, ?it/s]



      0%|          | 0/12 [00:00<?, ?it/s]


    Epoch [11], Train Loss : [0.12577] Val Loss : [0.17013] Val F1 Score : [0.79607]
    


      0%|          | 0/45 [00:00<?, ?it/s]



      0%|          | 0/12 [00:00<?, ?it/s]


    Epoch [12], Train Loss : [0.12194] Val Loss : [0.17080] Val F1 Score : [0.80714]
    Epoch 00013: reducing learning rate of group 0 to 3.1250e-04.
    


      0%|          | 0/45 [00:00<?, ?it/s]



      0%|          | 0/12 [00:00<?, ?it/s]


    Epoch [13], Train Loss : [0.12446] Val Loss : [0.16965] Val F1 Score : [0.80887]
    


      0%|          | 0/45 [00:00<?, ?it/s]



      0%|          | 0/12 [00:00<?, ?it/s]


    Epoch [14], Train Loss : [0.12167] Val Loss : [0.17113] Val F1 Score : [0.80277]
    Epoch 00015: reducing learning rate of group 0 to 1.5625e-04.
    


      0%|          | 0/45 [00:00<?, ?it/s]



      0%|          | 0/12 [00:00<?, ?it/s]


    Epoch [15], Train Loss : [0.12169] Val Loss : [0.17051] Val F1 Score : [0.79419]
    


      0%|          | 0/45 [00:00<?, ?it/s]



      0%|          | 0/12 [00:00<?, ?it/s]


    Epoch [16], Train Loss : [0.11998] Val Loss : [0.17401] Val F1 Score : [0.79603]
    Epoch 00017: reducing learning rate of group 0 to 7.8125e-05.
    


      0%|          | 0/45 [00:00<?, ?it/s]



      0%|          | 0/12 [00:00<?, ?it/s]


    Epoch [17], Train Loss : [0.11812] Val Loss : [0.16944] Val F1 Score : [0.80806]
    


      0%|          | 0/45 [00:00<?, ?it/s]



      0%|          | 0/12 [00:00<?, ?it/s]


    Epoch [18], Train Loss : [0.12036] Val Loss : [0.17244] Val F1 Score : [0.78809]
    Epoch 00019: reducing learning rate of group 0 to 3.9063e-05.
    


      0%|          | 0/45 [00:00<?, ?it/s]



      0%|          | 0/12 [00:00<?, ?it/s]


    Epoch [19], Train Loss : [0.11913] Val Loss : [0.17177] Val F1 Score : [0.80709]
    

## 학생 모델 정의하기.

다시 정리하자면 순전파를 이용하여 예측값을 구하고, 예측값과 실제값의 오차인 손실 함수 값을 구하고, 역전파를 이용하여 손실 함수가 최소가 되는 가중치를 찾아야 한다.

### 학생 모델 신경망 구현하기. (순전파를 이용하여 예측값 구하기.)


```python
class Student(nn.Module): # 학생 모델 신경망 만드는 코드이다.
    def __init__(self): # 생성자에서 사용하려는 모든 레이어를 선언하기.
        super(Student, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features=18, out_features=128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(in_features=128, out_features=512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(in_features=512, out_features=128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(in_features=128, out_features=1),
            nn.Sigmoid()
        )
        
    def forward(self, x): # 입력부터 출력까지 모델이 실행되는 방식을 선언하기.        
        output = self.classifier(x)
        return output
```

이 코드는 단순 선형 회귀, 1차원 배치 정규화, 그리고 Leaky Relu와 Sigmoid 함수가 이용된다.

### 교사 - 학생 지식 증류 손실 함수 값 정의하기.


```python
# 지식 증류 loss 계산하는 코드이다.
def distillation(student_logits, labels, teacher_logits, alpha):
    distillation_loss = nn.BCELoss()(student_logits, teacher_logits) # BCELoss 손실 함수 이용하기.
    # BCELoss는 마지막 레이어가 Sigmoid나 Softmax인 경우에 이용한다.
    student_loss = nn.BCELoss()(student_logits, labels.reshape(-1, 1)) # 학생 손실 삼수도 BCELoss 손실 함수 이용하기.
    return alpha * student_loss + (1-alpha) * distillation_loss 
```


```python
# 미니 배치 단위로 loss 계산하는 코드이다.
def distill_loss(output, target, teacher_output, loss_fn=distillation, opt=optimizer):
    loss_b = loss_fn(output, target, teacher_output, alpha=0.1)

    if opt is not None:
        opt.zero_grad()
        loss_b.backward()
        opt.step()

    return loss_b.item()
```

### 훈련 데이터 학생 모델과 검증 데이터 학생 모델 이용하여 loss를 최소로 하는 가중치 구하기.


```python
def student_train(s_model, t_model, optimizer, train_loader, val_loader, scheduler, device):
    s_model.to(device)
    t_model.to(device)
    
    best_score = 0
    best_model = None

    for epoch in range(CFG["EPOCHS"]):
        train_loss = []
        s_model.train()
        t_model.eval()
        
        for X_t, X_s, y in tqdm(train_loader):
            X_t = X_t.float().to(device)
            X_s = X_s.float().to(device)
            y = y.float().to(device)
            
            optimizer.zero_grad()

            output = s_model(X_s)
            with torch.no_grad():
                teacher_output = t_model(X_t)
                
            loss_b = distill_loss(output, y, teacher_output, loss_fn=distillation, opt=optimizer)

            train_loss.append(loss_b)

        val_loss, val_score = validation_student(s_model, t_model, val_loader, distill_loss, device)
        print(f'Epoch [{epoch}], Train Loss : [{np.mean(train_loss) :.5f}] Val Loss : [{np.mean(val_loss) :.5f}] Val F1 Score : [{val_score:.5f}]')
        
        if scheduler is not None:
            scheduler.step(val_score)
            
        if best_score < val_score:
            best_model = s_model
            best_score = val_score
        
    return best_model
```


```python
def validation_student(s_model, t_model, val_loader, criterion, device):
    s_model.eval()
    t_model.eval()

    val_loss = []
    pred_labels = []
    true_labels = []
    threshold = 0.35
    
    with torch.no_grad():
        for X_t, X_s, y in tqdm(val_loader):
            X_t = X_t.float().to(device)
            X_s = X_s.float().to(device)
            y = y.float().to(device)
            
            model_pred = s_model(X_s)
            teacher_output = t_model(X_t)
            
            loss_b = distill_loss(model_pred, y, teacher_output, loss_fn=distillation, opt=None)
            val_loss.append(loss_b)
            
            model_pred = model_pred.squeeze(1).to('cpu')
            pred_labels += model_pred.tolist()
            true_labels += y.tolist()
        
        pred_labels = np.where(np.array(pred_labels) > threshold, 1, 0)
        val_f1 = competition_metric(true_labels, pred_labels)
    return val_loss, val_f1    
```

### 학생 모델 돌리기.


```python
train_dataset = CustomDataset(train_X, train_y, True)
val_dataset = CustomDataset(val_X, val_y, True)

train_loader = DataLoader(train_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=False)
```


```python
student_model = Student()
student_model.eval()
optimizer = torch.optim.Adam(student_model.parameters(), lr=CFG['LEARNING_RATE'])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1, threshold_mode='abs',min_lr=1e-8, verbose=True)

best_student_model = student_train(student_model, teacher_model, optimizer, train_loader, val_loader, scheduler, device)
```


      0%|          | 0/45 [00:00<?, ?it/s]



      0%|          | 0/12 [00:00<?, ?it/s]


    Epoch [0], Train Loss : [0.32369] Val Loss : [0.28055] Val F1 Score : [0.47758]
    


      0%|          | 0/45 [00:00<?, ?it/s]



      0%|          | 0/12 [00:00<?, ?it/s]


    Epoch [1], Train Loss : [0.28634] Val Loss : [0.27604] Val F1 Score : [0.49009]
    


      0%|          | 0/45 [00:00<?, ?it/s]



      0%|          | 0/12 [00:00<?, ?it/s]


    Epoch [2], Train Loss : [0.29313] Val Loss : [0.27932] Val F1 Score : [0.49354]
    


      0%|          | 0/45 [00:00<?, ?it/s]



      0%|          | 0/12 [00:00<?, ?it/s]


    Epoch [3], Train Loss : [0.29020] Val Loss : [0.27624] Val F1 Score : [0.48979]
    


      0%|          | 0/45 [00:00<?, ?it/s]



      0%|          | 0/12 [00:00<?, ?it/s]


    Epoch [4], Train Loss : [0.28419] Val Loss : [0.27676] Val F1 Score : [0.48994]
    Epoch 00005: reducing learning rate of group 0 to 5.0000e-03.
    


      0%|          | 0/45 [00:00<?, ?it/s]



      0%|          | 0/12 [00:00<?, ?it/s]


    Epoch [5], Train Loss : [0.28730] Val Loss : [0.27633] Val F1 Score : [0.49946]
    


      0%|          | 0/45 [00:00<?, ?it/s]



      0%|          | 0/12 [00:00<?, ?it/s]


    Epoch [6], Train Loss : [0.28181] Val Loss : [0.27695] Val F1 Score : [0.49331]
    


      0%|          | 0/45 [00:00<?, ?it/s]



      0%|          | 0/12 [00:00<?, ?it/s]


    Epoch [7], Train Loss : [0.27913] Val Loss : [0.27706] Val F1 Score : [0.49607]
    Epoch 00008: reducing learning rate of group 0 to 2.5000e-03.
    


      0%|          | 0/45 [00:00<?, ?it/s]



      0%|          | 0/12 [00:00<?, ?it/s]


    Epoch [8], Train Loss : [0.27583] Val Loss : [0.27513] Val F1 Score : [0.50474]
    


      0%|          | 0/45 [00:00<?, ?it/s]



      0%|          | 0/12 [00:00<?, ?it/s]


    Epoch [9], Train Loss : [0.27357] Val Loss : [0.27445] Val F1 Score : [0.49765]
    


      0%|          | 0/45 [00:00<?, ?it/s]



      0%|          | 0/12 [00:00<?, ?it/s]


    Epoch [10], Train Loss : [0.27473] Val Loss : [0.27392] Val F1 Score : [0.49447]
    Epoch 00011: reducing learning rate of group 0 to 1.2500e-03.
    


      0%|          | 0/45 [00:00<?, ?it/s]



      0%|          | 0/12 [00:00<?, ?it/s]


    Epoch [11], Train Loss : [0.27339] Val Loss : [0.27486] Val F1 Score : [0.49514]
    


      0%|          | 0/45 [00:00<?, ?it/s]



      0%|          | 0/12 [00:00<?, ?it/s]


    Epoch [12], Train Loss : [0.27851] Val Loss : [0.27598] Val F1 Score : [0.49616]
    Epoch 00013: reducing learning rate of group 0 to 6.2500e-04.
    


      0%|          | 0/45 [00:00<?, ?it/s]



      0%|          | 0/12 [00:00<?, ?it/s]


    Epoch [13], Train Loss : [0.27191] Val Loss : [0.27507] Val F1 Score : [0.49712]
    


      0%|          | 0/45 [00:00<?, ?it/s]



      0%|          | 0/12 [00:00<?, ?it/s]


    Epoch [14], Train Loss : [0.27214] Val Loss : [0.27454] Val F1 Score : [0.49730]
    Epoch 00015: reducing learning rate of group 0 to 3.1250e-04.
    


      0%|          | 0/45 [00:00<?, ?it/s]



      0%|          | 0/12 [00:00<?, ?it/s]


    Epoch [15], Train Loss : [0.26982] Val Loss : [0.27433] Val F1 Score : [0.49211]
    


      0%|          | 0/45 [00:00<?, ?it/s]



      0%|          | 0/12 [00:00<?, ?it/s]


    Epoch [16], Train Loss : [0.27058] Val Loss : [0.27444] Val F1 Score : [0.49480]
    Epoch 00017: reducing learning rate of group 0 to 1.5625e-04.
    


      0%|          | 0/45 [00:00<?, ?it/s]



      0%|          | 0/12 [00:00<?, ?it/s]


    Epoch [17], Train Loss : [0.27104] Val Loss : [0.27405] Val F1 Score : [0.49910]
    


      0%|          | 0/45 [00:00<?, ?it/s]



      0%|          | 0/12 [00:00<?, ?it/s]


    Epoch [18], Train Loss : [0.27385] Val Loss : [0.27499] Val F1 Score : [0.49873]
    Epoch 00019: reducing learning rate of group 0 to 7.8125e-05.
    


      0%|          | 0/45 [00:00<?, ?it/s]



      0%|          | 0/12 [00:00<?, ?it/s]


    Epoch [19], Train Loss : [0.27273] Val Loss : [0.27476] Val F1 Score : [0.49855]
    

## 임계값 추론하기.

모델링을 통해 구한 확률을 임계값을 기준으로 0 또는 1로 변경한다. (정보 손실)

### 훈련용 데이터셋으로 최적의 임계값과 최적의 f1-score 구하기.


```python
def choose_threshold(model, val_loader, device):
    model.to(device)
    model.eval()
    
    thresholds = [0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    pred_labels = [] # 예측 라벨
    true_labels = [] # 실제 라벨
    
    best_score = 0
    best_thr = None
    with torch.no_grad():
        for _, x_s, y in tqdm(iter(val_loader)):
            x_s = x_s.float().to(device)
            y = y.float().to(device)
            
            model_pred = model(x_s)
            
            model_pred = model_pred.squeeze(1).to('cpu')
            pred_labels += model_pred.tolist()
            true_labels += y.tolist()
        
        for threshold in thresholds:
            pred_labels_thr = np.where(np.array(pred_labels) > threshold, 1, 0) # 예측 라벨을 임계값보다 크면 1, 작으면 0으로 변경하기.
            score_thr = competition_metric(true_labels, pred_labels_thr) # competitiom_metric : f1-score를 리턴해주는 함수이다.
            if best_score < score_thr:
                best_score = score_thr
                best_thr = threshold
    return best_thr, best_score # 계속 반복하여 최적의 임계값과 최적의 f1-score를 리턴한다.
```

최적의 f1-score (모델 평가 지표에 이용)와 최적의 임계값 (모델을 통하여 구한 확률을 0 또는 1로 변경하는 기준점)을 구할 수 있다.


```python
best_threshold, best_score = choose_threshold(best_student_model, val_loader, device)
print(f'Best Threshold : [{best_threshold}], Score : [{best_score:.5f}]')
```


      0%|          | 0/12 [00:00<?, ?it/s]


    Best Threshold : [0.2], Score : [0.56697]
    

### 위에서 구한 최적의 임계값을 이용하여 테스트 데이터셋의 확률을 0 또는 1로 변경하기.


```python
test_datasets = CustomDataset(test, None, False)
test_loaders = DataLoader(test_datasets, batch_size = CFG['BATCH_SIZE'], shuffle=False)
```


```python
def inference(model, test_loader, threshold, device):
    model.to(device)
    model.eval()
    
    test_predict = []
    with torch.no_grad():
        for x in tqdm(test_loader):
            x = x.float().to(device)
            model_pred = model(x)

            model_pred = model_pred.squeeze(1).to('cpu')
            test_predict += model_pred
        
    test_predict = np.where(np.array(test_predict) > threshold, 1, 0)
    print('Done.')
    return test_predict
```


```python
preds = inference(best_student_model, test_loaders, best_threshold, device)
```


      0%|          | 0/24 [00:00<?, ?it/s]


    Done.
    

## 이진 분류 결과 제출하기.


```python
submit = pd.read_csv('./sample_submission.csv')
submit['Y_LABEL'] = preds # 아까 0 또는 1로 바꾼 결과를 submit의 Y_LABEL에 넣어주기.
submit.head()
```





  <div id="df-30a63b55-bbd9-4669-a89b-080369940c20">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>Y_LABEL</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>TEST_0000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>TEST_0001</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>TEST_0002</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>TEST_0003</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>TEST_0004</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-30a63b55-bbd9-4669-a89b-080369940c20')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-30a63b55-bbd9-4669-a89b-080369940c20 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-30a63b55-bbd9-4669-a89b-080369940c20');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
submit.to_csv('./submit.csv', index=False)
```

최종 점수 : 0.56182 (상위 21%)
