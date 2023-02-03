---
layout: post
title: Oil Condition Classification
authors: [Minji Kim, Seoyeon Bae]
categories: [1기 AI/SW developers(개인 프로젝트)]
---

# 건설기계 오일 상태 분류

# 1. Intro


## 1.1 프로젝트 배경
건설 장비 내부 기계 부품의 마모 상태 및 윤활 성능을 
<br/>오일 데이터 분석을 통해 확인하고, 
<br/>AI를 활용한 <font color="#FFA93A">"분류 모델"</font> 개발을 통해 
<br/>적절한 🛠 교체 주기를 파악하고자 합니다!

![img](https://ifh.cc/g/ksWwhJ.jpg)




## 1.2 프로젝트 목적
건설장비에서 작동오일의 상태를 실시간으로 모니터링하기 위한 오일 상태 판단 모델 개발
<br/> 🌳 이진분류(정상 / 이상)


## 1.3 개념 정리
> **Knowledge Distillation**
<br/>지식(Knowledge)와 증류(Distillation)이 만나 지식증류 (Knowledge Distillation)라는 개념이 생겼다.

> **구성요소**
1. Model
- Teacher Model: 높은 예측 정확도를 가진 복잡한 모델
- Student Model: Teacher Model의 지식을 받는 단순한 모델
2. Function
- Activation Function: 입력값 활성화 여부 결정함수 (Softmax)
- Loss Function: Teacher 모델의 soft label과 Student 모델의 prediction 비교

![img](https://ifh.cc/g/jo4Onl.jpg)
<br/><font color="#636363">[출처: https://intellabs.github.io/distiller/knowledge_distillation.html ]</font>


# 2. 실습

## 2.1 환경설정

### 2.1.1 Module Import

글꼴 설치


```python
!sudo apt-get install -y fonts-nanum
!sudo fc-cache -fv
!rm ~/.cache/matplotlib -rf
```

    Reading package lists... Done
    Building dependency tree       
    Reading state information... Done
    fonts-nanum is already the newest version (20180306-3).
    The following package was automatically installed and is no longer required:
      libnvidia-common-510
    Use 'sudo apt autoremove' to remove it.
    0 upgraded, 0 newly installed, 0 to remove and 27 not upgraded.
    /usr/share/fonts: caching, new cache contents: 0 fonts, 1 dirs
    /usr/share/fonts/truetype: caching, new cache contents: 0 fonts, 3 dirs
    /usr/share/fonts/truetype/humor-sans: caching, new cache contents: 1 fonts, 0 dirs
    /usr/share/fonts/truetype/liberation: caching, new cache contents: 16 fonts, 0 dirs
    /usr/share/fonts/truetype/nanum: caching, new cache contents: 10 fonts, 0 dirs
    /usr/local/share/fonts: caching, new cache contents: 0 fonts, 0 dirs
    /root/.local/share/fonts: skipping, no such directory
    /root/.fonts: skipping, no such directory
    /usr/share/fonts/truetype: skipping, looped directory detected
    /usr/share/fonts/truetype/humor-sans: skipping, looped directory detected
    /usr/share/fonts/truetype/liberation: skipping, looped directory detected
    /usr/share/fonts/truetype/nanum: skipping, looped directory detected
    /var/cache/fontconfig: cleaning cache directory
    /root/.cache/fontconfig: not cleaning non-existent cache directory
    /root/.fontconfig: not cleaning non-existent cache directory
    fc-cache: succeeded
    


```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer


import os
import pandas as pd
import numpy as np
import missingno as msno
from tqdm.auto import tqdm
import random


import matplotlib.pyplot as plt
import seaborn as sns


import warnings
warnings.filterwarnings(action='ignore') 


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn import metrics
import lightgbm as lgb
import xgboost as xgb
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


```

### 2.1.2 Hyperparameter Setting


```python
CFG = {
    'EPOCHS': 30,
    'LEARNING_RATE':1e-2,
    'BATCH_SIZE':256,
    'SEED':41
}
```

### 2.1.3 Fixed RandomSeed


```python
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed_everything(CFG['SEED'])
```

## 2.2 데이터 불러오기


```python
from google.colab import drive
drive.mount('/content/drive')
```

    Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).
    


```python
train = pd.read_csv('/content/drive/MyDrive/ai sw/train.csv')
test = pd.read_csv('/content/drive/MyDrive/ai sw/test.csv')
submit = pd.read_csv('/content/drive/MyDrive/ai sw/sample_submission.csv')
```

## 2.3 데이터 관찰하기


```python
train.head(3)
```





  <div id="df-d54e4cca-7deb-467d-9773-eeb68623b181">
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
      <th>COMPONENT_ARBITRARY</th>
      <th>ANONYMOUS_1</th>
      <th>YEAR</th>
      <th>SAMPLE_TRANSFER_DAY</th>
      <th>ANONYMOUS_2</th>
      <th>AG</th>
      <th>AL</th>
      <th>B</th>
      <th>BA</th>
      <th>...</th>
      <th>U25</th>
      <th>U20</th>
      <th>U14</th>
      <th>U6</th>
      <th>U4</th>
      <th>V</th>
      <th>V100</th>
      <th>V40</th>
      <th>ZN</th>
      <th>Y_LABEL</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>TRAIN_00000</td>
      <td>COMPONENT3</td>
      <td>1486</td>
      <td>2011</td>
      <td>7</td>
      <td>200</td>
      <td>0</td>
      <td>3</td>
      <td>93</td>
      <td>0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>154.0</td>
      <td>75</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>TRAIN_00001</td>
      <td>COMPONENT2</td>
      <td>1350</td>
      <td>2021</td>
      <td>51</td>
      <td>375</td>
      <td>0</td>
      <td>2</td>
      <td>19</td>
      <td>0</td>
      <td>...</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>6.0</td>
      <td>216.0</td>
      <td>1454.0</td>
      <td>0</td>
      <td>NaN</td>
      <td>44.0</td>
      <td>652</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>TRAIN_00002</td>
      <td>COMPONENT2</td>
      <td>2415</td>
      <td>2015</td>
      <td>2</td>
      <td>200</td>
      <td>0</td>
      <td>110</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>39.0</td>
      <td>11261.0</td>
      <td>41081.0</td>
      <td>0</td>
      <td>NaN</td>
      <td>72.6</td>
      <td>412</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 54 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-d54e4cca-7deb-467d-9773-eeb68623b181')"
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
          document.querySelector('#df-d54e4cca-7deb-467d-9773-eeb68623b181 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-d54e4cca-7deb-467d-9773-eeb68623b181');
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
test.head(3)
```





  <div id="df-e93309de-8458-4e49-ac1d-5f014e886a31">
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
      <th>COMPONENT_ARBITRARY</th>
      <th>ANONYMOUS_1</th>
      <th>YEAR</th>
      <th>ANONYMOUS_2</th>
      <th>AG</th>
      <th>CO</th>
      <th>CR</th>
      <th>CU</th>
      <th>FE</th>
      <th>H2O</th>
      <th>MN</th>
      <th>MO</th>
      <th>NI</th>
      <th>PQINDEX</th>
      <th>TI</th>
      <th>V</th>
      <th>V40</th>
      <th>ZN</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>TEST_0000</td>
      <td>COMPONENT1</td>
      <td>2192</td>
      <td>2016</td>
      <td>200</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>12</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>10</td>
      <td>0</td>
      <td>0</td>
      <td>91.3</td>
      <td>1091</td>
    </tr>
    <tr>
      <th>1</th>
      <td>TEST_0001</td>
      <td>COMPONENT3</td>
      <td>2794</td>
      <td>2011</td>
      <td>200</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>278</td>
      <td>0.0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>2732</td>
      <td>1</td>
      <td>0</td>
      <td>126.9</td>
      <td>12</td>
    </tr>
    <tr>
      <th>2</th>
      <td>TEST_0002</td>
      <td>COMPONENT2</td>
      <td>1982</td>
      <td>2010</td>
      <td>200</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>16</td>
      <td>5</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>11</td>
      <td>0</td>
      <td>0</td>
      <td>44.3</td>
      <td>714</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-e93309de-8458-4e49-ac1d-5f014e886a31')"
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
          document.querySelector('#df-e93309de-8458-4e49-ac1d-5f014e886a31 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-e93309de-8458-4e49-ac1d-5f014e886a31');
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
train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 14095 entries, 0 to 14094
    Data columns (total 54 columns):
     #   Column               Non-Null Count  Dtype  
    ---  ------               --------------  -----  
     0   ID                   14095 non-null  object 
     1   COMPONENT_ARBITRARY  14095 non-null  object 
     2   ANONYMOUS_1          14095 non-null  int64  
     3   YEAR                 14095 non-null  int64  
     4   SAMPLE_TRANSFER_DAY  14095 non-null  int64  
     5   ANONYMOUS_2          14095 non-null  int64  
     6   AG                   14095 non-null  int64  
     7   AL                   14095 non-null  int64  
     8   B                    14095 non-null  int64  
     9   BA                   14095 non-null  int64  
     10  BE                   14095 non-null  int64  
     11  CA                   14095 non-null  int64  
     12  CD                   12701 non-null  float64
     13  CO                   14095 non-null  int64  
     14  CR                   14095 non-null  int64  
     15  CU                   14095 non-null  int64  
     16  FH2O                 3890 non-null   float64
     17  FNOX                 3890 non-null   float64
     18  FOPTIMETHGLY         3890 non-null   float64
     19  FOXID                3890 non-null   float64
     20  FSO4                 3890 non-null   float64
     21  FTBN                 3890 non-null   float64
     22  FE                   14095 non-null  int64  
     23  FUEL                 3890 non-null   float64
     24  H2O                  14095 non-null  float64
     25  K                    11796 non-null  float64
     26  LI                   14095 non-null  int64  
     27  MG                   14095 non-null  int64  
     28  MN                   14095 non-null  int64  
     29  MO                   14095 non-null  int64  
     30  NA                   14095 non-null  int64  
     31  NI                   14095 non-null  int64  
     32  P                    14095 non-null  int64  
     33  PB                   14095 non-null  int64  
     34  PQINDEX              14095 non-null  int64  
     35  S                    14095 non-null  int64  
     36  SB                   14095 non-null  int64  
     37  SI                   14095 non-null  int64  
     38  SN                   14095 non-null  int64  
     39  SOOTPERCENTAGE       3890 non-null   float64
     40  TI                   14095 non-null  int64  
     41  U100                 2316 non-null   float64
     42  U75                  2316 non-null   float64
     43  U50                  2316 non-null   float64
     44  U25                  2316 non-null   float64
     45  U20                  2316 non-null   float64
     46  U14                  2118 non-null   float64
     47  U6                   2118 non-null   float64
     48  U4                   2118 non-null   float64
     49  V                    14095 non-null  int64  
     50  V100                 3724 non-null   float64
     51  V40                  14095 non-null  float64
     52  ZN                   14095 non-null  int64  
     53  Y_LABEL              14095 non-null  int64  
    dtypes: float64(21), int64(31), object(2)
    memory usage: 5.8+ MB
    


```python
test.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 6041 entries, 0 to 6040
    Data columns (total 19 columns):
     #   Column               Non-Null Count  Dtype  
    ---  ------               --------------  -----  
     0   ID                   6041 non-null   object 
     1   COMPONENT_ARBITRARY  6041 non-null   object 
     2   ANONYMOUS_1          6041 non-null   int64  
     3   YEAR                 6041 non-null   int64  
     4   ANONYMOUS_2          6041 non-null   int64  
     5   AG                   6041 non-null   int64  
     6   CO                   6041 non-null   int64  
     7   CR                   6041 non-null   int64  
     8   CU                   6041 non-null   int64  
     9   FE                   6041 non-null   int64  
     10  H2O                  6041 non-null   float64
     11  MN                   6041 non-null   int64  
     12  MO                   6041 non-null   int64  
     13  NI                   6041 non-null   int64  
     14  PQINDEX              6041 non-null   int64  
     15  TI                   6041 non-null   int64  
     16  V                    6041 non-null   int64  
     17  V40                  6041 non-null   float64
     18  ZN                   6041 non-null   int64  
    dtypes: float64(2), int64(15), object(2)
    memory usage: 896.8+ KB
    

**변수 살펴보기**


- ID : ID
- COMPONENT_ARBITRARY : sample 오일 관련 부품 (Component 4종)
- ANONYMOUS_1 : 무명 Feature 1, 수치형 데이터
- YEAR : 오일 진단 년도(Year)
- SAMPLE_TRANSFER_DAY : 오일 샘플링 후 진단 기관으로 이동한 기간(Days)
- ANONYMOUS_2 : 무명 Feature 2, 수치형 데이터
- AG : 은(Silver) 함유량
- AL : 알루미늄(Aluminium) 함유량
- B : 붕소(Boron) 함유량
- BA : 바륨(Barium) 함유량
- BE : 베릴륨(Beryllium) 함유량
- CA : 칼슘(Calcium) 함유량
- CD : 카드뮴(Cadmium) 함유량
- CO : 코발트(Cobolt) 함유량
- CR : 크로뮴(Chromium) 함유량
- CR : 구리(Copper) 함유량
- FH2O : 물(Water) 수치(By FT-IR)
- FNOX : 질소산화물(Nox) 수치(By FT-IR)
- FOPTIMETHGLY : 비식별화
- FOXID : 산화(Oxidation) 수치(By FT-IR)
- FSO4 : 황산염(S04) 수치(By FT-IR)
- FTBN : 염기성 첨가제물질 수치(By FT-IR)
- FE : 철(Iron) 함유량
- FUEL : 연료 함유량
- H2O : 물 함유량
- K : 칼륨(Potassium) 함유량
- LI : 리튬(Lithium) 함유량
- MG : 마그네슘(Magnesium) 함유량
- MN : 망가니즈(Manganese) 함유량
- MO : 몰리브덴(Molybdenum) 함유량
- NA : 나트륨(Sodium) 함유량
- NI : 니켈(Nickel) 함유량
- P : 인(Phosphorus) 함유량
- PB : 납(Lead) 함유량
- PQINDEX : 입자 정량화 지수(Particle Quantifier Index)
- S : 황(Sulphur) 함유량
- SB : 안티몬(Antimony) 함유량
- SI : 규소(Silicone) 함유량
- SN : 주석(Tin) 함유량
- SOOTPERCENTAGE : 그을음 정도
- TI : 티타늄(Titanium) 함유량
- U100 : 100㎛ 이상 입자 크기(Particle Count)
- U75 : 75㎛ 이상 입자 크기(Particle Count)
- U50 : 50㎛ 이상 입자 크기(Particle Count)
- U25 : 25㎛ 이상 입자 크기(Particle Count)


## 2.4 데이터 전처리


```python
categorical_features = ['COMPONENT_ARBITRARY', 'YEAR']
# Inference(실제 진단 환경)에 사용하는 컬럼
test_stage_features = ['COMPONENT_ARBITRARY', 'ANONYMOUS_1', 'YEAR' , 'ANONYMOUS_2', 'AG', 'CO', 'CR', 'CU', 'FE', 'H2O', 'MN', 'MO', 'NI', 'PQINDEX', 'TI', 'V', 'V40', 'ZN']

```


```python
all_X = train.drop(['ID', 'Y_LABEL', 'SAMPLE_TRANSFER_DAY', 'FOPTIMETHGLY'], axis = 1)
all_y = train['Y_LABEL']

test = test.drop(['ID'], axis = 1)
```


```python
all_X.isnull().sum()
```




    COMPONENT_ARBITRARY        0
    ANONYMOUS_1                0
    YEAR                       0
    ANONYMOUS_2                0
    AG                         0
    AL                         0
    B                          0
    BA                         0
    BE                         0
    CA                         0
    CD                      1394
    CO                         0
    CR                         0
    CU                         0
    FH2O                   10205
    FNOX                   10205
    FOXID                  10205
    FSO4                   10205
    FTBN                   10205
    FE                         0
    FUEL                   10205
    H2O                        0
    K                       2299
    LI                         0
    MG                         0
    MN                         0
    MO                         0
    NA                         0
    NI                         0
    P                          0
    PB                         0
    PQINDEX                    0
    S                          0
    SB                         0
    SI                         0
    SN                         0
    SOOTPERCENTAGE         10205
    TI                         0
    U100                   11779
    U75                    11779
    U50                    11779
    U25                    11779
    U20                    11779
    U14                    11977
    U6                     11977
    U4                     11977
    V                          0
    V100                   10371
    V40                        0
    ZN                         0
    dtype: int64




```python
test.isnull().sum()
```




    COMPONENT_ARBITRARY    0
    ANONYMOUS_1            0
    YEAR                   0
    ANONYMOUS_2            0
    AG                     0
    CO                     0
    CR                     0
    CU                     0
    FE                     0
    H2O                    0
    MN                     0
    MO                     0
    NI                     0
    PQINDEX                0
    TI                     0
    V                      0
    V40                    0
    ZN                     0
    dtype: int64




```python
msno.matrix(all_X)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f43660177c0>




    
![png](/data/2023-02-04-Oil Condition Classification/output_29_1.png)
    


아래의 컬럼들에 결측치가 <font color="#FFA93A">너무 많다</font>.

- FH20
- FNOX
- FOXID
- FSO4
- FTBN
- FUEL
- SOOTPERCENTAGE
- U100, U75, U50, U25, U20, U14, U6, U4
- V100 



```python
# 제거할 컬럼 리스트 생성
drop_col = ['FH2O', 'FNOX', 'FOXID', 'FSO4','FTBN','FUEL', 'SOOTPERCENTAGE', 
            'U100', 'U75', 'U50', 'U25', 'U20', 'U14', 'U6', 'U4', 'V100' ]
```


```python
all_X = all_X.drop(drop_col, axis=1)
```


```python
all_X.isnull().sum()
```




    COMPONENT_ARBITRARY       0
    ANONYMOUS_1               0
    YEAR                      0
    ANONYMOUS_2               0
    AG                        0
    AL                        0
    B                         0
    BA                        0
    BE                        0
    CA                        0
    CD                     1394
    CO                        0
    CR                        0
    CU                        0
    FE                        0
    H2O                       0
    K                      2299
    LI                        0
    MG                        0
    MN                        0
    MO                        0
    NA                        0
    NI                        0
    P                         0
    PB                        0
    PQINDEX                   0
    S                         0
    SB                        0
    SI                        0
    SN                        0
    TI                        0
    V                         0
    V40                       0
    ZN                        0
    dtype: int64



KNN을 사용하여 결측치를 처리하기 전,
<br/>데이터 <font color="#FFA93A">정규화 및 표준화</font> 처리를 해준다.


```python
def get_values(value):
    return value.values.reshape(-1, 1)


for col in all_X.columns:
    if col not in categorical_features:
        scaler = StandardScaler() 
        #표준화 -> 각 특성의 평균을 0, 분산을 1로 변경(모든 특성이 같은 크기를 가지게 함)
        all_X[col] = scaler.fit_transform(get_values(all_X[col]))
        if col in test.columns:
            test[col] = scaler.transform(get_values(test[col]))
            
le = LabelEncoder()
# 카테고리형 데이터 -> 수치형 데이터 (변환)
for col in categorical_features:    
    all_X[col] = le.fit_transform(all_X[col])
    if col in test.columns:
        test[col] = le.transform(test[col])
```

남은 결측치 처리하기

> KNN

다른 데이터들의 레이블을 참조하여 분류하는 알고리즘

<br/>이 알고리즘에서의 <font color="FFA93A">hyperparameter</font>는 탐색할 이웃 수 <font color="FFA93A">k</font>와 거리 측정 방법

- k가 너무 <font color="FFA93A">작으면</font>, 모델이 데이터의 지역적인 특성을 지나치게 반영하여 overfitting 될 가능성 높아짐

- k가 너무 <font color="FFA93A">크면</font>, 모델이 과하게 정규화 되어서 underfitting 될 가능성이 높아짐

<br/>단순하고, 훈련 단계가 빨라서 효율적임.


```python
imputer = KNNImputer(n_neighbors = 5)
imputed = imputer.fit_transform(all_X)
all_X = pd.DataFrame(imputed, columns = all_X.columns)
all_X.head(5)
```





  <div id="df-d6208b6d-d7a9-4b6f-afc0-41683862bf7f">
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
      <th>COMPONENT_ARBITRARY</th>
      <th>ANONYMOUS_1</th>
      <th>YEAR</th>
      <th>ANONYMOUS_2</th>
      <th>AG</th>
      <th>AL</th>
      <th>B</th>
      <th>BA</th>
      <th>BE</th>
      <th>CA</th>
      <th>...</th>
      <th>PB</th>
      <th>PQINDEX</th>
      <th>S</th>
      <th>SB</th>
      <th>SI</th>
      <th>SN</th>
      <th>TI</th>
      <th>V</th>
      <th>V40</th>
      <th>ZN</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.0</td>
      <td>-0.393763</td>
      <td>4.0</td>
      <td>-0.340760</td>
      <td>-0.150214</td>
      <td>-0.111628</td>
      <td>0.281646</td>
      <td>-0.238453</td>
      <td>-0.041491</td>
      <td>1.141962</td>
      <td>...</td>
      <td>-0.160812</td>
      <td>5.293270</td>
      <td>1.001652</td>
      <td>-0.174727</td>
      <td>2.006643</td>
      <td>0.302478</td>
      <td>0.622282</td>
      <td>-0.10655</td>
      <td>0.899892</td>
      <td>-0.966002</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>-0.426022</td>
      <td>14.0</td>
      <td>-0.022576</td>
      <td>-0.150214</td>
      <td>-0.123127</td>
      <td>-0.437686</td>
      <td>-0.238453</td>
      <td>-0.041491</td>
      <td>1.087302</td>
      <td>...</td>
      <td>0.033010</td>
      <td>-0.259244</td>
      <td>-1.170187</td>
      <td>-0.174727</td>
      <td>-0.179489</td>
      <td>-0.252439</td>
      <td>-0.102635</td>
      <td>-0.10655</td>
      <td>-1.317376</td>
      <td>0.119147</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>-0.173409</td>
      <td>8.0</td>
      <td>-0.340760</td>
      <td>-0.150214</td>
      <td>1.118753</td>
      <td>-0.612659</td>
      <td>0.105735</td>
      <td>-0.041491</td>
      <td>-0.910846</td>
      <td>...</td>
      <td>-0.160812</td>
      <td>-0.260552</td>
      <td>-1.146917</td>
      <td>-0.174727</td>
      <td>-0.179489</td>
      <td>0.025019</td>
      <td>-0.102635</td>
      <td>-0.10655</td>
      <td>-0.740886</td>
      <td>-0.332215</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.0</td>
      <td>1.006399</td>
      <td>3.0</td>
      <td>-0.340760</td>
      <td>-0.150214</td>
      <td>-0.054133</td>
      <td>-0.593217</td>
      <td>-0.238453</td>
      <td>-0.041491</td>
      <td>0.400333</td>
      <td>...</td>
      <td>-0.063901</td>
      <td>-0.242884</td>
      <td>1.044975</td>
      <td>0.557916</td>
      <td>-0.174370</td>
      <td>0.025019</td>
      <td>-0.102635</td>
      <td>-0.10655</td>
      <td>0.482642</td>
      <td>-1.093888</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.0</td>
      <td>0.191634</td>
      <td>8.0</td>
      <td>-0.340760</td>
      <td>-0.150214</td>
      <td>-0.134626</td>
      <td>0.903771</td>
      <td>-0.238453</td>
      <td>-0.041491</td>
      <td>-0.874406</td>
      <td>...</td>
      <td>-0.160812</td>
      <td>-0.129674</td>
      <td>0.690669</td>
      <td>-0.174727</td>
      <td>-0.169250</td>
      <td>-0.252439</td>
      <td>-0.102635</td>
      <td>-0.10655</td>
      <td>0.478611</td>
      <td>-0.866326</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 34 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-d6208b6d-d7a9-4b6f-afc0-41683862bf7f')"
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
          document.querySelector('#df-d6208b6d-d7a9-4b6f-afc0-41683862bf7f button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-d6208b6d-d7a9-4b6f-afc0-41683862bf7f');
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




## 2.5 탐색적 데이터 분석(EDA, 시각화)


```python
all_X.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 14095 entries, 0 to 14094
    Data columns (total 34 columns):
     #   Column               Non-Null Count  Dtype  
    ---  ------               --------------  -----  
     0   COMPONENT_ARBITRARY  14095 non-null  float64
     1   ANONYMOUS_1          14095 non-null  float64
     2   YEAR                 14095 non-null  float64
     3   ANONYMOUS_2          14095 non-null  float64
     4   AG                   14095 non-null  float64
     5   AL                   14095 non-null  float64
     6   B                    14095 non-null  float64
     7   BA                   14095 non-null  float64
     8   BE                   14095 non-null  float64
     9   CA                   14095 non-null  float64
     10  CD                   14095 non-null  float64
     11  CO                   14095 non-null  float64
     12  CR                   14095 non-null  float64
     13  CU                   14095 non-null  float64
     14  FE                   14095 non-null  float64
     15  H2O                  14095 non-null  float64
     16  K                    14095 non-null  float64
     17  LI                   14095 non-null  float64
     18  MG                   14095 non-null  float64
     19  MN                   14095 non-null  float64
     20  MO                   14095 non-null  float64
     21  NA                   14095 non-null  float64
     22  NI                   14095 non-null  float64
     23  P                    14095 non-null  float64
     24  PB                   14095 non-null  float64
     25  PQINDEX              14095 non-null  float64
     26  S                    14095 non-null  float64
     27  SB                   14095 non-null  float64
     28  SI                   14095 non-null  float64
     29  SN                   14095 non-null  float64
     30  TI                   14095 non-null  float64
     31  V                    14095 non-null  float64
     32  V40                  14095 non-null  float64
     33  ZN                   14095 non-null  float64
    dtypes: float64(34)
    memory usage: 3.7 MB
    


```python
all_y
```




    0        0
    1        0
    2        1
    3        0
    4        0
            ..
    14090    0
    14091    0
    14092    0
    14093    0
    14094    0
    Name: Y_LABEL, Length: 14095, dtype: int64




```python
new_train = pd.concat([all_X, all_y], axis=1)
new_train.head(5)
```





  <div id="df-62b27724-925a-4b92-b1f1-88601ea2ac5f">
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
      <th>COMPONENT_ARBITRARY</th>
      <th>ANONYMOUS_1</th>
      <th>YEAR</th>
      <th>ANONYMOUS_2</th>
      <th>AG</th>
      <th>AL</th>
      <th>B</th>
      <th>BA</th>
      <th>BE</th>
      <th>CA</th>
      <th>...</th>
      <th>PQINDEX</th>
      <th>S</th>
      <th>SB</th>
      <th>SI</th>
      <th>SN</th>
      <th>TI</th>
      <th>V</th>
      <th>V40</th>
      <th>ZN</th>
      <th>Y_LABEL</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.0</td>
      <td>-0.393763</td>
      <td>4.0</td>
      <td>-0.340760</td>
      <td>-0.150214</td>
      <td>-0.111628</td>
      <td>0.281646</td>
      <td>-0.238453</td>
      <td>-0.041491</td>
      <td>1.141962</td>
      <td>...</td>
      <td>5.293270</td>
      <td>1.001652</td>
      <td>-0.174727</td>
      <td>2.006643</td>
      <td>0.302478</td>
      <td>0.622282</td>
      <td>-0.10655</td>
      <td>0.899892</td>
      <td>-0.966002</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>-0.426022</td>
      <td>14.0</td>
      <td>-0.022576</td>
      <td>-0.150214</td>
      <td>-0.123127</td>
      <td>-0.437686</td>
      <td>-0.238453</td>
      <td>-0.041491</td>
      <td>1.087302</td>
      <td>...</td>
      <td>-0.259244</td>
      <td>-1.170187</td>
      <td>-0.174727</td>
      <td>-0.179489</td>
      <td>-0.252439</td>
      <td>-0.102635</td>
      <td>-0.10655</td>
      <td>-1.317376</td>
      <td>0.119147</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>-0.173409</td>
      <td>8.0</td>
      <td>-0.340760</td>
      <td>-0.150214</td>
      <td>1.118753</td>
      <td>-0.612659</td>
      <td>0.105735</td>
      <td>-0.041491</td>
      <td>-0.910846</td>
      <td>...</td>
      <td>-0.260552</td>
      <td>-1.146917</td>
      <td>-0.174727</td>
      <td>-0.179489</td>
      <td>0.025019</td>
      <td>-0.102635</td>
      <td>-0.10655</td>
      <td>-0.740886</td>
      <td>-0.332215</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.0</td>
      <td>1.006399</td>
      <td>3.0</td>
      <td>-0.340760</td>
      <td>-0.150214</td>
      <td>-0.054133</td>
      <td>-0.593217</td>
      <td>-0.238453</td>
      <td>-0.041491</td>
      <td>0.400333</td>
      <td>...</td>
      <td>-0.242884</td>
      <td>1.044975</td>
      <td>0.557916</td>
      <td>-0.174370</td>
      <td>0.025019</td>
      <td>-0.102635</td>
      <td>-0.10655</td>
      <td>0.482642</td>
      <td>-1.093888</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.0</td>
      <td>0.191634</td>
      <td>8.0</td>
      <td>-0.340760</td>
      <td>-0.150214</td>
      <td>-0.134626</td>
      <td>0.903771</td>
      <td>-0.238453</td>
      <td>-0.041491</td>
      <td>-0.874406</td>
      <td>...</td>
      <td>-0.129674</td>
      <td>0.690669</td>
      <td>-0.174727</td>
      <td>-0.169250</td>
      <td>-0.252439</td>
      <td>-0.102635</td>
      <td>-0.10655</td>
      <td>0.478611</td>
      <td>-0.866326</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 35 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-62b27724-925a-4b92-b1f1-88601ea2ac5f')"
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
          document.querySelector('#df-62b27724-925a-4b92-b1f1-88601ea2ac5f button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-62b27724-925a-4b92-b1f1-88601ea2ac5f');
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
train_cor1 = new_train[['COMPONENT_ARBITRARY', 'ANONYMOUS_1', 'YEAR' , 'ANONYMOUS_2', 
                        'AG', 'CO', 'CR', 'CU', 'FE', 'H2O', 'MN', 'MO', 'NI', 'PQINDEX', 
                        'TI', 'V', 'V40', 'ZN', 'Y_LABEL']]
train_cor1.head(5)
```





  <div id="df-a7305caa-8080-4ebc-a0a4-b588447328ee">
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
      <th>COMPONENT_ARBITRARY</th>
      <th>ANONYMOUS_1</th>
      <th>YEAR</th>
      <th>ANONYMOUS_2</th>
      <th>AG</th>
      <th>CO</th>
      <th>CR</th>
      <th>CU</th>
      <th>FE</th>
      <th>H2O</th>
      <th>MN</th>
      <th>MO</th>
      <th>NI</th>
      <th>PQINDEX</th>
      <th>TI</th>
      <th>V</th>
      <th>V40</th>
      <th>ZN</th>
      <th>Y_LABEL</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.0</td>
      <td>-0.393763</td>
      <td>4.0</td>
      <td>-0.340760</td>
      <td>-0.150214</td>
      <td>-0.089633</td>
      <td>0.339245</td>
      <td>0.336858</td>
      <td>1.331290</td>
      <td>-0.041588</td>
      <td>1.186914</td>
      <td>-0.384284</td>
      <td>1.384414</td>
      <td>5.293270</td>
      <td>0.622282</td>
      <td>-0.10655</td>
      <td>0.899892</td>
      <td>-0.966002</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>-0.426022</td>
      <td>14.0</td>
      <td>-0.022576</td>
      <td>-0.150214</td>
      <td>-0.089633</td>
      <td>-0.115388</td>
      <td>-0.027612</td>
      <td>-0.330406</td>
      <td>-0.041588</td>
      <td>-0.250456</td>
      <td>-0.400998</td>
      <td>-0.191804</td>
      <td>-0.259244</td>
      <td>-0.102635</td>
      <td>-0.10655</td>
      <td>-1.317376</td>
      <td>0.119147</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>-0.173409</td>
      <td>8.0</td>
      <td>-0.340760</td>
      <td>-0.150214</td>
      <td>-0.089633</td>
      <td>-0.080416</td>
      <td>-0.252497</td>
      <td>-0.326655</td>
      <td>-0.041588</td>
      <td>-0.250456</td>
      <td>-0.400998</td>
      <td>-0.191804</td>
      <td>-0.260552</td>
      <td>-0.102635</td>
      <td>-0.10655</td>
      <td>-0.740886</td>
      <td>-0.332215</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.0</td>
      <td>1.006399</td>
      <td>3.0</td>
      <td>-0.340760</td>
      <td>-0.150214</td>
      <td>-0.089633</td>
      <td>-0.115388</td>
      <td>-0.260252</td>
      <td>-0.264764</td>
      <td>-0.041588</td>
      <td>-0.160621</td>
      <td>-0.400998</td>
      <td>-0.191804</td>
      <td>-0.242884</td>
      <td>-0.102635</td>
      <td>-0.10655</td>
      <td>0.482642</td>
      <td>-1.093888</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.0</td>
      <td>0.191634</td>
      <td>8.0</td>
      <td>-0.340760</td>
      <td>-0.150214</td>
      <td>-0.089633</td>
      <td>-0.115388</td>
      <td>-0.268007</td>
      <td>-0.200996</td>
      <td>-0.041588</td>
      <td>-0.250456</td>
      <td>-0.400998</td>
      <td>-0.191804</td>
      <td>-0.129674</td>
      <td>-0.102635</td>
      <td>-0.10655</td>
      <td>0.478611</td>
      <td>-0.866326</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-a7305caa-8080-4ebc-a0a4-b588447328ee')"
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
          document.querySelector('#df-a7305caa-8080-4ebc-a0a4-b588447328ee button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-a7305caa-8080-4ebc-a0a4-b588447328ee');
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
n_train_cor = round(train_cor1.corr(), 3)
n_train_cor
```





  <div id="df-aafc0b08-0fd8-4efc-a487-83b08782af1b">
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
      <th>COMPONENT_ARBITRARY</th>
      <th>ANONYMOUS_1</th>
      <th>YEAR</th>
      <th>ANONYMOUS_2</th>
      <th>AG</th>
      <th>CO</th>
      <th>CR</th>
      <th>CU</th>
      <th>FE</th>
      <th>H2O</th>
      <th>MN</th>
      <th>MO</th>
      <th>NI</th>
      <th>PQINDEX</th>
      <th>TI</th>
      <th>V</th>
      <th>V40</th>
      <th>ZN</th>
      <th>Y_LABEL</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>COMPONENT_ARBITRARY</th>
      <td>1.000</td>
      <td>0.021</td>
      <td>-0.002</td>
      <td>-0.001</td>
      <td>-0.001</td>
      <td>0.032</td>
      <td>0.038</td>
      <td>0.117</td>
      <td>0.199</td>
      <td>0.026</td>
      <td>0.149</td>
      <td>-0.346</td>
      <td>0.109</td>
      <td>0.178</td>
      <td>0.066</td>
      <td>0.042</td>
      <td>0.226</td>
      <td>-0.542</td>
      <td>0.003</td>
    </tr>
    <tr>
      <th>ANONYMOUS_1</th>
      <td>0.021</td>
      <td>1.000</td>
      <td>0.107</td>
      <td>0.072</td>
      <td>-0.026</td>
      <td>-0.004</td>
      <td>-0.007</td>
      <td>-0.014</td>
      <td>0.000</td>
      <td>0.004</td>
      <td>-0.004</td>
      <td>-0.006</td>
      <td>-0.008</td>
      <td>0.002</td>
      <td>0.002</td>
      <td>0.002</td>
      <td>0.020</td>
      <td>-0.020</td>
      <td>0.044</td>
    </tr>
    <tr>
      <th>YEAR</th>
      <td>-0.002</td>
      <td>0.107</td>
      <td>1.000</td>
      <td>0.138</td>
      <td>-0.129</td>
      <td>-0.052</td>
      <td>-0.029</td>
      <td>-0.138</td>
      <td>-0.058</td>
      <td>0.010</td>
      <td>-0.039</td>
      <td>-0.091</td>
      <td>-0.049</td>
      <td>-0.067</td>
      <td>0.006</td>
      <td>-0.028</td>
      <td>-0.052</td>
      <td>0.049</td>
      <td>-0.088</td>
    </tr>
    <tr>
      <th>ANONYMOUS_2</th>
      <td>-0.001</td>
      <td>0.072</td>
      <td>0.138</td>
      <td>1.000</td>
      <td>-0.006</td>
      <td>-0.000</td>
      <td>-0.002</td>
      <td>-0.002</td>
      <td>-0.005</td>
      <td>-0.004</td>
      <td>0.003</td>
      <td>-0.008</td>
      <td>-0.004</td>
      <td>-0.010</td>
      <td>-0.001</td>
      <td>-0.006</td>
      <td>-0.025</td>
      <td>0.033</td>
      <td>-0.034</td>
    </tr>
    <tr>
      <th>AG</th>
      <td>-0.001</td>
      <td>-0.026</td>
      <td>-0.129</td>
      <td>-0.006</td>
      <td>1.000</td>
      <td>0.009</td>
      <td>0.005</td>
      <td>0.051</td>
      <td>0.026</td>
      <td>-0.004</td>
      <td>0.020</td>
      <td>0.013</td>
      <td>0.054</td>
      <td>0.031</td>
      <td>0.004</td>
      <td>-0.004</td>
      <td>0.014</td>
      <td>0.003</td>
      <td>0.015</td>
    </tr>
    <tr>
      <th>CO</th>
      <td>0.032</td>
      <td>-0.004</td>
      <td>-0.052</td>
      <td>-0.000</td>
      <td>0.009</td>
      <td>1.000</td>
      <td>0.173</td>
      <td>0.265</td>
      <td>0.537</td>
      <td>0.060</td>
      <td>0.425</td>
      <td>0.001</td>
      <td>0.402</td>
      <td>0.175</td>
      <td>0.263</td>
      <td>0.271</td>
      <td>0.108</td>
      <td>-0.018</td>
      <td>0.008</td>
    </tr>
    <tr>
      <th>CR</th>
      <td>0.038</td>
      <td>-0.007</td>
      <td>-0.029</td>
      <td>-0.002</td>
      <td>0.005</td>
      <td>0.173</td>
      <td>1.000</td>
      <td>0.051</td>
      <td>0.314</td>
      <td>0.038</td>
      <td>0.248</td>
      <td>-0.008</td>
      <td>0.222</td>
      <td>0.147</td>
      <td>0.226</td>
      <td>0.391</td>
      <td>0.079</td>
      <td>-0.053</td>
      <td>0.014</td>
    </tr>
    <tr>
      <th>CU</th>
      <td>0.117</td>
      <td>-0.014</td>
      <td>-0.138</td>
      <td>-0.002</td>
      <td>0.051</td>
      <td>0.265</td>
      <td>0.051</td>
      <td>1.000</td>
      <td>0.287</td>
      <td>-0.002</td>
      <td>0.222</td>
      <td>-0.017</td>
      <td>0.487</td>
      <td>0.121</td>
      <td>0.103</td>
      <td>0.089</td>
      <td>-0.068</td>
      <td>0.041</td>
      <td>0.025</td>
    </tr>
    <tr>
      <th>FE</th>
      <td>0.199</td>
      <td>0.000</td>
      <td>-0.058</td>
      <td>-0.005</td>
      <td>0.026</td>
      <td>0.537</td>
      <td>0.314</td>
      <td>0.287</td>
      <td>1.000</td>
      <td>0.119</td>
      <td>0.622</td>
      <td>-0.064</td>
      <td>0.583</td>
      <td>0.428</td>
      <td>0.384</td>
      <td>0.341</td>
      <td>0.248</td>
      <td>-0.155</td>
      <td>0.048</td>
    </tr>
    <tr>
      <th>H2O</th>
      <td>0.026</td>
      <td>0.004</td>
      <td>0.010</td>
      <td>-0.004</td>
      <td>-0.004</td>
      <td>0.060</td>
      <td>0.038</td>
      <td>-0.002</td>
      <td>0.119</td>
      <td>1.000</td>
      <td>0.144</td>
      <td>-0.010</td>
      <td>0.052</td>
      <td>0.082</td>
      <td>0.154</td>
      <td>0.049</td>
      <td>0.371</td>
      <td>-0.031</td>
      <td>-0.004</td>
    </tr>
    <tr>
      <th>MN</th>
      <td>0.149</td>
      <td>-0.004</td>
      <td>-0.039</td>
      <td>0.003</td>
      <td>0.020</td>
      <td>0.425</td>
      <td>0.248</td>
      <td>0.222</td>
      <td>0.622</td>
      <td>0.144</td>
      <td>1.000</td>
      <td>-0.047</td>
      <td>0.528</td>
      <td>0.376</td>
      <td>0.585</td>
      <td>0.375</td>
      <td>0.150</td>
      <td>-0.044</td>
      <td>0.024</td>
    </tr>
    <tr>
      <th>MO</th>
      <td>-0.346</td>
      <td>-0.006</td>
      <td>-0.091</td>
      <td>-0.008</td>
      <td>0.013</td>
      <td>0.001</td>
      <td>-0.008</td>
      <td>-0.017</td>
      <td>-0.064</td>
      <td>-0.010</td>
      <td>-0.047</td>
      <td>1.000</td>
      <td>-0.043</td>
      <td>-0.082</td>
      <td>-0.031</td>
      <td>-0.010</td>
      <td>-0.008</td>
      <td>0.465</td>
      <td>0.001</td>
    </tr>
    <tr>
      <th>NI</th>
      <td>0.109</td>
      <td>-0.008</td>
      <td>-0.049</td>
      <td>-0.004</td>
      <td>0.054</td>
      <td>0.402</td>
      <td>0.222</td>
      <td>0.487</td>
      <td>0.583</td>
      <td>0.052</td>
      <td>0.528</td>
      <td>-0.043</td>
      <td>1.000</td>
      <td>0.356</td>
      <td>0.414</td>
      <td>0.283</td>
      <td>0.156</td>
      <td>-0.072</td>
      <td>0.047</td>
    </tr>
    <tr>
      <th>PQINDEX</th>
      <td>0.178</td>
      <td>0.002</td>
      <td>-0.067</td>
      <td>-0.010</td>
      <td>0.031</td>
      <td>0.175</td>
      <td>0.147</td>
      <td>0.121</td>
      <td>0.428</td>
      <td>0.082</td>
      <td>0.376</td>
      <td>-0.082</td>
      <td>0.356</td>
      <td>1.000</td>
      <td>0.258</td>
      <td>0.140</td>
      <td>0.226</td>
      <td>-0.177</td>
      <td>0.029</td>
    </tr>
    <tr>
      <th>TI</th>
      <td>0.066</td>
      <td>0.002</td>
      <td>0.006</td>
      <td>-0.001</td>
      <td>0.004</td>
      <td>0.263</td>
      <td>0.226</td>
      <td>0.103</td>
      <td>0.384</td>
      <td>0.154</td>
      <td>0.585</td>
      <td>-0.031</td>
      <td>0.414</td>
      <td>0.258</td>
      <td>1.000</td>
      <td>0.678</td>
      <td>0.170</td>
      <td>-0.081</td>
      <td>0.026</td>
    </tr>
    <tr>
      <th>V</th>
      <td>0.042</td>
      <td>0.002</td>
      <td>-0.028</td>
      <td>-0.006</td>
      <td>-0.004</td>
      <td>0.271</td>
      <td>0.391</td>
      <td>0.089</td>
      <td>0.341</td>
      <td>0.049</td>
      <td>0.375</td>
      <td>-0.010</td>
      <td>0.283</td>
      <td>0.140</td>
      <td>0.678</td>
      <td>1.000</td>
      <td>0.085</td>
      <td>-0.055</td>
      <td>0.021</td>
    </tr>
    <tr>
      <th>V40</th>
      <td>0.226</td>
      <td>0.020</td>
      <td>-0.052</td>
      <td>-0.025</td>
      <td>0.014</td>
      <td>0.108</td>
      <td>0.079</td>
      <td>-0.068</td>
      <td>0.248</td>
      <td>0.371</td>
      <td>0.150</td>
      <td>-0.008</td>
      <td>0.156</td>
      <td>0.226</td>
      <td>0.170</td>
      <td>0.085</td>
      <td>1.000</td>
      <td>-0.437</td>
      <td>0.023</td>
    </tr>
    <tr>
      <th>ZN</th>
      <td>-0.542</td>
      <td>-0.020</td>
      <td>0.049</td>
      <td>0.033</td>
      <td>0.003</td>
      <td>-0.018</td>
      <td>-0.053</td>
      <td>0.041</td>
      <td>-0.155</td>
      <td>-0.031</td>
      <td>-0.044</td>
      <td>0.465</td>
      <td>-0.072</td>
      <td>-0.177</td>
      <td>-0.081</td>
      <td>-0.055</td>
      <td>-0.437</td>
      <td>1.000</td>
      <td>-0.028</td>
    </tr>
    <tr>
      <th>Y_LABEL</th>
      <td>0.003</td>
      <td>0.044</td>
      <td>-0.088</td>
      <td>-0.034</td>
      <td>0.015</td>
      <td>0.008</td>
      <td>0.014</td>
      <td>0.025</td>
      <td>0.048</td>
      <td>-0.004</td>
      <td>0.024</td>
      <td>0.001</td>
      <td>0.047</td>
      <td>0.029</td>
      <td>0.026</td>
      <td>0.021</td>
      <td>0.023</td>
      <td>-0.028</td>
      <td>1.000</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-aafc0b08-0fd8-4efc-a487-83b08782af1b')"
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
          document.querySelector('#df-aafc0b08-0fd8-4efc-a487-83b08782af1b button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-aafc0b08-0fd8-4efc-a487-83b08782af1b');
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
plt.figure(figsize=(25,20))
sns.set(style="white")

plt.rc('font', family = 'NanumBarunGothic')
plt.title('train 상관관계', size=20)
palette = sns.color_palette('twilight') + sns.color_palette('bright')
cmap = sns.diverging_palette(200,10,as_cmap=True)

mask = np.zeros_like(n_train_cor, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# 히트맵을 그린다.
heatmap1 = sns.heatmap(n_train_cor,
            annot=True,
            mask=mask, 
            cmap=cmap,
            linewidth=.5,
            cbar_kws={"shrink": .5},
            vmin = -1, vmax = 1)

sns.set(font_scale = 1.0)
heatmap1.set_xticklabels(heatmap1.get_xmajorticklabels(),fontsize = 12)
heatmap1.set_yticklabels(heatmap1.get_ymajorticklabels(),fontsize = 12)

plt.xticks(rotation=45)

plt.show()
```


    
![png](/data/2023-02-04-Oil Condition Classification/output_46_0.png)
    


### 2.4.2 데이터셋 분리


```python
all_X
```





  <div id="df-6abd2afa-0282-4a7c-8a3b-3df63e89effa">
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
      <th>COMPONENT_ARBITRARY</th>
      <th>ANONYMOUS_1</th>
      <th>YEAR</th>
      <th>ANONYMOUS_2</th>
      <th>AG</th>
      <th>AL</th>
      <th>B</th>
      <th>BA</th>
      <th>BE</th>
      <th>CA</th>
      <th>...</th>
      <th>PB</th>
      <th>PQINDEX</th>
      <th>S</th>
      <th>SB</th>
      <th>SI</th>
      <th>SN</th>
      <th>TI</th>
      <th>V</th>
      <th>V40</th>
      <th>ZN</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.0</td>
      <td>-0.393763</td>
      <td>4.0</td>
      <td>-0.340760</td>
      <td>-0.150214</td>
      <td>-0.111628</td>
      <td>0.281646</td>
      <td>-0.238453</td>
      <td>-0.041491</td>
      <td>1.141962</td>
      <td>...</td>
      <td>-0.160812</td>
      <td>5.293270</td>
      <td>1.001652</td>
      <td>-0.174727</td>
      <td>2.006643</td>
      <td>0.302478</td>
      <td>0.622282</td>
      <td>-0.10655</td>
      <td>0.899892</td>
      <td>-0.966002</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>-0.426022</td>
      <td>14.0</td>
      <td>-0.022576</td>
      <td>-0.150214</td>
      <td>-0.123127</td>
      <td>-0.437686</td>
      <td>-0.238453</td>
      <td>-0.041491</td>
      <td>1.087302</td>
      <td>...</td>
      <td>0.033010</td>
      <td>-0.259244</td>
      <td>-1.170187</td>
      <td>-0.174727</td>
      <td>-0.179489</td>
      <td>-0.252439</td>
      <td>-0.102635</td>
      <td>-0.10655</td>
      <td>-1.317376</td>
      <td>0.119147</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>-0.173409</td>
      <td>8.0</td>
      <td>-0.340760</td>
      <td>-0.150214</td>
      <td>1.118753</td>
      <td>-0.612659</td>
      <td>0.105735</td>
      <td>-0.041491</td>
      <td>-0.910846</td>
      <td>...</td>
      <td>-0.160812</td>
      <td>-0.260552</td>
      <td>-1.146917</td>
      <td>-0.174727</td>
      <td>-0.179489</td>
      <td>0.025019</td>
      <td>-0.102635</td>
      <td>-0.10655</td>
      <td>-0.740886</td>
      <td>-0.332215</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.0</td>
      <td>1.006399</td>
      <td>3.0</td>
      <td>-0.340760</td>
      <td>-0.150214</td>
      <td>-0.054133</td>
      <td>-0.593217</td>
      <td>-0.238453</td>
      <td>-0.041491</td>
      <td>0.400333</td>
      <td>...</td>
      <td>-0.063901</td>
      <td>-0.242884</td>
      <td>1.044975</td>
      <td>0.557916</td>
      <td>-0.174370</td>
      <td>0.025019</td>
      <td>-0.102635</td>
      <td>-0.10655</td>
      <td>0.482642</td>
      <td>-1.093888</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.0</td>
      <td>0.191634</td>
      <td>8.0</td>
      <td>-0.340760</td>
      <td>-0.150214</td>
      <td>-0.134626</td>
      <td>0.903771</td>
      <td>-0.238453</td>
      <td>-0.041491</td>
      <td>-0.874406</td>
      <td>...</td>
      <td>-0.160812</td>
      <td>-0.129674</td>
      <td>0.690669</td>
      <td>-0.174727</td>
      <td>-0.169250</td>
      <td>-0.252439</td>
      <td>-0.102635</td>
      <td>-0.10655</td>
      <td>0.478611</td>
      <td>-0.866326</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>14090</th>
      <td>2.0</td>
      <td>-0.362928</td>
      <td>7.0</td>
      <td>-0.340760</td>
      <td>-0.150214</td>
      <td>-0.123127</td>
      <td>1.331482</td>
      <td>0.105735</td>
      <td>-0.041491</td>
      <td>-0.918269</td>
      <td>...</td>
      <td>-0.160812</td>
      <td>-0.248773</td>
      <td>1.069210</td>
      <td>-0.174727</td>
      <td>-0.169250</td>
      <td>-0.252439</td>
      <td>-0.102635</td>
      <td>-0.10655</td>
      <td>0.524972</td>
      <td>-1.076961</td>
    </tr>
    <tr>
      <th>14091</th>
      <td>0.0</td>
      <td>-0.085884</td>
      <td>6.0</td>
      <td>-0.340760</td>
      <td>-0.150214</td>
      <td>-0.111628</td>
      <td>0.203880</td>
      <td>-0.238453</td>
      <td>-0.041491</td>
      <td>1.065033</td>
      <td>...</td>
      <td>-0.063901</td>
      <td>-0.265787</td>
      <td>-0.700817</td>
      <td>-0.174727</td>
      <td>-0.153891</td>
      <td>-0.252439</td>
      <td>-0.102635</td>
      <td>-0.10655</td>
      <td>0.164162</td>
      <td>1.540935</td>
    </tr>
    <tr>
      <th>14092</th>
      <td>2.0</td>
      <td>-0.322130</td>
      <td>1.0</td>
      <td>0.295608</td>
      <td>-0.150214</td>
      <td>-0.077131</td>
      <td>-0.622379</td>
      <td>0.105735</td>
      <td>-0.041491</td>
      <td>-0.913545</td>
      <td>...</td>
      <td>-0.063901</td>
      <td>0.150406</td>
      <td>-0.256220</td>
      <td>0.557916</td>
      <td>-0.087334</td>
      <td>-0.252439</td>
      <td>-0.102635</td>
      <td>-0.10655</td>
      <td>-1.115806</td>
      <td>1.339703</td>
    </tr>
    <tr>
      <th>14093</th>
      <td>1.0</td>
      <td>-0.153722</td>
      <td>2.0</td>
      <td>0.295608</td>
      <td>-0.150214</td>
      <td>-0.123127</td>
      <td>-0.583497</td>
      <td>-0.238453</td>
      <td>-0.041491</td>
      <td>0.591983</td>
      <td>...</td>
      <td>-0.063901</td>
      <td>-0.264479</td>
      <td>-1.149169</td>
      <td>-0.174727</td>
      <td>-0.179489</td>
      <td>-0.252439</td>
      <td>-0.102635</td>
      <td>-0.10655</td>
      <td>-1.311328</td>
      <td>0.119147</td>
    </tr>
    <tr>
      <th>14094</th>
      <td>1.0</td>
      <td>-0.295090</td>
      <td>5.0</td>
      <td>-0.340760</td>
      <td>-0.150214</td>
      <td>-0.146124</td>
      <td>0.126115</td>
      <td>-0.238453</td>
      <td>-0.041491</td>
      <td>-0.588956</td>
      <td>...</td>
      <td>0.033010</td>
      <td>-0.218671</td>
      <td>-1.069386</td>
      <td>-0.174727</td>
      <td>-0.148771</td>
      <td>-0.252439</td>
      <td>0.187332</td>
      <td>-0.10655</td>
      <td>-1.256905</td>
      <td>0.043920</td>
    </tr>
  </tbody>
</table>
<p>14095 rows × 34 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-6abd2afa-0282-4a7c-8a3b-3df63e89effa')"
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
          document.querySelector('#df-6abd2afa-0282-4a7c-8a3b-3df63e89effa button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-6abd2afa-0282-4a7c-8a3b-3df63e89effa');
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
all_y
```




    0        0
    1        0
    2        1
    3        0
    4        0
            ..
    14090    0
    14091    0
    14092    0
    14093    0
    14094    0
    Name: Y_LABEL, Length: 14095, dtype: int64




```python
#train_test_split()함수는 기본적으로 테스트 세트를 20%로 나눔 (훈련 세트는 80%)
train_X, val_X, train_y, val_y = train_test_split(all_X, all_y, test_size=0.2, random_state=CFG['SEED'], stratify=all_y)

```

## 2.6 모델링

우선, BaseLine 에서 제공한 Modeling 코드를 사용하기 전
<br/>다른 여러 모델을 사용하여 각각 비교해본다.

> SVM (Support Vector Machine)


```python
model_1 = svm.SVC(gamma='scale')

model_1.fit(train_X, train_y)
 
y_pred = model_1.predict(val_X)
 
print('svm.SVC: %.2f' % (metrics.accuracy_score(y_pred, val_y) * 100))
```

    svm.SVC: 94.75
    

> DecisionTree


```python
model_1 = DecisionTreeClassifier()

model_1.fit(train_X, train_y)
 
y_pred = model_1.predict(val_X)
 
print('DecisionTreeClassifier: %.2f' % (metrics.accuracy_score(y_pred, val_y) * 100))
```

    DecisionTreeClassifier: 91.73
    

> K-Neighbors Classifier


```python
model_1 = KNeighborsClassifier()

model_1.fit(train_X, train_y)
 
y_pred = model_1.predict(val_X)

print('KNeighborsClassifier: %.2f' % (metrics.accuracy_score(y_pred, val_y) * 100))
```

    KNeighborsClassifier: 92.62
    

> Logistic Regression


```python
model_1 = LogisticRegression(solver='lbfgs', max_iter=2000)

model_1.fit(train_X, train_y)
 
y_pred = model_1.predict(val_X)
 
print('LogisticRegression: %.2f' % (metrics.accuracy_score(y_pred, val_y) * 100))
```

    LogisticRegression: 95.32
    

> RandomForest


```python
model_1 = RandomForestClassifier(n_estimators = 100)

model_1.fit(train_X, train_y)
 
y_pred = model_1.predict(val_X)
 
print('RandomForestClassifier: %.2f' % (metrics.accuracy_score(y_pred, val_y) * 100))
```

    RandomForestClassifier: 95.25
    

> LGBM Classifier (Light GBM)


```python
model_1 = lgb.LGBMClassifier()

model_1.fit(train_X, train_y)
 
y_pred = model_1.predict(val_X)
 
print('LightGBM Model accuracy score: %.2f' % (metrics.accuracy_score(y_pred, val_y) * 100))
```

    LightGBM Model accuracy score: 95.25
    

> XGB Classifer


```python
model_1 = xgb.XGBClassifier()

model_1.fit(train_X, train_y)
 
y_pred = model_1.predict(val_X)
 
print('LightGBM Model accuracy score: %.2f' % (metrics.accuracy_score(y_pred, val_y) * 100))
```

    LightGBM Model accuracy score: 95.14
    

### 2.6.1 Custom Dataset


```python
class CustomDataset(Dataset):
    def __init__(self, data_X, data_y, distillation=False):
        # 생성자, 데이터셋의 전처리를 해주는 부분
        super(CustomDataset, self).__init__()
        self.data_X = data_X
        self.data_y = data_y
        self.distillation = distillation
        
    def __len__(self):
        # 데이터셋의 길이. 총 데이터의 개수를 반환하는 함수
        return len(self.data_X)
    
    def __getitem__(self, index):
        # 데이터셋에서 특정 1개의 샘플을 가져와 반환하는 함수
        # dataset[i] 했을 때, i번째 데이터를 가져올 수 있도록 함.
        if self.distillation:
            # 지식 증류 학습 시
            teacher_X = torch.Tensor(self.data_X.iloc[index])
            student_X = torch.Tensor(self.data_X[test_stage_features].iloc[index])
            y = self.data_y.values[index]
            return teacher_X, student_X, y
        else:
            if self.data_y is None:
                test_X = torch.Tensor(self.data_X.iloc[index])
                return test_X
            else:
                teacher_X = torch.Tensor(self.data_X.iloc[index])
                y = self.data_y.values[index]
                return teacher_X, y

```


```python
train_dataset = CustomDataset(train_X, train_y, False)
val_dataset = CustomDataset(val_X, val_y, False)
```


```python
train_loader = DataLoader(train_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=False)
```

### 2.6.2 Define Teacher Model


```python
class Teacher(nn.Module):
    def __init__(self):
        super(Teacher, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features=34, out_features=256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(in_features=256, out_features=1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Linear(in_features=1024, out_features=256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(in_features=256, out_features=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        output = self.classifier(x)
        return output
```

### 2.6.3 Teacher Train / Validation


```python
def train(model, optimizer, train_loader, val_loader, scheduler, device):
    model.to(device)


    best_score = 0
    best_model = None
    criterion = nn.BCELoss().to(device)


    for epoch in range(CFG["EPOCHS"]):
        train_loss = []
  
        model.train()
        for X, y in tqdm(train_loader):
            X = X.float().to(device)
            y = y.float().to(device)
            
            optimizer.zero_grad()
            
            y_pred = model(X)
            
            loss = criterion(y_pred, y.reshape(-1, 1))
            loss.backward()
            
            optimizer.step()


            train_loss.append(loss.item())


        val_loss, val_score = validation_teacher(model, val_loader, criterion, device)
        print(f'Epoch [{epoch}], Train Loss : [{np.mean(train_loss) :.5f}] Val Loss : [{np.mean(val_loss) :.5f}] Val F1 Score : [{val_score:.5f}]')


        if scheduler is not None:
            scheduler.step(val_score)
            
        if best_score < val_score:
            best_model = model
            best_score = val_score
        
    return best_model 
```


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
            
            model_pred = model(X.to(device))
            
            loss = criterion(model_pred, y.reshape(-1, 1))
            val_loss.append(loss.item())      
            
            model_pred = model_pred.squeeze(1).to('cpu')  
            pred_labels += model_pred.tolist()
            true_labels += y.tolist()
        
        pred_labels = np.where(np.array(pred_labels) > threshold, 1, 0)
        val_f1 = competition_metric(true_labels, pred_labels)
    return val_loss, val_f1   

```

### 2.6.4 Run(Teacher Model)


```python
model = Teacher()
model.eval()
optimizer = torch.optim.Adam(model.parameters(), lr=CFG['LEARNING_RATE'])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1, threshold_mode='abs',min_lr=1e-8, verbose=True)


teacher_model = train(model, optimizer, train_loader, val_loader, scheduler, device)
```


      0%|          | 0/45 [00:00<?, ?it/s]



      0%|          | 0/12 [00:00<?, ?it/s]


    Epoch [0], Train Loss : [0.25386] Val Loss : [0.27375] Val F1 Score : [0.76599]
    


      0%|          | 0/45 [00:00<?, ?it/s]



      0%|          | 0/12 [00:00<?, ?it/s]


    Epoch [1], Train Loss : [0.19623] Val Loss : [0.23303] Val F1 Score : [0.77925]
    


      0%|          | 0/45 [00:00<?, ?it/s]



      0%|          | 0/12 [00:00<?, ?it/s]


    Epoch [2], Train Loss : [0.17424] Val Loss : [0.22012] Val F1 Score : [0.79782]
    


      0%|          | 0/45 [00:00<?, ?it/s]



      0%|          | 0/12 [00:00<?, ?it/s]


    Epoch [3], Train Loss : [0.16466] Val Loss : [0.29795] Val F1 Score : [0.79885]
    


      0%|          | 0/45 [00:00<?, ?it/s]



      0%|          | 0/12 [00:00<?, ?it/s]


    Epoch [4], Train Loss : [0.15722] Val Loss : [0.27205] Val F1 Score : [0.80131]
    


      0%|          | 0/45 [00:00<?, ?it/s]



      0%|          | 0/12 [00:00<?, ?it/s]


    Epoch [5], Train Loss : [0.16333] Val Loss : [0.32019] Val F1 Score : [0.78589]
    


      0%|          | 0/45 [00:00<?, ?it/s]



      0%|          | 0/12 [00:00<?, ?it/s]


    Epoch [6], Train Loss : [0.16928] Val Loss : [0.27536] Val F1 Score : [0.79727]
    Epoch 00007: reducing learning rate of group 0 to 5.0000e-03.
    


      0%|          | 0/45 [00:00<?, ?it/s]



      0%|          | 0/12 [00:00<?, ?it/s]


    Epoch [7], Train Loss : [0.14803] Val Loss : [0.25251] Val F1 Score : [0.79901]
    


      0%|          | 0/45 [00:00<?, ?it/s]



      0%|          | 0/12 [00:00<?, ?it/s]


    Epoch [8], Train Loss : [0.14723] Val Loss : [0.29791] Val F1 Score : [0.80104]
    Epoch 00009: reducing learning rate of group 0 to 2.5000e-03.
    


      0%|          | 0/45 [00:00<?, ?it/s]



      0%|          | 0/12 [00:00<?, ?it/s]


    Epoch [9], Train Loss : [0.14080] Val Loss : [0.28969] Val F1 Score : [0.80445]
    


      0%|          | 0/45 [00:00<?, ?it/s]



      0%|          | 0/12 [00:00<?, ?it/s]


    Epoch [10], Train Loss : [0.13199] Val Loss : [0.33207] Val F1 Score : [0.79666]
    


      0%|          | 0/45 [00:00<?, ?it/s]



      0%|          | 0/12 [00:00<?, ?it/s]


    Epoch [11], Train Loss : [0.15141] Val Loss : [0.29886] Val F1 Score : [0.79477]
    Epoch 00012: reducing learning rate of group 0 to 1.2500e-03.
    


      0%|          | 0/45 [00:00<?, ?it/s]



      0%|          | 0/12 [00:00<?, ?it/s]


    Epoch [12], Train Loss : [0.13733] Val Loss : [0.30850] Val F1 Score : [0.79807]
    


      0%|          | 0/45 [00:00<?, ?it/s]



      0%|          | 0/12 [00:00<?, ?it/s]


    Epoch [13], Train Loss : [0.12514] Val Loss : [0.31127] Val F1 Score : [0.79356]
    Epoch 00014: reducing learning rate of group 0 to 6.2500e-04.
    


      0%|          | 0/45 [00:00<?, ?it/s]



      0%|          | 0/12 [00:00<?, ?it/s]


    Epoch [14], Train Loss : [0.12362] Val Loss : [0.30225] Val F1 Score : [0.79438]
    


      0%|          | 0/45 [00:00<?, ?it/s]



      0%|          | 0/12 [00:00<?, ?it/s]


    Epoch [15], Train Loss : [0.12048] Val Loss : [0.31231] Val F1 Score : [0.79213]
    Epoch 00016: reducing learning rate of group 0 to 3.1250e-04.
    


      0%|          | 0/45 [00:00<?, ?it/s]



      0%|          | 0/12 [00:00<?, ?it/s]


    Epoch [16], Train Loss : [0.11990] Val Loss : [0.31361] Val F1 Score : [0.80094]
    


      0%|          | 0/45 [00:00<?, ?it/s]



      0%|          | 0/12 [00:00<?, ?it/s]


    Epoch [17], Train Loss : [0.12354] Val Loss : [0.32241] Val F1 Score : [0.79849]
    Epoch 00018: reducing learning rate of group 0 to 1.5625e-04.
    


      0%|          | 0/45 [00:00<?, ?it/s]



      0%|          | 0/12 [00:00<?, ?it/s]


    Epoch [18], Train Loss : [0.12005] Val Loss : [0.32464] Val F1 Score : [0.79368]
    


      0%|          | 0/45 [00:00<?, ?it/s]



      0%|          | 0/12 [00:00<?, ?it/s]


    Epoch [19], Train Loss : [0.11690] Val Loss : [0.31878] Val F1 Score : [0.79104]
    Epoch 00020: reducing learning rate of group 0 to 7.8125e-05.
    


      0%|          | 0/45 [00:00<?, ?it/s]



      0%|          | 0/12 [00:00<?, ?it/s]


    Epoch [20], Train Loss : [0.11789] Val Loss : [0.31385] Val F1 Score : [0.79419]
    


      0%|          | 0/45 [00:00<?, ?it/s]



      0%|          | 0/12 [00:00<?, ?it/s]


    Epoch [21], Train Loss : [0.12152] Val Loss : [0.33572] Val F1 Score : [0.78721]
    Epoch 00022: reducing learning rate of group 0 to 3.9063e-05.
    


      0%|          | 0/45 [00:00<?, ?it/s]



      0%|          | 0/12 [00:00<?, ?it/s]


    Epoch [22], Train Loss : [0.12934] Val Loss : [0.29623] Val F1 Score : [0.79575]
    


      0%|          | 0/45 [00:00<?, ?it/s]



      0%|          | 0/12 [00:00<?, ?it/s]


    Epoch [23], Train Loss : [0.11590] Val Loss : [0.30043] Val F1 Score : [0.77370]
    Epoch 00024: reducing learning rate of group 0 to 1.9531e-05.
    


      0%|          | 0/45 [00:00<?, ?it/s]



      0%|          | 0/12 [00:00<?, ?it/s]


    Epoch [24], Train Loss : [0.11642] Val Loss : [0.32704] Val F1 Score : [0.80716]
    


      0%|          | 0/45 [00:00<?, ?it/s]



      0%|          | 0/12 [00:00<?, ?it/s]


    Epoch [25], Train Loss : [0.11644] Val Loss : [0.34076] Val F1 Score : [0.79530]
    


      0%|          | 0/45 [00:00<?, ?it/s]



      0%|          | 0/12 [00:00<?, ?it/s]


    Epoch [26], Train Loss : [0.11658] Val Loss : [0.30919] Val F1 Score : [0.79384]
    Epoch 00027: reducing learning rate of group 0 to 9.7656e-06.
    


      0%|          | 0/45 [00:00<?, ?it/s]



      0%|          | 0/12 [00:00<?, ?it/s]


    Epoch [27], Train Loss : [0.11544] Val Loss : [0.31518] Val F1 Score : [0.80360]
    


      0%|          | 0/45 [00:00<?, ?it/s]



      0%|          | 0/12 [00:00<?, ?it/s]


    Epoch [28], Train Loss : [0.12797] Val Loss : [0.30333] Val F1 Score : [0.79039]
    Epoch 00029: reducing learning rate of group 0 to 4.8828e-06.
    


      0%|          | 0/45 [00:00<?, ?it/s]



      0%|          | 0/12 [00:00<?, ?it/s]


    Epoch [29], Train Loss : [0.11624] Val Loss : [0.31200] Val F1 Score : [0.79449]
    

### 2.6.5 Define Student Model


```python
class Student(nn.Module):
    def __init__(self):
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
        
    def forward(self, x):
        output = self.classifier(x)
        return output
```

### 2.6.6 Define Knowledge Distillation Loss


```python
def distillation(student_logits, labels, teacher_logits, alpha):
    distillation_loss = nn.BCELoss()(student_logits, teacher_logits)
    student_loss = nn.BCELoss()(student_logits, labels.reshape(-1, 1))
    return alpha * student_loss + (1-alpha) * distillation_loss
```


```python
def distill_loss(output, target, teacher_output, loss_fn=distillation, opt=optimizer):
    loss_b = loss_fn(output, target, teacher_output, alpha=0.1)


    if opt is not None:
        opt.zero_grad()
        loss_b.backward()
        opt.step()


    return loss_b.item()
```

### 2.6.7 Student Train / Validation


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

### 2.6.8 Run (Student Model)


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


    Epoch [0], Train Loss : [0.34100] Val Loss : [0.30850] Val F1 Score : [0.49910]
    


      0%|          | 0/45 [00:00<?, ?it/s]



      0%|          | 0/12 [00:00<?, ?it/s]


    Epoch [1], Train Loss : [0.30942] Val Loss : [0.30457] Val F1 Score : [0.51346]
    


      0%|          | 0/45 [00:00<?, ?it/s]



      0%|          | 0/12 [00:00<?, ?it/s]


    Epoch [2], Train Loss : [0.30595] Val Loss : [0.30768] Val F1 Score : [0.48608]
    


      0%|          | 0/45 [00:00<?, ?it/s]



      0%|          | 0/12 [00:00<?, ?it/s]


    Epoch [3], Train Loss : [0.30570] Val Loss : [0.29888] Val F1 Score : [0.49582]
    Epoch 00004: reducing learning rate of group 0 to 5.0000e-03.
    


      0%|          | 0/45 [00:00<?, ?it/s]



      0%|          | 0/12 [00:00<?, ?it/s]


    Epoch [4], Train Loss : [0.29965] Val Loss : [0.30245] Val F1 Score : [0.50112]
    


      0%|          | 0/45 [00:00<?, ?it/s]



      0%|          | 0/12 [00:00<?, ?it/s]


    Epoch [5], Train Loss : [0.29774] Val Loss : [0.30124] Val F1 Score : [0.50643]
    Epoch 00006: reducing learning rate of group 0 to 2.5000e-03.
    


      0%|          | 0/45 [00:00<?, ?it/s]



      0%|          | 0/12 [00:00<?, ?it/s]


    Epoch [6], Train Loss : [0.29454] Val Loss : [0.30022] Val F1 Score : [0.51485]
    


      0%|          | 0/45 [00:00<?, ?it/s]



      0%|          | 0/12 [00:00<?, ?it/s]


    Epoch [7], Train Loss : [0.29857] Val Loss : [0.30079] Val F1 Score : [0.51623]
    


      0%|          | 0/45 [00:00<?, ?it/s]



      0%|          | 0/12 [00:00<?, ?it/s]


    Epoch [8], Train Loss : [0.29768] Val Loss : [0.30305] Val F1 Score : [0.51011]
    


      0%|          | 0/45 [00:00<?, ?it/s]



      0%|          | 0/12 [00:00<?, ?it/s]


    Epoch [9], Train Loss : [0.29241] Val Loss : [0.30238] Val F1 Score : [0.51438]
    Epoch 00010: reducing learning rate of group 0 to 1.2500e-03.
    


      0%|          | 0/45 [00:00<?, ?it/s]



      0%|          | 0/12 [00:00<?, ?it/s]


    Epoch [10], Train Loss : [0.29246] Val Loss : [0.30039] Val F1 Score : [0.51369]
    


      0%|          | 0/45 [00:00<?, ?it/s]



      0%|          | 0/12 [00:00<?, ?it/s]


    Epoch [11], Train Loss : [0.29061] Val Loss : [0.30033] Val F1 Score : [0.51816]
    


      0%|          | 0/45 [00:00<?, ?it/s]



      0%|          | 0/12 [00:00<?, ?it/s]


    Epoch [12], Train Loss : [0.29335] Val Loss : [0.30349] Val F1 Score : [0.51508]
    


      0%|          | 0/45 [00:00<?, ?it/s]



      0%|          | 0/12 [00:00<?, ?it/s]


    Epoch [13], Train Loss : [0.28871] Val Loss : [0.29981] Val F1 Score : [0.52221]
    


      0%|          | 0/45 [00:00<?, ?it/s]



      0%|          | 0/12 [00:00<?, ?it/s]


    Epoch [14], Train Loss : [0.28952] Val Loss : [0.30384] Val F1 Score : [0.52211]
    


      0%|          | 0/45 [00:00<?, ?it/s]



      0%|          | 0/12 [00:00<?, ?it/s]


    Epoch [15], Train Loss : [0.29373] Val Loss : [0.30562] Val F1 Score : [0.52579]
    


      0%|          | 0/45 [00:00<?, ?it/s]



      0%|          | 0/12 [00:00<?, ?it/s]


    Epoch [16], Train Loss : [0.29239] Val Loss : [0.30211] Val F1 Score : [0.51699]
    


      0%|          | 0/45 [00:00<?, ?it/s]



      0%|          | 0/12 [00:00<?, ?it/s]


    Epoch [17], Train Loss : [0.28973] Val Loss : [0.30094] Val F1 Score : [0.52650]
    


      0%|          | 0/45 [00:00<?, ?it/s]



      0%|          | 0/12 [00:00<?, ?it/s]


    Epoch [18], Train Loss : [0.28574] Val Loss : [0.30739] Val F1 Score : [0.52597]
    


      0%|          | 0/45 [00:00<?, ?it/s]



      0%|          | 0/12 [00:00<?, ?it/s]


    Epoch [19], Train Loss : [0.29196] Val Loss : [0.30142] Val F1 Score : [0.52949]
    


      0%|          | 0/45 [00:00<?, ?it/s]



      0%|          | 0/12 [00:00<?, ?it/s]


    Epoch [20], Train Loss : [0.28639] Val Loss : [0.30043] Val F1 Score : [0.52366]
    


      0%|          | 0/45 [00:00<?, ?it/s]



      0%|          | 0/12 [00:00<?, ?it/s]


    Epoch [21], Train Loss : [0.29203] Val Loss : [0.30206] Val F1 Score : [0.52757]
    Epoch 00022: reducing learning rate of group 0 to 6.2500e-04.
    


      0%|          | 0/45 [00:00<?, ?it/s]



      0%|          | 0/12 [00:00<?, ?it/s]


    Epoch [22], Train Loss : [0.28658] Val Loss : [0.30026] Val F1 Score : [0.53372]
    


      0%|          | 0/45 [00:00<?, ?it/s]



      0%|          | 0/12 [00:00<?, ?it/s]


    Epoch [23], Train Loss : [0.28386] Val Loss : [0.30378] Val F1 Score : [0.52314]
    


      0%|          | 0/45 [00:00<?, ?it/s]



      0%|          | 0/12 [00:00<?, ?it/s]


    Epoch [24], Train Loss : [0.28414] Val Loss : [0.30249] Val F1 Score : [0.53472]
    


      0%|          | 0/45 [00:00<?, ?it/s]



      0%|          | 0/12 [00:00<?, ?it/s]


    Epoch [25], Train Loss : [0.28433] Val Loss : [0.30188] Val F1 Score : [0.53430]
    


      0%|          | 0/45 [00:00<?, ?it/s]



      0%|          | 0/12 [00:00<?, ?it/s]


    Epoch [26], Train Loss : [0.28871] Val Loss : [0.30604] Val F1 Score : [0.53264]
    Epoch 00027: reducing learning rate of group 0 to 3.1250e-04.
    


      0%|          | 0/45 [00:00<?, ?it/s]



      0%|          | 0/12 [00:00<?, ?it/s]


    Epoch [27], Train Loss : [0.28440] Val Loss : [0.30558] Val F1 Score : [0.53647]
    


      0%|          | 0/45 [00:00<?, ?it/s]



      0%|          | 0/12 [00:00<?, ?it/s]


    Epoch [28], Train Loss : [0.28185] Val Loss : [0.30201] Val F1 Score : [0.53979]
    


      0%|          | 0/45 [00:00<?, ?it/s]



      0%|          | 0/12 [00:00<?, ?it/s]


    Epoch [29], Train Loss : [0.28271] Val Loss : [0.30270] Val F1 Score : [0.54218]
    

### 2.6.9 Choose Inference Threshold


```python
def choose_threshold(model, val_loader, device):
    model.to(device)
    model.eval()
    
    thresholds = [0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    pred_labels = []
    true_labels = []
    
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
            pred_labels_thr = np.where(np.array(pred_labels) > threshold, 1, 0)
            score_thr = competition_metric(true_labels, pred_labels_thr)
            if best_score < score_thr:
                best_score = score_thr
                best_thr = threshold
    return best_thr, best_score
```


```python
best_threshold, best_score = choose_threshold(best_student_model, val_loader, device)
print(f'Best Threshold : [{best_threshold}], Score : [{best_score:.5f}]')
```


      0%|          | 0/12 [00:00<?, ?it/s]


    Best Threshold : [0.2], Score : [0.55298]
    

## 2.7 Inference


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
    

## 2.8 Submit


```python
submit['Y_LABEL'] = preds
submit.head()
# submit.to_csv('./submit.csv', index=False)
```





  <div id="df-fab8777e-73b3-4525-95a8-5ef687c05a77">
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
      <td>1</td>
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
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-fab8777e-73b3-4525-95a8-5ef687c05a77')"
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
          document.querySelector('#df-fab8777e-73b3-4525-95a8-5ef687c05a77 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-fab8777e-73b3-4525-95a8-5ef687c05a77');
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

```
