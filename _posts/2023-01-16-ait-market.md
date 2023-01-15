# # WHAT? 

# "AIT마켓" : AI트레이더 거래 플랫폼
![image.png](attachment:image.png)

## AIT란?
> "AIT마켓"에서 **`AIT`**는 "AI Trader"의 약자이며, "로보 어드바이저"에서 파생 및 차별화하여 고안한 용어로,  
AI로 매수/매도를 결정하는 단기 투자에 대한 AI모델입니다. 

## AIT 마켓이란?

> ![image-2.png](attachment:image-2.png)  
"AIT마켓"은 운영자, 개발자, 구매자, 총 3명의 참여자가 상호작용하여 AI트레이더를 거래하는 플랫폼입니다.


# # WHY?

## 왜 AIT 마켓인가?
> **1) 빠르게 성장 중인 로보 어드바이저 시장**  
![image-2.png](attachment:image-2.png)
국내에서는 대표적으로 에임, 파운트, 핀트와 같은 로보어드바이저 서비스가 활성화되어있습니다. 이러한 로보어드바이저 서비스의 가입자수와 관리자산 금액이 세계적으로 급격히 증가하면서, 로보어드바이저의 시장규모는 크게 성장하고 있습니다.  

<br>
    
> **2) C2C 시장으로의 도전**  
기존 로보어드바이저 시장은 위탁운용(D2C)서비스로 이루어져있어 서비스의 개방성에 제한이 있다는 문제점이 존재합니다. 
![image-3.png](attachment:image-3.png)
당근마켓, 유튜브와 같이 C2C 시장이 형성된 후, 시장이 폭발적으로 성장한 예시들이 다수 존재합니다. "AIT 마켓"은 개발자와 구매자의 AI트레이더 거래를 연결하는 C2C 시장을 개척함으로써 인공지능 기반의 로보어드바이저 시장 성장에 기여할 수 있습니다. 

# # HOW?

## AIT 마켓 참여자


> "AIT 마켓"에서는 **`운영자`**, **`개발자`**, **`구매자`**, 총 3명의 참여자가 활동합니다.
![image.png](attachment:image.png)  
**`운영자`**는 학습 데이터 및 수익률 벤치마크를 제공함으로써 개발자와 구매자 간의 중개 역할을 하며, AI트레이더 시장 플랫폼을 제공합니다. <br> <br> 
**`개발자`**는 인공지능 모델 학습을 통하여 AI 트레이더를 개발하고, 생성한 AI트레이더를 AIT마켓에 제출합니다.  <br> <br> 
**`구매자`**는 AIT마켓에서 매매 히스토리 및 벤치마크 정보를 제공받아 AI트레이더를 구매하고, 이를  투자에 활용합니다.

## AIT 마켓 프로세스 

> AIT마켓은 크게 4단계의 프로세스로 진행됩니다.  
![image.png](attachment:image.png) <br>  1) **`AIT 마켓 운영자`**는 KRX 및 각종 포털, 신문사에서 데이터를 수집하고, 구조화하여  AI트레이더 개발자에게 API 방식으로 제공합니다.   <br><br>
2) **`AI트레이더 개발자`**는 AI트레이더를 개발하여 AIT마켓에 제출합니다. <br>  
3) **`AI트레이더 구매자`**는 AIT마켓으로부터 AI트레이더 수익률 리더보드를 제공받습니다.<br>  
4) **`AI트레이더 구매자`**는 3)에서 제공받은 정보를 토대로 AI트레이더 결제 및 구독을 진행합니다.

## "AIT마켓"에 대한 ipynb 개념증명 시나리오
> **본 ipynb 파일은 다음의 순서로 "AI트레이더 거래 플랫폼"을 시연합니다.**


*본 ipynb 파일은 시간 복잡도보다는 코드의 가독성을 중심으로 작성한 코드임을 알립니다.   


    
------------------------------------------------------------------------------------------------------------------------------    

### 1) AIT 운영자 : 데이터 수집 / 구조화하여 제공 
+ 종목 선정

+ 일별 주가 데이터 다운로드

+ 주가 데이터에 보조지표 독립변수 추가하기

+ 주가 데이터 스케일 표준화 

+ 학습/시험 데이터셋으로 분할하고 학습 데이터셋을 AIT 개발 커뮤니티에 공개 

### 2) AIT 생산자 : AI 트레이더 생성 
+ 수원랩 XGBoost팀
    + 데이터 전처리

    + 데이터 모델링 

    + 상승 확률 경계값 설정

    + 수익성 검증

    + 모델 저장 후 AIT마켓에 제출 (출시) 
    
+ 경기랩 Logistic Regression팀
    + 데이터 전처리

    + 데이터 모델링 

    + 상승 확률 경계값 설정

    + 수익성 검증

    + 모델 저장 후 AIT마켓에 제출 (출시)
+ 뽀로로 LightGBM팀
    + 데이터 전처리

    + 데이터 모델링 

    + 상승 확률 경계값 설정

    + 수익성 검증

    + 모델 저장 후 AIT마켓에 제출 (출시)
+ 티라노 Gaussian naive bayes팀
    + 데이터 전처리

    + 데이터 모델링 

    + 상승 확률 경계값 설정

    + 수익성 검증

    + 모델 저장 후 AIT마켓에 제출 (출시)
+ 폴리 Random Forest팀
    + 데이터 전처리

    + 데이터 모델링 

    + 상승 확률 경계값 설정

    + 수익성 검증

    + 모델 저장 후 AIT마켓에 제출 (출시)
+ 쌍둥이 K nearest neighbor팀
    + 데이터 전처리

    + 데이터 모델링 

    + 상승 확률 경계값 설정

    + 수익성 검증

    + 모델 저장 후 AIT마켓에 제출 (출시)

### 3) AIT 운영자 : 모델 별 수익률 비교

+ 비공개 데이터 및 제출된 모델 불러오기 

+ 주문일지 작성하기 

+ 매매 시뮬레이션을 실행하여 수익률 계산

+ 수익률 리더보드 및 시각화 

### 4) AIT 구매자 : AI트레이더 구독

+ 구독 시스템 챗봇 예시

### 추후 웹/앱 개발예정

requirement library 설치


```python
! pip install pandas
! pip install numpy
! pip install -U finance-datareader
! pip install pykrx
! pip install lightgbm
! pip install tqdm
! pip install ta
! pip install xgboost
! pip install -U scikit-learn
! pip install seaborn
```

    Requirement already satisfied: pandas in c:\programdata\anaconda3\lib\site-packages (1.3.1)
    Requirement already satisfied: python-dateutil>=2.7.3 in c:\programdata\anaconda3\lib\site-packages (from pandas) (2.8.2)
    Requirement already satisfied: pytz>=2017.3 in c:\programdata\anaconda3\lib\site-packages (from pandas) (2021.1)
    Requirement already satisfied: numpy>=1.17.3 in c:\programdata\anaconda3\lib\site-packages (from pandas) (1.20.3)
    Requirement already satisfied: six>=1.5 in c:\programdata\anaconda3\lib\site-packages (from python-dateutil>=2.7.3->pandas) (1.16.0)
    Requirement already satisfied: numpy in c:\programdata\anaconda3\lib\site-packages (1.20.3)
    Requirement already satisfied: finance-datareader in c:\programdata\anaconda3\lib\site-packages (0.9.31)
    Collecting finance-datareader
      Downloading finance_datareader-0.9.34-py3-none-any.whl (17 kB)
    Requirement already satisfied: lxml in c:\programdata\anaconda3\lib\site-packages (from finance-datareader) (4.6.3)
    Requirement already satisfied: requests>=2.3.0 in c:\programdata\anaconda3\lib\site-packages (from finance-datareader) (2.25.1)
    Requirement already satisfied: requests-file in c:\programdata\anaconda3\lib\site-packages (from finance-datareader) (1.5.1)
    Requirement already satisfied: tqdm in c:\programdata\anaconda3\lib\site-packages (from finance-datareader) (4.62.1)
    Requirement already satisfied: pandas>=0.19.2 in c:\programdata\anaconda3\lib\site-packages (from finance-datareader) (1.3.1)
    Requirement already satisfied: pytz>=2017.3 in c:\programdata\anaconda3\lib\site-packages (from pandas>=0.19.2->finance-datareader) (2021.1)
    Requirement already satisfied: python-dateutil>=2.7.3 in c:\programdata\anaconda3\lib\site-packages (from pandas>=0.19.2->finance-datareader) (2.8.2)
    Requirement already satisfied: numpy>=1.17.3 in c:\programdata\anaconda3\lib\site-packages (from pandas>=0.19.2->finance-datareader) (1.20.3)
    Requirement already satisfied: six>=1.5 in c:\programdata\anaconda3\lib\site-packages (from python-dateutil>=2.7.3->pandas>=0.19.2->finance-datareader) (1.16.0)
    Requirement already satisfied: certifi>=2017.4.17 in c:\programdata\anaconda3\lib\site-packages (from requests>=2.3.0->finance-datareader) (2021.10.8)
    Requirement already satisfied: urllib3<1.27,>=1.21.1 in c:\programdata\anaconda3\lib\site-packages (from requests>=2.3.0->finance-datareader) (1.26.6)
    Requirement already satisfied: idna<3,>=2.5 in c:\programdata\anaconda3\lib\site-packages (from requests>=2.3.0->finance-datareader) (2.10)
    Requirement already satisfied: chardet<5,>=3.0.2 in c:\programdata\anaconda3\lib\site-packages (from requests>=2.3.0->finance-datareader) (4.0.0)
    Requirement already satisfied: colorama in c:\programdata\anaconda3\lib\site-packages (from tqdm->finance-datareader) (0.4.4)
    Installing collected packages: finance-datareader
      Attempting uninstall: finance-datareader
        Found existing installation: finance-datareader 0.9.31
        Uninstalling finance-datareader-0.9.31:
          Successfully uninstalled finance-datareader-0.9.31
    Successfully installed finance-datareader-0.9.34
    Collecting pykrx
      Downloading pykrx-1.0.37-py3-none-any.whl (97 kB)
    Collecting datetime
      Downloading DateTime-4.5-py2.py3-none-any.whl (52 kB)
    Requirement already satisfied: requests in c:\programdata\anaconda3\lib\site-packages (from pykrx) (2.25.1)
    Requirement already satisfied: xlrd in c:\programdata\anaconda3\lib\site-packages (from pykrx) (2.0.1)
    Requirement already satisfied: pandas in c:\programdata\anaconda3\lib\site-packages (from pykrx) (1.3.1)
    Collecting deprecated
      Downloading Deprecated-1.2.13-py2.py3-none-any.whl (9.6 kB)
    Requirement already satisfied: numpy in c:\programdata\anaconda3\lib\site-packages (from pykrx) (1.20.3)
    Requirement already satisfied: pytz in c:\programdata\anaconda3\lib\site-packages (from datetime->pykrx) (2021.1)
    Requirement already satisfied: zope.interface in c:\programdata\anaconda3\lib\site-packages (from datetime->pykrx) (5.4.0)
    Requirement already satisfied: wrapt<2,>=1.10 in c:\programdata\anaconda3\lib\site-packages (from deprecated->pykrx) (1.12.1)
    Requirement already satisfied: python-dateutil>=2.7.3 in c:\programdata\anaconda3\lib\site-packages (from pandas->pykrx) (2.8.2)
    Requirement already satisfied: six>=1.5 in c:\programdata\anaconda3\lib\site-packages (from python-dateutil>=2.7.3->pandas->pykrx) (1.16.0)
    Requirement already satisfied: idna<3,>=2.5 in c:\programdata\anaconda3\lib\site-packages (from requests->pykrx) (2.10)
    Requirement already satisfied: chardet<5,>=3.0.2 in c:\programdata\anaconda3\lib\site-packages (from requests->pykrx) (4.0.0)
    Requirement already satisfied: certifi>=2017.4.17 in c:\programdata\anaconda3\lib\site-packages (from requests->pykrx) (2021.10.8)
    Requirement already satisfied: urllib3<1.27,>=1.21.1 in c:\programdata\anaconda3\lib\site-packages (from requests->pykrx) (1.26.6)
    Requirement already satisfied: setuptools in c:\programdata\anaconda3\lib\site-packages (from zope.interface->datetime->pykrx) (52.0.0.post20210125)
    Installing collected packages: deprecated, datetime, pykrx
    Successfully installed datetime-4.5 deprecated-1.2.13 pykrx-1.0.37
    Collecting lightgbm
      Downloading lightgbm-3.3.2-py3-none-win_amd64.whl (1.0 MB)
    Requirement already satisfied: numpy in c:\programdata\anaconda3\lib\site-packages (from lightgbm) (1.20.3)
    Requirement already satisfied: wheel in c:\programdata\anaconda3\lib\site-packages (from lightgbm) (0.37.0)
    Requirement already satisfied: scikit-learn!=0.22.0 in c:\programdata\anaconda3\lib\site-packages (from lightgbm) (0.24.2)
    Requirement already satisfied: scipy in c:\programdata\anaconda3\lib\site-packages (from lightgbm) (1.6.2)
    Requirement already satisfied: joblib>=0.11 in c:\programdata\anaconda3\lib\site-packages (from scikit-learn!=0.22.0->lightgbm) (1.0.1)
    Requirement already satisfied: threadpoolctl>=2.0.0 in c:\programdata\anaconda3\lib\site-packages (from scikit-learn!=0.22.0->lightgbm) (2.2.0)
    Installing collected packages: lightgbm
    Successfully installed lightgbm-3.3.2
    Requirement already satisfied: tqdm in c:\programdata\anaconda3\lib\site-packages (4.62.1)
    Requirement already satisfied: colorama in c:\programdata\anaconda3\lib\site-packages (from tqdm) (0.4.4)
    Collecting ta
      Downloading ta-0.10.1.tar.gz (24 kB)
    Requirement already satisfied: numpy in c:\programdata\anaconda3\lib\site-packages (from ta) (1.20.3)
    Requirement already satisfied: pandas in c:\programdata\anaconda3\lib\site-packages (from ta) (1.3.1)
    Requirement already satisfied: pytz>=2017.3 in c:\programdata\anaconda3\lib\site-packages (from pandas->ta) (2021.1)
    Requirement already satisfied: python-dateutil>=2.7.3 in c:\programdata\anaconda3\lib\site-packages (from pandas->ta) (2.8.2)
    Requirement already satisfied: six>=1.5 in c:\programdata\anaconda3\lib\site-packages (from python-dateutil>=2.7.3->pandas->ta) (1.16.0)
    Building wheels for collected packages: ta
      Building wheel for ta (setup.py): started
      Building wheel for ta (setup.py): finished with status 'done'
      Created wheel for ta: filename=ta-0.10.1-py3-none-any.whl size=28987 sha256=6db9189e6bb59c36cbca11de59dfff9c875c7de20066074a67592cd7d9d885e7
      Stored in directory: c:\users\hongr\appdata\local\pip\cache\wheels\18\9a\81\694fa8602da445fa009fd13c8da25001be19efdfb67a9cc348
    Successfully built ta
    Installing collected packages: ta
    Successfully installed ta-0.10.1
    Requirement already satisfied: xgboost in c:\programdata\anaconda3\lib\site-packages (1.4.2)
    Requirement already satisfied: numpy in c:\programdata\anaconda3\lib\site-packages (from xgboost) (1.20.3)
    Requirement already satisfied: scipy in c:\programdata\anaconda3\lib\site-packages (from xgboost) (1.6.2)
    Requirement already satisfied: scikit-learn in c:\programdata\anaconda3\lib\site-packages (0.24.2)
    Collecting scikit-learn
      Downloading scikit_learn-1.1.2-cp38-cp38-win_amd64.whl (7.3 MB)
    Requirement already satisfied: scipy>=1.3.2 in c:\programdata\anaconda3\lib\site-packages (from scikit-learn) (1.6.2)
    Requirement already satisfied: numpy>=1.17.3 in c:\programdata\anaconda3\lib\site-packages (from scikit-learn) (1.20.3)
    Requirement already satisfied: threadpoolctl>=2.0.0 in c:\programdata\anaconda3\lib\site-packages (from scikit-learn) (2.2.0)
    Requirement already satisfied: joblib>=1.0.0 in c:\programdata\anaconda3\lib\site-packages (from scikit-learn) (1.0.1)
    Installing collected packages: scikit-learn
      Attempting uninstall: scikit-learn
        Found existing installation: scikit-learn 0.24.2
        Uninstalling scikit-learn-0.24.2:
    

    ERROR: Could not install packages due to an OSError: [WinError 5] 액세스가 거부되었습니다: 'c:\\programdata\\anaconda3\\lib\\site-packages\\scikit_learn-0.24.2.dist-info\\COPYING'
    Consider using the `--user` option or check the permissions.
    
    

    Requirement already satisfied: seaborn in c:\programdata\anaconda3\lib\site-packages (0.11.2)
    Requirement already satisfied: matplotlib>=2.2 in c:\programdata\anaconda3\lib\site-packages (from seaborn) (3.4.2)
    Requirement already satisfied: numpy>=1.15 in c:\programdata\anaconda3\lib\site-packages (from seaborn) (1.20.3)
    Requirement already satisfied: pandas>=0.23 in c:\programdata\anaconda3\lib\site-packages (from seaborn) (1.3.1)
    Requirement already satisfied: scipy>=1.0 in c:\programdata\anaconda3\lib\site-packages (from seaborn) (1.6.2)
    Requirement already satisfied: python-dateutil>=2.7 in c:\programdata\anaconda3\lib\site-packages (from matplotlib>=2.2->seaborn) (2.8.2)
    Requirement already satisfied: pillow>=6.2.0 in c:\programdata\anaconda3\lib\site-packages (from matplotlib>=2.2->seaborn) (8.3.1)
    Requirement already satisfied: kiwisolver>=1.0.1 in c:\programdata\anaconda3\lib\site-packages (from matplotlib>=2.2->seaborn) (1.3.1)
    Requirement already satisfied: cycler>=0.10 in c:\programdata\anaconda3\lib\site-packages (from matplotlib>=2.2->seaborn) (0.10.0)
    Requirement already satisfied: pyparsing>=2.2.1 in c:\programdata\anaconda3\lib\site-packages (from matplotlib>=2.2->seaborn) (2.4.7)
    Requirement already satisfied: six in c:\programdata\anaconda3\lib\site-packages (from cycler>=0.10->matplotlib>=2.2->seaborn) (1.16.0)
    Requirement already satisfied: pytz>=2017.3 in c:\programdata\anaconda3\lib\site-packages (from pandas>=0.23->seaborn) (2021.1)
    


```python
import os
import time
import pandas as pd
import numpy as np
import datetime
import warnings
from tqdm import tqdm
import platform
import matplotlib
import pickle


warnings.filterwarnings('ignore')
```

# 1. AIT 운영자 : 데이터 수집 / 구조화하여 제공 
 > AIT 마켓 운영자가 KRX 등 주가 데이터 데이터베이스로부터 데이터를 수집/가공하여 AIT 개발자들이 모델을 만들 수 있도록 학습 데이터를 제공하는 단계입니다. 
![image-2.png](attachment:image-2.png) 

## 1.1. 종목 선정


2017년 1월 2일에 KOSPI 및 KOSDAQ 시장에 존속하고 있었던 약 2000개 종목을 대상으로 선정합니다.


```python
from pykrx import stock
lst_code = stock.get_market_ticker_list(date='20170102', market='KOSPI') +\
         stock.get_market_ticker_list(date='20170102', market='KOSDAQ')
print("종목 갯수:", len(lst_code))
print(lst_code[:5], '...')
```

    종목 갯수: 2106
    ['095570', '068400', '006840', '027410', '138930'] ...
    

## 1.2. 일별 주가 데이터 다운로드

FinanceDataReader 라이브러리를 통해 위에서 선정한 종목들의 일별 주가 데이터를 다운로드 받습니다. 

최근 6.5년 (2016년~2022년 6월 30일) 데이터를 다운로드 받습니다. 


```python
############# FinanceDataReader 라이브러리를 이용하여 최근 6.5년 주가 데이터를 불러옴 #############
import FinanceDataReader as fdr
start = '2016-01-01' 
end = '2022-06-30'

df = pd.DataFrame()
for code in tqdm(lst_code):
    stock_df = fdr.DataReader(code, start, end).reset_index()
    
    if (len(stock_df) == 0) or (len(stock_df) < 1596): # 최근 6년치 데이터(260 개장일 기준)가 쌓여있지 않으면, 보조지표를 생성 못하므로 조건을 걸어 줌
        continue
        
    stock_df.insert(1,'Code',code)
    df = df.append(stock_df)

df.to_csv("raw_data.csv", index=False)
df
```

    100%|███████████████████████████████████████| 2106/2106 [03:12<00:00, 10.93it/s]
    




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
      <th>Date</th>
      <th>Code</th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Volume</th>
      <th>Change</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2016-01-04</td>
      <td>095570</td>
      <td>9560</td>
      <td>9730</td>
      <td>9420</td>
      <td>9600</td>
      <td>20135</td>
      <td>0.019108</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2016-01-05</td>
      <td>095570</td>
      <td>9450</td>
      <td>9550</td>
      <td>9360</td>
      <td>9540</td>
      <td>8225</td>
      <td>-0.006250</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2016-01-06</td>
      <td>095570</td>
      <td>9610</td>
      <td>9610</td>
      <td>9440</td>
      <td>9560</td>
      <td>7271</td>
      <td>0.002096</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2016-01-07</td>
      <td>095570</td>
      <td>9580</td>
      <td>9580</td>
      <td>9120</td>
      <td>9300</td>
      <td>5358</td>
      <td>-0.027197</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2016-01-08</td>
      <td>095570</td>
      <td>9050</td>
      <td>9330</td>
      <td>9040</td>
      <td>9220</td>
      <td>7056</td>
      <td>-0.008602</td>
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
    </tr>
    <tr>
      <th>1591</th>
      <td>2022-06-24</td>
      <td>037440</td>
      <td>4970</td>
      <td>5500</td>
      <td>4970</td>
      <td>5390</td>
      <td>387835</td>
      <td>0.069444</td>
    </tr>
    <tr>
      <th>1592</th>
      <td>2022-06-27</td>
      <td>037440</td>
      <td>5700</td>
      <td>5840</td>
      <td>5430</td>
      <td>5780</td>
      <td>290045</td>
      <td>0.072356</td>
    </tr>
    <tr>
      <th>1593</th>
      <td>2022-06-28</td>
      <td>037440</td>
      <td>5690</td>
      <td>6190</td>
      <td>5690</td>
      <td>5800</td>
      <td>364417</td>
      <td>0.003460</td>
    </tr>
    <tr>
      <th>1594</th>
      <td>2022-06-29</td>
      <td>037440</td>
      <td>5640</td>
      <td>7540</td>
      <td>5640</td>
      <td>7540</td>
      <td>11733817</td>
      <td>0.300000</td>
    </tr>
    <tr>
      <th>1595</th>
      <td>2022-06-30</td>
      <td>037440</td>
      <td>7950</td>
      <td>9630</td>
      <td>7860</td>
      <td>8870</td>
      <td>29214629</td>
      <td>0.176393</td>
    </tr>
  </tbody>
</table>
<p>2960580 rows × 8 columns</p>
</div>



2016년 1월 1일 부터 2022년 6월 30일까지의 일 단위 주가 데이터를 불러와 생성한 데이터프레임 입니다. 

## 1.3. 주가 데이터에 보조지표 독립변수 추가하기

보조지표를 추가하여 총 55개의 독립 변수를 생성합니다.

- **원본 데이터 독립 변수: 6개**

    - Open, High, Low, Close, Volume, Change

- **보조지표 독립 변수: 49개**

    - TA라이브러리 이용한 이동평균선, 볼린저 밴드 등등의 보조지표


```python
import ta

df = pd.read_csv('raw_data.csv')
df['Code'] = df['Code'].apply(lambda x : str(x).zfill(6))
df2 = pd.DataFrame()
for code, stock_df in tqdm(df.groupby('Code')):
    
    # 이평선 생성
    ma = [5,20,60,120]
    for days in ma:
        stock_df['ma_'+str(days)] = stock_df['Close'].rolling(window = days).mean()
    
    # 여러 보조 지표 생성
    H, L, C, V = stock_df['High'], stock_df['Low'], stock_df['Close'], stock_df['Volume']
    
    stock_df['trading_value'] = stock_df['Close']*stock_df['Volume']
    
    stock_df['MFI'] = ta.volume.money_flow_index(
        high=H, low=L, close=C, volume=V, fillna=True)
    
    stock_df['ADI'] = ta.volume.acc_dist_index(
        high=H, low=L, close=C, volume=V, fillna=True)
    
    stock_df['OBV'] = ta.volume.on_balance_volume(close=C, volume=V, fillna=True)
    stock_df['CMF'] = ta.volume.chaikin_money_flow(
        high=H, low=L, close=C, volume=V, fillna=True)
    
    stock_df['FI'] = ta.volume.force_index(close=C, volume=V, fillna=True)
    stock_df['EOMEMV'] = ta.volume.ease_of_movement(
        high=H, low=L, volume=V, fillna=True)
    
    stock_df['VPT'] = ta.volume.volume_price_trend(close=C, volume=V, fillna=True)
    stock_df['NVI'] = ta.volume.negative_volume_index(close=C, volume=V, fillna=True)
    stock_df['VMAP'] = ta.volume.volume_weighted_average_price(
        high=H, low=L, close=C, volume=V, fillna=True)
    
    # Volatility
    stock_df['ATR'] = ta.volatility.average_true_range(
        high=H, low=L, close=C, fillna=True)
    stock_df['BHB'] = ta.volatility.bollinger_hband(close=C, fillna=True)
    stock_df['BLB'] = ta.volatility.bollinger_lband(close=C, fillna=True)
    stock_df['KCH'] = ta.volatility.keltner_channel_hband(
        high=H, low=L, close=C, fillna=True)
    stock_df['KCL'] = ta.volatility.keltner_channel_lband(
        high=H, low=L, close=C, fillna=True)
    stock_df['KCM'] = ta.volatility.keltner_channel_mband(
        high=H, low=L, close=C, fillna=True)
    stock_df['DCH'] = ta.volatility.donchian_channel_hband(
        high=H, low=L, close=C, fillna=True)
    stock_df['DCL'] = ta.volatility.donchian_channel_lband(
        high=H, low=L, close=C, fillna=True)
    stock_df['DCM'] = ta.volatility.donchian_channel_mband(
        high=H, low=L, close=C, fillna=True)
    stock_df['UI'] = ta.volatility.ulcer_index(close=C, fillna=True)
    # Trend
    stock_df['SMA'] = ta.trend.sma_indicator(close=C, fillna=True)
    stock_df['EMA'] = ta.trend.ema_indicator(close=C, fillna=True)
    stock_df['WMA'] = ta.trend.wma_indicator(close=C, fillna=True)
    stock_df['MACD'] = ta.trend.macd(close=C, fillna=True)
    stock_df['ADX'] = ta.trend.adx(high=H, low=L, close=C, fillna=True)
    stock_df['VIneg'] = ta.trend.vortex_indicator_neg(
        high=H, low=L, close=C, fillna=True)
    stock_df['VIpos'] = ta.trend.vortex_indicator_pos(
        high=H, low=L, close=C, fillna=True)
    stock_df['TRIX'] = ta.trend.trix(close=C, fillna=True)
    stock_df['MI'] = ta.trend.mass_index(high=H, low=L, fillna=True)
    stock_df['CCI'] = ta.trend.cci(high=H, low=L, close=C, fillna=True)
    stock_df['DPO'] = ta.trend.dpo(close=C, fillna=True)
    stock_df['KST'] = ta.trend.kst(close=C, fillna=True)
    stock_df['Ichimoku'] = ta.trend.ichimoku_a(high=H, low=L, fillna=True)
    stock_df['ParabolicSAR'] = ta.trend.psar_down(
        high=H, low=L, close=C, fillna=True)
    stock_df['STC'] = ta.trend.stc(close=C, fillna=True)
    # Momentum
    stock_df['RSI'] = ta.momentum.rsi(close=C, fillna=True)
    stock_df['SRSI'] = ta.momentum.stochrsi(close=C, fillna=True)
    stock_df['TSI'] = ta.momentum.tsi(close=C, fillna=True)
    stock_df['UO'] = ta.momentum.ultimate_oscillator(
        high=H, low=L, close=C, fillna=True)
    stock_df['SR'] = ta.momentum.stoch(close=C, high=H, low=L, fillna=True)
    stock_df['WR'] = ta.momentum.williams_r(high=H, low=L, close=C, fillna=True)
    stock_df['AO'] = ta.momentum.awesome_oscillator(high=H, low=L, fillna=True)
    stock_df['KAMA'] = ta.momentum.kama(close=C, fillna=True)
    stock_df['ROC'] = ta.momentum.roc(close=C, fillna=True)
    stock_df['PPO'] = ta.momentum.ppo(close=C, fillna=True)
    stock_df['PVO'] = ta.momentum.pvo(volume=V, fillna=True)
    
    df2 = df2.append(stock_df) 

df2.to_csv('raw_data_index_added.csv', index=False)
df2
```

    100%|███████████████████████████████████████| 1855/1855 [18:12<00:00,  1.70it/s]
    




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
      <th>Date</th>
      <th>Code</th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Volume</th>
      <th>Change</th>
      <th>ma_5</th>
      <th>ma_20</th>
      <th>...</th>
      <th>SRSI</th>
      <th>TSI</th>
      <th>UO</th>
      <th>SR</th>
      <th>WR</th>
      <th>AO</th>
      <th>KAMA</th>
      <th>ROC</th>
      <th>PPO</th>
      <th>PVO</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>459648</th>
      <td>2016-01-04</td>
      <td>000020</td>
      <td>8130</td>
      <td>8150</td>
      <td>7920</td>
      <td>8140</td>
      <td>281440</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>95.652174</td>
      <td>-4.347826</td>
      <td>0.000000</td>
      <td>8140.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>459649</th>
      <td>2016-01-05</td>
      <td>000020</td>
      <td>8040</td>
      <td>8250</td>
      <td>8000</td>
      <td>8190</td>
      <td>243179</td>
      <td>0.006143</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.000000</td>
      <td>100.000000</td>
      <td>39.583333</td>
      <td>81.818182</td>
      <td>-18.181818</td>
      <td>0.000000</td>
      <td>8153.710776</td>
      <td>0.000000</td>
      <td>0.048978</td>
      <td>-1.095512</td>
    </tr>
    <tr>
      <th>459650</th>
      <td>2016-01-06</td>
      <td>000020</td>
      <td>8200</td>
      <td>8590</td>
      <td>8110</td>
      <td>8550</td>
      <td>609906</td>
      <td>0.043956</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.000000</td>
      <td>100.000000</td>
      <td>65.625000</td>
      <td>94.029851</td>
      <td>-5.970149</td>
      <td>0.000000</td>
      <td>8210.298275</td>
      <td>0.000000</td>
      <td>0.437814</td>
      <td>7.866130</td>
    </tr>
    <tr>
      <th>459651</th>
      <td>2016-01-07</td>
      <td>000020</td>
      <td>8470</td>
      <td>8690</td>
      <td>8190</td>
      <td>8380</td>
      <td>704752</td>
      <td>-0.019883</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.000000</td>
      <td>93.488920</td>
      <td>56.164384</td>
      <td>59.740260</td>
      <td>-40.259740</td>
      <td>0.000000</td>
      <td>8240.741109</td>
      <td>0.000000</td>
      <td>0.570633</td>
      <td>15.684879</td>
    </tr>
    <tr>
      <th>459652</th>
      <td>2016-01-08</td>
      <td>000020</td>
      <td>8210</td>
      <td>8900</td>
      <td>8130</td>
      <td>8770</td>
      <td>802330</td>
      <td>0.046539</td>
      <td>8406.0</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.000000</td>
      <td>89.637123</td>
      <td>65.470852</td>
      <td>86.734694</td>
      <td>-13.265306</td>
      <td>0.000000</td>
      <td>8249.882296</td>
      <td>0.000000</td>
      <td>1.043258</td>
      <td>22.201829</td>
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
      <th>2183323</th>
      <td>2022-06-24</td>
      <td>950130</td>
      <td>12850</td>
      <td>13650</td>
      <td>12850</td>
      <td>13600</td>
      <td>319363</td>
      <td>0.050193</td>
      <td>13340.0</td>
      <td>15275.0</td>
      <td>...</td>
      <td>0.421113</td>
      <td>-34.868854</td>
      <td>33.375301</td>
      <td>17.441860</td>
      <td>-82.558140</td>
      <td>-2672.500000</td>
      <td>13626.769118</td>
      <td>-18.072289</td>
      <td>-6.067552</td>
      <td>-9.215386</td>
    </tr>
    <tr>
      <th>2183324</th>
      <td>2022-06-27</td>
      <td>950130</td>
      <td>13800</td>
      <td>13950</td>
      <td>13400</td>
      <td>13750</td>
      <td>271801</td>
      <td>0.011029</td>
      <td>13450.0</td>
      <td>15155.0</td>
      <td>...</td>
      <td>0.493412</td>
      <td>-33.771203</td>
      <td>39.860299</td>
      <td>22.222222</td>
      <td>-77.777778</td>
      <td>-2570.000000</td>
      <td>13638.933767</td>
      <td>-17.664671</td>
      <td>-5.910892</td>
      <td>-9.643873</td>
    </tr>
    <tr>
      <th>2183325</th>
      <td>2022-06-28</td>
      <td>950130</td>
      <td>14000</td>
      <td>14300</td>
      <td>13600</td>
      <td>13850</td>
      <td>530492</td>
      <td>0.007273</td>
      <td>13500.0</td>
      <td>15037.5</td>
      <td>...</td>
      <td>0.542858</td>
      <td>-32.505777</td>
      <td>36.768126</td>
      <td>25.316456</td>
      <td>-74.683544</td>
      <td>-2392.352941</td>
      <td>13651.451839</td>
      <td>-16.314199</td>
      <td>-5.668384</td>
      <td>-3.532023</td>
    </tr>
    <tr>
      <th>2183326</th>
      <td>2022-06-29</td>
      <td>950130</td>
      <td>13650</td>
      <td>13950</td>
      <td>13650</td>
      <td>13750</td>
      <td>147032</td>
      <td>-0.007220</td>
      <td>13580.0</td>
      <td>14895.0</td>
      <td>...</td>
      <td>0.549629</td>
      <td>-31.687694</td>
      <td>38.200222</td>
      <td>23.076923</td>
      <td>-76.923077</td>
      <td>-2311.911765</td>
      <td>13653.769562</td>
      <td>-11.003236</td>
      <td>-5.472416</td>
      <td>-7.996994</td>
    </tr>
    <tr>
      <th>2183327</th>
      <td>2022-06-30</td>
      <td>950130</td>
      <td>13900</td>
      <td>14600</td>
      <td>13800</td>
      <td>14600</td>
      <td>893557</td>
      <td>0.061818</td>
      <td>13910.0</td>
      <td>14795.0</td>
      <td>...</td>
      <td>1.000000</td>
      <td>-27.705397</td>
      <td>42.932039</td>
      <td>49.295775</td>
      <td>-50.704225</td>
      <td>-2069.852941</td>
      <td>13678.652174</td>
      <td>-2.341137</td>
      <td>-4.780093</td>
      <td>5.630385</td>
    </tr>
  </tbody>
</table>
<p>2960580 rows × 58 columns</p>
</div>



보조지표를 추가한 데이터프레임 입니다. 보조지표는 일정 기간 쌓인 주가 데이터로 계산하기 때문에 1년치를 미리 가져온 뒤 전처리하였습니다.   

## 1.4. 주가 데이터 스케일 표준화

주가의 스케일(액면가)가 서로 다른 종목들을 동일한 인공지능 모델에서 모델링하기 위해서 주가 관련 23개의 지표 스케일을 표준화 합니다.

주가 스케일 방법:
- new_x = (x / prev_close) - 1

- [해당 스케일링 방식이 궁금하시면, 링크에 설명이 자세히 적혀있습니다.](https://inhovation97.tistory.com/60)


```python
df3 = pd.DataFrame([])
target_columns = ['Open', 'High', 'Low', 'ma_5', 'ma_20', 'ma_60', 'ma_120', 
               'VMAP', 'BHB', 'BLB', 'KCH', 'KCL', 'KCM', 'DCH', 'DCL', 'DCM',
               'SMA', 'EMA', 'WMA', 'Ichimoku', 'ParabolicSAR', 'KAMA','MACD']
for code, stock_df in tqdm(df2.groupby('Code')):
    for target in target_columns:
        stock_df[target] = stock_df[target]/stock_df['Close'].shift(1)  - 1 
    df3 = df3.append(stock_df)
    
df3 = df3.dropna(axis=0)
df3.to_csv('raw_data_index_added_normalize.csv', index=False)
df3
```

    100%|███████████████████████████████████████| 1855/1855 [02:26<00:00, 12.63it/s]
    




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
      <th>Date</th>
      <th>Code</th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Volume</th>
      <th>Change</th>
      <th>ma_5</th>
      <th>ma_20</th>
      <th>...</th>
      <th>SRSI</th>
      <th>TSI</th>
      <th>UO</th>
      <th>SR</th>
      <th>WR</th>
      <th>AO</th>
      <th>KAMA</th>
      <th>ROC</th>
      <th>PPO</th>
      <th>PVO</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>459767</th>
      <td>2016-06-29</td>
      <td>000020</td>
      <td>0.009221</td>
      <td>0.034836</td>
      <td>-0.006148</td>
      <td>9750</td>
      <td>352292</td>
      <td>-0.001025</td>
      <td>-0.022131</td>
      <td>0.046107</td>
      <td>...</td>
      <td>0.529126</td>
      <td>-8.116135</td>
      <td>48.063493</td>
      <td>41.176471</td>
      <td>-58.823529</td>
      <td>-827.794118</td>
      <td>0.024090</td>
      <td>-3.940887</td>
      <td>-1.408719</td>
      <td>-11.019576</td>
    </tr>
    <tr>
      <th>459768</th>
      <td>2016-06-30</td>
      <td>000020</td>
      <td>0.010256</td>
      <td>0.066667</td>
      <td>0.001026</td>
      <td>10100</td>
      <td>466039</td>
      <td>0.035897</td>
      <td>-0.013538</td>
      <td>0.045641</td>
      <td>...</td>
      <td>0.875363</td>
      <td>-6.553028</td>
      <td>48.684638</td>
      <td>54.901961</td>
      <td>-45.098039</td>
      <td>-765.088235</td>
      <td>0.025411</td>
      <td>-2.415459</td>
      <td>-1.124410</td>
      <td>-9.123558</td>
    </tr>
    <tr>
      <th>459769</th>
      <td>2016-07-01</td>
      <td>000020</td>
      <td>0.009901</td>
      <td>0.009901</td>
      <td>-0.013861</td>
      <td>9960</td>
      <td>208228</td>
      <td>-0.013861</td>
      <td>-0.030297</td>
      <td>0.004752</td>
      <td>...</td>
      <td>0.767924</td>
      <td>-5.911240</td>
      <td>49.021117</td>
      <td>49.411765</td>
      <td>-50.588235</td>
      <td>-624.205882</td>
      <td>-0.010171</td>
      <td>-4.230769</td>
      <td>-1.001426</td>
      <td>-12.562340</td>
    </tr>
    <tr>
      <th>459770</th>
      <td>2016-07-04</td>
      <td>000020</td>
      <td>0.004016</td>
      <td>0.044177</td>
      <td>-0.006024</td>
      <td>10400</td>
      <td>275210</td>
      <td>0.044177</td>
      <td>0.003414</td>
      <td>0.017620</td>
      <td>...</td>
      <td>1.000000</td>
      <td>-3.312304</td>
      <td>52.287743</td>
      <td>66.666667</td>
      <td>-33.333333</td>
      <td>-427.205882</td>
      <td>0.004218</td>
      <td>-0.952381</td>
      <td>-0.541344</td>
      <td>-13.977866</td>
    </tr>
    <tr>
      <th>459771</th>
      <td>2016-07-05</td>
      <td>000020</td>
      <td>0.000000</td>
      <td>0.004808</td>
      <td>-0.019231</td>
      <td>10350</td>
      <td>156010</td>
      <td>-0.004808</td>
      <td>-0.027692</td>
      <td>-0.027115</td>
      <td>...</td>
      <td>0.961700</td>
      <td>-1.483696</td>
      <td>58.584716</td>
      <td>64.705882</td>
      <td>-35.294118</td>
      <td>-266.529412</td>
      <td>-0.037783</td>
      <td>1.970443</td>
      <td>-0.216142</td>
      <td>-17.713482</td>
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
      <th>2183323</th>
      <td>2022-06-24</td>
      <td>950130</td>
      <td>-0.007722</td>
      <td>0.054054</td>
      <td>-0.007722</td>
      <td>13600</td>
      <td>319363</td>
      <td>0.050193</td>
      <td>0.030116</td>
      <td>0.179537</td>
      <td>...</td>
      <td>0.421113</td>
      <td>-34.868854</td>
      <td>33.375301</td>
      <td>17.441860</td>
      <td>-82.558140</td>
      <td>-2672.500000</td>
      <td>0.052260</td>
      <td>-18.072289</td>
      <td>-6.067552</td>
      <td>-9.215386</td>
    </tr>
    <tr>
      <th>2183324</th>
      <td>2022-06-27</td>
      <td>950130</td>
      <td>0.014706</td>
      <td>0.025735</td>
      <td>-0.014706</td>
      <td>13750</td>
      <td>271801</td>
      <td>0.011029</td>
      <td>-0.011029</td>
      <td>0.114338</td>
      <td>...</td>
      <td>0.493412</td>
      <td>-33.771203</td>
      <td>39.860299</td>
      <td>22.222222</td>
      <td>-77.777778</td>
      <td>-2570.000000</td>
      <td>0.002863</td>
      <td>-17.664671</td>
      <td>-5.910892</td>
      <td>-9.643873</td>
    </tr>
    <tr>
      <th>2183325</th>
      <td>2022-06-28</td>
      <td>950130</td>
      <td>0.018182</td>
      <td>0.040000</td>
      <td>-0.010909</td>
      <td>13850</td>
      <td>530492</td>
      <td>0.007273</td>
      <td>-0.018182</td>
      <td>0.093636</td>
      <td>...</td>
      <td>0.542858</td>
      <td>-32.505777</td>
      <td>36.768126</td>
      <td>25.316456</td>
      <td>-74.683544</td>
      <td>-2392.352941</td>
      <td>-0.007167</td>
      <td>-16.314199</td>
      <td>-5.668384</td>
      <td>-3.532023</td>
    </tr>
    <tr>
      <th>2183326</th>
      <td>2022-06-29</td>
      <td>950130</td>
      <td>-0.014440</td>
      <td>0.007220</td>
      <td>-0.014440</td>
      <td>13750</td>
      <td>147032</td>
      <td>-0.007220</td>
      <td>-0.019495</td>
      <td>0.075451</td>
      <td>...</td>
      <td>0.549629</td>
      <td>-31.687694</td>
      <td>38.200222</td>
      <td>23.076923</td>
      <td>-76.923077</td>
      <td>-2311.911765</td>
      <td>-0.014168</td>
      <td>-11.003236</td>
      <td>-5.472416</td>
      <td>-7.996994</td>
    </tr>
    <tr>
      <th>2183327</th>
      <td>2022-06-30</td>
      <td>950130</td>
      <td>0.010909</td>
      <td>0.061818</td>
      <td>0.003636</td>
      <td>14600</td>
      <td>893557</td>
      <td>0.061818</td>
      <td>0.011636</td>
      <td>0.076000</td>
      <td>...</td>
      <td>1.000000</td>
      <td>-27.705397</td>
      <td>42.932039</td>
      <td>49.295775</td>
      <td>-50.704225</td>
      <td>-2069.852941</td>
      <td>-0.005189</td>
      <td>-2.341137</td>
      <td>-4.780093</td>
      <td>5.630385</td>
    </tr>
  </tbody>
</table>
<p>2739835 rows × 58 columns</p>
</div>



위 데이터를 확인해보면, 가격 관련 변수들의 스케일이 표준화된 것을 알 수 있습니다.  

## 1.5. 공개/비공개 데이터셋으로 분할하고 공개 데이터셋을 AIT 개발 커뮤니티에 공개 
**데이터셋 분할 생성**  
- **`공개 데이터셋`** : 2017 ~ 2021 (4년)   
- **`비공개 데이터셋`** : 2022 상반기 (6개월)  


```python
df3 = pd.read_csv("raw_data_index_added_normalize.csv")
df3['Code'] = df3['Code'].apply(lambda x : str(x).zfill(6))

split_date_opened = (df3['Date'] >= '2017-01-02') & (df3['Date'] <= '2021-12-31')
split_date_closed = (df3['Date'] >= '2022-01-02') & (df3['Date'] <= '2022-06-30')

opened_df = df3[split_date_opened]
closed_df = df3[split_date_closed]

opened_df.to_csv('opened_df.csv', index=False)
closed_df.to_csv('closed_df.csv', index=False)

print("opened dataset: ", opened_df.shape)
print("closed dataset: ", closed_df.shape)
```

    opened dataset:  (2279795, 58)
    closed dataset:  (224455, 58)
    

# 2. AIT 생산자 : AI트레이더 생성

총 6명의 생산자가 AI 트레이더를 학습하여 AIT 마켓에 제출한다고 가정합니다. 

## 2.1. 수원랩 XGBoost 팀    
- **생산자 이름:** 수원랩 XGBoost
- **예측 목표:** 변동성 종목에 대해서 다음 날 종가 2% 상승 여부인공지능 모델로 예측
    - 변동성 기준: 거래대금 10억 & 금일 주가 변화율 5% 이상
- **매매 전략:** 상승 예측 확률 x% 시 매수 then 다음 날 종가 매도 
- **인공지능 모델:** XGBoost
- **학습 데이터:** 3년치 주가 일별 데이터
- **검증 데이터:** 1년치 주가 일별 데이터

### 2.1.1. 데이터 전처리 

**1) 변동성 종목을 선정**

- 변동성 기준 : 거래대금 10억 & 금일 주가 변화율 5% 이상

**2) 종속 변수를 이진 분류 예측 문제로 설정**

- 종속 변수 : 다음 날 종가 상승률(next_change) 2% 이상 상승 여부 (상승시 1, 상승 안할 시 0)

- next_change = (next_day_close - today_close) / today_close 


```python
# df2 = pd.read_csv('raw_data_index_added.csv')
# df2['Code'] = df2['Code'].apply(lambda x : str(x).zfill(6))

opened_df = pd.read_csv('opened_df.csv')
opened_df['Code'] = opened_df['Code'].apply(lambda x : str(x).zfill(6))

opened_df['next_change'] = opened_df['Change'].shift(-1)
opened_df = opened_df.dropna()

openedX = opened_df.iloc[:, :-1]
openedY = opened_df[['next_change']]

# 변동성 조건 
print('변동성 조건 설정 전:', openedX.shape)
def sampling(df):
    condition1 = (-0.3 <= df.Change) & (df.Change <= 0.3) # 상한가, 하한가 초과하는 예외 제거 
    condition2 = df.trading_value >= 1000000000 # 변동성 조건 1: 거래대금 10억 이상 
    condition3 = (-0.05 >= df.Change) | (0.05 <= df.Change) # 변동성 조건 2: 금일 주가 변화율 5%이상 
    condition = condition1 & condition2 & condition3
    return condition

condition = sampling(openedX)

# 변동성 조건 적용 
openedX = openedX[condition]
openedY = openedY[condition]

print('변동성 조건 설정 후:', openedX.shape)

# 학습 후 검증을 위한 학습 / 검증 데이터셋으로 분할 
split_date_train = (openedX['Date'] >= '2017-01-02') & (openedX['Date'] <= '2020-12-31')
split_date_valid = (openedX['Date'] >= '2021-01-02') & (openedX['Date'] <= '2021-12-31')

openedX = openedX.drop(columns=['Date', 'Code'])

trainX = openedX[split_date_train]
trainY = openedY[split_date_train]
# 종속변수: 2% 상승여부에 대한 분류 문제를 위해 이진 값으로 변환 
trainY_classification = (trainY >= 0.02).astype('int')

validX = openedX[split_date_valid]
validY = openedY[split_date_valid]
validY_classification = (validY >= 0.02).astype('int')
```

    변동성 조건 설정 전: (2279794, 58)
    변동성 조건 설정 후: (146907, 58)
    

### 2.1.2. 데이터 모델링 
XGBoost를 사용하여 trainX, trainY 데이터셋에 대하여 예측 모델 학습 


```python
##### XGBoost
from xgboost import XGBClassifier
scale_pos_weight = round(72/28 , 2)

xgb = XGBClassifier(random_state = 42,
                   n_jobs=30,
                   scale_pos_weight=scale_pos_weight,
                   learning_rate=0.1,
                   max_depth=4,
                   n_estimators=1000,
                   ) 
xgb.fit(trainX, trainY_classification)
```

    [20:40:56] WARNING: ../src/learner.cc:1115: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.
    




    XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                  colsample_bynode=1, colsample_bytree=1, enable_categorical=False,
                  gamma=0, gpu_id=-1, importance_type=None,
                  interaction_constraints='', learning_rate=0.1, max_delta_step=0,
                  max_depth=4, min_child_weight=1, missing=nan,
                  monotone_constraints='()', n_estimators=1000, n_jobs=30,
                  num_parallel_tree=1, predictor='auto', random_state=42,
                  reg_alpha=0, reg_lambda=1, scale_pos_weight=2.57, subsample=1,
                  tree_method='exact', validate_parameters=1, verbosity=None)



### 2.2.3. 상승 확률 경계값 설정 


```python
from sklearn.preprocessing import Binarizer
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib
thresholds = list(np.arange(0.1, 1, 0.01))
matplotlib.rcParams['font.family'] ='NanumSquareRound'

fig = plt.figure(figsize=(10,6))
def get_eval_by_threshold(y_test, pred_proba, thresholds):
    for i in thresholds:
        binarizer = Binarizer(threshold = i).fit(pred_proba)
        pred = binarizer.transform(pred_proba)
        
        plt.scatter(i, precision_score(y_test, pred), color='b', label='정밀도') # 정밀도
        plt.scatter(i, recall_score(y_test, pred), color='r', label ='재현율') # 재현율 
        plt.scatter(i, f1_score(y_test, pred), color='g', label='f1 score') # f1 score
        if i == 0.1:
            plt.legend(fontsize = 15)
        plt.title('XG Boost 정밀도, 재현율, f1 score',fontsize=20)
        plt.ylabel("score", fontsize=20)
        plt.xlabel("Threshhold", fontsize=20)
        plt.axvline(0.815, color = 'b')
        
xgb_valid_prob = xgb.predict_proba(validX)[:, 1] # XGBoost validation set 확률값리스트

get_eval_by_threshold(validY_classification['next_change'], xgb_valid_prob.reshape(-1,1), thresholds)

plt.grid()
plt.show()
```


    
![png](output_36_0.png)
    


### 2.2.4. 수익성 검증


```python
import seaborn as sns
upper_80=[]
lower_80=[]

xgb_valid_prob = xgb.predict_proba(validX)[:, 1]

for prob, change in zip(xgb_valid_prob,100*(validY['next_change'])):
    if prob >= 0.815:
        upper_80.append(change)
    else:
        lower_80.append(change)

        
fig = plt.figure(figsize=(16,10))
sns.distplot(upper_80, color= 'r', label='80% 이상 예측한 값들의 다음날 종가 변화율 분포')
sns.distplot(lower_80, label='80% 이외의 나머지 추론 값들의 다음날 종가 변화율 분포')
plt.axvline(np.mean(upper_80),color='r')
plt.axvline(np.mean(lower_80),color='b')
plt.legend(fontsize=15)

print('붉은색 분포의 평균 :',np.mean(upper_80))
print('푸른색 분포의 평균 :',np.mean(lower_80))
```

    붉은색 분포의 평균 : 3.0641545151381817
    푸른색 분포의 평균 : 0.20760016914099816
    


    
![png](output_38_1.png)
    


### 2.2.5. 모델 저장 후 AIT마켓에 제출(출시)


```python
import pickle

result = {
    'name' : '수원랩XGboost',
    'sampling' : sampling,
    'model' : xgb,
    'threshold' : 0.815    
}

# pickle.dump('수원랩XGboost.pickle', result)
# Store data (serialize)
with open('수원랩XGboost.pickle', 'wb') as handle:
    pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
```

## 2.2. 경기랩 Logistic Regression 팀    
- **생산자 이름:** 경기랩 Logistic Regression
- **예측 목표:** 변동성 종목에 대해서 다음 날 종가 2% 상승 여부인공지능 모델로 예측
    - 변동성 기준: 거래대금 10억 & 금일 주가 변화율 5% 이상
- **매매 전략:** 상승 예측 확률 x% 시 매수 then 다음 날 종가 매도 
- **인공지능 모델:** Logistic Regression
- **학습 데이터:** 3년치 주가 일별 데이터
- **검증 데이터:** 1년치 주가 일별 데이터

### 2.2.1. 데이터 전처리 

**1) 변동성 종목을 선정**

- 변동성 기준 : 거래대금 10억 & 금일 주가 변화율 5% 이상

**2) 종속 변수를 이진 분류 예측 문제로 설정**

- 종속 변수 : 다음 날 종가 상승률(next_change) 2% 이상 상승 여부 (상승시 1, 상승 안할 시 0)

- next_change = (next_day_close - today_close) / today_close 


```python
opened_df = pd.read_csv('opened_df.csv')
opened_df['Code'] = opened_df['Code'].apply(lambda x : str(x).zfill(6))

opened_df['next_change'] = opened_df['Change'].shift(-1)
opened_df = opened_df.dropna()

openedX = opened_df.iloc[:, :-1]
openedY = opened_df[['next_change']]

# 변동성 조건 
print('변동성 조건 설정 전:', openedX.shape)
def sampling(df):
    condition1 = (-0.3 <= df.Change) & (df.Change <= 0.3) # 상한가, 하한가 초과하는 예외 제거 
    condition2 = df.trading_value >= 1000000000 # 변동성 조건 1: 거래대금 10억 이상 
    condition3 = (-0.05 >= df.Change) | (0.05 <= df.Change) # 변동성 조건 2: 금일 주가 변화율 5%이상 
    condition = condition1 & condition2 & condition3
    return condition

condition = sampling(openedX)

# 변동성 조건 적용 
openedX = openedX[condition]
openedY = openedY[condition]

print('변동성 조건 설정 후:', openedX.shape)

# 학습 후 검증을 위한 학습 / 검증 데이터셋으로 분할 
split_date_train = (openedX['Date'] >= '2017-01-02') & (openedX['Date'] <= '2020-12-31')
split_date_valid = (openedX['Date'] >= '2021-01-02') & (openedX['Date'] <= '2021-12-31')

openedX = openedX.drop(columns=['Date', 'Code'])

trainX = openedX[split_date_train]
trainY = openedY[split_date_train]
# 종속변수: 2% 상승여부에 대한 분류 문제를 위해 이진 값으로 변환 
trainY_classification = (trainY >= 0.02).astype('int')

validX = openedX[split_date_valid]
validY = openedY[split_date_valid]
validY_classification = (validY >= 0.02).astype('int')
```

    변동성 조건 설정 전: (2279794, 58)
    변동성 조건 설정 후: (146907, 58)
    

### 2.2.2. 데이터 모델링 
Logistic Regression를 사용하여 trainX, trainY 데이터셋에 대하여 예측 모델 학습 


```python
##### Logistic Regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state = 42, max_iter=1000)
lr.fit(trainX, trainY_classification)
```




    LogisticRegression(max_iter=1000, random_state=42)



### 2.2.3. 상승 확률 경계값 설정 


```python
from sklearn.preprocessing import Binarizer
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib
thresholds = list(np.arange(0.1, 1, 0.01))
matplotlib.rcParams['font.family'] ='NanumSquareRound'

fig = plt.figure(figsize=(10,6))
def get_eval_by_threshold(y_test, pred_proba, thresholds):
    for i in thresholds:
        binarizer = Binarizer(threshold = i).fit(pred_proba)
        pred = binarizer.transform(pred_proba)
        
        plt.scatter(i, precision_score(y_test, pred), color='b', label='정밀도') # 정밀도
        plt.scatter(i, recall_score(y_test, pred), color='r', label ='재현율') # 재현율 
        plt.scatter(i, f1_score(y_test, pred), color='g', label='f1 score') # f1 score
        if i == 0.1:
            plt.legend(fontsize = 15)
        plt.title('Logistic Regression 정밀도, 재현율, f1 score',fontsize=20)
        plt.ylabel("score", fontsize=20)
        plt.xlabel("Threshhold", fontsize=20)
        plt.axvline(0.47, color = 'b')
        
lr_valid_prob = lr.predict_proba(validX)[:, 1] # XGBoost validation set 확률값리스트

get_eval_by_threshold(validY_classification['next_change'], lr_valid_prob.reshape(-1,1), thresholds)

plt.grid()
plt.show()
```


    
![png](output_47_0.png)
    


### 2.2.4. 수익성 검증


```python
import seaborn as sns
upper_80=[]
lower_80=[]

lr_valid_prob = lr.predict_proba(validX)[:, 1]

for prob, change in zip(lr_valid_prob,100*(validY['next_change'])):
    if prob >= 0.47:
        upper_80.append(change)
    else:
        lower_80.append(change)

        
fig = plt.figure(figsize=(16,10))
sns.distplot(upper_80, color= 'r', label='80% 이상 예측한 값들의 다음날 종가 변화율 분포')
sns.distplot(lower_80, label='80% 이외의 나머지 추론 값들의 다음날 종가 변화율 분포')
plt.axvline(np.mean(upper_80),color='r')
plt.axvline(np.mean(lower_80),color='b')
plt.legend(fontsize=15)
lr
print('붉은색 분포의 평균 :',np.mean(upper_80))
print('푸른색 분포의 평균 :',np.mean(lower_80))
```

    붉은색 분포의 평균 : 0.34888469574511616
    푸른색 분포의 평균 : 0.1826507707516834
    


    
![png](output_49_1.png)
    


### 2.2.5. 모델 저장 후 AIT마켓에 제출(출시)


```python
import pickle

result = {
    'name' : '경기랩LogisticRegression',
    'sampling' : sampling,
    'model' : lr,
    'threshold' : 0.47    
}

# pickle.dump('수원랩XGboost.pickle', result)
# Store data (serialize)
with open('경기랩LogisticRegression.pickle', 'wb') as handle:
    pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
```

## 2.3. 뽀로로 LightGBM 팀    
- **생산자 이름:** 뽀로로 LightGBM
- **예측 목표:** 변동성 종목에 대해서 다음 날 종가 2% 상승 여부인공지능 모델로 예측
    - 변동성 기준: 거래대금 10억 & 금일 주가 변화율 5% 이상
- **매매 전략:** 상승 예측 확률 x% 시 매수 then 다음 날 종가 매도 
- **인공지능 모델:** LightGBM
- **학습 데이터:** 3년치 주가 일별 데이터
- **검증 데이터:** 1년치 주가 일별 데이터

### 2.3.1. 데이터 전처리 

**1) 변동성 종목을 선정**

- 변동성 기준 : 거래대금 10억 & 금일 주가 변화율 5% 이상

**2) 종속 변수를 이진 분류 예측 문제로 설정**

- 종속 변수 : 다음 날 종가 상승률(next_change) 2% 이상 상승 여부 (상승시 1, 상승 안할 시 0)

- next_change = (next_day_close - today_close) / today_close 


```python
opened_df = pd.read_csv('opened_df.csv')
opened_df['Code'] = opened_df['Code'].apply(lambda x : str(x).zfill(6))

opened_df['next_change'] = opened_df['Change'].shift(-1)
opened_df = opened_df.dropna()

openedX = opened_df.iloc[:, :-1]
openedY = opened_df[['next_change']]

# 변동성 조건 
print('변동성 조건 설정 전:', openedX.shape)
def sampling(df):
    condition1 = (-0.3 <= df.Change) & (df.Change <= 0.3) # 상한가, 하한가 초과하는 예외 제거 
    condition2 = df.trading_value >= 1000000000 # 변동성 조건 1: 거래대금 10억 이상 
    condition3 = (-0.05 >= df.Change) | (0.05 <= df.Change) # 변동성 조건 2: 금일 주가 변화율 5%이상 
    condition = condition1 & condition2 & condition3
    return condition

condition = sampling(openedX)

# 변동성 조건 적용 
openedX = openedX[condition]
openedY = openedY[condition]

print('변동성 조건 설정 후:', openedX.shape)

# 학습 후 검증을 위한 학습 / 검증 데이터셋으로 분할 
split_date_train = (openedX['Date'] >= '2017-01-02') & (openedX['Date'] <= '2020-12-31')
split_date_valid = (openedX['Date'] >= '2021-01-02') & (openedX['Date'] <= '2021-12-31')

openedX = openedX.drop(columns=['Date', 'Code'])

trainX = openedX[split_date_train]
trainY = openedY[split_date_train]
# 종속변수: 2% 상승여부에 대한 분류 문제를 위해 이진 값으로 변환 
trainY_classification = (trainY >= 0.02).astype('int')

validX = openedX[split_date_valid]
validY = openedY[split_date_valid]
validY_classification = (validY >= 0.02).astype('int')
```

    변동성 조건 설정 전: (2279794, 58)
    변동성 조건 설정 후: (146907, 58)
    

### 2.3.2. 데이터 모델링 
LightBGM를 사용하여 trainX, trainY 데이터셋에 대하여 예측 모델 학습 


```python
##### LightBGM
from lightgbm import LGBMClassifier
import os
scale_pos_weight = round(72/28 , 2)

params = {  'random_state' : 42,
            'scale_pos_weight' : scale_pos_weight,
            'learning_rate' : 0.1, 
            'num_iterations' : 1000,
            'max_depth' : 4,
            'n_jobs' : 30,
            'boost_from_average' : False,
            'objective' : 'binary' }

lgb = LGBMClassifier( **params )
lgb.fit(trainX, trainY_classification)
```




    LGBMClassifier(boost_from_average=False, max_depth=4, n_jobs=30,
                   num_iterations=1000, objective='binary', random_state=42,
                   scale_pos_weight=2.57)



### 2.3.3. 상승 확률 경계값 설정 


```python
from sklearn.preprocessing import Binarizer
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib
thresholds = list(np.arange(0.1, 1, 0.01))
matplotlib.rcParams['font.family'] ='NanumSquareRound'

fig = plt.figure(figsize=(10,6))
def get_eval_by_threshold(y_test, pred_proba, thresholds):
    for i in thresholds:
        binarizer = Binarizer(threshold = i).fit(pred_proba)
        pred = binarizer.transform(pred_proba)
        
        plt.scatter(i, precision_score(y_test, pred), color='b', label='정밀도') # 정밀도
        plt.scatter(i, recall_score(y_test, pred), color='r', label ='재현율') # 재현율 
        plt.scatter(i, f1_score(y_test, pred), color='g', label='f1 score') # f1 score
        if i == 0.1:
            plt.legend(fontsize = 15)
        plt.title('LightGBM 정밀도, 재현율, f1 score',fontsize=20)
        plt.ylabel("score", fontsize=20)
        plt.xlabel("Threshhold", fontsize=20)
        plt.axvline(0.82, color = 'b')
        
lgb_valid_prob = lgb.predict_proba(validX)[:, 1] # XGBoost validation set 확률값리스트

get_eval_by_threshold(validY_classification['next_change'], lgb_valid_prob.reshape(-1,1), thresholds)

plt.grid()
plt.show()
```


    
![png](output_58_0.png)
    


### 2.3.4. 수익성 검증


```python
import seaborn as sns
upper_80=[]
lower_80=[]

lgb_valid_prob = lgb.predict_proba(validX)[:, 1]

for prob, change in zip(lgb_valid_prob,100*(validY['next_change'])):
    if prob >= 0.82:
        upper_80.append(change)
    else:
        lower_80.append(change)

        
fig = plt.figure(figsize=(16,10))
sns.distplot(upper_80, color= 'r', label='80% 이상 예측한 값들의 다음날 종가 변화율 분포')
sns.distplot(lower_80, label='80% 이외의 나머지 추론 값들의 다음날 종가 변화율 분포')
plt.axvline(np.mean(upper_80),color='r')
plt.axvline(np.mean(lower_80),color='b')
plt.legend(fontsize=15)

print('붉은색 분포의 평균 :',np.mean(upper_80))
print('푸른색 분포의 평균 :',np.mean(lower_80))
```

    붉은색 분포의 평균 : 4.134808887967427
    푸른색 분포의 평균 : 0.20164355194696085
    


    
![png](output_60_1.png)
    


### 2.3.5. 모델 저장 후 AIT마켓에 제출(출시)


```python
import pickle

result = {
    'name' : '뽀로로LightGBM',
    'sampling' : sampling,
    'model' : lgb,
    'threshold' : 0.82    
}

with open('뽀로로LightGBM.pickle', 'wb') as handle:
    pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
```

## 2.4. 티라노 Gaussian naive bayes 팀    
- **생산자 이름:** 티라노 Gaussian naive bayes
- **예측 목표:** 변동성 종목에 대해서 다음 날 종가 2% 상승 여부인공지능 모델로 예측
    - 변동성 기준: 거래대금 10억 & 금일 주가 변화율 5% 이상
- **매매 전략:** 상승 예측 확률 x% 시 매수 then 다음 날 종가 매도 
- **인공지능 모델:** Gaussian naive bayes
- **학습 데이터:** 3년치 주가 일별 데이터
- **검증 데이터:** 1년치 주가 일별 데이터

### 2.4.1. 데이터 전처리 

**1) 변동성 종목을 선정**

- 변동성 기준 : 거래대금 10억 & 금일 주가 변화율 5% 이상

**2) 종속 변수를 이진 분류 예측 문제로 설정**

- 종속 변수 : 다음 날 종가 상승률(next_change) 2% 이상 상승 여부 (상승시 1, 상승 안할 시 0)

- next_change = (next_day_close - today_close) / today_close 


```python
opened_df = pd.read_csv('opened_df.csv')
opened_df['Code'] = opened_df['Code'].apply(lambda x : str(x).zfill(6))

opened_df['next_change'] = opened_df['Change'].shift(-1)
opened_df = opened_df.dropna()

openedX = opened_df.iloc[:, :-1]
openedY = opened_df[['next_change']]

# 변동성 조건 
print('변동성 조건 설정 전:', openedX.shape)
def sampling(df):
    condition1 = (-0.3 <= df.Change) & (df.Change <= 0.3) # 상한가, 하한가 초과하는 예외 제거 
    condition2 = df.trading_value >= 1000000000 # 변동성 조건 1: 거래대금 10억 이상 
    condition3 = (-0.05 >= df.Change) | (0.05 <= df.Change) # 변동성 조건 2: 금일 주가 변화율 5%이상 
    condition = condition1 & condition2 & condition3
    return condition

condition = sampling(openedX)

# 변동성 조건 적용 
openedX = openedX[condition]
openedY = openedY[condition]

print('변동성 조건 설정 후:', openedX.shape)

# 학습 후 검증을 위한 학습 / 검증 데이터셋으로 분할 
split_date_train = (openedX['Date'] >= '2017-01-02') & (openedX['Date'] <= '2020-12-31')
split_date_valid = (openedX['Date'] >= '2021-01-02') & (openedX['Date'] <= '2021-12-31')

openedX = openedX.drop(columns=['Date', 'Code'])

trainX = openedX[split_date_train]
trainY = openedY[split_date_train]
# 종속변수: 2% 상승여부에 대한 분류 문제를 위해 이진 값으로 변환 
trainY_classification = (trainY >= 0.02).astype('int')

validX = openedX[split_date_valid]
validY = openedY[split_date_valid]
validY_classification = (validY >= 0.02).astype('int')
```

    변동성 조건 설정 전: (2279794, 58)
    변동성 조건 설정 후: (146907, 58)
    

### 2.4.2. 데이터 모델링 
Gaussian naive bayes를 사용하여 trainX, trainY 데이터셋에 대하여 예측 모델 학습 


```python
##### Gaussian naive bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(trainX, trainY_classification)
```




    GaussianNB()



### 2.4.3. 상승 확률 경계값 설정 


```python
from sklearn.preprocessing import Binarizer
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib
thresholds = list(np.arange(0.1, 1, 0.01))
matplotlib.rcParams['font.family'] ='NanumSquareRound'

fig = plt.figure(figsize=(10,6))
def get_eval_by_threshold(y_test, pred_proba, thresholds):
    for i in thresholds:
        binarizer = Binarizer(threshold = i).fit(pred_proba)
        pred = binarizer.transform(pred_proba)
        
        plt.scatter(i, precision_score(y_test, pred), color='b', label='정밀도') # 정밀도
        plt.scatter(i, recall_score(y_test, pred), color='r', label ='재현율') # 재현율 
        plt.scatter(i, f1_score(y_test, pred), color='g', label='f1 score') # f1 score
        if i == 0.1:
            plt.legend(fontsize = 15)
        plt.title('Gaussian naive bayes 정밀도, 재현율, f1 score',fontsize=20)
        plt.ylabel("score", fontsize=20)
        plt.xlabel("Threshhold", fontsize=20)
        plt.axvline(0.97, color = 'b')
        
gnb_valid_prob = gnb.predict_proba(validX)[:, 1] # XGBoost validation set 확률값리스트

get_eval_by_threshold(validY_classification['next_change'], gnb_valid_prob.reshape(-1,1), thresholds)

plt.grid()
plt.show()
```


    
![png](output_69_0.png)
    


### 2.4.4. 수익성 검증


```python
import seaborn as sns
upper_80=[]
lower_80=[]

gnb_valid_prob = gnb.predict_proba(validX)[:, 1]

for prob, change in zip(gnb_valid_prob,100*(validY['next_change'])):
    if prob >= 0.97:
        upper_80.append(change)
    else:
        lower_80.append(change)

        
fig = plt.figure(figsize=(16,10))
sns.distplot(upper_80, color= 'r', label='80% 이상 예측한 값들의 다음날 종가 변화율 분포')
sns.distplot(lower_80, label='80% 이외의 나머지 추론 값들의 다음날 종가 변화율 분포')
plt.axvline(np.mean(upper_80),color='r')
plt.axvline(np.mean(lower_80),color='b')
plt.legend(fontsize=15)

print('붉은색 분포의 평균 :',np.mean(upper_80))
print('푸른색 분포의 평균 :',np.mean(lower_80))
```

    붉은색 분포의 평균 : 0.590020789846276
    푸른색 분포의 평균 : 0.2233166627152597
    


    
![png](output_71_1.png)
    


### 2.4.5. 모델 저장 후 AIT마켓에 제출(출시)


```python
import pickle

result = {
    'name' : '티라노Gaussiannaivebayes',
    'sampling' : sampling,
    'model' : gnb,
    'threshold' : 0.97    
}

with open('티라노Gaussiannaivebayes.pickle', 'wb') as handle:
    pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
```

## 2.5. 폴리 Random Forest 팀    
- **생산자 이름:** 폴리 Random Forest
- **예측 목표:** 변동성 종목에 대해서 다음 날 종가 2% 상승 여부인공지능 모델로 예측
    - 변동성 기준: 거래대금 10억 & 금일 주가 변화율 5% 이상
- **매매 전략:** 상승 예측 확률 x% 시 매수 then 다음 날 종가 매도 
- **인공지능 모델:** Random Forest
- **학습 데이터:** 3년치 주가 일별 데이터
- **검증 데이터:** 1년치 주가 일별 데이터

### 2.5.1. 데이터 전처리 

**1) 변동성 종목을 선정**

- 변동성 기준 : 거래대금 10억 & 금일 주가 변화율 5% 이상

**2) 종속 변수를 이진 분류 예측 문제로 설정**

- 종속 변수 : 다음 날 종가 상승률(next_change) 2% 이상 상승 여부 (상승시 1, 상승 안할 시 0)

- next_change = (next_day_close - today_close) / today_close 


```python
opened_df = pd.read_csv('opened_df.csv')
opened_df['Code'] = opened_df['Code'].apply(lambda x : str(x).zfill(6))

opened_df['next_change'] = opened_df['Change'].shift(-1)
opened_df = opened_df.dropna()

openedX = opened_df.iloc[:, :-1]
openedY = opened_df[['next_change']]

# 변동성 조건 
print('변동성 조건 설정 전:', openedX.shape)
def sampling(df):
    condition1 = (-0.3 <= df.Change) & (df.Change <= 0.3) # 상한가, 하한가 초과하는 예외 제거 
    condition2 = df.trading_value >= 1000000000 # 변동성 조건 1: 거래대금 10억 이상 
    condition3 = (-0.05 >= df.Change) | (0.05 <= df.Change) # 변동성 조건 2: 금일 주가 변화율 5%이상 
    condition = condition1 & condition2 & condition3
    return condition

condition = sampling(openedX)

# 변동성 조건 적용 
openedX = openedX[condition]
openedY = openedY[condition]

print('변동성 조건 설정 후:', openedX.shape)

# 학습 후 검증을 위한 학습 / 검증 데이터셋으로 분할 
split_date_train = (openedX['Date'] >= '2017-01-02') & (openedX['Date'] <= '2020-12-31')
split_date_valid = (openedX['Date'] >= '2021-01-02') & (openedX['Date'] <= '2021-12-31')

openedX = openedX.drop(columns=['Date', 'Code'])

trainX = openedX[split_date_train]
trainY = openedY[split_date_train]
# 종속변수: 2% 상승여부에 대한 분류 문제를 위해 이진 값으로 변환 
trainY_classification = (trainY >= 0.02).astype('int')

validX = openedX[split_date_valid]
validY = openedY[split_date_valid]
validY_classification = (validY >= 0.02).astype('int')
```

    변동성 조건 설정 전: (2279794, 58)
    변동성 조건 설정 후: (146907, 58)
    

### 2.5.2. 데이터 모델링 
Random forest를 사용하여 trainX, trainY 데이터셋에 대하여 예측 모델 학습 


```python
##### Random forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state = 42, max_depth=4, n_estimators=150)
rf.fit(trainX, trainY_classification)
```




    RandomForestClassifier(max_depth=4, n_estimators=150, random_state=42)



### 2.5.3. 상승 확률 경계값 설정 


```python
from sklearn.preprocessing import Binarizer
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib
thresholds = list(np.arange(0.1, 1, 0.01))
matplotlib.rcParams['font.family'] ='NanumSquareRound'

fig = plt.figure(figsize=(10,6))
def get_eval_by_threshold(y_test, pred_proba, thresholds):
    for i in thresholds:
        binarizer = Binarizer(threshold = i).fit(pred_proba)
        pred = binarizer.transform(pred_proba)
        
        plt.scatter(i, precision_score(y_test, pred), color='b', label='정밀도') # 정밀도
        plt.scatter(i, recall_score(y_test, pred), color='r', label ='재현율') # 재현율 
        plt.scatter(i, f1_score(y_test, pred), color='g', label='f1 score') # f1 score
        if i == 0.1:
            plt.legend(fontsize = 15)
        plt.title('Random Forest 정밀도, 재현율, f1 score',fontsize=20)
        plt.ylabel("score", fontsize=20)
        plt.xlabel("Threshhold", fontsize=20)
        plt.axvline(0.54, color = 'b')
        
rf_valid_prob = rf.predict_proba(validX)[:, 1] # XGBoost validation set 확률값리스트

get_eval_by_threshold(validY_classification['next_change'], rf_valid_prob.reshape(-1,1), thresholds)

plt.grid()
plt.show()
```


    
![png](output_80_0.png)
    


### 2.5.4. 수익성 검증


```python
import seaborn as sns
upper_80=[]
lower_80=[]

rf_valid_prob = rf.predict_proba(validX)[:, 1]

for prob, change in zip(rf_valid_prob,100*(validY['next_change'])):
    if prob >= 0.54:
        upper_80.append(change)
    else:
        lower_80.append(change)

        
fig = plt.figure(figsize=(16,10))
sns.distplot(upper_80, color= 'r', label='80% 이상 예측한 값들의 다음날 종가 변화율 분포')
sns.distplot(lower_80, label='80% 이외의 나머지 추론 값들의 다음날 종가 변화율 분포')
plt.axvline(np.mean(upper_80),color='r')
plt.axvline(np.mean(lower_80),color='b')
plt.legend(fontsize=15)

print('붉은색 분포의 평균 :',np.mean(upper_80))
print('푸른색 분포의 평균 :',np.mean(lower_80))
```

    붉은색 분포의 평균 : 2.9850184972449414
    푸른색 분포의 평균 : 0.22071345772061618
    


    
![png](output_82_1.png)
    


### 2.5.5. 모델 저장 후 AIT마켓에 제출(출시)


```python
import pickle

result = {
    'name' : '폴리RandomForest',
    'sampling' : sampling,
    'model' : rf,
    'threshold' : 0.54  
}

with open('폴리RandomForest.pickle', 'wb') as handle:
    pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
```

## 2.6. 쌍둥이 K nearest neighbor 팀    
- **생산자 이름:** 쌍둥이 K nearest neighbor
- **예측 목표:** 변동성 종목에 대해서 다음 날 종가 2% 상승 여부인공지능 모델로 예측
    - 변동성 기준: 거래대금 10억 & 금일 주가 변화율 5% 이상
- **매매 전략:** 상승 예측 확률 x% 시 매수 then 다음 날 종가 매도 
- **인공지능 모델:** K nearest neighbor
- **학습 데이터:** 3년치 주가 일별 데이터
- **검증 데이터:** 1년치 주가 일별 데이터

### 2.5.1. 데이터 전처리 

**1) 변동성 종목을 선정**

- 변동성 기준 : 거래대금 10억 & 금일 주가 변화율 5% 이상

**2) 종속 변수를 이진 분류 예측 문제로 설정**

- 종속 변수 : 다음 날 종가 상승률(next_change) 2% 이상 상승 여부 (상승시 1, 상승 안할 시 0)

- next_change = (next_day_close - today_close) / today_close 


```python
opened_df = pd.read_csv('opened_df.csv')
opened_df['Code'] = opened_df['Code'].apply(lambda x : str(x).zfill(6))

opened_df['next_change'] = opened_df['Change'].shift(-1)
opened_df = opened_df.dropna()

openedX = opened_df.iloc[:, :-1]
openedY = opened_df[['next_change']]

# 변동성 조건 
print('변동성 조건 설정 전:', openedX.shape)
def sampling(df):
    condition1 = (-0.3 <= df.Change) & (df.Change <= 0.3) # 상한가, 하한가 초과하는 예외 제거 
    condition2 = df.trading_value >= 1000000000 # 변동성 조건 1: 거래대금 10억 이상 
    condition3 = (-0.05 >= df.Change) | (0.05 <= df.Change) # 변동성 조건 2: 금일 주가 변화율 5%이상 
    condition = condition1 & condition2 & condition3
    return condition

condition = sampling(openedX)

# 변동성 조건 적용 
openedX = openedX[condition]
openedY = openedY[condition]

print('변동성 조건 설정 후:', openedX.shape)

# 학습 후 검증을 위한 학습 / 검증 데이터셋으로 분할 
split_date_train = (openedX['Date'] >= '2017-01-02') & (openedX['Date'] <= '2020-12-31')
split_date_valid = (openedX['Date'] >= '2021-01-02') & (openedX['Date'] <= '2021-12-31')

openedX = openedX.drop(columns=['Date', 'Code'])

trainX = openedX[split_date_train]
trainY = openedY[split_date_train]
# 종속변수: 2% 상승여부에 대한 분류 문제를 위해 이진 값으로 변환 
trainY_classification = (trainY >= 0.02).astype('int')

validX = openedX[split_date_valid]
validY = openedY[split_date_valid]
validY_classification = (validY >= 0.02).astype('int')
```

    변동성 조건 설정 전: (2279794, 58)
    변동성 조건 설정 후: (146907, 58)
    

### 2.5.2. 데이터 모델링 
K nearest neighbor를 사용하여 trainX, trainY 데이터셋에 대하여 예측 모델 학습 


```python
##### K nearest neighbor
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(trainX, trainY_classification)
```




    KNeighborsClassifier()



### 2.5.3. 상승 확률 경계값 설정 


```python
from sklearn.preprocessing import Binarizer
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib
thresholds = list(np.arange(0.1, 1, 0.01))
matplotlib.rcParams['font.family'] ='NanumSquareRound'

fig = plt.figure(figsize=(10,6))
def get_eval_by_threshold(y_test, pred_proba, thresholds):
    for i in thresholds:
        binarizer = Binarizer(threshold = i).fit(pred_proba)
        pred = binarizer.transform(pred_proba)
        
        plt.scatter(i, precision_score(y_test, pred), color='b', label='정밀도') # 정밀도
        plt.scatter(i, recall_score(y_test, pred), color='r', label ='재현율') # 재현율 
        plt.scatter(i, f1_score(y_test, pred), color='g', label='f1 score') # f1 score
        if i == 0.1:
            plt.legend(fontsize = 15)
        plt.title('K nearest neighbor 정밀도, 재현율, f1 score',fontsize=20)
        plt.ylabel("score", fontsize=20)
        plt.xlabel("Threshhold", fontsize=20)
        plt.axvline(0.6, color = 'b')
        
knn_valid_prob = knn.predict_proba(validX)[:, 1] # XGBoost validation set 확률값리스트

get_eval_by_threshold(validY_classification['next_change'], knn_valid_prob.reshape(-1,1), thresholds)

plt.grid()
plt.show()
```


    
![png](output_91_0.png)
    


### 2.5.4. 수익성 검증


```python
import seaborn as sns
upper_80=[]
lower_80=[]

xgb_valid_prob = xgb.predict_proba(validX)[:, 1]

for prob, change in zip(xgb_valid_prob,100*(validY['next_change'])):
    if prob >= 0.6:
        upper_80.append(change)
    else:
        lower_80.append(change)

        
fig = plt.figure(figsize=(16,10))
sns.distplot(upper_80, color= 'r', label='80% 이상 예측한 값들의 다음날 종가 변화율 분포')
sns.distplot(lower_80, label='80% 이외의 나머지 추론 값들의 다음날 종가 변화율 분포')
plt.axvline(np.mean(upper_80),color='r')
plt.axvline(np.mean(lower_80),color='b')
plt.legend(fontsize=15)

print('붉은색 분포의 평균 :',np.mean(upper_80))
print('푸른색 분포의 평균 :',np.mean(lower_80))
```

    붉은색 분포의 평균 : 1.6412434414011048
    푸른색 분포의 평균 : 0.06998500698053335
    


    
![png](output_93_1.png)
    


### 2.5.5. 모델 저장 후 AIT마켓에 제출(출시)


```python
import pickle

result = {
    'name' : '쌍둥이Knearestneighbor',
    'sampling' : sampling,
    'model' : knn,
    'threshold' : 0.6    
}

with open('쌍둥이Knearestneighbor.pickle', 'wb') as handle:
    pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
```

# 3. AIT 운영자 : 모델 별 수익률 비교

## 3.1. 비공개 데이터 및 제출된 모델 불러오기

1) 비공개 주가 데이터를 불러옵니다.

2) AIT마켓에 제출된 모델들을 불러옵니다.


```python
# 비공개 주가 데이터를 불러옵니다.
closed_df = pd.read_csv('./closed_df.csv')
closed_df['Code'] = closed_df['Code'].apply(lambda x : str(x).zfill(6))

# AIT마켓에 제출된 모델들을 불러옵니다.
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

lst_model_file = ['수원랩XGboost.pickle',
                    '경기랩LogisticRegression.pickle',
                    '뽀로로LightGBM.pickle',
                    '티라노Gaussiannaivebayes.pickle',
                    '폴리RandomForest.pickle',
                    '쌍둥이Knearestneighbor.pickle']

lst_model_file = [file for file in lst_model_file]

lst_model = []
for model_file in lst_model_file:
    with open(model_file, 'rb') as handle:
        lst_model.append(pickle.load(handle))
```

## 3.2. 주문일지 작성하기

비공개 주가 데이터(2022.01.01~2022.06.30)에서 제출된 모델들의 주문일지를 작성합니다.


```python
# 주문일지를 저장할 리스트를 선언합니다.
lst_name = []
lst_order_dict = []
for x in tqdm(lst_model): 
    name = x['name']
    sampling = x['sampling']
    model = x['model']
    threshold = x['threshold']
    
    # 샘플링 조건에 맞는 일자만 선별합니다.
    condition = sampling(closed_df)
    closed_df_sampled = closed_df[condition]
    closed_df_sampled_X = closed_df_sampled.drop(columns=['Date', 'Code'])
    
    # 선별된 일자에 대해서 예측을 수행합니다.
    lst_pred_y = model.predict_proba(closed_df_sampled_X)[:,1]
    
    ############################ 주문 일지 작성  ############################
    order_dict={} # order_dict : 주문 일지
    for date, code, pred_y in zip(closed_df_sampled['Date'], closed_df_sampled['Code'], lst_pred_y):
        # 만약 예측 값이 경계값 보다 높은 경우, 매수 주문을 실행합니다.
        if pred_y >= threshold:
            if date not in order_dict.keys():
                order_dict[date] = [[code], "buy", 1.0, [round(pred_y*100,2)]]
            else:
                order_dict[date][0].append(code)
                order_dict[date][2] = (1/len(order_dict[date][0]))
                order_dict[date][3].append(round(pred_y*100,2))
    order_dict = sorted(order_dict.items())
    
    lst_name.append(name)
    lst_order_dict.append(order_dict)
```

    100%|█████████████████████████████████████████████| 6/6 [00:22<00:00,  3.83s/it]
    

## 3.3. 매매 시뮬레이션을 실행하여 수익률 계산

작성된 주문일지에 대해서 매매 시뮬레이션을 수행하여 수익률을 계산 및 추적합니다.


```python
import FinanceDataReader as fdr
import pandas as pd
import numpy as np

printing = 'off' # 매매 시뮬레이션 로그의 출력을 조절합니다.
start_date = pd.to_datetime('2022-01-01') # 매매 시뮬레이션을 시작할 날짜
end_date = pd.to_datetime('2022-07-01') # 매매 시뮬레이션을 종료할 날짜

lst_result = []
for name, order_dict in tqdm(zip(lst_name,lst_order_dict)): 
    ############################ 수익률 시뮬레이션  ############################
    start_money = 1000000 # 초기 현금 1천만원
    money = start_money
    
    sell_change_lst =[]
    result=[]
    if printing == 'on':
        print('시작 금액 : {} 만원'.format(start_money/100000))
    for i, row in enumerate(order_dict): #주문 일지를 한 줄 읽어 옴
        buy_date = pd.to_datetime(row[0])
        code_lst = row[1][0]    
        buy_ratio = row[1][2] 

        sell_date_dict={} # 매도를 위한 딕셔너리
        no_subtract_money = money # 예측한 기업이 여러 개일 때 매수 금액을 계산하는 돈
        for code in code_lst:
            stock = fdr.DataReader(code, start_date, end_date).reset_index()
            stock_lst = stock.values.tolist()

            for ii,stock_row in enumerate(stock_lst):

                buy_stock_date = stock_row[0]
                buy_close_price = stock_row[4]
                
                sell_stock_date = stock_lst[ii+1][0]
                sell_close_price = stock_lst[ii+1][4]
                sell_change = stock_lst[ii+1][6]


                    # 예측한 기업이 1개일때
                if buy_date == buy_stock_date and sell_stock_date not in sell_date_dict.keys():
                # 매수    
                    buy_stock_count = int((money*buy_ratio)/buy_close_price)
                    money -= buy_stock_count*buy_close_price

                    sell_date_dict[sell_stock_date] = [[code, buy_stock_count, buy_close_price, sell_close_price, sell_change, ]]
                    
                    if printing == 'on':
                        print('\n{} : BUY {} -> {}주 구입, 매수 금액: {}만원'.format(
                                                          str(buy_date).split(" ")[0], 
                                                          code, 
                                                          buy_stock_count,
                                                          (buy_stock_count * buy_close_price)/10000)
                                                           )
                    break

                    # 예측한 기업이 2개 이상일때
                elif buy_date == buy_stock_date and sell_stock_date in sell_date_dict.keys():
                # 매수    
                    buy_stock_count = int((no_subtract_money * buy_ratio) / buy_close_price)
                    money -= buy_stock_count*buy_close_price

                    sell_date_dict[sell_stock_date].append([code, buy_stock_count, buy_close_price, sell_close_price, sell_change])

                    if printing == 'on':
                        print('{} : BUY {} -> {}주 구입, 매수 금액: {}만원'.format(
                                                          str(buy_date).split(" ")[0], 
                                                          code, 
                                                          buy_stock_count,
                                                          (buy_stock_count * buy_close_price)/10000)
                                                           )
                    break

        for sell_stock in sell_date_dict[sell_stock_date]:
            sell_stock_code = sell_stock[0] # 팔아야 할 기업 코드
            buy_stock_count = sell_stock[1] # 매수 수량
            buy_close_price = sell_stock[2] # 매수 가격
            sell_close_price = sell_stock[3] # 매도 가격
            close_change = sell_stock[4] # 다음 날 d+1의 종가 변화량

        # 매도
            earnings = (buy_stock_count * sell_close_price) - (buy_stock_count * buy_close_price)
            money += (buy_stock_count * sell_close_price)
            if printing == 'on':
                print('{} : SELL {} -> 매도 금액 : {}만원, 전날 대비 상승률 {}:, 이익금 : {}만원, 현재 자산 : {:.2f}만원'.format(str(sell_stock_date).split(" ")[0],
                                                                                        sell_stock_code, 
                                                                                        (buy_stock_count * sell_close_price)/10000,                     
                                                                                        round(close_change*100, 2),
                                                                                          (earnings/10000), 
                                                                                        (money/10000)))
            sell_change_lst.append(close_change)
        result.append([str(sell_stock_date).split(" ")[0], (money/10000)-100])
    if printing == 'on':        
        print('2022 상반기 수익률은 :', 100*(money/start_money) - 100,'% 입니다.')
    lst_result.append(pd.DataFrame(result, columns=['sell_date', 'money']))

from functools import reduce        
# 6개의 df를 한번에 merge하기
df_merge = reduce(lambda left, right: pd.merge(left, right, on='sell_date', how='outer',sort=True), lst_result)

df_merge.columns=['sell_date'] + lst_name
df_merge = df_merge.sort_values(by=['sell_date'], axis=0).reset_index()
del df_merge['index']
df_merge = df_merge.set_index('sell_date').fillna(method='ffill').fillna(0)

# 일자별 수익률 추적 데이터입니다.
df_merge
```

    6it [07:39, 76.61s/it] 
    




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
      <th>수원랩XGboost</th>
      <th>경기랩LogisticRegression</th>
      <th>뽀로로LightGBM</th>
      <th>티라노Gaussiannaivebayes</th>
      <th>폴리RandomForest</th>
      <th>쌍둥이Knearestneighbor</th>
    </tr>
    <tr>
      <th>sell_date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2022-01-04</th>
      <td>0.0000</td>
      <td>-0.1520</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>4.6800</td>
    </tr>
    <tr>
      <th>2022-01-05</th>
      <td>0.0000</td>
      <td>-0.1965</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>7.2819</td>
    </tr>
    <tr>
      <th>2022-01-06</th>
      <td>0.0000</td>
      <td>-2.5690</td>
      <td>0.0000</td>
      <td>-6.5765</td>
      <td>0.0000</td>
      <td>5.0368</td>
    </tr>
    <tr>
      <th>2022-01-07</th>
      <td>1.7450</td>
      <td>-0.9060</td>
      <td>1.8850</td>
      <td>-4.8715</td>
      <td>0.0000</td>
      <td>6.9533</td>
    </tr>
    <tr>
      <th>2022-01-10</th>
      <td>1.7450</td>
      <td>-1.2140</td>
      <td>1.8850</td>
      <td>-4.8715</td>
      <td>0.0000</td>
      <td>6.4128</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2022-06-27</th>
      <td>137.0468</td>
      <td>-21.0606</td>
      <td>147.0539</td>
      <td>-25.1050</td>
      <td>61.7662</td>
      <td>5.6073</td>
    </tr>
    <tr>
      <th>2022-06-28</th>
      <td>161.1810</td>
      <td>-21.0661</td>
      <td>152.1374</td>
      <td>-25.7650</td>
      <td>70.6452</td>
      <td>6.5757</td>
    </tr>
    <tr>
      <th>2022-06-29</th>
      <td>142.4810</td>
      <td>-20.5206</td>
      <td>152.1374</td>
      <td>-25.7650</td>
      <td>58.5452</td>
      <td>6.6912</td>
    </tr>
    <tr>
      <th>2022-06-30</th>
      <td>142.4810</td>
      <td>-21.1086</td>
      <td>95.4374</td>
      <td>-25.7650</td>
      <td>23.8952</td>
      <td>3.9831</td>
    </tr>
    <tr>
      <th>2022-07-01</th>
      <td>142.4810</td>
      <td>-21.6028</td>
      <td>95.4374</td>
      <td>-25.7650</td>
      <td>23.8952</td>
      <td>3.4180</td>
    </tr>
  </tbody>
</table>
<p>121 rows × 6 columns</p>
</div>



## 3.4. 수익률 리더보드 및 시각화
1) 리더보드를 작성합니다.

2) 시계열 플랏을 작성합니다.


```python
# 리더보드
leader_board = pd.DataFrame(df_merge.iloc[-1]).sort_values(by=df_merge.index[-1], axis=0, ascending=False)
leader_board
```




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
      <th>2022-07-01</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>수원랩XGboost</th>
      <td>142.4810</td>
    </tr>
    <tr>
      <th>뽀로로LightGBM</th>
      <td>95.4374</td>
    </tr>
    <tr>
      <th>폴리RandomForest</th>
      <td>23.8952</td>
    </tr>
    <tr>
      <th>쌍둥이Knearestneighbor</th>
      <td>3.4180</td>
    </tr>
    <tr>
      <th>경기랩LogisticRegression</th>
      <td>-21.6028</td>
    </tr>
    <tr>
      <th>티라노Gaussiannaivebayes</th>
      <td>-25.7650</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 수익률 플랏
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
matplotlib.rcParams['font.family'] ='NanumSquareRound'
mpl.rcParams['axes.unicode_minus'] =False
f,ax = plt.subplots(1,1,figsize=(25,12),sharex=False)
for i in range(len(lst_name)):
    ax = sns.lineplot(data = df_merge, x=df_merge.index, y = lst_name[i], label=lst_name[i], linewidth=4)

_=plt.xticks(range(0,91, 10), fontsize=30, rotation = 30)
_=plt.yticks(range(-100,100,10), fontsize=30)
ax.set_xlabel('거래일', fontsize=35)
ax.set_ylabel('수익률', fontsize=35)
_=plt.legend(fontsize=30)
########################### 그래프에 표시되지 않은 모델은 80% 이상 확률 값의 매수신호가 없는 것입니다. #####################
```


    
![png](output_105_0.png)
    


# 4. AIT 구매자 : AI트레이더 시그널 구독
2022년 상반기 XGBoost모델이 약 60% 수익을 내며 1위를 차지했습니다.   
이제 이 모델을 구독하여 서비스를 예시를 설명합니다.   

위에서 작성한 simulator함수는 매수 주문서인 order_dict를 작성합니다.   
매수 주문서인 order_dict에 기반하여 구독자에게 챗봇을 통해 매수 신호를 전달합니다.   

-----------------------------------------------------------------------------------------------------------------------------------
**아래 챗봇은 예시를 위해 생성한 Slack 챗봇입니다.**


```python
order_dict = lst_order_dict[lst_name.index("수원랩XGboost")] # 1등 알고리즘 수원랩XGboost의 매매일지를 불러옵니다.
example_order = order_dict[-1] 
example_order # 가장 마지막에 매매한 주문을 메신저로 포워딩합니다.
```




    ('2022-06-30', [['016385'], 'buy', 1.0, [81.86]])




```python
from datetime import datetime
import requests
def dbgout(message):
    """인자로 받은 문자열을 파이썬 셸과 슬랙으로 동시에 출력한다."""
    # 필요한 인자 토큰, 슬랙 채널 이름
    token = 'xoxb-3837103642832-3810483855669-'
    channel = '#ai-구독서비스-알리미'

    strbuf = datetime.now().strftime('[%m/%d %H:%M:%S] ') + message
    response = requests.post("https://slack.com/api/chat.postMessage",
        headers={"Authorization": "Bearer "+token},
        data={"channel": channel,"text": message}
    )
```


```python
date = example_order[0]
code = example_order[1][0]
prob = example_order[1][-1]
dbgout('구독하신 KRX AI알고리즘 서비스 알리미입니다.')

for (code, prob) in zip(code, prob):
    dbgout('금일 : {}, 종목코드 {} -> {} %확률로 내일의 주가 상승을 예측하였습니다.'.format(date, code, prob))

```

# 구독 시스템 챗봇 예시

<img src='https://ifh.cc/g/3B7TAb.gif' width=400 >

-------

향후에는 이러한 AI트레이더를 제작자들이 업로드하고 이 AI트레이더를 비교하여 선택적으로 구독할 수 있는 플랫폼 서비스를 계획하고 있습니다. 현재 slack 메시지로 구독 정보를 보내는 방식도 kakaotalk 또는 api 또는 증권사의  HTS 연계 또는 자체 거래 시스템을 개발하여 구독과 동시에 매매와도 연동 될 수 있도록 서비스를 개발하고자 합니다.
