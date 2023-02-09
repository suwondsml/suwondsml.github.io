---
layout: post
title: Dacon tengery
authors: [chanho]
categories: [1기 AI/SW developers(개인 프로젝트)]
---


<head>
  <style>
    table.dataframe {
      white-space: normal;
      width: 100%;
      height: 240px;
      display: block;
      overflow: auto;
      font-family: Arial, sans-serif;
      font-size: 0.9rem;
      line-height: 20px;
      text-align: center;
      border: 0px !important;
    }

    table.dataframe th {
      text-align: center;
      font-weight: bold;
      padding: 8px;
    }

    table.dataframe td {
      text-align: center;
      padding: 8px;
    }

    table.dataframe tr:hover {
      background: #b8d1f3; 
    }

    .output_prompt {
      overflow: auto;
      font-size: 0.9rem;
      line-height: 1.45;
      border-radius: 0.3rem;
      -webkit-overflow-scrolling: touch;
      padding: 0.8rem;
      margin-top: 0;
      margin-bottom: 15px;
      font: 1rem Consolas, "Liberation Mono", Menlo, Courier, monospace;
      color: $code-text-color;
      border: solid 1px $border-color;
      border-radius: 0.3rem;
      word-break: normal;
      white-space: pre;
    }

  .dataframe tbody tr th:only-of-type {
      vertical-align: middle;
  }

  .dataframe tbody tr th {
      vertical-align: top;
  }

  .dataframe thead th {
      text-align: center !important;
      padding: 8px;
  }

  .page__content p {
      margin: 0 0 0px !important;
  }

  .page__content p > strong {
    font-size: 0.8rem !important;
  }

  </style>
</head>

2022년 12월 12일~14일까지 진행됐던 데이콘공모전 입니다.<!--more-->

# 모듈부르기



```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

trainX, testX, trainY, testY = train_test_split( train_df.iloc[:,2:], train_df['착과량(int)'], test_size=0.1, random_state=42 )
```


```python
!pip install lightgbm
```

<pre>
Collecting lightgbm
  Downloading lightgbm-3.3.3-py3-none-win_amd64.whl (1.0 MB)
Requirement already satisfied: wheel in c:\users\kdh23\anaconda3\lib\site-packages (from lightgbm) (0.36.2)
Requirement already satisfied: scikit-learn!=0.22.0 in c:\users\kdh23\anaconda3\lib\site-packages (from lightgbm) (0.24.2)
Requirement already satisfied: scipy in c:\users\kdh23\anaconda3\lib\site-packages (from lightgbm) (1.6.2)
Requirement already satisfied: numpy in c:\users\kdh23\anaconda3\lib\site-packages (from lightgbm) (1.20.3)
Requirement already satisfied: joblib>=0.11 in c:\users\kdh23\anaconda3\lib\site-packages (from scikit-learn!=0.22.0->lightgbm) (1.0.1)
Requirement already satisfied: threadpoolctl>=2.0.0 in c:\users\kdh23\anaconda3\lib\site-packages (from scikit-learn!=0.22.0->lightgbm) (2.2.0)
Installing collected packages: lightgbm
Successfully installed lightgbm-3.3.3
</pre>
# 데이터 관찰하기

    1. 2207그루의 나무가있음

    

    2. 나무 생육 상태 Features(5개)

        1. 수고(m)

        2. 수관폭1(min)

        3. 수관폭2(max)

        4. 수관폭평균(수관폭1과 수관폭2의 평균)

        

    3. 새순 Features(89개)

        2022. 09. 01 ~ 2022. 11. 28까지 측정된 새순데이터

        

    4. 엽록소 Features(89개)

        2022. 09. 01 ~ 2022. 11. 28까지 측정된 엽록소 데이터



# 모델설계


## LGBM을 사용한 추측



```python
from lightgbm import LGBMRegressor
import os
import time

# pos 8.2 & lr 0.08하면 더 좋음
start_time=time.time()

model = LGBMRegressor(random_state = 42,
                       max_depth = 3, 
    n_estimators = 1000,
    learning_rate = 0.1,)

lgbm_trained_model = model.fit( trainX, trainY, 
          eval_set=[(testX,testY)],
          early_stopping_rounds=50, 
          verbose = 5, 
          eval_metric = 'rmse')

# train_pred = trained_model.predict(trainX)
# train_prob = trained_model.predict_proba(trainX)[:, 1]

# test_pred = trained_model.predict(testX)
# test_prob = trained_model.predict_proba(testX)[:, 1]
```

<pre>
[5]	valid_0's rmse: 134.841	valid_0's l2: 18182.2
[10]	valid_0's rmse: 85.5184	valid_0's l2: 7313.4
[15]	valid_0's rmse: 59.3022	valid_0's l2: 3516.75
[20]	valid_0's rmse: 46.7741	valid_0's l2: 2187.81
[25]	valid_0's rmse: 41.4175	valid_0's l2: 1715.41
[30]	valid_0's rmse: 39.3243	valid_0's l2: 1546.4
[35]	valid_0's rmse: 38.5315	valid_0's l2: 1484.68
[40]	valid_0's rmse: 38.3732	valid_0's l2: 1472.51
[45]	valid_0's rmse: 38.18	valid_0's l2: 1457.71
[50]	valid_0's rmse: 38.1013	valid_0's l2: 1451.71
[55]	valid_0's rmse: 38.1764	valid_0's l2: 1457.43
[60]	valid_0's rmse: 38.2284	valid_0's l2: 1461.41
[65]	valid_0's rmse: 38.3148	valid_0's l2: 1468.03
[70]	valid_0's rmse: 38.3422	valid_0's l2: 1470.13
[75]	valid_0's rmse: 38.4117	valid_0's l2: 1475.46
[80]	valid_0's rmse: 38.436	valid_0's l2: 1477.32
[85]	valid_0's rmse: 38.5037	valid_0's l2: 1482.54
[90]	valid_0's rmse: 38.5726	valid_0's l2: 1487.85
[95]	valid_0's rmse: 38.5956	valid_0's l2: 1489.62
[100]	valid_0's rmse: 38.5374	valid_0's l2: 1485.13
</pre>
## submission 제출



```python
lgbm_pred = lgbm_trained_model.predict(test_df.drop(['ID'], axis=1))
sample_submission = pd.read_csv('./sample_submission.csv')

sample_submission['착과량(int)'] = lgbm_pred
sample_submission.to_csv('./submit3.csv', index=False)
```

# xgb를 사용한 추측



```python
import xgboost as xgb
import os
import time

start_time=time.time()

model = xgb.XGBRegressor(
    max_depth = 3, 
    n_estimators = 1000,
    random_state = 42,
    learning_rate = 0.1,

#     n_jobs = 30
                    
)

xgb_trained_model = model.fit( trainX, trainY, 
          eval_set=[(testX,testY)],
          early_stopping_rounds=50, 
          verbose = 5, 
          eval_metric = 'rmse')

# train_pred = trained_model.predict(trainX)
# train_prob = trained_model.predict_proba(trainX)[:, 1]

# test_pred = trained_model.predict(testX)
# test_prob = trained_model.predict_proba(testX)[:, 1]

print("---%s seconds ---" % (time.time() - start_time))
```

<pre>
[0]	validation_0-rmse:430.37021
[5]	validation_0-rmse:258.31488
[10]	validation_0-rmse:157.59738
[15]	validation_0-rmse:99.69896
[20]	validation_0-rmse:67.86095
[25]	validation_0-rmse:51.67970
[30]	validation_0-rmse:44.21014
[35]	validation_0-rmse:40.97302
[40]	validation_0-rmse:39.42442
[45]	validation_0-rmse:38.76698
[50]	validation_0-rmse:38.42331
[55]	validation_0-rmse:38.33013
[60]	validation_0-rmse:38.29298
[65]	validation_0-rmse:38.38273
[70]	validation_0-rmse:38.40695
[75]	validation_0-rmse:38.43439
[80]	validation_0-rmse:38.53521
[85]	validation_0-rmse:38.58548
[90]	validation_0-rmse:38.66847
[95]	validation_0-rmse:38.79533
[100]	validation_0-rmse:38.83015
[105]	validation_0-rmse:38.91818
[108]	validation_0-rmse:38.96620
---0.9385890960693359 seconds ---
</pre>
## submission 제출



```python
xgb_pred = xgb_trained_model.predict(test_df.drop(['ID'], axis=1))
sample_submission = pd.read_csv('./sample_submission.csv')

sample_submission['착과량(int)'] = xgb_pred
sample_submission.to_csv('./submit3.csv', index=False)
```

# xgb + LGBM 

각 추측값을 더한후 2로 나눈뒤 제출했습니다.



```python
# xgb_pred = xgb_trained_model.predict(test_df.drop(['ID'], axis=1))
sample_submission = pd.read_csv('./sample_submission.csv')

sample_submission['착과량(int)'] = (xgb_pred + lgbm_pred) / 2
sample_submission.to_csv('./ensemble.csv', index=False)
```
