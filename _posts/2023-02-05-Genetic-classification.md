---
layout: post
title: Genetic classification
authors: [Taeyoung Lee, Taehyun Kim]
categories: [1기 AI/SW developers(팀 프로젝트)]
---


```python
!sudo apt-get install -y fonts-nanum
!sudo fc-cache -fv
!rm ~/.cache/matplotlib -rf
```

    Reading package lists... Done
    Building dependency tree       
    Reading state information... Done
    fonts-nanum is already the newest version (20170925-1).
    The following package was automatically installed and is no longer required:
      libnvidia-common-460
    Use 'sudo apt autoremove' to remove it.
    0 upgraded, 0 newly installed, 0 to remove and 21 not upgraded.
    /usr/share/fonts: caching, new cache contents: 0 fonts, 1 dirs
    /usr/share/fonts/truetype: caching, new cache contents: 0 fonts, 3 dirs
    /usr/share/fonts/truetype/humor-sans: caching, new cache contents: 1 fonts, 0 dirs
    /usr/share/fonts/truetype/liberation: caching, new cache contents: 16 fonts, 0 dirs
    /usr/share/fonts/truetype/nanum: caching, new cache contents: 10 fonts, 0 dirs
    /usr/local/share/fonts: caching, new cache contents: 0 fonts, 0 dirs
    /root/.local/share/fonts: skipping, no such directory
    /root/.fonts: skipping, no such directory
    /var/cache/fontconfig: cleaning cache directory
    /root/.cache/fontconfig: not cleaning non-existent cache directory
    /root/.fontconfig: not cleaning non-existent cache directory
    fc-cache: succeeded
    


```python
import pandas as pd
import random
import os
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns
%matplotlib inline

from sklearn import preprocessing
import matplotlib.pyplot as plt
plt.rc('font', family='NanumBarunGothic')

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
# plt.rc('font', family='AppleGothic')
# from catboost import CatBoostClassifier
```


```python
class CFG:
    SEED = 42

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
seed_everything(CFG.SEED) # Seed 고정
```


```python
from google.colab import drive
drive.mount('/content/drive')
```

    Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).
    


```python
snp_info= pd.read_csv("/content/drive/MyDrive/유전체/snp_info.csv")
train= pd.read_csv("/content/drive/MyDrive/유전체/train.csv")
test= pd.read_csv("/content/drive/MyDrive/유전체/test.csv")
sample_submission= pd.read_csv("/content/drive/MyDrive/유전체/sample_submission.csv")
```

Dataset Info.

### train.csv [파일]
1. id : 개체 고유 ID
개체정보
2. father : 개체의 가계 고유 번호 (0 : Unknown)
3. mother : 개체의 모계 고유 번호 (0 : Unknown)
4. gender : 개체 성별 (0 : Unknown, 1 : female, 2 : male)
5. trait : 개체 표현형 정보 
15개의 SNP 정보 : SNP_01 ~ SNP_15
6. class : 개체의 품종 (A,B,C)


### test.csv [파일]
1. id : 개체 샘플 별 고유 ID
개체정보
2. father : 개체의 가계 고유 번호 (0 : Unknown)
3. mother : 개체의 모계 고유 번호 (0 : Unknown)
4. gender : 개체 성별 (0 : Unknown, 1 : female, 2 : male)
5. trait : 개체 표현형 정보 
15개의 SNP 정보 : SNP_01 ~ SNP_15


### snp_info.csv [파일]
15개의 SNP 세부 정보
1. name : SNP 명
2. chrom : 염색체 정보
3. cm : Genetic distance
4. pos : 각 마커의 유전체상 위치 정보


### sample_submission.csv [파일] - 제출 양식
1. id : 개체 샘플 별 고유 ID
2. class : 예측한 개체의 품종 (A,B,C)


```python
snp_info
```





  <div id="df-3a2dac6e-15c5-4e43-a0bb-5f69ddbce63f">
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
      <th>SNP_id</th>
      <th>name</th>
      <th>chrom</th>
      <th>cm</th>
      <th>pos</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>SNP_01</td>
      <td>BTA-19852-no-rs</td>
      <td>2</td>
      <td>67.05460</td>
      <td>42986890</td>
    </tr>
    <tr>
      <th>1</th>
      <td>SNP_02</td>
      <td>ARS-USMARC-Parent-DQ647190-rs29013632</td>
      <td>6</td>
      <td>31.15670</td>
      <td>13897068</td>
    </tr>
    <tr>
      <th>2</th>
      <td>SNP_03</td>
      <td>ARS-BFGL-NGS-117009</td>
      <td>6</td>
      <td>68.28920</td>
      <td>44649549</td>
    </tr>
    <tr>
      <th>3</th>
      <td>SNP_04</td>
      <td>ARS-BFGL-NGS-60567</td>
      <td>6</td>
      <td>77.87490</td>
      <td>53826064</td>
    </tr>
    <tr>
      <th>4</th>
      <td>SNP_05</td>
      <td>BovineHD0600017032</td>
      <td>6</td>
      <td>80.50150</td>
      <td>61779512</td>
    </tr>
    <tr>
      <th>5</th>
      <td>SNP_06</td>
      <td>BovineHD0600017424</td>
      <td>6</td>
      <td>80.59540</td>
      <td>63048481</td>
    </tr>
    <tr>
      <th>6</th>
      <td>SNP_07</td>
      <td>Hapmap49442-BTA-111073</td>
      <td>6</td>
      <td>80.78000</td>
      <td>64037334</td>
    </tr>
    <tr>
      <th>7</th>
      <td>SNP_08</td>
      <td>BovineHD0600018638</td>
      <td>6</td>
      <td>82.68560</td>
      <td>67510588</td>
    </tr>
    <tr>
      <th>8</th>
      <td>SNP_09</td>
      <td>ARS-BFGL-NGS-37727</td>
      <td>6</td>
      <td>86.87400</td>
      <td>73092782</td>
    </tr>
    <tr>
      <th>9</th>
      <td>SNP_10</td>
      <td>BTB-01558306</td>
      <td>7</td>
      <td>62.06920</td>
      <td>40827112</td>
    </tr>
    <tr>
      <th>10</th>
      <td>SNP_11</td>
      <td>ARS-BFGL-NGS-44247</td>
      <td>8</td>
      <td>97.17310</td>
      <td>92485682</td>
    </tr>
    <tr>
      <th>11</th>
      <td>SNP_12</td>
      <td>Hapmap32827-BTA-146530</td>
      <td>9</td>
      <td>62.74630</td>
      <td>55007839</td>
    </tr>
    <tr>
      <th>12</th>
      <td>SNP_13</td>
      <td>BTB-00395482</td>
      <td>9</td>
      <td>63.41810</td>
      <td>59692848</td>
    </tr>
    <tr>
      <th>13</th>
      <td>SNP_14</td>
      <td>Hapmap40256-BTA-84189</td>
      <td>9</td>
      <td>66.81970</td>
      <td>72822507</td>
    </tr>
    <tr>
      <th>14</th>
      <td>SNP_15</td>
      <td>BovineHD1000000224</td>
      <td>10</td>
      <td>1.78774</td>
      <td>814291</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-3a2dac6e-15c5-4e43-a0bb-5f69ddbce63f')"
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
          document.querySelector('#df-3a2dac6e-15c5-4e43-a0bb-5f69ddbce63f button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-3a2dac6e-15c5-4e43-a0bb-5f69ddbce63f');
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
train
```





  <div id="df-8edf21f2-0fa3-463f-8a0b-1ce72e6f6750">
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
      <th>id</th>
      <th>father</th>
      <th>mother</th>
      <th>gender</th>
      <th>trait</th>
      <th>SNP_01</th>
      <th>SNP_02</th>
      <th>SNP_03</th>
      <th>SNP_04</th>
      <th>SNP_05</th>
      <th>...</th>
      <th>SNP_07</th>
      <th>SNP_08</th>
      <th>SNP_09</th>
      <th>SNP_10</th>
      <th>SNP_11</th>
      <th>SNP_12</th>
      <th>SNP_13</th>
      <th>SNP_14</th>
      <th>SNP_15</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>TRAIN_000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>G G</td>
      <td>A G</td>
      <td>A A</td>
      <td>G A</td>
      <td>C A</td>
      <td>...</td>
      <td>A A</td>
      <td>G G</td>
      <td>A A</td>
      <td>G G</td>
      <td>A G</td>
      <td>A A</td>
      <td>A A</td>
      <td>A A</td>
      <td>A A</td>
      <td>B</td>
    </tr>
    <tr>
      <th>1</th>
      <td>TRAIN_001</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>A G</td>
      <td>A G</td>
      <td>C A</td>
      <td>A A</td>
      <td>A A</td>
      <td>...</td>
      <td>A A</td>
      <td>G A</td>
      <td>A A</td>
      <td>A G</td>
      <td>A A</td>
      <td>G A</td>
      <td>G G</td>
      <td>A A</td>
      <td>A A</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>TRAIN_002</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>G G</td>
      <td>G G</td>
      <td>A A</td>
      <td>G A</td>
      <td>C C</td>
      <td>...</td>
      <td>A A</td>
      <td>G A</td>
      <td>G A</td>
      <td>A G</td>
      <td>A A</td>
      <td>A A</td>
      <td>A A</td>
      <td>A A</td>
      <td>A A</td>
      <td>B</td>
    </tr>
    <tr>
      <th>3</th>
      <td>TRAIN_003</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>A A</td>
      <td>G G</td>
      <td>A A</td>
      <td>G A</td>
      <td>A A</td>
      <td>...</td>
      <td>G G</td>
      <td>A A</td>
      <td>G G</td>
      <td>A G</td>
      <td>G G</td>
      <td>G G</td>
      <td>G G</td>
      <td>A A</td>
      <td>G G</td>
      <td>A</td>
    </tr>
    <tr>
      <th>4</th>
      <td>TRAIN_004</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>G G</td>
      <td>G G</td>
      <td>C C</td>
      <td>A A</td>
      <td>C C</td>
      <td>...</td>
      <td>A A</td>
      <td>A A</td>
      <td>A A</td>
      <td>G G</td>
      <td>A A</td>
      <td>A A</td>
      <td>A G</td>
      <td>A A</td>
      <td>G A</td>
      <td>C</td>
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
      <th>257</th>
      <td>TRAIN_257</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>A G</td>
      <td>A G</td>
      <td>A A</td>
      <td>G A</td>
      <td>C C</td>
      <td>...</td>
      <td>A A</td>
      <td>G A</td>
      <td>A A</td>
      <td>G G</td>
      <td>A G</td>
      <td>G A</td>
      <td>A A</td>
      <td>A A</td>
      <td>A A</td>
      <td>B</td>
    </tr>
    <tr>
      <th>258</th>
      <td>TRAIN_258</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>G G</td>
      <td>A A</td>
      <td>C A</td>
      <td>A A</td>
      <td>A A</td>
      <td>...</td>
      <td>G A</td>
      <td>G A</td>
      <td>A A</td>
      <td>A G</td>
      <td>A G</td>
      <td>A A</td>
      <td>A G</td>
      <td>A A</td>
      <td>G A</td>
      <td>C</td>
    </tr>
    <tr>
      <th>259</th>
      <td>TRAIN_259</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>A G</td>
      <td>G G</td>
      <td>A A</td>
      <td>G A</td>
      <td>A A</td>
      <td>...</td>
      <td>G G</td>
      <td>G A</td>
      <td>G A</td>
      <td>A A</td>
      <td>G G</td>
      <td>G G</td>
      <td>G G</td>
      <td>C A</td>
      <td>G G</td>
      <td>A</td>
    </tr>
    <tr>
      <th>260</th>
      <td>TRAIN_260</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>A A</td>
      <td>G G</td>
      <td>A A</td>
      <td>G A</td>
      <td>A A</td>
      <td>...</td>
      <td>G G</td>
      <td>A A</td>
      <td>G A</td>
      <td>A G</td>
      <td>A G</td>
      <td>G A</td>
      <td>G G</td>
      <td>C A</td>
      <td>G G</td>
      <td>A</td>
    </tr>
    <tr>
      <th>261</th>
      <td>TRAIN_261</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>G G</td>
      <td>A G</td>
      <td>C A</td>
      <td>G G</td>
      <td>C C</td>
      <td>...</td>
      <td>A A</td>
      <td>A A</td>
      <td>A A</td>
      <td>G G</td>
      <td>A A</td>
      <td>A A</td>
      <td>G G</td>
      <td>A A</td>
      <td>G A</td>
      <td>B</td>
    </tr>
  </tbody>
</table>
<p>262 rows × 21 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-8edf21f2-0fa3-463f-8a0b-1ce72e6f6750')"
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
          document.querySelector('#df-8edf21f2-0fa3-463f-8a0b-1ce72e6f6750 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-8edf21f2-0fa3-463f-8a0b-1ce72e6f6750');
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
test
```





  <div id="df-c485bfbf-b9b5-4842-84ce-d84d7f8e2d49">
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
      <th>id</th>
      <th>father</th>
      <th>mother</th>
      <th>gender</th>
      <th>trait</th>
      <th>SNP_01</th>
      <th>SNP_02</th>
      <th>SNP_03</th>
      <th>SNP_04</th>
      <th>SNP_05</th>
      <th>SNP_06</th>
      <th>SNP_07</th>
      <th>SNP_08</th>
      <th>SNP_09</th>
      <th>SNP_10</th>
      <th>SNP_11</th>
      <th>SNP_12</th>
      <th>SNP_13</th>
      <th>SNP_14</th>
      <th>SNP_15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>TEST_000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>A G</td>
      <td>G G</td>
      <td>A A</td>
      <td>G A</td>
      <td>A A</td>
      <td>A G</td>
      <td>G G</td>
      <td>G A</td>
      <td>G A</td>
      <td>A G</td>
      <td>A G</td>
      <td>G A</td>
      <td>G G</td>
      <td>C A</td>
      <td>G A</td>
    </tr>
    <tr>
      <th>1</th>
      <td>TEST_001</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>G G</td>
      <td>A G</td>
      <td>C C</td>
      <td>G G</td>
      <td>C C</td>
      <td>A A</td>
      <td>A A</td>
      <td>A A</td>
      <td>A A</td>
      <td>G G</td>
      <td>A G</td>
      <td>A A</td>
      <td>A A</td>
      <td>A A</td>
      <td>A A</td>
    </tr>
    <tr>
      <th>2</th>
      <td>TEST_002</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>G G</td>
      <td>A G</td>
      <td>A A</td>
      <td>A A</td>
      <td>C A</td>
      <td>A G</td>
      <td>A A</td>
      <td>A A</td>
      <td>A A</td>
      <td>A G</td>
      <td>A A</td>
      <td>G A</td>
      <td>G G</td>
      <td>A A</td>
      <td>G G</td>
    </tr>
    <tr>
      <th>3</th>
      <td>TEST_003</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>G G</td>
      <td>A G</td>
      <td>C A</td>
      <td>A A</td>
      <td>C C</td>
      <td>A A</td>
      <td>A A</td>
      <td>A A</td>
      <td>A A</td>
      <td>G G</td>
      <td>A A</td>
      <td>G A</td>
      <td>A G</td>
      <td>A A</td>
      <td>A A</td>
    </tr>
    <tr>
      <th>4</th>
      <td>TEST_004</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>A A</td>
      <td>G G</td>
      <td>A A</td>
      <td>G G</td>
      <td>A A</td>
      <td>G G</td>
      <td>G G</td>
      <td>A A</td>
      <td>G G</td>
      <td>A G</td>
      <td>G G</td>
      <td>G A</td>
      <td>G G</td>
      <td>A A</td>
      <td>G G</td>
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
    </tr>
    <tr>
      <th>170</th>
      <td>TEST_170</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>A G</td>
      <td>G G</td>
      <td>C C</td>
      <td>A A</td>
      <td>C A</td>
      <td>A G</td>
      <td>A A</td>
      <td>G G</td>
      <td>A A</td>
      <td>G G</td>
      <td>G G</td>
      <td>A A</td>
      <td>A A</td>
      <td>A A</td>
      <td>G A</td>
    </tr>
    <tr>
      <th>171</th>
      <td>TEST_171</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>G G</td>
      <td>A A</td>
      <td>A A</td>
      <td>A A</td>
      <td>C A</td>
      <td>A G</td>
      <td>A A</td>
      <td>A A</td>
      <td>A A</td>
      <td>A G</td>
      <td>A A</td>
      <td>A A</td>
      <td>A G</td>
      <td>A A</td>
      <td>G A</td>
    </tr>
    <tr>
      <th>172</th>
      <td>TEST_172</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>G G</td>
      <td>A A</td>
      <td>A A</td>
      <td>A A</td>
      <td>C A</td>
      <td>A G</td>
      <td>A A</td>
      <td>A A</td>
      <td>A A</td>
      <td>G G</td>
      <td>A G</td>
      <td>A A</td>
      <td>A G</td>
      <td>A A</td>
      <td>G G</td>
    </tr>
    <tr>
      <th>173</th>
      <td>TEST_173</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>A G</td>
      <td>G G</td>
      <td>C A</td>
      <td>G A</td>
      <td>C C</td>
      <td>G G</td>
      <td>A A</td>
      <td>G A</td>
      <td>A A</td>
      <td>G G</td>
      <td>A G</td>
      <td>A A</td>
      <td>A A</td>
      <td>A A</td>
      <td>A A</td>
    </tr>
    <tr>
      <th>174</th>
      <td>TEST_174</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>G G</td>
      <td>G G</td>
      <td>C C</td>
      <td>G A</td>
      <td>C A</td>
      <td>A A</td>
      <td>G A</td>
      <td>G G</td>
      <td>A A</td>
      <td>G G</td>
      <td>G G</td>
      <td>A A</td>
      <td>A A</td>
      <td>A A</td>
      <td>A A</td>
    </tr>
  </tbody>
</table>
<p>175 rows × 20 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-c485bfbf-b9b5-4842-84ce-d84d7f8e2d49')"
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
          document.querySelector('#df-c485bfbf-b9b5-4842-84ce-d84d7f8e2d49 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-c485bfbf-b9b5-4842-84ce-d84d7f8e2d49');
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
print(f"train : {train.shape}")
print(f"test : {test.shape}")
print(f"submission : {sample_submission.shape}")
```

    train : (262, 21)
    test : (175, 20)
    submission : (175, 2)
    


```python
snp_info.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 15 entries, 0 to 14
    Data columns (total 5 columns):
     #   Column  Non-Null Count  Dtype  
    ---  ------  --------------  -----  
     0   SNP_id  15 non-null     object 
     1   name    15 non-null     object 
     2   chrom   15 non-null     int64  
     3   cm      15 non-null     float64
     4   pos     15 non-null     int64  
    dtypes: float64(1), int64(2), object(2)
    memory usage: 728.0+ bytes
    


```python
train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 262 entries, 0 to 261
    Data columns (total 21 columns):
     #   Column  Non-Null Count  Dtype 
    ---  ------  --------------  ----- 
     0   id      262 non-null    object
     1   father  262 non-null    int64 
     2   mother  262 non-null    int64 
     3   gender  262 non-null    int64 
     4   trait   262 non-null    int64 
     5   SNP_01  262 non-null    object
     6   SNP_02  262 non-null    object
     7   SNP_03  262 non-null    object
     8   SNP_04  262 non-null    object
     9   SNP_05  262 non-null    object
     10  SNP_06  262 non-null    object
     11  SNP_07  262 non-null    object
     12  SNP_08  262 non-null    object
     13  SNP_09  262 non-null    object
     14  SNP_10  262 non-null    object
     15  SNP_11  262 non-null    object
     16  SNP_12  262 non-null    object
     17  SNP_13  262 non-null    object
     18  SNP_14  262 non-null    object
     19  SNP_15  262 non-null    object
     20  class   262 non-null    object
    dtypes: int64(4), object(17)
    memory usage: 43.1+ KB
    

### SNP_info 분석


```python
snp_info
```





  <div id="df-9cd1a77a-5726-496a-ac9a-f4a92d3e66f2">
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
      <th>SNP_id</th>
      <th>name</th>
      <th>chrom</th>
      <th>cm</th>
      <th>pos</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>SNP_01</td>
      <td>BTA-19852-no-rs</td>
      <td>2</td>
      <td>67.05460</td>
      <td>42986890</td>
    </tr>
    <tr>
      <th>1</th>
      <td>SNP_02</td>
      <td>ARS-USMARC-Parent-DQ647190-rs29013632</td>
      <td>6</td>
      <td>31.15670</td>
      <td>13897068</td>
    </tr>
    <tr>
      <th>2</th>
      <td>SNP_03</td>
      <td>ARS-BFGL-NGS-117009</td>
      <td>6</td>
      <td>68.28920</td>
      <td>44649549</td>
    </tr>
    <tr>
      <th>3</th>
      <td>SNP_04</td>
      <td>ARS-BFGL-NGS-60567</td>
      <td>6</td>
      <td>77.87490</td>
      <td>53826064</td>
    </tr>
    <tr>
      <th>4</th>
      <td>SNP_05</td>
      <td>BovineHD0600017032</td>
      <td>6</td>
      <td>80.50150</td>
      <td>61779512</td>
    </tr>
    <tr>
      <th>5</th>
      <td>SNP_06</td>
      <td>BovineHD0600017424</td>
      <td>6</td>
      <td>80.59540</td>
      <td>63048481</td>
    </tr>
    <tr>
      <th>6</th>
      <td>SNP_07</td>
      <td>Hapmap49442-BTA-111073</td>
      <td>6</td>
      <td>80.78000</td>
      <td>64037334</td>
    </tr>
    <tr>
      <th>7</th>
      <td>SNP_08</td>
      <td>BovineHD0600018638</td>
      <td>6</td>
      <td>82.68560</td>
      <td>67510588</td>
    </tr>
    <tr>
      <th>8</th>
      <td>SNP_09</td>
      <td>ARS-BFGL-NGS-37727</td>
      <td>6</td>
      <td>86.87400</td>
      <td>73092782</td>
    </tr>
    <tr>
      <th>9</th>
      <td>SNP_10</td>
      <td>BTB-01558306</td>
      <td>7</td>
      <td>62.06920</td>
      <td>40827112</td>
    </tr>
    <tr>
      <th>10</th>
      <td>SNP_11</td>
      <td>ARS-BFGL-NGS-44247</td>
      <td>8</td>
      <td>97.17310</td>
      <td>92485682</td>
    </tr>
    <tr>
      <th>11</th>
      <td>SNP_12</td>
      <td>Hapmap32827-BTA-146530</td>
      <td>9</td>
      <td>62.74630</td>
      <td>55007839</td>
    </tr>
    <tr>
      <th>12</th>
      <td>SNP_13</td>
      <td>BTB-00395482</td>
      <td>9</td>
      <td>63.41810</td>
      <td>59692848</td>
    </tr>
    <tr>
      <th>13</th>
      <td>SNP_14</td>
      <td>Hapmap40256-BTA-84189</td>
      <td>9</td>
      <td>66.81970</td>
      <td>72822507</td>
    </tr>
    <tr>
      <th>14</th>
      <td>SNP_15</td>
      <td>BovineHD1000000224</td>
      <td>10</td>
      <td>1.78774</td>
      <td>814291</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-9cd1a77a-5726-496a-ac9a-f4a92d3e66f2')"
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
          document.querySelector('#df-9cd1a77a-5726-496a-ac9a-f4a92d3e66f2 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-9cd1a77a-5726-496a-ac9a-f4a92d3e66f2');
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




#### SNP chrom별로 분류해 데이터 살펴보기


```python
plt.figure(figsize=(10,6))
sns.countplot(x='chrom', data=snp_info)
plt.title('chrom 종류별 개수')
```




    Text(0.5, 1.0, 'chrom 종류별 개수')




    
![output_15_1](https://user-images.githubusercontent.com/113446739/216803658-771e0fd9-c690-4f7a-a260-58474a4ee958.png)
    



```python
plt.figure(figsize=(10,6))
sns.barplot(x='SNP_id', y='cm', data=snp_info)
plt.title('유전거리 분포')
```




    Text(0.5, 1.0, '유전거리 분포')




    
![output_16_1](https://user-images.githubusercontent.com/113446739/216803659-fa60fd6c-d9eb-4b76-a97d-8c14b74462b3.png)
    



```python
plt.figure(figsize=(10,6))
sns.barplot(x='SNP_id', y='pos', data=snp_info)
plt.title('각 마커 유전체상 위치정보')
```




    Text(0.5, 1.0, '각 마커 유전체상 위치정보')




    
![output_17_1](https://user-images.githubusercontent.com/113446739/216803660-93ce4c33-c9e6-43df-85af-1ed6f6703ebb.png)
    



```python
snp_6=snp_info[snp_info['chrom']==6]
snp_6
```





  <div id="df-2d19ed8b-b55d-4938-956e-857af2d190dd">
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
      <th>SNP_id</th>
      <th>name</th>
      <th>chrom</th>
      <th>cm</th>
      <th>pos</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>SNP_02</td>
      <td>ARS-USMARC-Parent-DQ647190-rs29013632</td>
      <td>6</td>
      <td>31.1567</td>
      <td>13897068</td>
    </tr>
    <tr>
      <th>2</th>
      <td>SNP_03</td>
      <td>ARS-BFGL-NGS-117009</td>
      <td>6</td>
      <td>68.2892</td>
      <td>44649549</td>
    </tr>
    <tr>
      <th>3</th>
      <td>SNP_04</td>
      <td>ARS-BFGL-NGS-60567</td>
      <td>6</td>
      <td>77.8749</td>
      <td>53826064</td>
    </tr>
    <tr>
      <th>4</th>
      <td>SNP_05</td>
      <td>BovineHD0600017032</td>
      <td>6</td>
      <td>80.5015</td>
      <td>61779512</td>
    </tr>
    <tr>
      <th>5</th>
      <td>SNP_06</td>
      <td>BovineHD0600017424</td>
      <td>6</td>
      <td>80.5954</td>
      <td>63048481</td>
    </tr>
    <tr>
      <th>6</th>
      <td>SNP_07</td>
      <td>Hapmap49442-BTA-111073</td>
      <td>6</td>
      <td>80.7800</td>
      <td>64037334</td>
    </tr>
    <tr>
      <th>7</th>
      <td>SNP_08</td>
      <td>BovineHD0600018638</td>
      <td>6</td>
      <td>82.6856</td>
      <td>67510588</td>
    </tr>
    <tr>
      <th>8</th>
      <td>SNP_09</td>
      <td>ARS-BFGL-NGS-37727</td>
      <td>6</td>
      <td>86.8740</td>
      <td>73092782</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-2d19ed8b-b55d-4938-956e-857af2d190dd')"
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
          document.querySelector('#df-2d19ed8b-b55d-4938-956e-857af2d190dd button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-2d19ed8b-b55d-4938-956e-857af2d190dd');
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
snp_9=snp_info[snp_info['chrom']==9]
snp_9
```





  <div id="df-ee54ca3f-8c25-4ee4-9d0a-6965ed5c7e8a">
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
      <th>SNP_id</th>
      <th>name</th>
      <th>chrom</th>
      <th>cm</th>
      <th>pos</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>11</th>
      <td>SNP_12</td>
      <td>Hapmap32827-BTA-146530</td>
      <td>9</td>
      <td>62.7463</td>
      <td>55007839</td>
    </tr>
    <tr>
      <th>12</th>
      <td>SNP_13</td>
      <td>BTB-00395482</td>
      <td>9</td>
      <td>63.4181</td>
      <td>59692848</td>
    </tr>
    <tr>
      <th>13</th>
      <td>SNP_14</td>
      <td>Hapmap40256-BTA-84189</td>
      <td>9</td>
      <td>66.8197</td>
      <td>72822507</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-ee54ca3f-8c25-4ee4-9d0a-6965ed5c7e8a')"
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
          document.querySelector('#df-ee54ca3f-8c25-4ee4-9d0a-6965ed5c7e8a button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-ee54ca3f-8c25-4ee4-9d0a-6965ed5c7e8a');
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
plt.figure(figsize=(10,6))
sns.barplot(x='SNP_id', y='cm', data=snp_6)
plt.title('chrom=6 인 SNP의 유전거리')
```




    Text(0.5, 1.0, 'chrom=6 인 SNP의 유전거리')




    
![output_20_1](https://user-images.githubusercontent.com/113446739/216803662-9370cf58-f749-4b17-bd60-392a30af4b6d.png)
    



```python
plt.figure(figsize=(10,6))
sns.barplot(x='SNP_id', y='pos', data=snp_6)
plt.title('chrom=6 인 SNP의 유전체상 위치정보')
```




    Text(0.5, 1.0, 'chrom=6 인 SNP의 유전체상 위치정보')




    
![output_21_1](https://user-images.githubusercontent.com/113446739/216803664-ac2a17bd-e0cd-4222-8d27-c1d2d09e41d8.png)
    



```python
plt.figure(figsize=(10,6))
sns.barplot(x='SNP_id', y='cm', data=snp_9)
plt.title('chrom=9 인 SNP의 유전거리')
```




    Text(0.5, 1.0, 'chrom=9 인 SNP의 유전거리')




    
![output_22_1](https://user-images.githubusercontent.com/113446739/216803665-8ad2eb7d-991e-4927-b3e4-97fbdff48e89.png)
    



```python
plt.figure(figsize=(10,6))
sns.barplot(x='SNP_id', y='pos', data=snp_9)
plt.title('chrom=9 인 SNP의 유전체상 위치정보')
```




    Text(0.5, 1.0, 'chrom=9 인 SNP의 유전체상 위치정보')




    
![output_23_1](https://user-images.githubusercontent.com/113446739/216803666-1aab23d4-f296-4337-ac67-b9a559896c18.png)
    


### 숫자형으로 변경


```python
def get_x_y(df):
    if 'class' in df.columns:
        df_x = df.drop(columns=['id', 'class'])
        df_y = df['class']
        return df_x, df_y
    else:
        df_x = df.drop(columns=['id'])
        return df_x
```


```python
train_x, train_y = get_x_y(train)
test_x = get_x_y(test)
```


```python
class_le = preprocessing.LabelEncoder()
snp_le = preprocessing.LabelEncoder()
snp_col = [f'SNP_{str(x).zfill(2)}' for x in range(1,16)]
```


```python
snp_data = []
for col in snp_col:
    snp_data += list(train_x[col].values)
```


```python
train_y = class_le.fit_transform(train_y)
snp_le.fit(snp_data)
```




    LabelEncoder()




```python
for col in train_x.columns:
    if col in snp_col:
        train_x[col] = snp_le.transform(train_x[col])
        test_x[col] = snp_le.transform(test_x[col])
```


```python
train_x
```





  <div id="df-86a98125-e518-484c-a263-a18103ca0e73">
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
      <th>father</th>
      <th>mother</th>
      <th>gender</th>
      <th>trait</th>
      <th>SNP_01</th>
      <th>SNP_02</th>
      <th>SNP_03</th>
      <th>SNP_04</th>
      <th>SNP_05</th>
      <th>SNP_06</th>
      <th>SNP_07</th>
      <th>SNP_08</th>
      <th>SNP_09</th>
      <th>SNP_10</th>
      <th>SNP_11</th>
      <th>SNP_12</th>
      <th>SNP_13</th>
      <th>SNP_14</th>
      <th>SNP_15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>5</td>
      <td>5</td>
      <td>0</td>
      <td>4</td>
      <td>3</td>
      <td>5</td>
      <td>0</td>
      <td>4</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>5</td>
      <td>5</td>
      <td>0</td>
      <td>5</td>
      <td>1</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>5</td>
      <td>5</td>
      <td>3</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
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
    </tr>
    <tr>
      <th>257</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>5</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>258</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>5</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>259</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>5</td>
      <td>4</td>
      <td>4</td>
      <td>0</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>2</td>
      <td>5</td>
    </tr>
    <tr>
      <th>260</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>5</td>
      <td>5</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>5</td>
      <td>2</td>
      <td>5</td>
    </tr>
    <tr>
      <th>261</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>5</td>
      <td>1</td>
      <td>2</td>
      <td>5</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
<p>262 rows × 19 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-86a98125-e518-484c-a263-a18103ca0e73')"
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
          document.querySelector('#df-86a98125-e518-484c-a263-a18103ca0e73 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-86a98125-e518-484c-a263-a18103ca0e73');
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
snp_le.classes_
```




    array(['A A', 'A G', 'C A', 'C C', 'G A', 'G G'], dtype='<U3')




```python
trait_1=train_x[train_x['trait']==1]
trait_2=train_x[train_x['trait']==2]
```


```python
trait_1
```





  <div id="df-32b1d3f8-315a-4438-aae9-92ee711b28f1">
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
      <th>father</th>
      <th>mother</th>
      <th>gender</th>
      <th>trait</th>
      <th>SNP_01</th>
      <th>SNP_02</th>
      <th>SNP_03</th>
      <th>SNP_04</th>
      <th>SNP_05</th>
      <th>SNP_06</th>
      <th>SNP_07</th>
      <th>SNP_08</th>
      <th>SNP_09</th>
      <th>SNP_10</th>
      <th>SNP_11</th>
      <th>SNP_12</th>
      <th>SNP_13</th>
      <th>SNP_14</th>
      <th>SNP_15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>5</td>
      <td>5</td>
      <td>0</td>
      <td>5</td>
      <td>1</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>5</td>
      <td>5</td>
      <td>0</td>
      <td>5</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>5</td>
      <td>3</td>
      <td>5</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>5</td>
      <td>5</td>
      <td>0</td>
      <td>4</td>
      <td>5</td>
      <td>1</td>
      <td>5</td>
      <td>5</td>
      <td>2</td>
      <td>5</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>5</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>5</td>
      <td>4</td>
      <td>5</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>5</td>
      <td>5</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>2</td>
      <td>5</td>
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
    </tr>
    <tr>
      <th>248</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>5</td>
      <td>5</td>
      <td>0</td>
      <td>5</td>
      <td>1</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>2</td>
      <td>4</td>
    </tr>
    <tr>
      <th>250</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>5</td>
      <td>5</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>253</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>5</td>
      <td>5</td>
      <td>4</td>
      <td>4</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>259</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>5</td>
      <td>4</td>
      <td>4</td>
      <td>0</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>2</td>
      <td>5</td>
    </tr>
    <tr>
      <th>260</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>5</td>
      <td>5</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>5</td>
      <td>2</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
<p>69 rows × 19 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-32b1d3f8-315a-4438-aae9-92ee711b28f1')"
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
          document.querySelector('#df-32b1d3f8-315a-4438-aae9-92ee711b28f1 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-32b1d3f8-315a-4438-aae9-92ee711b28f1');
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
trait_2
```





  <div id="df-56a04ef5-5ca6-405a-ad31-1d7d07376226">
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
      <th>father</th>
      <th>mother</th>
      <th>gender</th>
      <th>trait</th>
      <th>SNP_01</th>
      <th>SNP_02</th>
      <th>SNP_03</th>
      <th>SNP_04</th>
      <th>SNP_05</th>
      <th>SNP_06</th>
      <th>SNP_07</th>
      <th>SNP_08</th>
      <th>SNP_09</th>
      <th>SNP_10</th>
      <th>SNP_11</th>
      <th>SNP_12</th>
      <th>SNP_13</th>
      <th>SNP_14</th>
      <th>SNP_15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>5</td>
      <td>5</td>
      <td>0</td>
      <td>4</td>
      <td>3</td>
      <td>5</td>
      <td>0</td>
      <td>4</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>5</td>
      <td>5</td>
      <td>3</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>5</td>
      <td>5</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
    </tr>
    <tr>
      <th>255</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>5</td>
      <td>2</td>
      <td>4</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
      <td>5</td>
      <td>0</td>
      <td>5</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>256</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>5</td>
      <td>5</td>
      <td>2</td>
      <td>4</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>5</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>257</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>5</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>258</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>5</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>261</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>5</td>
      <td>1</td>
      <td>2</td>
      <td>5</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
<p>193 rows × 19 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-56a04ef5-5ca6-405a-ad31-1d7d07376226')"
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
          document.querySelector('#df-56a04ef5-5ca6-405a-ad31-1d7d07376226 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-56a04ef5-5ca6-405a-ad31-1d7d07376226');
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




#### Trait 종류에 따른 데이터 형태


```python
for i in trait_1.columns[4:]:
    print(trait_1[i].unique())
```

    [0 1 5]
    [5 1]
    [0]
    [4 5 0]
    [0 2]
    [5 1]
    [5 4]
    [0 4]
    [5 4 0]
    [1 5 0]
    [5 1]
    [5 4 0]
    [5 1]
    [0 3 2]
    [5 4 0]
    


```python
for i in trait_2.columns[4:]:
    print(trait_2[i].unique())
```

    [5 1 0]
    [1 5 0]
    [0 2 3]
    [4 0 5]
    [2 0 3]
    [0 1 5]
    [0 4]
    [5 4 0]
    [0 4 5]
    [5 1 0]
    [1 0 5]
    [0 4 5]
    [0 5 1]
    [0 2 3]
    [0 4 5]
    

- trait에 따른 데이터 차이 없음 파악


```python
train_num=train_x.drop(columns=['father', 'mother', 'gender'])
train_num
```





  <div id="df-aec7039a-7b9f-49f7-9b3d-f434bfa9ce52">
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
      <th>trait</th>
      <th>SNP_01</th>
      <th>SNP_02</th>
      <th>SNP_03</th>
      <th>SNP_04</th>
      <th>SNP_05</th>
      <th>SNP_06</th>
      <th>SNP_07</th>
      <th>SNP_08</th>
      <th>SNP_09</th>
      <th>SNP_10</th>
      <th>SNP_11</th>
      <th>SNP_12</th>
      <th>SNP_13</th>
      <th>SNP_14</th>
      <th>SNP_15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>5</td>
      <td>5</td>
      <td>0</td>
      <td>4</td>
      <td>3</td>
      <td>5</td>
      <td>0</td>
      <td>4</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>5</td>
      <td>5</td>
      <td>0</td>
      <td>5</td>
      <td>1</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>5</td>
      <td>5</td>
      <td>3</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
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
    </tr>
    <tr>
      <th>257</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>5</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>258</th>
      <td>2</td>
      <td>5</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>259</th>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>5</td>
      <td>4</td>
      <td>4</td>
      <td>0</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>2</td>
      <td>5</td>
    </tr>
    <tr>
      <th>260</th>
      <td>1</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>5</td>
      <td>5</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>5</td>
      <td>2</td>
      <td>5</td>
    </tr>
    <tr>
      <th>261</th>
      <td>2</td>
      <td>5</td>
      <td>1</td>
      <td>2</td>
      <td>5</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
<p>262 rows × 16 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-aec7039a-7b9f-49f7-9b3d-f434bfa9ce52')"
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
          document.querySelector('#df-aec7039a-7b9f-49f7-9b3d-f434bfa9ce52 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-aec7039a-7b9f-49f7-9b3d-f434bfa9ce52');
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
colormap = plt.cm.PuBu
plt.figure(figsize=(15,15), dpi=200)

sns.heatmap(train_num.astype(float).corr(), linewidths = 0.1, vmax = 1.0,
           square = True, cmap = colormap, linecolor = "white", annot = True, annot_kws = {"size" : 5})
```

    /usr/local/lib/python3.8/dist-packages/matplotlib/backends/backend_agg.py:214: RuntimeWarning: Glyph 8722 missing from current font.
      font.set_text(s, 0.0, flags=flags)
    /usr/local/lib/python3.8/dist-packages/matplotlib/backends/backend_agg.py:183: RuntimeWarning: Glyph 8722 missing from current font.
      font.set_text(s, 0, flags=flags)
    




    <matplotlib.axes._subplots.AxesSubplot at 0x7f3b4dd43130>




    
![output_41_2](https://user-images.githubusercontent.com/113446739/216803667-83f20750-6e0e-4479-abc0-ed22501cbbf3.png)
    


### 모델 돌려보기


```python
model_y=pd.DataFrame(train_y, columns=['class'])
model_y
```





  <div id="df-b18364af-7d33-4db5-9b4c-46265b90067e">
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
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>257</th>
      <td>1</td>
    </tr>
    <tr>
      <th>258</th>
      <td>2</td>
    </tr>
    <tr>
      <th>259</th>
      <td>0</td>
    </tr>
    <tr>
      <th>260</th>
      <td>0</td>
    </tr>
    <tr>
      <th>261</th>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>262 rows × 1 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-b18364af-7d33-4db5-9b4c-46265b90067e')"
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
          document.querySelector('#df-b18364af-7d33-4db5-9b4c-46265b90067e button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-b18364af-7d33-4db5-9b4c-46265b90067e');
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
model_X=train_x.drop(columns=['father','mother','gender','SNP_01','SNP_02','SNP_03','SNP_09','SNP_10', 'SNP_11', 'SNP_12', 'SNP_13','SNP_14', 'SNP_15'])
model_X
```





  <div id="df-da793b6e-45e7-4133-85b9-8d4c286c9f30">
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
      <th>trait</th>
      <th>SNP_04</th>
      <th>SNP_05</th>
      <th>SNP_06</th>
      <th>SNP_07</th>
      <th>SNP_08</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>4</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>4</td>
      <td>3</td>
      <td>5</td>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>5</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
      <th>257</th>
      <td>2</td>
      <td>4</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>258</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>259</th>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>5</td>
      <td>4</td>
    </tr>
    <tr>
      <th>260</th>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>5</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>261</th>
      <td>2</td>
      <td>5</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>262 rows × 6 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-da793b6e-45e7-4133-85b9-8d4c286c9f30')"
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
          document.querySelector('#df-da793b6e-45e7-4133-85b9-8d4c286c9f30 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-da793b6e-45e7-4133-85b9-8d4c286c9f30');
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




### Automl 사용해 제출해보기


```python
test_x=test_x.drop(columns=['father','mother','gender','SNP_01','SNP_02','SNP_03','SNP_09','SNP_10', 'SNP_11', 'SNP_12', 'SNP_13','SNP_14', 'SNP_15'])
```


```python
test_x
```





  <div id="df-b3b7d717-ab56-4b97-81e3-6993234d6eb0">
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
      <th>trait</th>
      <th>SNP_04</th>
      <th>SNP_05</th>
      <th>SNP_06</th>
      <th>SNP_07</th>
      <th>SNP_08</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>5</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>5</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>5</td>
      <td>0</td>
      <td>5</td>
      <td>5</td>
      <td>0</td>
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
      <th>170</th>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>171</th>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>172</th>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>173</th>
      <td>2</td>
      <td>4</td>
      <td>3</td>
      <td>5</td>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>174</th>
      <td>2</td>
      <td>4</td>
      <td>2</td>
      <td>0</td>
      <td>4</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
<p>175 rows × 6 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-b3b7d717-ab56-4b97-81e3-6993234d6eb0')"
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
          document.querySelector('#df-b3b7d717-ab56-4b97-81e3-6993234d6eb0 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-b3b7d717-ab56-4b97-81e3-6993234d6eb0');
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
model_y
```





  <div id="df-0faf5085-4ad3-41ee-aeef-3686efaeb735">
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
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>257</th>
      <td>1</td>
    </tr>
    <tr>
      <th>258</th>
      <td>2</td>
    </tr>
    <tr>
      <th>259</th>
      <td>0</td>
    </tr>
    <tr>
      <th>260</th>
      <td>0</td>
    </tr>
    <tr>
      <th>261</th>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>262 rows × 1 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-0faf5085-4ad3-41ee-aeef-3686efaeb735')"
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
          document.querySelector('#df-0faf5085-4ad3-41ee-aeef-3686efaeb735 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-0faf5085-4ad3-41ee-aeef-3686efaeb735');
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
model_X
```





  <div id="df-5c8e6708-8628-488c-b777-756d407198aa">
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
      <th>trait</th>
      <th>SNP_04</th>
      <th>SNP_05</th>
      <th>SNP_06</th>
      <th>SNP_07</th>
      <th>SNP_08</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>4</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>4</td>
      <td>3</td>
      <td>5</td>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>5</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
      <th>257</th>
      <td>2</td>
      <td>4</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>258</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>259</th>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>5</td>
      <td>4</td>
    </tr>
    <tr>
      <th>260</th>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>5</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>261</th>
      <td>2</td>
      <td>5</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>262 rows × 6 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-5c8e6708-8628-488c-b777-756d407198aa')"
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
          document.querySelector('#df-5c8e6708-8628-488c-b777-756d407198aa button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-5c8e6708-8628-488c-b777-756d407198aa');
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
model_train=pd.concat([model_X, model_y], axis=1)
model_train
```





  <div id="df-072e4463-d22b-464e-aa20-b8865ea3608c">
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
      <th>trait</th>
      <th>SNP_04</th>
      <th>SNP_05</th>
      <th>SNP_06</th>
      <th>SNP_07</th>
      <th>SNP_08</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>4</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>4</td>
      <td>3</td>
      <td>5</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>5</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
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
    </tr>
    <tr>
      <th>257</th>
      <td>2</td>
      <td>4</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>258</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>4</td>
      <td>2</td>
    </tr>
    <tr>
      <th>259</th>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>5</td>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>260</th>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>5</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>261</th>
      <td>2</td>
      <td>5</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>262 rows × 7 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-072e4463-d22b-464e-aa20-b8865ea3608c')"
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
          document.querySelector('#df-072e4463-d22b-464e-aa20-b8865ea3608c button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-072e4463-d22b-464e-aa20-b8865ea3608c');
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
pip install pycaret
```

    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Requirement already satisfied: pycaret in /usr/local/lib/python3.8/dist-packages (2.3.10)
    Requirement already satisfied: lightgbm>=2.3.1 in /usr/local/lib/python3.8/dist-packages (from pycaret) (3.3.4)
    Requirement already satisfied: cufflinks>=0.17.0 in /usr/local/lib/python3.8/dist-packages (from pycaret) (0.17.3)
    Requirement already satisfied: pyod in /usr/local/lib/python3.8/dist-packages (from pycaret) (1.0.7)
    Requirement already satisfied: kmodes>=0.10.1 in /usr/local/lib/python3.8/dist-packages (from pycaret) (0.12.2)
    Requirement already satisfied: pyyaml<6.0.0 in /usr/local/lib/python3.8/dist-packages (from pycaret) (5.4.1)
    Requirement already satisfied: numba<0.55 in /usr/local/lib/python3.8/dist-packages (from pycaret) (0.54.1)
    Requirement already satisfied: scipy<=1.5.4 in /usr/local/lib/python3.8/dist-packages (from pycaret) (1.5.4)
    Requirement already satisfied: mlxtend>=0.17.0 in /usr/local/lib/python3.8/dist-packages (from pycaret) (0.19.0)
    Requirement already satisfied: yellowbrick>=1.0.1 in /usr/local/lib/python3.8/dist-packages (from pycaret) (1.3.post1)
    Requirement already satisfied: pyLDAvis in /usr/local/lib/python3.8/dist-packages (from pycaret) (3.2.2)
    Requirement already satisfied: joblib in /usr/local/lib/python3.8/dist-packages (from pycaret) (1.2.0)
    Requirement already satisfied: gensim<4.0.0 in /usr/local/lib/python3.8/dist-packages (from pycaret) (3.6.0)
    Requirement already satisfied: scikit-learn==0.23.2 in /usr/local/lib/python3.8/dist-packages (from pycaret) (0.23.2)
    Requirement already satisfied: mlflow in /usr/local/lib/python3.8/dist-packages (from pycaret) (2.1.1)
    Requirement already satisfied: IPython in /usr/local/lib/python3.8/dist-packages (from pycaret) (7.9.0)
    Requirement already satisfied: seaborn in /usr/local/lib/python3.8/dist-packages (from pycaret) (0.11.2)
    Requirement already satisfied: nltk in /usr/local/lib/python3.8/dist-packages (from pycaret) (3.7)
    Requirement already satisfied: pandas-profiling>=2.8.0 in /usr/local/lib/python3.8/dist-packages (from pycaret) (3.6.2)
    Requirement already satisfied: scikit-plot in /usr/local/lib/python3.8/dist-packages (from pycaret) (0.3.7)
    Requirement already satisfied: plotly>=4.4.1 in /usr/local/lib/python3.8/dist-packages (from pycaret) (5.5.0)
    Requirement already satisfied: wordcloud in /usr/local/lib/python3.8/dist-packages (from pycaret) (1.8.2.2)
    Requirement already satisfied: ipywidgets in /usr/local/lib/python3.8/dist-packages (from pycaret) (7.7.1)
    Requirement already satisfied: matplotlib in /usr/local/lib/python3.8/dist-packages (from pycaret) (3.2.2)
    Requirement already satisfied: pandas in /usr/local/lib/python3.8/dist-packages (from pycaret) (1.3.5)
    Requirement already satisfied: imbalanced-learn==0.7.0 in /usr/local/lib/python3.8/dist-packages (from pycaret) (0.7.0)
    Requirement already satisfied: Boruta in /usr/local/lib/python3.8/dist-packages (from pycaret) (0.3)
    Requirement already satisfied: umap-learn in /usr/local/lib/python3.8/dist-packages (from pycaret) (0.5.3)
    Requirement already satisfied: textblob in /usr/local/lib/python3.8/dist-packages (from pycaret) (0.15.3)
    Requirement already satisfied: spacy<2.4.0 in /usr/local/lib/python3.8/dist-packages (from pycaret) (2.3.9)
    Requirement already satisfied: numpy>=1.13.3 in /usr/local/lib/python3.8/dist-packages (from imbalanced-learn==0.7.0->pycaret) (1.20.0)
    Requirement already satisfied: threadpoolctl>=2.0.0 in /usr/local/lib/python3.8/dist-packages (from scikit-learn==0.23.2->pycaret) (3.1.0)
    Requirement already satisfied: six>=1.9.0 in /usr/local/lib/python3.8/dist-packages (from cufflinks>=0.17.0->pycaret) (1.15.0)
    Requirement already satisfied: colorlover>=0.2.1 in /usr/local/lib/python3.8/dist-packages (from cufflinks>=0.17.0->pycaret) (0.3.0)
    Requirement already satisfied: setuptools>=34.4.1 in /usr/local/lib/python3.8/dist-packages (from cufflinks>=0.17.0->pycaret) (57.4.0)
    Requirement already satisfied: smart-open>=1.2.1 in /usr/local/lib/python3.8/dist-packages (from gensim<4.0.0->pycaret) (6.3.0)
    Requirement already satisfied: pickleshare in /usr/local/lib/python3.8/dist-packages (from IPython->pycaret) (0.7.5)
    Requirement already satisfied: traitlets>=4.2 in /usr/local/lib/python3.8/dist-packages (from IPython->pycaret) (5.7.1)
    Requirement already satisfied: jedi>=0.10 in /usr/local/lib/python3.8/dist-packages (from IPython->pycaret) (0.18.2)
    Requirement already satisfied: pygments in /usr/local/lib/python3.8/dist-packages (from IPython->pycaret) (2.6.1)
    Requirement already satisfied: pexpect in /usr/local/lib/python3.8/dist-packages (from IPython->pycaret) (4.8.0)
    Requirement already satisfied: decorator in /usr/local/lib/python3.8/dist-packages (from IPython->pycaret) (4.4.2)
    Requirement already satisfied: prompt-toolkit<2.1.0,>=2.0.0 in /usr/local/lib/python3.8/dist-packages (from IPython->pycaret) (2.0.10)
    Requirement already satisfied: backcall in /usr/local/lib/python3.8/dist-packages (from IPython->pycaret) (0.2.0)
    Requirement already satisfied: jupyterlab-widgets>=1.0.0 in /usr/local/lib/python3.8/dist-packages (from ipywidgets->pycaret) (3.0.5)
    Requirement already satisfied: ipykernel>=4.5.1 in /usr/local/lib/python3.8/dist-packages (from ipywidgets->pycaret) (5.3.4)
    Requirement already satisfied: ipython-genutils~=0.2.0 in /usr/local/lib/python3.8/dist-packages (from ipywidgets->pycaret) (0.2.0)
    Requirement already satisfied: widgetsnbextension~=3.6.0 in /usr/local/lib/python3.8/dist-packages (from ipywidgets->pycaret) (3.6.1)
    Requirement already satisfied: wheel in /usr/local/lib/python3.8/dist-packages (from lightgbm>=2.3.1->pycaret) (0.38.4)
    Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.8/dist-packages (from matplotlib->pycaret) (0.11.0)
    Requirement already satisfied: python-dateutil>=2.1 in /usr/local/lib/python3.8/dist-packages (from matplotlib->pycaret) (2.8.2)
    Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.8/dist-packages (from matplotlib->pycaret) (1.4.4)
    Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /usr/local/lib/python3.8/dist-packages (from matplotlib->pycaret) (3.0.9)
    Requirement already satisfied: llvmlite<0.38,>=0.37.0rc1 in /usr/local/lib/python3.8/dist-packages (from numba<0.55->pycaret) (0.37.0)
    Requirement already satisfied: pytz>=2017.3 in /usr/local/lib/python3.8/dist-packages (from pandas->pycaret) (2022.7)
    Requirement already satisfied: multimethod<1.10,>=1.4 in /usr/local/lib/python3.8/dist-packages (from pandas-profiling>=2.8.0->pycaret) (1.9.1)
    Requirement already satisfied: htmlmin==0.1.12 in /usr/local/lib/python3.8/dist-packages (from pandas-profiling>=2.8.0->pycaret) (0.1.12)
    Requirement already satisfied: requests<2.29,>=2.24.0 in /usr/local/lib/python3.8/dist-packages (from pandas-profiling>=2.8.0->pycaret) (2.28.2)
    Requirement already satisfied: jinja2<3.2,>=2.11.1 in /usr/local/lib/python3.8/dist-packages (from pandas-profiling>=2.8.0->pycaret) (2.11.3)
    Requirement already satisfied: statsmodels<0.14,>=0.13.2 in /usr/local/lib/python3.8/dist-packages (from pandas-profiling>=2.8.0->pycaret) (0.13.5)
    Requirement already satisfied: visions[type_image_path]==0.7.5 in /usr/local/lib/python3.8/dist-packages (from pandas-profiling>=2.8.0->pycaret) (0.7.5)
    Requirement already satisfied: typeguard<2.14,>=2.13.2 in /usr/local/lib/python3.8/dist-packages (from pandas-profiling>=2.8.0->pycaret) (2.13.3)
    Requirement already satisfied: phik<0.13,>=0.11.1 in /usr/local/lib/python3.8/dist-packages (from pandas-profiling>=2.8.0->pycaret) (0.12.3)
    Requirement already satisfied: pydantic<1.11,>=1.8.1 in /usr/local/lib/python3.8/dist-packages (from pandas-profiling>=2.8.0->pycaret) (1.10.4)
    Requirement already satisfied: tqdm<4.65,>=4.48.2 in /usr/local/lib/python3.8/dist-packages (from pandas-profiling>=2.8.0->pycaret) (4.64.1)
    Requirement already satisfied: tangled-up-in-unicode>=0.0.4 in /usr/local/lib/python3.8/dist-packages (from visions[type_image_path]==0.7.5->pandas-profiling>=2.8.0->pycaret) (0.2.0)
    Requirement already satisfied: networkx>=2.4 in /usr/local/lib/python3.8/dist-packages (from visions[type_image_path]==0.7.5->pandas-profiling>=2.8.0->pycaret) (2.8.8)
    Requirement already satisfied: attrs>=19.3.0 in /usr/local/lib/python3.8/dist-packages (from visions[type_image_path]==0.7.5->pandas-profiling>=2.8.0->pycaret) (22.2.0)
    Requirement already satisfied: Pillow in /usr/local/lib/python3.8/dist-packages (from visions[type_image_path]==0.7.5->pandas-profiling>=2.8.0->pycaret) (7.1.2)
    Requirement already satisfied: imagehash in /usr/local/lib/python3.8/dist-packages (from visions[type_image_path]==0.7.5->pandas-profiling>=2.8.0->pycaret) (4.3.1)
    Requirement already satisfied: tenacity>=6.2.0 in /usr/local/lib/python3.8/dist-packages (from plotly>=4.4.1->pycaret) (8.1.0)
    Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /usr/local/lib/python3.8/dist-packages (from spacy<2.4.0->pycaret) (1.0.9)
    Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /usr/local/lib/python3.8/dist-packages (from spacy<2.4.0->pycaret) (1.0.6)
    Requirement already satisfied: thinc<7.5.0,>=7.4.1 in /usr/local/lib/python3.8/dist-packages (from spacy<2.4.0->pycaret) (7.4.6)
    Requirement already satisfied: blis<0.8.0,>=0.4.0 in /usr/local/lib/python3.8/dist-packages (from spacy<2.4.0->pycaret) (0.7.9)
    Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /usr/local/lib/python3.8/dist-packages (from spacy<2.4.0->pycaret) (1.0.2)
    Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /usr/local/lib/python3.8/dist-packages (from spacy<2.4.0->pycaret) (2.0.7)
    Requirement already satisfied: plac<1.2.0,>=0.9.6 in /usr/local/lib/python3.8/dist-packages (from spacy<2.4.0->pycaret) (1.1.3)
    Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /usr/local/lib/python3.8/dist-packages (from spacy<2.4.0->pycaret) (3.0.8)
    Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /usr/local/lib/python3.8/dist-packages (from spacy<2.4.0->pycaret) (0.10.1)
    Collecting numpy>=1.13.3
      Using cached numpy-1.19.5-cp38-cp38-manylinux2010_x86_64.whl (14.9 MB)
    Requirement already satisfied: pyarrow<11,>=4.0.0 in /usr/local/lib/python3.8/dist-packages (from mlflow->pycaret) (9.0.0)
    Requirement already satisfied: sqlalchemy<2,>=1.4.0 in /usr/local/lib/python3.8/dist-packages (from mlflow->pycaret) (1.4.46)
    Requirement already satisfied: sqlparse<1,>=0.4.0 in /usr/local/lib/python3.8/dist-packages (from mlflow->pycaret) (0.4.3)
    Requirement already satisfied: packaging<23 in /usr/local/lib/python3.8/dist-packages (from mlflow->pycaret) (21.3)
    Requirement already satisfied: alembic<2 in /usr/local/lib/python3.8/dist-packages (from mlflow->pycaret) (1.9.2)
    Requirement already satisfied: click<9,>=7.0 in /usr/local/lib/python3.8/dist-packages (from mlflow->pycaret) (7.1.2)
    Requirement already satisfied: Flask<3 in /usr/local/lib/python3.8/dist-packages (from mlflow->pycaret) (1.1.4)
    Requirement already satisfied: databricks-cli<1,>=0.8.7 in /usr/local/lib/python3.8/dist-packages (from mlflow->pycaret) (0.17.4)
    Requirement already satisfied: markdown<4,>=3.3 in /usr/local/lib/python3.8/dist-packages (from mlflow->pycaret) (3.4.1)
    Requirement already satisfied: importlib-metadata!=4.7.0,<6,>=3.7.0 in /usr/local/lib/python3.8/dist-packages (from mlflow->pycaret) (5.2.0)
    Requirement already satisfied: protobuf<5,>=3.12.0 in /usr/local/lib/python3.8/dist-packages (from mlflow->pycaret) (3.19.6)
    Requirement already satisfied: shap<1,>=0.40 in /usr/local/lib/python3.8/dist-packages (from mlflow->pycaret) (0.41.0)
    Requirement already satisfied: cloudpickle<3 in /usr/local/lib/python3.8/dist-packages (from mlflow->pycaret) (2.2.0)
    Requirement already satisfied: gunicorn<21 in /usr/local/lib/python3.8/dist-packages (from mlflow->pycaret) (20.1.0)
    Requirement already satisfied: querystring-parser<2 in /usr/local/lib/python3.8/dist-packages (from mlflow->pycaret) (1.2.4)
    Requirement already satisfied: gitpython<4,>=2.1.0 in /usr/local/lib/python3.8/dist-packages (from mlflow->pycaret) (3.1.30)
    Requirement already satisfied: docker<7,>=4.0.0 in /usr/local/lib/python3.8/dist-packages (from mlflow->pycaret) (6.0.1)
    Requirement already satisfied: entrypoints<1 in /usr/local/lib/python3.8/dist-packages (from mlflow->pycaret) (0.4)
    Requirement already satisfied: regex>=2021.8.3 in /usr/local/lib/python3.8/dist-packages (from nltk->pycaret) (2022.6.2)
    Requirement already satisfied: numexpr in /usr/local/lib/python3.8/dist-packages (from pyLDAvis->pycaret) (2.8.4)
    Requirement already satisfied: funcy in /usr/local/lib/python3.8/dist-packages (from pyLDAvis->pycaret) (1.17)
    Requirement already satisfied: future in /usr/local/lib/python3.8/dist-packages (from pyLDAvis->pycaret) (0.16.0)
    Requirement already satisfied: pynndescent>=0.5 in /usr/local/lib/python3.8/dist-packages (from umap-learn->pycaret) (0.5.8)
    Requirement already satisfied: Mako in /usr/local/lib/python3.8/dist-packages (from alembic<2->mlflow->pycaret) (1.2.4)
    Requirement already satisfied: importlib-resources in /usr/local/lib/python3.8/dist-packages (from alembic<2->mlflow->pycaret) (5.10.2)
    Requirement already satisfied: oauthlib>=3.1.0 in /usr/local/lib/python3.8/dist-packages (from databricks-cli<1,>=0.8.7->mlflow->pycaret) (3.2.2)
    Requirement already satisfied: pyjwt>=1.7.0 in /usr/local/lib/python3.8/dist-packages (from databricks-cli<1,>=0.8.7->mlflow->pycaret) (2.6.0)
    Requirement already satisfied: tabulate>=0.7.7 in /usr/local/lib/python3.8/dist-packages (from databricks-cli<1,>=0.8.7->mlflow->pycaret) (0.8.10)
    Requirement already satisfied: websocket-client>=0.32.0 in /usr/local/lib/python3.8/dist-packages (from docker<7,>=4.0.0->mlflow->pycaret) (1.4.2)
    Requirement already satisfied: urllib3>=1.26.0 in /usr/local/lib/python3.8/dist-packages (from docker<7,>=4.0.0->mlflow->pycaret) (1.26.14)
    Requirement already satisfied: Werkzeug<2.0,>=0.15 in /usr/local/lib/python3.8/dist-packages (from Flask<3->mlflow->pycaret) (1.0.1)
    Requirement already satisfied: itsdangerous<2.0,>=0.24 in /usr/local/lib/python3.8/dist-packages (from Flask<3->mlflow->pycaret) (1.1.0)
    Requirement already satisfied: gitdb<5,>=4.0.1 in /usr/local/lib/python3.8/dist-packages (from gitpython<4,>=2.1.0->mlflow->pycaret) (4.0.10)
    Requirement already satisfied: zipp>=0.5 in /usr/local/lib/python3.8/dist-packages (from importlib-metadata!=4.7.0,<6,>=3.7.0->mlflow->pycaret) (3.11.0)
    Requirement already satisfied: tornado>=4.2 in /usr/local/lib/python3.8/dist-packages (from ipykernel>=4.5.1->ipywidgets->pycaret) (6.0.4)
    Requirement already satisfied: jupyter-client in /usr/local/lib/python3.8/dist-packages (from ipykernel>=4.5.1->ipywidgets->pycaret) (6.1.12)
    Requirement already satisfied: parso<0.9.0,>=0.8.0 in /usr/local/lib/python3.8/dist-packages (from jedi>=0.10->IPython->pycaret) (0.8.3)
    Requirement already satisfied: MarkupSafe>=0.23 in /usr/local/lib/python3.8/dist-packages (from jinja2<3.2,>=2.11.1->pandas-profiling>=2.8.0->pycaret) (2.0.1)
    Requirement already satisfied: wcwidth in /usr/local/lib/python3.8/dist-packages (from prompt-toolkit<2.1.0,>=2.0.0->IPython->pycaret) (0.2.5)
    Requirement already satisfied: typing-extensions>=4.2.0 in /usr/local/lib/python3.8/dist-packages (from pydantic<1.11,>=1.8.1->pandas-profiling>=2.8.0->pycaret) (4.4.0)
    Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.8/dist-packages (from requests<2.29,>=2.24.0->pandas-profiling>=2.8.0->pycaret) (2.1.1)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.8/dist-packages (from requests<2.29,>=2.24.0->pandas-profiling>=2.8.0->pycaret) (2022.12.7)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.8/dist-packages (from requests<2.29,>=2.24.0->pandas-profiling>=2.8.0->pycaret) (2.10)
    Requirement already satisfied: slicer==0.0.7 in /usr/local/lib/python3.8/dist-packages (from shap<1,>=0.40->mlflow->pycaret) (0.0.7)
    Requirement already satisfied: greenlet!=0.4.17 in /usr/local/lib/python3.8/dist-packages (from sqlalchemy<2,>=1.4.0->mlflow->pycaret) (2.0.1)
    Requirement already satisfied: patsy>=0.5.2 in /usr/local/lib/python3.8/dist-packages (from statsmodels<0.14,>=0.13.2->pandas-profiling>=2.8.0->pycaret) (0.5.3)
    Requirement already satisfied: notebook>=4.4.1 in /usr/local/lib/python3.8/dist-packages (from widgetsnbextension~=3.6.0->ipywidgets->pycaret) (5.7.16)
    Requirement already satisfied: ptyprocess>=0.5 in /usr/local/lib/python3.8/dist-packages (from pexpect->IPython->pycaret) (0.7.0)
    Requirement already satisfied: smmap<6,>=3.0.1 in /usr/local/lib/python3.8/dist-packages (from gitdb<5,>=4.0.1->gitpython<4,>=2.1.0->mlflow->pycaret) (5.0.0)
    Requirement already satisfied: prometheus-client in /usr/local/lib/python3.8/dist-packages (from notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets->pycaret) (0.15.0)
    Requirement already satisfied: nbformat in /usr/local/lib/python3.8/dist-packages (from notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets->pycaret) (5.7.1)
    Requirement already satisfied: terminado>=0.8.1 in /usr/local/lib/python3.8/dist-packages (from notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets->pycaret) (0.13.3)
    Requirement already satisfied: pyzmq>=17 in /usr/local/lib/python3.8/dist-packages (from notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets->pycaret) (23.2.1)
    Requirement already satisfied: nbconvert<6.0 in /usr/local/lib/python3.8/dist-packages (from notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets->pycaret) (5.6.1)
    Requirement already satisfied: jupyter-core>=4.4.0 in /usr/local/lib/python3.8/dist-packages (from notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets->pycaret) (5.1.2)
    Requirement already satisfied: Send2Trash in /usr/local/lib/python3.8/dist-packages (from notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets->pycaret) (1.8.0)
    Requirement already satisfied: PyWavelets in /usr/local/lib/python3.8/dist-packages (from imagehash->visions[type_image_path]==0.7.5->pandas-profiling>=2.8.0->pycaret) (1.4.1)
    Requirement already satisfied: platformdirs>=2.5 in /usr/local/lib/python3.8/dist-packages (from jupyter-core>=4.4.0->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets->pycaret) (2.6.2)
    Requirement already satisfied: mistune<2,>=0.8.1 in /usr/local/lib/python3.8/dist-packages (from nbconvert<6.0->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets->pycaret) (0.8.4)
    Requirement already satisfied: bleach in /usr/local/lib/python3.8/dist-packages (from nbconvert<6.0->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets->pycaret) (5.0.1)
    Requirement already satisfied: defusedxml in /usr/local/lib/python3.8/dist-packages (from nbconvert<6.0->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets->pycaret) (0.7.1)
    Requirement already satisfied: testpath in /usr/local/lib/python3.8/dist-packages (from nbconvert<6.0->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets->pycaret) (0.6.0)
    Requirement already satisfied: pandocfilters>=1.4.1 in /usr/local/lib/python3.8/dist-packages (from nbconvert<6.0->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets->pycaret) (1.5.0)
    Requirement already satisfied: fastjsonschema in /usr/local/lib/python3.8/dist-packages (from nbformat->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets->pycaret) (2.16.2)
    Requirement already satisfied: jsonschema>=2.6 in /usr/local/lib/python3.8/dist-packages (from nbformat->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets->pycaret) (4.3.3)
    Requirement already satisfied: pyrsistent!=0.17.0,!=0.17.1,!=0.17.2,>=0.14.0 in /usr/local/lib/python3.8/dist-packages (from jsonschema>=2.6->nbformat->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets->pycaret) (0.19.3)
    Requirement already satisfied: webencodings in /usr/local/lib/python3.8/dist-packages (from bleach->nbconvert<6.0->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets->pycaret) (0.5.1)
    Installing collected packages: numpy
      Attempting uninstall: numpy
        Found existing installation: numpy 1.20.0
        Uninstalling numpy-1.20.0:
          Successfully uninstalled numpy-1.20.0
    [31mERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    xarray 2022.12.0 requires numpy>=1.20, but you have numpy 1.19.5 which is incompatible.
    xarray-einstats 0.4.0 requires numpy>=1.20, but you have numpy 1.19.5 which is incompatible.
    xarray-einstats 0.4.0 requires scipy>=1.6, but you have scipy 1.5.4 which is incompatible.
    tensorflow 2.9.2 requires numpy>=1.20, but you have numpy 1.19.5 which is incompatible.
    jaxlib 0.3.25+cuda11.cudnn805 requires numpy>=1.20, but you have numpy 1.19.5 which is incompatible.
    jax 0.3.25 requires numpy>=1.20, but you have numpy 1.19.5 which is incompatible.
    en-core-web-sm 3.4.1 requires spacy<3.5.0,>=3.4.0, but you have spacy 2.3.9 which is incompatible.
    cmdstanpy 1.0.8 requires numpy>=1.21, but you have numpy 1.19.5 which is incompatible.[0m[31m
    [0mSuccessfully installed numpy-1.19.5
    


```python
pip install numpy==1.20
```

    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Collecting numpy==1.20
      Using cached numpy-1.20.0-cp38-cp38-manylinux2010_x86_64.whl (15.4 MB)
    Installing collected packages: numpy
      Attempting uninstall: numpy
        Found existing installation: numpy 1.19.5
        Uninstalling numpy-1.19.5:
          Successfully uninstalled numpy-1.19.5
    [31mERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    yellowbrick 1.3.post1 requires numpy<1.20,>=1.16.0, but you have numpy 1.20.0 which is incompatible.
    xarray-einstats 0.4.0 requires scipy>=1.6, but you have scipy 1.5.4 which is incompatible.
    en-core-web-sm 3.4.1 requires spacy<3.5.0,>=3.4.0, but you have spacy 2.3.9 which is incompatible.
    cmdstanpy 1.0.8 requires numpy>=1.21, but you have numpy 1.20.0 which is incompatible.[0m[31m
    [0mSuccessfully installed numpy-1.20.0
    


```python
pip install scikit-learn==0.23.2
```

    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Requirement already satisfied: scikit-learn==0.23.2 in /usr/local/lib/python3.8/dist-packages (0.23.2)
    Requirement already satisfied: joblib>=0.11 in /usr/local/lib/python3.8/dist-packages (from scikit-learn==0.23.2) (1.2.0)
    Requirement already satisfied: numpy>=1.13.3 in /usr/local/lib/python3.8/dist-packages (from scikit-learn==0.23.2) (1.20.0)
    Requirement already satisfied: scipy>=0.19.1 in /usr/local/lib/python3.8/dist-packages (from scikit-learn==0.23.2) (1.5.4)
    Requirement already satisfied: threadpoolctl>=2.0.0 in /usr/local/lib/python3.8/dist-packages (from scikit-learn==0.23.2) (3.1.0)
    


```python
from pycaret.classification import *
setup_clf = setup(data=model_train, target='class',
                  session_id=777, fold_shuffle=True)
```



  <div id="df-ba162f17-97dc-487b-b402-8155aa864fa7">
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
      <th>Description</th>
      <th>Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>session_id</td>
      <td>777</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Target</td>
      <td>class</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Target Type</td>
      <td>Multiclass</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Label Encoded</td>
      <td>None</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Original Data</td>
      <td>(262, 7)</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Missing Values</td>
      <td>False</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Numeric Features</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Categorical Features</td>
      <td>6</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Ordinal Features</td>
      <td>False</td>
    </tr>
    <tr>
      <th>9</th>
      <td>High Cardinality Features</td>
      <td>False</td>
    </tr>
    <tr>
      <th>10</th>
      <td>High Cardinality Method</td>
      <td>None</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Transformed Train Set</td>
      <td>(183, 16)</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Transformed Test Set</td>
      <td>(79, 16)</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Shuffle Train-Test</td>
      <td>True</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Stratify Train-Test</td>
      <td>False</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Fold Generator</td>
      <td>StratifiedKFold</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Fold Number</td>
      <td>10</td>
    </tr>
    <tr>
      <th>17</th>
      <td>CPU Jobs</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Use GPU</td>
      <td>False</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Log Experiment</td>
      <td>False</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Experiment Name</td>
      <td>clf-default-name</td>
    </tr>
    <tr>
      <th>21</th>
      <td>USI</td>
      <td>f926</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Imputation Type</td>
      <td>simple</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Iterative Imputation Iteration</td>
      <td>None</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Numeric Imputer</td>
      <td>mean</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Iterative Imputation Numeric Model</td>
      <td>None</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Categorical Imputer</td>
      <td>constant</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Iterative Imputation Categorical Model</td>
      <td>None</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Unknown Categoricals Handling</td>
      <td>least_frequent</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Normalize</td>
      <td>False</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Normalize Method</td>
      <td>None</td>
    </tr>
    <tr>
      <th>31</th>
      <td>Transformation</td>
      <td>False</td>
    </tr>
    <tr>
      <th>32</th>
      <td>Transformation Method</td>
      <td>None</td>
    </tr>
    <tr>
      <th>33</th>
      <td>PCA</td>
      <td>False</td>
    </tr>
    <tr>
      <th>34</th>
      <td>PCA Method</td>
      <td>None</td>
    </tr>
    <tr>
      <th>35</th>
      <td>PCA Components</td>
      <td>None</td>
    </tr>
    <tr>
      <th>36</th>
      <td>Ignore Low Variance</td>
      <td>False</td>
    </tr>
    <tr>
      <th>37</th>
      <td>Combine Rare Levels</td>
      <td>False</td>
    </tr>
    <tr>
      <th>38</th>
      <td>Rare Level Threshold</td>
      <td>None</td>
    </tr>
    <tr>
      <th>39</th>
      <td>Numeric Binning</td>
      <td>False</td>
    </tr>
    <tr>
      <th>40</th>
      <td>Remove Outliers</td>
      <td>False</td>
    </tr>
    <tr>
      <th>41</th>
      <td>Outliers Threshold</td>
      <td>None</td>
    </tr>
    <tr>
      <th>42</th>
      <td>Remove Multicollinearity</td>
      <td>False</td>
    </tr>
    <tr>
      <th>43</th>
      <td>Multicollinearity Threshold</td>
      <td>None</td>
    </tr>
    <tr>
      <th>44</th>
      <td>Remove Perfect Collinearity</td>
      <td>True</td>
    </tr>
    <tr>
      <th>45</th>
      <td>Clustering</td>
      <td>False</td>
    </tr>
    <tr>
      <th>46</th>
      <td>Clustering Iteration</td>
      <td>None</td>
    </tr>
    <tr>
      <th>47</th>
      <td>Polynomial Features</td>
      <td>False</td>
    </tr>
    <tr>
      <th>48</th>
      <td>Polynomial Degree</td>
      <td>None</td>
    </tr>
    <tr>
      <th>49</th>
      <td>Trignometry Features</td>
      <td>False</td>
    </tr>
    <tr>
      <th>50</th>
      <td>Polynomial Threshold</td>
      <td>None</td>
    </tr>
    <tr>
      <th>51</th>
      <td>Group Features</td>
      <td>False</td>
    </tr>
    <tr>
      <th>52</th>
      <td>Feature Selection</td>
      <td>False</td>
    </tr>
    <tr>
      <th>53</th>
      <td>Feature Selection Method</td>
      <td>classic</td>
    </tr>
    <tr>
      <th>54</th>
      <td>Features Selection Threshold</td>
      <td>None</td>
    </tr>
    <tr>
      <th>55</th>
      <td>Feature Interaction</td>
      <td>False</td>
    </tr>
    <tr>
      <th>56</th>
      <td>Feature Ratio</td>
      <td>False</td>
    </tr>
    <tr>
      <th>57</th>
      <td>Interaction Threshold</td>
      <td>None</td>
    </tr>
    <tr>
      <th>58</th>
      <td>Fix Imbalance</td>
      <td>False</td>
    </tr>
    <tr>
      <th>59</th>
      <td>Fix Imbalance Method</td>
      <td>SMOTE</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-ba162f17-97dc-487b-b402-8155aa864fa7')"
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
          document.querySelector('#df-ba162f17-97dc-487b-b402-8155aa864fa7 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-ba162f17-97dc-487b-b402-8155aa864fa7');
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



    INFO:logs:create_model_container: 0
    INFO:logs:master_model_container: 0
    INFO:logs:display_container: 1
    INFO:logs:Pipeline(memory=None,
             steps=[('dtypes',
                     DataTypes_Auto_infer(categorical_features=[],
                                          display_types=True, features_todrop=[],
                                          id_columns=[],
                                          ml_usecase='classification',
                                          numerical_features=[], target='class',
                                          time_features=[])),
                    ('imputer',
                     Simple_Imputer(categorical_strategy='not_available',
                                    fill_value_categorical=None,
                                    fill_value_numerical=None,
                                    numeric_strate...
                    ('scaling', 'passthrough'), ('P_transform', 'passthrough'),
                    ('binn', 'passthrough'), ('rem_outliers', 'passthrough'),
                    ('cluster_all', 'passthrough'),
                    ('dummy', Dummify(target='class')),
                    ('fix_perfect', Remove_100(target='class')),
                    ('clean_names', Clean_Colum_Names()),
                    ('feature_select', 'passthrough'), ('fix_multi', 'passthrough'),
                    ('dfs', 'passthrough'), ('pca', 'passthrough')],
             verbose=False)
    INFO:logs:setup() succesfully completed......................................
    


```python
best_model = compare_models(sort='F1')
```



  <div id="df-2edc3dee-ecd5-4ba0-a24c-88c52732606d">
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
      <th>Model</th>
      <th>Accuracy</th>
      <th>AUC</th>
      <th>Recall</th>
      <th>Prec.</th>
      <th>F1</th>
      <th>Kappa</th>
      <th>MCC</th>
      <th>TT (Sec)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>dt</th>
      <td>Decision Tree Classifier</td>
      <td>0.9564</td>
      <td>0.9620</td>
      <td>0.9613</td>
      <td>0.9617</td>
      <td>0.9560</td>
      <td>0.9333</td>
      <td>0.9365</td>
      <td>0.028</td>
    </tr>
    <tr>
      <th>ridge</th>
      <td>Ridge Classifier</td>
      <td>0.9509</td>
      <td>0.0000</td>
      <td>0.9585</td>
      <td>0.9597</td>
      <td>0.9505</td>
      <td>0.9248</td>
      <td>0.9300</td>
      <td>0.033</td>
    </tr>
    <tr>
      <th>et</th>
      <td>Extra Trees Classifier</td>
      <td>0.9512</td>
      <td>0.9754</td>
      <td>0.9550</td>
      <td>0.9583</td>
      <td>0.9505</td>
      <td>0.9248</td>
      <td>0.9293</td>
      <td>0.539</td>
    </tr>
    <tr>
      <th>rf</th>
      <td>Random Forest Classifier</td>
      <td>0.9456</td>
      <td>0.9758</td>
      <td>0.9508</td>
      <td>0.9521</td>
      <td>0.9451</td>
      <td>0.9162</td>
      <td>0.9203</td>
      <td>0.443</td>
    </tr>
    <tr>
      <th>gbc</th>
      <td>Gradient Boosting Classifier</td>
      <td>0.9401</td>
      <td>0.9767</td>
      <td>0.9460</td>
      <td>0.9485</td>
      <td>0.9395</td>
      <td>0.9080</td>
      <td>0.9133</td>
      <td>0.606</td>
    </tr>
    <tr>
      <th>lr</th>
      <td>Logistic Regression</td>
      <td>0.9345</td>
      <td>0.9840</td>
      <td>0.9391</td>
      <td>0.9452</td>
      <td>0.9342</td>
      <td>0.8995</td>
      <td>0.9055</td>
      <td>0.523</td>
    </tr>
    <tr>
      <th>svm</th>
      <td>SVM - Linear Kernel</td>
      <td>0.9342</td>
      <td>0.0000</td>
      <td>0.9466</td>
      <td>0.9511</td>
      <td>0.9329</td>
      <td>0.8998</td>
      <td>0.9100</td>
      <td>0.032</td>
    </tr>
    <tr>
      <th>lightgbm</th>
      <td>Light Gradient Boosting Machine</td>
      <td>0.9289</td>
      <td>0.9818</td>
      <td>0.9371</td>
      <td>0.9375</td>
      <td>0.9285</td>
      <td>0.8911</td>
      <td>0.8963</td>
      <td>0.383</td>
    </tr>
    <tr>
      <th>knn</th>
      <td>K Neighbors Classifier</td>
      <td>0.9193</td>
      <td>0.9695</td>
      <td>0.9233</td>
      <td>0.9255</td>
      <td>0.9167</td>
      <td>0.8748</td>
      <td>0.8811</td>
      <td>0.052</td>
    </tr>
    <tr>
      <th>lda</th>
      <td>Linear Discriminant Analysis</td>
      <td>0.9126</td>
      <td>0.9818</td>
      <td>0.9052</td>
      <td>0.9374</td>
      <td>0.9128</td>
      <td>0.8650</td>
      <td>0.8770</td>
      <td>0.045</td>
    </tr>
    <tr>
      <th>ada</th>
      <td>Ada Boost Classifier</td>
      <td>0.9126</td>
      <td>0.9683</td>
      <td>0.9268</td>
      <td>0.9249</td>
      <td>0.9123</td>
      <td>0.8667</td>
      <td>0.8739</td>
      <td>0.217</td>
    </tr>
    <tr>
      <th>nb</th>
      <td>Naive Bayes</td>
      <td>0.7637</td>
      <td>0.9809</td>
      <td>0.8095</td>
      <td>0.8652</td>
      <td>0.7411</td>
      <td>0.6485</td>
      <td>0.7117</td>
      <td>0.030</td>
    </tr>
    <tr>
      <th>dummy</th>
      <td>Dummy Classifier</td>
      <td>0.4208</td>
      <td>0.5000</td>
      <td>0.3333</td>
      <td>0.1776</td>
      <td>0.2496</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.026</td>
    </tr>
    <tr>
      <th>qda</th>
      <td>Quadratic Discriminant Analysis</td>
      <td>0.2512</td>
      <td>0.0000</td>
      <td>0.3333</td>
      <td>0.0637</td>
      <td>0.1014</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.025</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-2edc3dee-ecd5-4ba0-a24c-88c52732606d')"
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
          document.querySelector('#df-2edc3dee-ecd5-4ba0-a24c-88c52732606d button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-2edc3dee-ecd5-4ba0-a24c-88c52732606d');
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



    INFO:logs:create_model_container: 14
    INFO:logs:master_model_container: 14
    INFO:logs:display_container: 2
    INFO:logs:DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                           max_depth=None, max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, presort='deprecated',
                           random_state=777, splitter='best')
    INFO:logs:compare_models() succesfully completed......................................
    


```python
fold=StratifiedKFold(n_splits=5, shuffle=True)
ridge = create_model('ridge', fold=fold)
```



  <div id="df-61267939-cbf6-43c3-b3eb-983305b67c53">
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
      <th>Accuracy</th>
      <th>AUC</th>
      <th>Recall</th>
      <th>Prec.</th>
      <th>F1</th>
      <th>Kappa</th>
      <th>MCC</th>
    </tr>
    <tr>
      <th>Fold</th>
      <th></th>
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
      <th>0</th>
      <td>0.8919</td>
      <td>0.0</td>
      <td>0.8944</td>
      <td>0.8960</td>
      <td>0.8904</td>
      <td>0.8345</td>
      <td>0.8383</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.9730</td>
      <td>0.0</td>
      <td>0.9792</td>
      <td>0.9751</td>
      <td>0.9731</td>
      <td>0.9585</td>
      <td>0.9596</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.9730</td>
      <td>0.0</td>
      <td>0.9792</td>
      <td>0.9751</td>
      <td>0.9731</td>
      <td>0.9585</td>
      <td>0.9596</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.9444</td>
      <td>0.0</td>
      <td>0.9556</td>
      <td>0.9524</td>
      <td>0.9446</td>
      <td>0.9155</td>
      <td>0.9198</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.8889</td>
      <td>0.0</td>
      <td>0.9056</td>
      <td>0.8965</td>
      <td>0.8892</td>
      <td>0.8310</td>
      <td>0.8349</td>
    </tr>
    <tr>
      <th>Mean</th>
      <td>0.9342</td>
      <td>0.0</td>
      <td>0.9428</td>
      <td>0.9390</td>
      <td>0.9341</td>
      <td>0.8996</td>
      <td>0.9024</td>
    </tr>
    <tr>
      <th>Std</th>
      <td>0.0373</td>
      <td>0.0</td>
      <td>0.0361</td>
      <td>0.0359</td>
      <td>0.0376</td>
      <td>0.0568</td>
      <td>0.0557</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-61267939-cbf6-43c3-b3eb-983305b67c53')"
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
          document.querySelector('#df-61267939-cbf6-43c3-b3eb-983305b67c53 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-61267939-cbf6-43c3-b3eb-983305b67c53');
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



    INFO:logs:create_model_container: 15
    INFO:logs:master_model_container: 15
    INFO:logs:display_container: 3
    INFO:logs:RidgeClassifier(alpha=1.0, class_weight=None, copy_X=True, fit_intercept=True,
                    max_iter=None, normalize=False, random_state=777, solver='auto',
                    tol=0.001)
    INFO:logs:create_model() succesfully completed......................................
    


```python
tuned_rd = tune_model(ridge, optimize='F1')
```



  <div id="df-66f28e8c-5cc9-47ee-aab9-abcd3041b29e">
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
      <th>Accuracy</th>
      <th>AUC</th>
      <th>Recall</th>
      <th>Prec.</th>
      <th>F1</th>
      <th>Kappa</th>
      <th>MCC</th>
    </tr>
    <tr>
      <th>Fold</th>
      <th></th>
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
      <th>0</th>
      <td>0.8421</td>
      <td>0.0</td>
      <td>0.8333</td>
      <td>0.8852</td>
      <td>0.8283</td>
      <td>0.7522</td>
      <td>0.7846</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.8947</td>
      <td>0.0</td>
      <td>0.9028</td>
      <td>0.8947</td>
      <td>0.8947</td>
      <td>0.8390</td>
      <td>0.8390</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0000</td>
      <td>0.0</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.9444</td>
      <td>0.0</td>
      <td>0.9524</td>
      <td>0.9524</td>
      <td>0.9444</td>
      <td>0.9163</td>
      <td>0.9206</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.9444</td>
      <td>0.0</td>
      <td>0.9524</td>
      <td>0.9524</td>
      <td>0.9444</td>
      <td>0.9163</td>
      <td>0.9206</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.0000</td>
      <td>0.0</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1.0000</td>
      <td>0.0</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.8889</td>
      <td>0.0</td>
      <td>0.9167</td>
      <td>0.9167</td>
      <td>0.8889</td>
      <td>0.8302</td>
      <td>0.8462</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.8333</td>
      <td>0.0</td>
      <td>0.8472</td>
      <td>0.8346</td>
      <td>0.8307</td>
      <td>0.7379</td>
      <td>0.7415</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.9444</td>
      <td>0.0</td>
      <td>0.9583</td>
      <td>0.9524</td>
      <td>0.9447</td>
      <td>0.9143</td>
      <td>0.9187</td>
    </tr>
    <tr>
      <th>Mean</th>
      <td>0.9292</td>
      <td>0.0</td>
      <td>0.9363</td>
      <td>0.9388</td>
      <td>0.9276</td>
      <td>0.8906</td>
      <td>0.8971</td>
    </tr>
    <tr>
      <th>Std</th>
      <td>0.0594</td>
      <td>0.0</td>
      <td>0.0576</td>
      <td>0.0529</td>
      <td>0.0620</td>
      <td>0.0931</td>
      <td>0.0873</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-66f28e8c-5cc9-47ee-aab9-abcd3041b29e')"
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
          document.querySelector('#df-66f28e8c-5cc9-47ee-aab9-abcd3041b29e button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-66f28e8c-5cc9-47ee-aab9-abcd3041b29e');
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



    INFO:logs:create_model_container: 16
    INFO:logs:master_model_container: 16
    INFO:logs:display_container: 4
    INFO:logs:RidgeClassifier(alpha=4.24, class_weight=None, copy_X=True, fit_intercept=True,
                    max_iter=None, normalize=False, random_state=777, solver='auto',
                    tol=0.001)
    INFO:logs:tune_model() succesfully completed......................................
    


```python
evaluate_model(tuned_rd)
```

    INFO:logs:Initializing evaluate_model()
    INFO:logs:evaluate_model(estimator=RidgeClassifier(alpha=4.24, class_weight=None, copy_X=True, fit_intercept=True,
                    max_iter=None, normalize=False, random_state=777, solver='auto',
                    tol=0.001), fold=None, fit_kwargs=None, plot_kwargs=None, feature_name=None, groups=None, use_train_data=False)
    


    interactive(children=(ToggleButtons(description='Plot Type:', icons=('',), options=(('Hyperparameters', 'param…



```python
final_model = finalize_model(tuned_rd)
```

    INFO:logs:Initializing finalize_model()
    INFO:logs:finalize_model(estimator=RidgeClassifier(alpha=4.24, class_weight=None, copy_X=True, fit_intercept=True,
                    max_iter=None, normalize=False, random_state=777, solver='auto',
                    tol=0.001), fit_kwargs=None, groups=None, model_only=True, display=None, experiment_custom_tags=None, return_train_score=False)
    INFO:logs:Finalizing RidgeClassifier(alpha=4.24, class_weight=None, copy_X=True, fit_intercept=True,
                    max_iter=None, normalize=False, random_state=777, solver='auto',
                    tol=0.001)
    INFO:logs:Initializing create_model()
    INFO:logs:create_model(estimator=RidgeClassifier(alpha=4.24, class_weight=None, copy_X=True, fit_intercept=True,
                    max_iter=None, normalize=False, random_state=777, solver='auto',
                    tol=0.001), fold=None, round=4, cross_validation=True, predict=True, fit_kwargs={}, groups=None, refit=True, verbose=False, system=False, metrics=None, experiment_custom_tags=None, add_to_model_list=False, probability_threshold=None, display=None, return_train_score=False, kwargs={})
    INFO:logs:Checking exceptions
    INFO:logs:Importing libraries
    INFO:logs:Copying training dataset
    INFO:logs:Defining folds
    INFO:logs:Declaring metric variables
    INFO:logs:Importing untrained model
    INFO:logs:Declaring custom model
    INFO:logs:Ridge Classifier Imported succesfully
    INFO:logs:Starting cross validation
    INFO:logs:Cross validating with StratifiedKFold(n_splits=10, random_state=777, shuffle=True), n_jobs=-1
    INFO:logs:Calculating mean and std
    INFO:logs:Creating metrics dataframe
    INFO:logs:Finalizing model
    INFO:logs:create_model_container: 16
    INFO:logs:master_model_container: 16
    INFO:logs:display_container: 5
    INFO:logs:RidgeClassifier(alpha=4.24, class_weight=None, copy_X=True, fit_intercept=True,
                    max_iter=None, normalize=False, random_state=777, solver='auto',
                    tol=0.001)
    INFO:logs:create_model() succesfully completed......................................
    INFO:logs:create_model_container: 16
    INFO:logs:master_model_container: 16
    INFO:logs:display_container: 4
    INFO:logs:RidgeClassifier(alpha=4.24, class_weight=None, copy_X=True, fit_intercept=True,
                    max_iter=None, normalize=False, random_state=777, solver='auto',
                    tol=0.001)
    INFO:logs:finalize_model() succesfully completed......................................
    


```python
prediction = predict_model(final_model, data = test_x)
```

    INFO:logs:Initializing predict_model()
    INFO:logs:predict_model(estimator=RidgeClassifier(alpha=4.24, class_weight=None, copy_X=True, fit_intercept=True,
                    max_iter=None, normalize=False, random_state=777, solver='auto',
                    tol=0.001), probability_threshold=None, encoded_labels=False, drift_report=False, raw_score=False, round=4, verbose=True, ml_usecase=MLUsecase.CLASSIFICATION, display=None, drift_kwargs=None)
    INFO:logs:Checking exceptions
    INFO:logs:Preloading libraries
    INFO:logs:Preparing display monitor
    


```python
prediction
```





  <div id="df-4e9f472e-db17-40e3-a275-9e13b5fabf64">
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
      <th>trait</th>
      <th>SNP_04</th>
      <th>SNP_05</th>
      <th>SNP_06</th>
      <th>SNP_07</th>
      <th>SNP_08</th>
      <th>Label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>5</td>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>5</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>5</td>
      <td>0</td>
      <td>5</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
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
    </tr>
    <tr>
      <th>170</th>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>171</th>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>172</th>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>173</th>
      <td>2</td>
      <td>4</td>
      <td>3</td>
      <td>5</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>174</th>
      <td>2</td>
      <td>4</td>
      <td>2</td>
      <td>0</td>
      <td>4</td>
      <td>5</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>175 rows × 7 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-4e9f472e-db17-40e3-a275-9e13b5fabf64')"
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
          document.querySelector('#df-4e9f472e-db17-40e3-a275-9e13b5fabf64 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-4e9f472e-db17-40e3-a275-9e13b5fabf64');
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
prediction['Label']
```




    0      0
    1      1
    2      2
    3      2
    4      0
          ..
    170    1
    171    2
    172    2
    173    1
    174    1
    Name: Label, Length: 175, dtype: int64




```python
submit = pd.read_csv('/content/drive/MyDrive/유전체/sample_submission.csv')

submit['class'] = class_le.inverse_transform(prediction['Label'])

submit.to_csv('/content/drive/MyDrive/유전체/sample_submission.csv', index=False)
```


```python
submit
```





  <div id="df-8542d274-8902-44d4-a782-338e0117345f">
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
      <th>id</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>TEST_000</td>
      <td>A</td>
    </tr>
    <tr>
      <th>1</th>
      <td>TEST_001</td>
      <td>B</td>
    </tr>
    <tr>
      <th>2</th>
      <td>TEST_002</td>
      <td>C</td>
    </tr>
    <tr>
      <th>3</th>
      <td>TEST_003</td>
      <td>C</td>
    </tr>
    <tr>
      <th>4</th>
      <td>TEST_004</td>
      <td>A</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>170</th>
      <td>TEST_170</td>
      <td>B</td>
    </tr>
    <tr>
      <th>171</th>
      <td>TEST_171</td>
      <td>C</td>
    </tr>
    <tr>
      <th>172</th>
      <td>TEST_172</td>
      <td>C</td>
    </tr>
    <tr>
      <th>173</th>
      <td>TEST_173</td>
      <td>B</td>
    </tr>
    <tr>
      <th>174</th>
      <td>TEST_174</td>
      <td>B</td>
    </tr>
  </tbody>
</table>
<p>175 rows × 2 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-8542d274-8902-44d4-a782-338e0117345f')"
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
          document.querySelector('#df-8542d274-8902-44d4-a782-338e0117345f button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-8542d274-8902-44d4-a782-338e0117345f');
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



