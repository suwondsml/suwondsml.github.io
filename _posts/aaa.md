# 경로 설정


```python
from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import StratifiedKFold
```


```python
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df["target"] = iris.target
print(df["target"].value_counts())
df

```

    0    50
    1    50
    2    50
    Name: target, dtype: int64





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
      <th>sepal length (cm)</th>
      <th>sepal width (cm)</th>
      <th>petal length (cm)</th>
      <th>petal width (cm)</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>145</th>
      <td>6.7</td>
      <td>3.0</td>
      <td>5.2</td>
      <td>2.3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>146</th>
      <td>6.3</td>
      <td>2.5</td>
      <td>5.0</td>
      <td>1.9</td>
      <td>2</td>
    </tr>
    <tr>
      <th>147</th>
      <td>6.5</td>
      <td>3.0</td>
      <td>5.2</td>
      <td>2.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>148</th>
      <td>6.2</td>
      <td>3.4</td>
      <td>5.4</td>
      <td>2.3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>149</th>
      <td>5.9</td>
      <td>3.0</td>
      <td>5.1</td>
      <td>1.8</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>150 rows × 5 columns</p>
</div>




```python
skf = StratifiedKFold(n_splits=3)
n_iter = 0

```


```python
for train_index, test_index in skf.split(df, df["target"]):
    n_iter += 1
    target_train = df["target"].iloc[train_index]
    target_test = df["target"].iloc[test_index]
    print(f"교차검증: {n_iter}")
    print(f"학습 타겟 데이터 분포:\n{target_train.value_counts()}")
    print(f"검증 타겟 데이터 분포:\n{target_test.value_counts()}\n")
```

    교차검증: 1
    학습 타겟 데이터 분포:
    2    34
    0    33
    1    33
    Name: target, dtype: int64
    검증 타겟 데이터 분포:
    0    17
    1    17
    2    16
    Name: target, dtype: int64
    
    교차검증: 2
    학습 타겟 데이터 분포:
    1    34
    0    33
    2    33
    Name: target, dtype: int64
    검증 타겟 데이터 분포:
    0    17
    2    17
    1    16
    Name: target, dtype: int64
    
    교차검증: 3
    학습 타겟 데이터 분포:
    0    34
    1    33
    2    33
    Name: target, dtype: int64
    검증 타겟 데이터 분포:
    1    17
    2    17
    0    16
    Name: target, dtype: int64
    



```python

```


```python

```


```python

```


```python

```


```python

```


```python

```

# module import


```python
# files and system
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
# working with images
import cv2
import imageio
import scipy.ndimage
# import skimage.transform
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

# 랜덤성을 배제한 환경 고정


```python
random_seed= 42
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
```


```python
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # select device for training, i.e. gpu or cpu
print(DEVICE)
```

    cuda:0


# Data Load

## train data


```python
df = pd.read_csv("./data/open/train.csv")
df.head(5)
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
      <th>id</th>
      <th>img_path</th>
      <th>artist</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>./train/0000.jpg</td>
      <td>Diego Velazquez</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>./train/0001.jpg</td>
      <td>Vincent van Gogh</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>./train/0002.jpg</td>
      <td>Claude Monet</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>./train/0003.jpg</td>
      <td>Edgar Degas</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>./train/0004.jpg</td>
      <td>Hieronymus Bosch</td>
    </tr>
  </tbody>
</table>
</div>



## Evaluation data


```python
df_eval = pd.read_csv("./data/open/test.csv")
df_eval.head()
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
      <th>id</th>
      <th>img_path</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>TEST_00000</td>
      <td>./test/TEST_00000.jpg</td>
    </tr>
    <tr>
      <th>1</th>
      <td>TEST_00001</td>
      <td>./test/TEST_00001.jpg</td>
    </tr>
    <tr>
      <th>2</th>
      <td>TEST_00002</td>
      <td>./test/TEST_00002.jpg</td>
    </tr>
    <tr>
      <th>3</th>
      <td>TEST_00003</td>
      <td>./test/TEST_00003.jpg</td>
    </tr>
    <tr>
      <th>4</th>
      <td>TEST_00004</td>
      <td>./test/TEST_00004.jpg</td>
    </tr>
  </tbody>
</table>
</div>



# Label encoding


```python
artist_name = list(df['artist'].value_counts().index)
artist_name
```




    ['Vincent van Gogh',
     'Edgar Degas',
     'Pablo Picasso',
     'Pierre-Auguste Renoir',
     'Albrecht Du rer',
     'Paul Gauguin',
     'Francisco Goya',
     'Rembrandt',
     'Titian',
     'Marc Chagall',
     'Alfred Sisley',
     'Paul Klee',
     'Rene Magritte',
     'Amedeo Modigliani',
     'Andy Warhol',
     'Henri Matisse',
     'Sandro Botticelli',
     'Mikhail Vrubel',
     'Hieronymus Bosch',
     'Leonardo da Vinci',
     'Salvador Dali',
     'Peter Paul Rubens',
     'Kazimir Malevich',
     'Pieter Bruegel',
     'Frida Kahlo',
     'Diego Velazquez',
     'Joan Miro',
     'Andrei Rublev',
     'Raphael',
     'Giotto di Bondone',
     'Gustav Klimt',
     'El Greco',
     'Camille Pissarro',
     'Jan van Eyck',
     'Edouard Manet',
     'Henri de Toulouse-Lautrec',
     'Vasiliy Kandinskiy',
     'Piet Mondrian',
     'Claude Monet',
     'Henri Rousseau',
     'Diego Rivera',
     'Edvard Munch',
     'William Turner',
     'Gustave Courbet',
     'Michelangelo',
     'Paul Cezanne',
     'Caravaggio',
     'Georges Seurat',
     'Eugene Delacroix',
     'Jackson Pollock']




```python
add_list = []
for i in range(len(df)):
    add_list.append(artist_name.index(df.iloc[i]['artist']))

print(f"nums of artist : {len(artist_name)}")
df['label'] = add_list
df

```

    nums of artist : 50





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
      <th>img_path</th>
      <th>artist</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>./train/0000.jpg</td>
      <td>Diego Velazquez</td>
      <td>25</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>./train/0001.jpg</td>
      <td>Vincent van Gogh</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>./train/0002.jpg</td>
      <td>Claude Monet</td>
      <td>38</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>./train/0003.jpg</td>
      <td>Edgar Degas</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>./train/0004.jpg</td>
      <td>Hieronymus Bosch</td>
      <td>18</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>5906</th>
      <td>5906</td>
      <td>./train/5906.jpg</td>
      <td>Pieter Bruegel</td>
      <td>23</td>
    </tr>
    <tr>
      <th>5907</th>
      <td>5907</td>
      <td>./train/5907.jpg</td>
      <td>Peter Paul Rubens</td>
      <td>21</td>
    </tr>
    <tr>
      <th>5908</th>
      <td>5908</td>
      <td>./train/5908.jpg</td>
      <td>Paul Gauguin</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5909</th>
      <td>5909</td>
      <td>./train/5909.jpg</td>
      <td>Paul Gauguin</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5910</th>
      <td>5910</td>
      <td>./train/5910.jpg</td>
      <td>Andrei Rublev</td>
      <td>27</td>
    </tr>
  </tbody>
</table>
<p>5911 rows × 4 columns</p>
</div>



# 데이터 수 비교


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


    
![png](output_25_0.png)
    


- 데이터의 불균형이 있음

# Data split


```python
X_train, X_val, y_train, y_val = train_test_split(df, df['label'].values, test_size=0.2)



print("Number of posters for training: ", len(X_train))
print("Number of posters for validation: ", len(X_val))
```

    Number of posters for training:  4728
    Number of posters for validation:  1183



```python
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=3)
n_iter = 0
for train_index, test_index in skf.split(df, df["label"]):
    X_train = df.iloc[train_index]
    X_val = df.iloc[test_index]
```


```python
train_index, test_index, a = skf.split(df, df["label"])
print(len(train_index[0]))
print(len(train_index[1]))

```

    3940
    1971



```python
X_train

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
      <th>id</th>
      <th>img_path</th>
      <th>artist</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>./train/0000.jpg</td>
      <td>Diego Velazquez</td>
      <td>25</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>./train/0001.jpg</td>
      <td>Vincent van Gogh</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>./train/0002.jpg</td>
      <td>Claude Monet</td>
      <td>38</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>./train/0003.jpg</td>
      <td>Edgar Degas</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>./train/0004.jpg</td>
      <td>Hieronymus Bosch</td>
      <td>18</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4505</th>
      <td>4505</td>
      <td>./train/4505.jpg</td>
      <td>Gustav Klimt</td>
      <td>30</td>
    </tr>
    <tr>
      <th>4541</th>
      <td>4541</td>
      <td>./train/4541.jpg</td>
      <td>Gustav Klimt</td>
      <td>30</td>
    </tr>
    <tr>
      <th>4593</th>
      <td>4593</td>
      <td>./train/4593.jpg</td>
      <td>Gustav Klimt</td>
      <td>30</td>
    </tr>
    <tr>
      <th>4594</th>
      <td>4594</td>
      <td>./train/4594.jpg</td>
      <td>Caravaggio</td>
      <td>46</td>
    </tr>
    <tr>
      <th>4618</th>
      <td>4618</td>
      <td>./train/4618.jpg</td>
      <td>Giotto di Bondone</td>
      <td>29</td>
    </tr>
  </tbody>
</table>
<p>3941 rows × 4 columns</p>
</div>




```python
# count the number of images per category in train_ds
import collections
counter_train = collections.Counter(y_train)
print(counter_train)
```

    Counter({0: 508, 1: 392, 2: 238, 4: 184, 3: 182, 5: 178, 6: 159, 8: 144, 9: 142, 7: 141, 10: 133, 11: 115, 12: 112, 13: 109, 14: 103, 15: 97, 16: 95, 18: 94, 17: 94, 19: 84, 20: 81, 23: 71, 22: 70, 21: 70, 27: 66, 24: 61, 25: 61, 28: 60, 26: 60, 30: 55, 29: 54, 33: 53, 31: 51, 35: 51, 32: 49, 36: 49, 38: 46, 37: 45, 39: 44, 34: 44, 40: 42, 42: 37, 41: 34, 43: 33, 46: 27, 44: 26, 47: 25, 45: 23, 48: 18, 49: 18})


- test dataset을 만들기 위해 한번더 split을 해줌


```python
X_train
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
      <th>id</th>
      <th>img_path</th>
      <th>artist</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5867</th>
      <td>5867</td>
      <td>./train/5867.jpg</td>
      <td>Gustave Courbet</td>
      <td>43</td>
    </tr>
    <tr>
      <th>4584</th>
      <td>4584</td>
      <td>./train/4584.jpg</td>
      <td>Francisco Goya</td>
      <td>6</td>
    </tr>
    <tr>
      <th>2451</th>
      <td>2451</td>
      <td>./train/2451.jpg</td>
      <td>Pablo Picasso</td>
      <td>2</td>
    </tr>
    <tr>
      <th>5120</th>
      <td>5120</td>
      <td>./train/5120.jpg</td>
      <td>Pablo Picasso</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1328</th>
      <td>1328</td>
      <td>./train/1328.jpg</td>
      <td>Titian</td>
      <td>8</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3772</th>
      <td>3772</td>
      <td>./train/3772.jpg</td>
      <td>William Turner</td>
      <td>42</td>
    </tr>
    <tr>
      <th>5191</th>
      <td>5191</td>
      <td>./train/5191.jpg</td>
      <td>Hieronymus Bosch</td>
      <td>18</td>
    </tr>
    <tr>
      <th>5226</th>
      <td>5226</td>
      <td>./train/5226.jpg</td>
      <td>Claude Monet</td>
      <td>38</td>
    </tr>
    <tr>
      <th>5390</th>
      <td>5390</td>
      <td>./train/5390.jpg</td>
      <td>Andrei Rublev</td>
      <td>27</td>
    </tr>
    <tr>
      <th>860</th>
      <td>860</td>
      <td>./train/0860.jpg</td>
      <td>Pablo Picasso</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>4728 rows × 4 columns</p>
</div>



## Train set -> Train set + Test set (0.9 : 0.1)


```python
X_train, X_test, y_train, y_test = train_test_split(X_train, X_train['label'].values, test_size=0.1)
print("Number of posters for training: ", len(X_train))
print("Number of posters for validation: ", len(X_test))
```

    Number of posters for training:  4255
    Number of posters for validation:  473



```python
counter_train = collections.Counter(y_train)
print(counter_train)
X_train
```

    Counter({0: 457, 1: 359, 2: 217, 4: 166, 3: 160, 5: 155, 6: 142, 8: 135, 9: 130, 7: 125, 10: 116, 11: 105, 12: 104, 13: 98, 14: 91, 17: 86, 15: 83, 18: 82, 16: 79, 19: 73, 20: 73, 22: 66, 23: 65, 21: 63, 27: 61, 25: 55, 28: 54, 26: 52, 29: 52, 33: 50, 24: 50, 30: 50, 35: 47, 37: 44, 31: 44, 36: 43, 39: 42, 32: 42, 38: 40, 34: 38, 40: 36, 42: 35, 41: 32, 43: 31, 44: 24, 46: 23, 45: 22, 47: 22, 48: 18, 49: 18})





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
      <th>img_path</th>
      <th>artist</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>620</th>
      <td>620</td>
      <td>./train/0620.jpg</td>
      <td>Mikhail Vrubel</td>
      <td>17</td>
    </tr>
    <tr>
      <th>4688</th>
      <td>4688</td>
      <td>./train/4688.jpg</td>
      <td>Claude Monet</td>
      <td>38</td>
    </tr>
    <tr>
      <th>2502</th>
      <td>2502</td>
      <td>./train/2502.jpg</td>
      <td>Rene Magritte</td>
      <td>12</td>
    </tr>
    <tr>
      <th>3594</th>
      <td>3594</td>
      <td>./train/3594.jpg</td>
      <td>Albrecht Du rer</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4248</th>
      <td>4248</td>
      <td>./train/4248.jpg</td>
      <td>Marc Chagall</td>
      <td>9</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>5597</th>
      <td>5597</td>
      <td>./train/5597.jpg</td>
      <td>Francisco Goya</td>
      <td>6</td>
    </tr>
    <tr>
      <th>3402</th>
      <td>3402</td>
      <td>./train/3402.jpg</td>
      <td>Pierre-Auguste Renoir</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3699</th>
      <td>3699</td>
      <td>./train/3699.jpg</td>
      <td>Pablo Picasso</td>
      <td>2</td>
    </tr>
    <tr>
      <th>868</th>
      <td>868</td>
      <td>./train/0868.jpg</td>
      <td>Vincent van Gogh</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3975</th>
      <td>3975</td>
      <td>./train/3975.jpg</td>
      <td>Vincent van Gogh</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>4255 rows × 4 columns</p>
</div>




```python
X_test
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
      <th>id</th>
      <th>img_path</th>
      <th>artist</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5724</th>
      <td>5724</td>
      <td>./train/5724.jpg</td>
      <td>Giotto di Bondone</td>
      <td>29</td>
    </tr>
    <tr>
      <th>921</th>
      <td>921</td>
      <td>./train/0921.jpg</td>
      <td>Vincent van Gogh</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1412</th>
      <td>1412</td>
      <td>./train/1412.jpg</td>
      <td>Albrecht Du rer</td>
      <td>4</td>
    </tr>
    <tr>
      <th>5048</th>
      <td>5048</td>
      <td>./train/5048.jpg</td>
      <td>Claude Monet</td>
      <td>38</td>
    </tr>
    <tr>
      <th>462</th>
      <td>462</td>
      <td>./train/0462.jpg</td>
      <td>Sandro Botticelli</td>
      <td>16</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2957</th>
      <td>2957</td>
      <td>./train/2957.jpg</td>
      <td>Alfred Sisley</td>
      <td>10</td>
    </tr>
    <tr>
      <th>2104</th>
      <td>2104</td>
      <td>./train/2104.jpg</td>
      <td>Frida Kahlo</td>
      <td>24</td>
    </tr>
    <tr>
      <th>5486</th>
      <td>5486</td>
      <td>./train/5486.jpg</td>
      <td>Marc Chagall</td>
      <td>9</td>
    </tr>
    <tr>
      <th>3331</th>
      <td>3331</td>
      <td>./train/3331.jpg</td>
      <td>Frida Kahlo</td>
      <td>24</td>
    </tr>
    <tr>
      <th>1997</th>
      <td>1997</td>
      <td>./train/1997.jpg</td>
      <td>Amedeo Modigliani</td>
      <td>13</td>
    </tr>
  </tbody>
</table>
<p>473 rows × 4 columns</p>
</div>




```python
X_val
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
      <th>id</th>
      <th>img_path</th>
      <th>artist</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5577</th>
      <td>5577</td>
      <td>./train/5577.jpg</td>
      <td>Pablo Picasso</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1881</th>
      <td>1881</td>
      <td>./train/1881.jpg</td>
      <td>Pieter Bruegel</td>
      <td>23</td>
    </tr>
    <tr>
      <th>4204</th>
      <td>4204</td>
      <td>./train/4204.jpg</td>
      <td>Albrecht Du rer</td>
      <td>4</td>
    </tr>
    <tr>
      <th>5312</th>
      <td>5312</td>
      <td>./train/5312.jpg</td>
      <td>Francisco Goya</td>
      <td>6</td>
    </tr>
    <tr>
      <th>2741</th>
      <td>2741</td>
      <td>./train/2741.jpg</td>
      <td>Hieronymus Bosch</td>
      <td>18</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3809</th>
      <td>3809</td>
      <td>./train/3809.jpg</td>
      <td>Albrecht Du rer</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2024</th>
      <td>2024</td>
      <td>./train/2024.jpg</td>
      <td>Edgar Degas</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3450</th>
      <td>3450</td>
      <td>./train/3450.jpg</td>
      <td>Pierre-Auguste Renoir</td>
      <td>3</td>
    </tr>
    <tr>
      <th>653</th>
      <td>653</td>
      <td>./train/0653.jpg</td>
      <td>Diego Velazquez</td>
      <td>25</td>
    </tr>
    <tr>
      <th>429</th>
      <td>429</td>
      <td>./train/0429.jpg</td>
      <td>Diego Rivera</td>
      <td>40</td>
    </tr>
  </tbody>
</table>
<p>1183 rows × 4 columns</p>
</div>



## train set (dataset)


```python
base_path = '/project/segmentation/smcho1201/my_project/Classification_of_paintings'
```


```python
X_train = X_train.sort_values(by=['id']) # id 기준 정렬
X_train['img_path'] = [base_path + '/data/open/' + path[2:] for path in X_train['img_path']]

print(f"train set 개수 : {len(X_train)}")
X_train.head()
```

    train set 개수 : 4255





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
      <th>img_path</th>
      <th>artist</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>/project/segmentation/smcho1201/my_project/Cla...</td>
      <td>Diego Velazquez</td>
      <td>25</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>/project/segmentation/smcho1201/my_project/Cla...</td>
      <td>Vincent van Gogh</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>/project/segmentation/smcho1201/my_project/Cla...</td>
      <td>Claude Monet</td>
      <td>38</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>/project/segmentation/smcho1201/my_project/Cla...</td>
      <td>Edgar Degas</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>/project/segmentation/smcho1201/my_project/Cla...</td>
      <td>Rene Magritte</td>
      <td>12</td>
    </tr>
  </tbody>
</table>
</div>



## validation set (dataset)


```python
X_val = X_val.sort_values(by=['id'])
X_val['img_path'] = [base_path + '/data/open/' + path[2:] for path in X_val['img_path']]

print(f"validation set 개수 : {len(X_val)}")
X_val.head()
```

    validation set 개수 : 1183





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
      <th>img_path</th>
      <th>artist</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>/project/segmentation/smcho1201/my_project/Cla...</td>
      <td>Michelangelo</td>
      <td>44</td>
    </tr>
    <tr>
      <th>12</th>
      <td>12</td>
      <td>/project/segmentation/smcho1201/my_project/Cla...</td>
      <td>Edouard Manet</td>
      <td>34</td>
    </tr>
    <tr>
      <th>14</th>
      <td>14</td>
      <td>/project/segmentation/smcho1201/my_project/Cla...</td>
      <td>Francisco Goya</td>
      <td>6</td>
    </tr>
    <tr>
      <th>15</th>
      <td>15</td>
      <td>/project/segmentation/smcho1201/my_project/Cla...</td>
      <td>Edgar Degas</td>
      <td>1</td>
    </tr>
    <tr>
      <th>17</th>
      <td>17</td>
      <td>/project/segmentation/smcho1201/my_project/Cla...</td>
      <td>Edgar Degas</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



## Test set (dataset)


```python
X_test = X_test.sort_values(by=['id'])
X_test['img_path'] = [base_path + '/data/open/' + path[2:] for path in X_test['img_path']]

print(f"Test set 개수 : {len(X_test)}")
X_test.head()
```

    Test set 개수 : 473





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
      <th>img_path</th>
      <th>artist</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>/project/segmentation/smcho1201/my_project/Cla...</td>
      <td>Hieronymus Bosch</td>
      <td>18</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>/project/segmentation/smcho1201/my_project/Cla...</td>
      <td>Pierre-Auguste Renoir</td>
      <td>3</td>
    </tr>
    <tr>
      <th>16</th>
      <td>16</td>
      <td>/project/segmentation/smcho1201/my_project/Cla...</td>
      <td>Vincent van Gogh</td>
      <td>0</td>
    </tr>
    <tr>
      <th>31</th>
      <td>31</td>
      <td>/project/segmentation/smcho1201/my_project/Cla...</td>
      <td>Francisco Goya</td>
      <td>6</td>
    </tr>
    <tr>
      <th>39</th>
      <td>39</td>
      <td>/project/segmentation/smcho1201/my_project/Cla...</td>
      <td>Salvador Dali</td>
      <td>20</td>
    </tr>
  </tbody>
</table>
</div>



## Evaluation set (dataset)


```python
df_eval['img_path'] = [base_path + '/data/open/' + path[2:] for path in df_eval['img_path']]

print(f"Evaluation set 개수 : {len(df_eval)}")
df_eval.head()
```

    Evaluation set 개수 : 12670





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
      <th>img_path</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>TEST_00000</td>
      <td>/project/segmentation/smcho1201/my_project/Cla...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>TEST_00001</td>
      <td>/project/segmentation/smcho1201/my_project/Cla...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>TEST_00002</td>
      <td>/project/segmentation/smcho1201/my_project/Cla...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>TEST_00003</td>
      <td>/project/segmentation/smcho1201/my_project/Cla...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>TEST_00004</td>
      <td>/project/segmentation/smcho1201/my_project/Cla...</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_eval['img_path'][0]
```




    '/project/segmentation/smcho1201/my_project/Classification_of_paintings/data/open/test/TEST_00000.jpg'



# custom DataSet

## 데이터 확인

### Train dataset 확인


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

    Diego Velazquez
    25



    
![png](output_53_1.png)
    


    (1300, 1024, 3)
    Vincent van Gogh
    0



    
![png](output_53_3.png)
    


    (1024, 568, 3)
    Claude Monet
    38



    
![png](output_53_5.png)
    


    (722, 1024, 3)
    Edgar Degas
    1



    
![png](output_53_7.png)
    


    (836, 1053, 3)
    Rene Magritte
    12



    
![png](output_53_9.png)
    


    (584, 807, 3)


### Evaluation dataset 확인


```python
for i in range(0, 5):
    path = list(df_eval['img_path'])[i]
    ndarray = img.imread(path)

    plt.imshow(ndarray)
    plt.axis('off')
    plt.show()
    print(ndarray.shape)
```


    
![png](output_55_0.png)
    


    (1088, 788, 3)



    
![png](output_55_2.png)
    


    (403, 269, 3)



    
![png](output_55_4.png)
    


    (1227, 1012, 3)



    
![png](output_55_6.png)
    


    (440, 640, 3)



    
![png](output_55_8.png)
    


    (384, 400, 3)


## transform


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

- cv2 : HWC
- PIL : CHW
    - torch.tensor : CHW ( PIL 로 변환후 tensor 변환)


```python
# cv2 : H W C
cv2.imread(list(X_train['img_path'])[0]).shape
```




    (1300, 1024, 3)



## custom dataset


```python
import cv2
class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, transform=None):
        self.data = dataframe
        self.transform = transform   # 데이터 전처리
#         self.seed = seed
        

        self.lst_input = list(self.data['img_path'])
        self.lst_label = list(self.data['label'])

    def __len__(self):
        return len(self.lst_label)

    def __getitem__(self, index):

        label = self.lst_label[index] 
        img_input = cv2.imread(self.lst_input[index]) # H W C

#         if img_input.ndim == 2:
#             img_input = img_input[:, :, np.newaxis]

        if self.transform:
#             torch.manual_seed(self.seed)
            img_input = self.transform(img_input)

        return img_input, label
```


```python
# 데이터셋 클래스 적용
custom_dataset_train = Dataset(X_train, transform=transform_train)
print("My custom training-dataset has {} elements".format(len(custom_dataset_train)))

custom_dataset_val = Dataset(X_val, transform=transform_val)
print("My custom valing-dataset has {} elements".format(len(custom_dataset_val)))
```

    My custom training-dataset has 4255 elements
    My custom valing-dataset has 1183 elements


## 각 라벨마다 개수


```python
artist_name
```




    ['Vincent van Gogh',
     'Edgar Degas',
     'Pablo Picasso',
     'Pierre-Auguste Renoir',
     'Albrecht Du rer',
     'Paul Gauguin',
     'Francisco Goya',
     'Rembrandt',
     'Titian',
     'Marc Chagall',
     'Alfred Sisley',
     'Paul Klee',
     'Rene Magritte',
     'Amedeo Modigliani',
     'Andy Warhol',
     'Henri Matisse',
     'Sandro Botticelli',
     'Mikhail Vrubel',
     'Hieronymus Bosch',
     'Leonardo da Vinci',
     'Salvador Dali',
     'Peter Paul Rubens',
     'Kazimir Malevich',
     'Pieter Bruegel',
     'Frida Kahlo',
     'Diego Velazquez',
     'Joan Miro',
     'Andrei Rublev',
     'Raphael',
     'Giotto di Bondone',
     'Gustav Klimt',
     'El Greco',
     'Camille Pissarro',
     'Jan van Eyck',
     'Edouard Manet',
     'Henri de Toulouse-Lautrec',
     'Vasiliy Kandinskiy',
     'Piet Mondrian',
     'Claude Monet',
     'Henri Rousseau',
     'Diego Rivera',
     'Edvard Munch',
     'William Turner',
     'Gustave Courbet',
     'Michelangelo',
     'Paul Cezanne',
     'Caravaggio',
     'Georges Seurat',
     'Eugene Delacroix',
     'Jackson Pollock']




```python
num_artist = []

for name in artist_name:
    num = len(X_train.loc[X_train['artist'] == name])
    num_artist.append(num)
num_artist
```




    [457,
     359,
     217,
     160,
     166,
     155,
     142,
     125,
     135,
     130,
     116,
     105,
     104,
     98,
     91,
     83,
     79,
     86,
     82,
     73,
     73,
     63,
     66,
     65,
     50,
     55,
     52,
     61,
     54,
     52,
     50,
     44,
     42,
     50,
     38,
     47,
     43,
     44,
     40,
     42,
     36,
     32,
     35,
     31,
     24,
     22,
     23,
     22,
     18,
     18]



## 데이터로더 잘 구현되었는지 확인


```python
# 데이터로더 잘 구현되었는지 확인
# dataset_train = Dataset(dataframe = X_train, transform = transform_train)

img_input, label = custom_dataset_train.__getitem__(0) 
# input = data['input']
# label = data['label']

print(f"Input shape : {img_input.shape} \nLabel : {label}")

# # 불러온 이미지 시각화

# cv2.imshow('aa', img_input)
# cv2.waitKey(0) 
img_input_np = img_input.numpy()
img_input_np = img_input_np.transpose(1,2,0)
plt.imshow(img_input_np)
plt.title('input')
plt.show()


plt.hist(img_input.flatten(), bins=20)
plt.title('input')
plt.show()
```

    Input shape : torch.Size([3, 224, 224]) 
    Label : 25


    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).



    
![png](output_67_2.png)
    



    
![png](output_67_3.png)
    


## 가중 무작위 샘플링 (Weighted Random Sampling) 

### Choice


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
```


```python
weights_trian = make_weights(custom_dataset_train.lst_label, 50)
weights_trian = torch.DoubleTensor(weights_trian)

weights_val = make_weights(custom_dataset_val.lst_label, 50)
weights_val = torch.DoubleTensor(weights_val)

```

## Data Loader 생성


```python
sampler_train = torch.utils.data.sampler.WeightedRandomSampler(weights_trian, len(weights_trian))
sampler_val = torch.utils.data.sampler.WeightedRandomSampler(weights_val, len(weights_val))
```


```python
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


```python

```


```python

```

# Model

- [wide_resnet50_2], [vit_small_patch8_224], [vit_small_patch8_224_dino], [visformer_small]
- [mnasnet], [EfficientNet_b4], [resnet18]


```python
def mk_model(model_name, num_classes=50):
    if model_name == 'resnet18':
        from torchvision import models
        import torch

        resnet18_pretrained = models.resnet18(pretrained=True)
        num_ftrs = resnet18_pretrained.fc.in_features
        resnet18_pretrained.fc = nn.Linear(num_ftrs, num_classes)

        resnet18_pretrained.to(DEVICE)
        
        net = resnet18_pretrained
        
        
        
        
    if model_name == 'wide_resnet50_2':
        from torchvision import models
        import torch
        
        wide_resnet50_2 = models.wide_resnet50_2(pretrained=True)
        
        num_ftr = wide_resnet50_2.fc.in_features
        wide_resnet50_2.fc = nn.Linear(num_ftr, num_classes)
        
        wide_resnet50_2.to(DEVICE)
        
        net = wide_resnet50_2
        
        
        
        
    if model_name == 'EfficientNet_b4':
        from torchvision import models
        import torch
        import timm

        class EfficientNet_b4(nn.Module):
            def __init__(self, num_classes):
                super(EfficientNet_b4, self).__init__()
                self.efficient = timm.create_model(model_name='efficientnet_b4',
                    num_classes=num_classes,
                    pretrained=True
                    )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.efficient(x)
        EfficientNet_b4 = EfficientNet_b4(50)
        EfficientNet_b4.to(DEVICE)
        
        net = EfficientNet_b4
        
        
        
        
    if model_name == 'visformer_small':
        from torchvision import models
        import torch
        import timm

        visformer_small = timm.create_model('visformer_small', num_classes=num_classes, pretrained=True)

        visformer_small.to(DEVICE)

        net = visformer_small
        
        
        
        
    if model_name == 'vit_base_patch8_224':
        from torchvision import models
        import torch
        from pytorch_image_models.timm.models.vision_transformer import vit_base_patch8_224

        vit_base_patch8_224 = vit_base_patch8_224(pretrained=True)



        vit_base_patch8_224.to(DEVICE)

        num_ftr = vit_base_patch8_224.head.in_features
        vit_base_patch8_224.head = nn.Linear(num_ftr, num_classes)

        vit_base_patch8_224.to(DEVICE)

        net = vit_base_patch8_224
        
        
        
        
    if model_name == 'vit_small_patch8_224_dino':
        from torchvision import models
        import torch
        import timm

        vit_small_patch8_224_dino = timm.create_model('vit_small_patch8_224_dino', num_classes=num_classes, pretrained=True)

        vit_small_patch8_224_dino.to(DEVICE)

        net = vit_small_patch8_224_dino
        
    return net
```


```python

```


```python

```

# Loss Function

## 가중 손실 함수 (Choice)


```python
# 가중치 손실 함수 사용 여부
Use = True

if Use:
    num_artist # 각 artist당 그림 개수
    weights = [1 - (x/sum(num_artist)) for x in num_artist]
    class_weights = torch.FloatTensor(weights).to(DEVICE)
    class_weights = (class_weights*0.25)**2
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
else:
    criterion = nn.CrossEntropyLoss()
```

# Early Stopping


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

# Train

## Training function


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
```

## training


```python
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

    ##### wide_resnet50_2 ##### wide_resnet50_2 ##### wide_resnet50_2 ##### wide_resnet50_2 ##### wide_resnet50_2 ##### wide_resnet50_2 #####
    
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
    
    [ Train epoch: 3 ]
    
    Current batch: 0
    Current batch average train accuracy: 0.8
    Current batch average train loss: 0.08031809329986572
    
    Current batch: 100
    Current batch average train accuracy: 0.7
    Current batch average train loss: 0.11667827367782593
    
    Current batch: 200
    Current batch average train accuracy: 0.9
    Current batch average train loss: 0.03392101228237152
    
    Current batch: 300
    Current batch average train accuracy: 0.9
    Current batch average train loss: 0.05113908052444458
    
    Current batch: 400
    Current batch average train accuracy: 0.8
    Current batch average train loss: 0.0751347541809082
    
    Total average train accuarcy: 0.8124559341950647
    Total average train loss: 0.07431414920210698
    
    [ Test epoch: 3 ]
    
    Total average test accuarcy: 0.6855452240067624
    Total average test loss: 0.12458974260298357
    Validation loss decreased (79.540437 --> 71.609651).  Saving model ...
    
    [ Train epoch: 4 ]
    
    Current batch: 0
    Current batch average train accuracy: 0.9
    Current batch average train loss: 0.035314160585403445
    
    Current batch: 100
    Current batch average train accuracy: 0.9
    Current batch average train loss: 0.0584869921207428
    
    Current batch: 200
    Current batch average train accuracy: 0.8
    Current batch average train loss: 0.07751970887184143
    
    Current batch: 300
    Current batch average train accuracy: 0.8
    Current batch average train loss: 0.07503929734230042
    
    Current batch: 400
    Current batch average train accuracy: 0.9
    Current batch average train loss: 0.07168535590171814
    
    Total average train accuarcy: 0.899412455934195
    Total average train loss: 0.043144246739730434
    
    [ Test epoch: 4 ]
    
    Total average test accuarcy: 0.6846999154691462
    Total average test loss: 0.11776891399241059
    Validation loss decreased (71.609651 --> 68.971296).  Saving model ...
    
    [ Train epoch: 5 ]
    
    Current batch: 0
    Current batch average train accuracy: 1.0
    Current batch average train loss: 0.01839989423751831
    
    Current batch: 100
    Current batch average train accuracy: 1.0
    Current batch average train loss: 0.01448403149843216
    
    Current batch: 200
    Current batch average train accuracy: 1.0
    Current batch average train loss: 0.008771134167909622
    
    Current batch: 300
    Current batch average train accuracy: 1.0
    Current batch average train loss: 0.02215784788131714
    
    Current batch: 400
    Current batch average train accuracy: 0.9
    Current batch average train loss: 0.022818927466869355
    
    Total average train accuarcy: 0.9407755581668625
    Total average train loss: 0.025357265622474333
    
    [ Test epoch: 5 ]
    
    Total average test accuarcy: 0.7049873203719358
    Total average test loss: 0.12210358198754181
    EarlyStopping counter: 1 out of 7
    
    [ Train epoch: 6 ]
    
    Current batch: 0
    Current batch average train accuracy: 1.0
    Current batch average train loss: 0.005834012851119041
    
    Current batch: 100
    Current batch average train accuracy: 0.9
    Current batch average train loss: 0.052578520774841306
    
    Current batch: 200
    Current batch average train accuracy: 1.0
    Current batch average train loss: 0.008138114213943481
    
    Current batch: 300
    Current batch average train accuracy: 1.0
    Current batch average train loss: 0.003540213406085968
    
    Current batch: 400
    Current batch average train accuracy: 1.0
    Current batch average train loss: 0.009987757354974747
    
    Total average train accuarcy: 0.9687426556991774
    Total average train loss: 0.015824957134269974
    
    [ Test epoch: 6 ]
    
    Total average test accuarcy: 0.7075232459847844
    Total average test loss: 0.1263978029030516
    EarlyStopping counter: 2 out of 7
    
    [ Train epoch: 7 ]
    
    Current batch: 0
    Current batch average train accuracy: 1.0
    Current batch average train loss: 0.0122738316655159
    
    Current batch: 100
    Current batch average train accuracy: 0.9
    Current batch average train loss: 0.022086846828460693
    
    Current batch: 200
    Current batch average train accuracy: 0.9
    Current batch average train loss: 0.04830687642097473
    
    Current batch: 300
    Current batch average train accuracy: 1.0
    Current batch average train loss: 0.021113552153110504
    
    Current batch: 400
    Current batch average train accuracy: 1.0
    Current batch average train loss: 0.0100735105574131
    
    Total average train accuarcy: 0.9522914218566393
    Total average train loss: 0.020046860073527843
    
    [ Test epoch: 7 ]
    
    Total average test accuarcy: 0.6677937447168216
    Total average test loss: 0.14606230006894114
    EarlyStopping counter: 3 out of 7
    
    [ Train epoch: 8 ]
    
    Current batch: 0
    Current batch average train accuracy: 0.9
    Current batch average train loss: 0.048132511973381045
    
    Current batch: 100
    Current batch average train accuracy: 1.0
    Current batch average train loss: 0.022430966794490814
    
    Current batch: 200
    Current batch average train accuracy: 1.0
    Current batch average train loss: 0.007398919761180877
    
    Current batch: 300
    Current batch average train accuracy: 1.0
    Current batch average train loss: 0.002246011421084404
    
    Current batch: 400
    Current batch average train accuracy: 1.0
    Current batch average train loss: 0.00789462924003601
    
    Total average train accuarcy: 0.9417156286721504
    Total average train loss: 0.021679584561988326
    
    [ Test epoch: 8 ]
    
    Total average test accuarcy: 0.6694843617920541
    Total average test loss: 0.14282638389091318
    EarlyStopping counter: 4 out of 7
    
    [ Train epoch: 9 ]
    
    Current batch: 0
    Current batch average train accuracy: 0.9
    Current batch average train loss: 0.024816781282424927
    
    Current batch: 100
    Current batch average train accuracy: 1.0
    Current batch average train loss: 0.010340771079063416
    
    Current batch: 200
    Current batch average train accuracy: 1.0
    Current batch average train loss: 0.012292558699846268
    
    Current batch: 300
    Current batch average train accuracy: 1.0
    Current batch average train loss: 0.004036666452884674
    
    Current batch: 400
    Current batch average train accuracy: 1.0
    Current batch average train loss: 0.004669356718659401
    
    Total average train accuarcy: 0.9576968272620446
    Total average train loss: 0.017115938725862673
    
    [ Test epoch: 9 ]
    
    Total average test accuarcy: 0.6821639898562976
    Total average test loss: 0.14327239659587307
    EarlyStopping counter: 5 out of 7
    
    [ Train epoch: 10 ]
    
    Current batch: 0
    Current batch average train accuracy: 1.0
    Current batch average train loss: 0.0006041845306754112
    
    Current batch: 100
    Current batch average train accuracy: 0.9
    Current batch average train loss: 0.034868082404136656
    
    Current batch: 200
    Current batch average train accuracy: 0.9
    Current batch average train loss: 0.019551892578601838
    
    Current batch: 300
    Current batch average train accuracy: 1.0
    Current batch average train loss: 0.003930402547121048
    
    Current batch: 400
    Current batch average train accuracy: 1.0
    Current batch average train loss: 0.0003784839995205402
    
    Total average train accuarcy: 0.9692126909518214
    Total average train loss: 0.012606453303232368
    
    [ Test epoch: 10 ]
    
    Total average test accuarcy: 0.6931530008453085
    Total average test loss: 0.13341308500387863
    EarlyStopping counter: 6 out of 7
    
    [ Train epoch: 11 ]
    
    Current batch: 0
    Current batch average train accuracy: 1.0
    Current batch average train loss: 0.006071718037128448
    
    Current batch: 100
    Current batch average train accuracy: 0.9
    Current batch average train loss: 0.014630226790904999
    
    Current batch: 200
    Current batch average train accuracy: 1.0
    Current batch average train loss: 0.005484155938029289
    
    Current batch: 300
    Current batch average train accuracy: 1.0
    Current batch average train loss: 0.0022720765322446824
    
    Current batch: 400
    Current batch average train accuracy: 1.0
    Current batch average train loss: 0.004606072604656219
    
    Total average train accuarcy: 0.9699177438307873
    Total average train loss: 0.011917608937294552
    
    [ Test epoch: 11 ]
    
    Total average test accuarcy: 0.6931530008453085
    Total average test loss: 0.13351807541143865
    EarlyStopping counter: 7 out of 7
    Early stopping
    #######################################################################################################################
    #######################################################################################################################
    #######################################################################################################################
    #######################################################################################################################
    #######################################################################################################################
    #END# wide_resnet50_2 #END# wide_resnet50_2 #END# wide_resnet50_2 #END# wide_resnet50_2 #END# wide_resnet50_2 #END# wide_resnet50_2 #END#


# Test


```python
for model_name in models:
    model = mk_model(model_name)
    model.load_state_dict(torch.load(f'checkpoints/final/ckpt_{model_name}.pth'))

    size = (224,224)
    
    # 예측한 값 list
    pred_li = []


    for i in range(len(X_test)):
        test_images = list(X_test['img_path'])
        path_img = test_images[i]

        # eval 전 이미지 전처리
        img = cv2.imread(path_img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # 양성형 이웃 보간 (2x2 픽셀 참조하여 보간함.)
        img = cv2.resize(img, size, interpolation = cv2.INTER_LINEAR)

        # eval
        model.eval()
        eval_image = img / 255.0
        eval_image = eval_image.astype(np.float32)
        eval_image = eval_image.transpose((2,0,1))
        eval_image = torch.from_numpy(eval_image).unsqueeze(0) # Batch 채널 추가 -> (1, 3, 256, 256)
        eval_image = eval_image.to( device=DEVICE, dtype = torch.float32 )

        with torch.no_grad():
            pred = model(eval_image)


        pred = pred.cpu().numpy()
        pred = np.argmax(pred)
        pred_li.append(pred)
    
    
    
    
    #####################################
    # f1-score
    from sklearn.metrics import f1_score
    
    y_true = list(X_test['label'])
    y_pred = pred_li
    print("##########################################################################################################")
    print(f"########## {model_name} ########## {model_name} ########## {model_name} ########## {model_name} ##########")
    print(f"정답 라벨 개수 : {len(y_true)}\t예측 라벨 개수 : {len(y_pred)}")
    score = f1_score(y_true, y_pred, average='macro')
    print(f"f1 score (macro) : {score}")
```

    ##########################################################################################################
    ########## wide_resnet50_2 ########## wide_resnet50_2 ########## wide_resnet50_2 ########## wide_resnet50_2 ##########
    정답 라벨 개수 : 473	예측 라벨 개수 : 473
    f1 score (macro) : 0.10945009206236594


# Evaluation


```python
for model_name in models:
    model = mk_model(model_name)
    model.load_state_dict(torch.load(f'checkpoints/final/ckpt_{model_name}.pth'))

    size = (224,224)

    pred_li = []
    for i in range(len(df_eval)):
        test_images = list(df_eval['img_path'])
        path_img = test_images[i]

        # eval 전 이미지 전처리
        img = cv2.imread(path_img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # 양성형 이웃 보간 (2x2 픽셀 참조하여 보간함.)
        img = cv2.resize(img, size, interpolation = cv2.INTER_LINEAR)

        # eval
        model.eval()
        eval_image = img / 255.0
        eval_image = eval_image.astype(np.float32)
        eval_image = eval_image.transpose((2,0,1))
        eval_image = torch.from_numpy(eval_image).unsqueeze(0) # Batch 채널 추가 -> (1, 3, 256, 256)
        eval_image = eval_image.to( device=DEVICE, dtype = torch.float32 )

        with torch.no_grad():
            pred = model(eval_image)


        pred = pred.cpu().numpy()
        pred = np.argmax(pred)
        pred_li.append(pred)
    
    
    # label -> artist
    pred_artist_li = []
    for idx in pred_li:
        pred_artist_li.append(artist_name[idx])
        
        
    #  Evaluation dataframe 생성
    complete = pd.read_csv("./data/open/sample_submission.csv")
    complete['artist'] = pred_artist_li
    
    
    # csv 파일 추출
    complete.to_csv(f'/project/segmentation/smcho1201/my_project/Classification_of_paintings/data/open/final/csv_{model_name}.csv', index=False)
```


```python

```


```python

```
