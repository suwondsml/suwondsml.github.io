---
layout: post
title: Movie-prediction
authors: [Taeyoung Lee]
categories: [1기 AI/SW developers(개인 프로젝트)]
---


# 영화 관객수 예측 모델

## 라이브러리 및 Data set 불러오기


```python
pip install lightgbm
```

    Defaulting to user installation because normal site-packages is not writeable
    Requirement already satisfied: lightgbm in c:\users\user\appdata\roaming\python\python39\site-packages (3.3.3)
    Requirement already satisfied: wheel in c:\programdata\anaconda3\lib\site-packages (from lightgbm) (0.37.1)
    Requirement already satisfied: scikit-learn!=0.22.0 in c:\programdata\anaconda3\lib\site-packages (from lightgbm) (1.0.2)
    Requirement already satisfied: scipy in c:\programdata\anaconda3\lib\site-packages (from lightgbm) (1.7.3)
    Requirement already satisfied: numpy in c:\programdata\anaconda3\lib\site-packages (from lightgbm) (1.21.5)
    Requirement already satisfied: threadpoolctl>=2.0.0 in c:\programdata\anaconda3\lib\site-packages (from scikit-learn!=0.22.0->lightgbm) (2.2.0)
    Requirement already satisfied: joblib>=0.11 in c:\programdata\anaconda3\lib\site-packages (from scikit-learn!=0.22.0->lightgbm) (1.1.0)
    Note: you may need to restart the kernel to use updated packages.
    


```python
pip install scikit-learn
```

    Defaulting to user installation because normal site-packages is not writeable
    Requirement already satisfied: scikit-learn in c:\programdata\anaconda3\lib\site-packages (1.0.2)
    Requirement already satisfied: numpy>=1.14.6 in c:\programdata\anaconda3\lib\site-packages (from scikit-learn) (1.21.5)
    Requirement already satisfied: scipy>=1.1.0 in c:\programdata\anaconda3\lib\site-packages (from scikit-learn) (1.7.3)
    Requirement already satisfied: threadpoolctl>=2.0.0 in c:\programdata\anaconda3\lib\site-packages (from scikit-learn) (2.2.0)
    Requirement already satisfied: joblib>=0.11 in c:\programdata\anaconda3\lib\site-packages (from scikit-learn) (1.1.0)
    Note: you may need to restart the kernel to use updated packages.
    


```python
import pandas as pd
import lightgbm as lgb
import seaborn as sns
import matplotlib.pyplot as plt
import re
%matplotlib inline
```


```python
# 데이터 불러오기
train = pd.read_csv("C:/Users/user/Desktop/영화 관객 예측 모델/movies_train.csv")
test = pd.read_csv("C:/Users/user/Desktop/영화 관객 예측 모델/movies_test.csv")
submission = pd.read_csv("C:/Users/user/Desktop/영화 관객 예측 모델/submission.csv")
```


```python
import platform

from matplotlib import font_manager, rc
plt.rcParams['axes.unicode_minus'] = False

if platform.system() == 'Darwin':
    rc('font', family='AppleGothic')
    print('Mac version')
elif platform.system() == 'Windows':
    path = "c:/Windows/Fonts/malgun.ttf"
    font_name = font_manager.FontProperties(fname=path).get_name()
    rc('font', family=font_name)
    print('Windows version')
```

    Windows version
    

## EDA

+ title : 영화의 제목
+ distributor : 배급사
+ genre : 장르
+ release_time : 개봉일
+ time : 상영시간(분)
+ screening_rat : 상영등급
+ director : 감독이름
+ dir_prev_bfnum : 해당 감독이 이 영화를 만들기 전 제작에 참여한 영화에서의 평균 관객수(단 관객수가 알려지지 않은 영화 제외)
+ dir_prev_num : 해당 감독이 이 영화를 만들기 전 제작에 참여한 영화의 개수(단 관객수가 알려지지 않은 영화 제외)
+ num_staff : 스텝수
+ num_actor : 주연배우수
+ box_off_num : 관객수


```python
train.head()
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
      <th>title</th>
      <th>distributor</th>
      <th>genre</th>
      <th>release_time</th>
      <th>time</th>
      <th>screening_rat</th>
      <th>director</th>
      <th>dir_prev_bfnum</th>
      <th>dir_prev_num</th>
      <th>num_staff</th>
      <th>num_actor</th>
      <th>box_off_num</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>개들의 전쟁</td>
      <td>롯데엔터테인먼트</td>
      <td>액션</td>
      <td>2012-11-22</td>
      <td>96</td>
      <td>청소년 관람불가</td>
      <td>조병옥</td>
      <td>NaN</td>
      <td>0</td>
      <td>91</td>
      <td>2</td>
      <td>23398</td>
    </tr>
    <tr>
      <th>1</th>
      <td>내부자들</td>
      <td>(주)쇼박스</td>
      <td>느와르</td>
      <td>2015-11-19</td>
      <td>130</td>
      <td>청소년 관람불가</td>
      <td>우민호</td>
      <td>1161602.50</td>
      <td>2</td>
      <td>387</td>
      <td>3</td>
      <td>7072501</td>
    </tr>
    <tr>
      <th>2</th>
      <td>은밀하게 위대하게</td>
      <td>(주)쇼박스</td>
      <td>액션</td>
      <td>2013-06-05</td>
      <td>123</td>
      <td>15세 관람가</td>
      <td>장철수</td>
      <td>220775.25</td>
      <td>4</td>
      <td>343</td>
      <td>4</td>
      <td>6959083</td>
    </tr>
    <tr>
      <th>3</th>
      <td>나는 공무원이다</td>
      <td>(주)NEW</td>
      <td>코미디</td>
      <td>2012-07-12</td>
      <td>101</td>
      <td>전체 관람가</td>
      <td>구자홍</td>
      <td>23894.00</td>
      <td>2</td>
      <td>20</td>
      <td>6</td>
      <td>217866</td>
    </tr>
    <tr>
      <th>4</th>
      <td>불량남녀</td>
      <td>쇼박스(주)미디어플렉스</td>
      <td>코미디</td>
      <td>2010-11-04</td>
      <td>108</td>
      <td>15세 관람가</td>
      <td>신근호</td>
      <td>1.00</td>
      <td>1</td>
      <td>251</td>
      <td>2</td>
      <td>483387</td>
    </tr>
  </tbody>
</table>
</div>




```python
movie_best = train.sort_values(ascending = False,by = 'box_off_num').head(10)
```


```python
movie_best
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
      <th>title</th>
      <th>distributor</th>
      <th>genre</th>
      <th>release_time</th>
      <th>time</th>
      <th>screening_rat</th>
      <th>director</th>
      <th>dir_prev_bfnum</th>
      <th>dir_prev_num</th>
      <th>num_staff</th>
      <th>num_actor</th>
      <th>box_off_num</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>372</th>
      <td>국제시장</td>
      <td>CJ 엔터테인먼트</td>
      <td>드라마</td>
      <td>2014-12-17</td>
      <td>126</td>
      <td>12세 관람가</td>
      <td>윤제균</td>
      <td>NaN</td>
      <td>0</td>
      <td>869</td>
      <td>4</td>
      <td>14262766</td>
    </tr>
    <tr>
      <th>362</th>
      <td>도둑들</td>
      <td>(주)쇼박스</td>
      <td>느와르</td>
      <td>2012-07-25</td>
      <td>135</td>
      <td>15세 관람가</td>
      <td>최동훈</td>
      <td>2.564692e+06</td>
      <td>3</td>
      <td>462</td>
      <td>10</td>
      <td>12983841</td>
    </tr>
    <tr>
      <th>530</th>
      <td>7번방의 선물</td>
      <td>(주)NEW</td>
      <td>코미디</td>
      <td>2013-01-23</td>
      <td>127</td>
      <td>15세 관람가</td>
      <td>이환경</td>
      <td>8.190495e+05</td>
      <td>2</td>
      <td>300</td>
      <td>8</td>
      <td>12811435</td>
    </tr>
    <tr>
      <th>498</th>
      <td>암살</td>
      <td>(주)쇼박스</td>
      <td>액션</td>
      <td>2015-07-22</td>
      <td>139</td>
      <td>15세 관람가</td>
      <td>최동훈</td>
      <td>5.169479e+06</td>
      <td>4</td>
      <td>628</td>
      <td>3</td>
      <td>12706663</td>
    </tr>
    <tr>
      <th>460</th>
      <td>광해, 왕이 된 남자</td>
      <td>CJ 엔터테인먼트</td>
      <td>드라마</td>
      <td>2012-09-13</td>
      <td>131</td>
      <td>15세 관람가</td>
      <td>추창민</td>
      <td>1.552541e+06</td>
      <td>2</td>
      <td>402</td>
      <td>3</td>
      <td>12323595</td>
    </tr>
    <tr>
      <th>122</th>
      <td>변호인</td>
      <td>(주)NEW</td>
      <td>드라마</td>
      <td>2013-12-18</td>
      <td>127</td>
      <td>15세 관람가</td>
      <td>양우석</td>
      <td>NaN</td>
      <td>0</td>
      <td>311</td>
      <td>5</td>
      <td>11374879</td>
    </tr>
    <tr>
      <th>496</th>
      <td>설국열차</td>
      <td>CJ 엔터테인먼트</td>
      <td>SF</td>
      <td>2013-08-01</td>
      <td>125</td>
      <td>15세 관람가</td>
      <td>봉준호</td>
      <td>NaN</td>
      <td>0</td>
      <td>67</td>
      <td>10</td>
      <td>9350351</td>
    </tr>
    <tr>
      <th>101</th>
      <td>관상</td>
      <td>(주)쇼박스</td>
      <td>드라마</td>
      <td>2013-09-11</td>
      <td>139</td>
      <td>15세 관람가</td>
      <td>한재림</td>
      <td>1.242778e+06</td>
      <td>2</td>
      <td>298</td>
      <td>6</td>
      <td>9135806</td>
    </tr>
    <tr>
      <th>505</th>
      <td>해적: 바다로 간 산적</td>
      <td>롯데엔터테인먼트</td>
      <td>SF</td>
      <td>2014-08-06</td>
      <td>130</td>
      <td>12세 관람가</td>
      <td>이석훈</td>
      <td>1.843895e+06</td>
      <td>3</td>
      <td>868</td>
      <td>2</td>
      <td>8666208</td>
    </tr>
    <tr>
      <th>476</th>
      <td>수상한 그녀</td>
      <td>CJ 엔터테인먼트</td>
      <td>코미디</td>
      <td>2014-01-22</td>
      <td>124</td>
      <td>15세 관람가</td>
      <td>황동혁</td>
      <td>2.781990e+06</td>
      <td>2</td>
      <td>437</td>
      <td>5</td>
      <td>8659725</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=[10,10])
sns.scatterplot(data=train, x='num_staff', y = 'box_off_num')
```




    <AxesSubplot:xlabel='num_staff', ylabel='box_off_num'>




    
![output_12_1](https://user-images.githubusercontent.com/113446739/216802934-7aaaece0-d0f4-4717-98b3-8f7857803dcf.png)
    


스탭수가 많으면 관객수가 조금 많은거 같다.


```python
plt.figure(figsize=[10,10])
sns.scatterplot(data=train, x='time', y = 'box_off_num')
```




    <AxesSubplot:xlabel='time', ylabel='box_off_num'>




    
![output_14_1](https://user-images.githubusercontent.com/113446739/216802936-d975d316-b184-4a07-a8e9-f194a804fc1d.png)
    


상영시간 120~140분 사이에 관객수가 많은걸 볼수있다


```python
plt.figure(figsize=[10,10])
sns.scatterplot(data=train, x='genre', y = 'box_off_num')
```




    <AxesSubplot:xlabel='genre', ylabel='box_off_num'>




    
![output_16_1](https://user-images.githubusercontent.com/113446739/216802938-660c5d72-3c64-4394-9728-f44c6d80806b.png)
    


장르와 관객수의 그래프이다.


```python
plt.figure(figsize=[10,10])
sns.scatterplot(data=train, x='num_actor', y = 'box_off_num')
```




    <AxesSubplot:xlabel='num_actor', ylabel='box_off_num'>




    
![output_18_1](https://user-images.githubusercontent.com/113446739/216802939-adaae5d9-5e12-410b-a757-728eaa52d154.png)
    


주연 배우와 관객수 그래프이다.


```python
train[['title','box_off_num']].head()
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
      <th>title</th>
      <th>box_off_num</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>개들의 전쟁</td>
      <td>23398</td>
    </tr>
    <tr>
      <th>1</th>
      <td>내부자들</td>
      <td>7072501</td>
    </tr>
    <tr>
      <th>2</th>
      <td>은밀하게 위대하게</td>
      <td>6959083</td>
    </tr>
    <tr>
      <th>3</th>
      <td>나는 공무원이다</td>
      <td>217866</td>
    </tr>
    <tr>
      <th>4</th>
      <td>불량남녀</td>
      <td>483387</td>
    </tr>
  </tbody>
</table>
</div>




```python
test.head()
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
      <th>title</th>
      <th>distributor</th>
      <th>genre</th>
      <th>release_time</th>
      <th>time</th>
      <th>screening_rat</th>
      <th>director</th>
      <th>dir_prev_bfnum</th>
      <th>dir_prev_num</th>
      <th>num_staff</th>
      <th>num_actor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>용서는 없다</td>
      <td>시네마서비스</td>
      <td>느와르</td>
      <td>2010-01-07</td>
      <td>125</td>
      <td>청소년 관람불가</td>
      <td>김형준</td>
      <td>3.005290e+05</td>
      <td>2</td>
      <td>304</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>아빠가 여자를 좋아해</td>
      <td>(주)쇼박스</td>
      <td>멜로/로맨스</td>
      <td>2010-01-14</td>
      <td>113</td>
      <td>12세 관람가</td>
      <td>이광재</td>
      <td>3.427002e+05</td>
      <td>4</td>
      <td>275</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>하모니</td>
      <td>CJ 엔터테인먼트</td>
      <td>드라마</td>
      <td>2010-01-28</td>
      <td>115</td>
      <td>12세 관람가</td>
      <td>강대규</td>
      <td>4.206611e+06</td>
      <td>3</td>
      <td>419</td>
      <td>7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>의형제</td>
      <td>(주)쇼박스</td>
      <td>액션</td>
      <td>2010-02-04</td>
      <td>116</td>
      <td>15세 관람가</td>
      <td>장훈</td>
      <td>6.913420e+05</td>
      <td>2</td>
      <td>408</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>평행 이론</td>
      <td>CJ 엔터테인먼트</td>
      <td>공포</td>
      <td>2010-02-18</td>
      <td>110</td>
      <td>15세 관람가</td>
      <td>권호영</td>
      <td>3.173800e+04</td>
      <td>1</td>
      <td>380</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
submission.head()
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
      <th>title</th>
      <th>box_off_num</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>용서는 없다</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>아빠가 여자를 좋아해</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>하모니</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>의형제</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>평행 이론</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(train.shape)
print(test.shape)
print(submission.shape)
```

    (600, 12)
    (243, 11)
    (243, 2)
    

### 데이터 형태 파악


```python
train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 600 entries, 0 to 599
    Data columns (total 12 columns):
     #   Column          Non-Null Count  Dtype  
    ---  ------          --------------  -----  
     0   title           600 non-null    object 
     1   distributor     600 non-null    object 
     2   genre           600 non-null    object 
     3   release_time    600 non-null    object 
     4   time            600 non-null    int64  
     5   screening_rat   600 non-null    object 
     6   director        600 non-null    object 
     7   dir_prev_bfnum  270 non-null    float64
     8   dir_prev_num    600 non-null    int64  
     9   num_staff       600 non-null    int64  
     10  num_actor       600 non-null    int64  
     11  box_off_num     600 non-null    int64  
    dtypes: float64(1), int64(5), object(6)
    memory usage: 56.4+ KB
    

위 train데이터를 보면 dir_prev_dfnum 330개는 null 값임을 알 수 있다.


```python
test.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 243 entries, 0 to 242
    Data columns (total 11 columns):
     #   Column          Non-Null Count  Dtype  
    ---  ------          --------------  -----  
     0   title           243 non-null    object 
     1   distributor     243 non-null    object 
     2   genre           243 non-null    object 
     3   release_time    243 non-null    object 
     4   time            243 non-null    int64  
     5   screening_rat   243 non-null    object 
     6   director        243 non-null    object 
     7   dir_prev_bfnum  107 non-null    float64
     8   dir_prev_num    243 non-null    int64  
     9   num_staff       243 non-null    int64  
     10  num_actor       243 non-null    int64  
    dtypes: float64(1), int64(4), object(6)
    memory usage: 21.0+ KB
    

위 test데이터도 보면 dir_prev_dfnum은 107개가 채워져 있고 나머지 채워지지 않았다는것을 알수 있다.

## groupby함수로 그룹을 만들기


```python
train[['genre', 'box_off_num']].groupby('genre').mean().sort_values('box_off_num')
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
      <th>box_off_num</th>
    </tr>
    <tr>
      <th>genre</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>뮤지컬</th>
      <td>6.627000e+03</td>
    </tr>
    <tr>
      <th>다큐멘터리</th>
      <td>6.717226e+04</td>
    </tr>
    <tr>
      <th>서스펜스</th>
      <td>8.261100e+04</td>
    </tr>
    <tr>
      <th>애니메이션</th>
      <td>1.819267e+05</td>
    </tr>
    <tr>
      <th>멜로/로맨스</th>
      <td>4.259680e+05</td>
    </tr>
    <tr>
      <th>미스터리</th>
      <td>5.275482e+05</td>
    </tr>
    <tr>
      <th>공포</th>
      <td>5.908325e+05</td>
    </tr>
    <tr>
      <th>드라마</th>
      <td>6.256898e+05</td>
    </tr>
    <tr>
      <th>코미디</th>
      <td>1.193914e+06</td>
    </tr>
    <tr>
      <th>SF</th>
      <td>1.788346e+06</td>
    </tr>
    <tr>
      <th>액션</th>
      <td>2.203974e+06</td>
    </tr>
    <tr>
      <th>느와르</th>
      <td>2.263695e+06</td>
    </tr>
  </tbody>
</table>
</div>



genre,box_off_num를 genre로 그룹바이한후 mean으로 평균을 산출해서 sort_values로 정렬해준다.

액션 느와르가 가장 인기가 좋은것을 확인


```python
train.describe()
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
      <th>time</th>
      <th>dir_prev_bfnum</th>
      <th>dir_prev_num</th>
      <th>num_staff</th>
      <th>num_actor</th>
      <th>box_off_num</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>600.000000</td>
      <td>2.700000e+02</td>
      <td>600.000000</td>
      <td>600.000000</td>
      <td>600.000000</td>
      <td>6.000000e+02</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>100.863333</td>
      <td>1.050443e+06</td>
      <td>0.876667</td>
      <td>151.118333</td>
      <td>3.706667</td>
      <td>7.081818e+05</td>
    </tr>
    <tr>
      <th>std</th>
      <td>18.097528</td>
      <td>1.791408e+06</td>
      <td>1.183409</td>
      <td>165.654671</td>
      <td>2.446889</td>
      <td>1.828006e+06</td>
    </tr>
    <tr>
      <th>min</th>
      <td>45.000000</td>
      <td>1.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>89.000000</td>
      <td>2.038000e+04</td>
      <td>0.000000</td>
      <td>17.000000</td>
      <td>2.000000</td>
      <td>1.297250e+03</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>100.000000</td>
      <td>4.784236e+05</td>
      <td>0.000000</td>
      <td>82.500000</td>
      <td>3.000000</td>
      <td>1.259100e+04</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>114.000000</td>
      <td>1.286569e+06</td>
      <td>2.000000</td>
      <td>264.000000</td>
      <td>4.000000</td>
      <td>4.798868e+05</td>
    </tr>
    <tr>
      <th>max</th>
      <td>180.000000</td>
      <td>1.761531e+07</td>
      <td>5.000000</td>
      <td>869.000000</td>
      <td>25.000000</td>
      <td>1.426277e+07</td>
    </tr>
  </tbody>
</table>
</div>



### corr함수 상관계수를 확인


```python
train.corr()
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
      <th>time</th>
      <th>dir_prev_bfnum</th>
      <th>dir_prev_num</th>
      <th>num_staff</th>
      <th>num_actor</th>
      <th>box_off_num</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>time</th>
      <td>1.000000</td>
      <td>0.264675</td>
      <td>0.306727</td>
      <td>0.623205</td>
      <td>0.114153</td>
      <td>0.441452</td>
    </tr>
    <tr>
      <th>dir_prev_bfnum</th>
      <td>0.264675</td>
      <td>1.000000</td>
      <td>0.131822</td>
      <td>0.323521</td>
      <td>0.083818</td>
      <td>0.283184</td>
    </tr>
    <tr>
      <th>dir_prev_num</th>
      <td>0.306727</td>
      <td>0.131822</td>
      <td>1.000000</td>
      <td>0.450706</td>
      <td>0.014006</td>
      <td>0.259674</td>
    </tr>
    <tr>
      <th>num_staff</th>
      <td>0.623205</td>
      <td>0.323521</td>
      <td>0.450706</td>
      <td>1.000000</td>
      <td>0.077871</td>
      <td>0.544265</td>
    </tr>
    <tr>
      <th>num_actor</th>
      <td>0.114153</td>
      <td>0.083818</td>
      <td>0.014006</td>
      <td>0.077871</td>
      <td>1.000000</td>
      <td>0.111179</td>
    </tr>
    <tr>
      <th>box_off_num</th>
      <td>0.441452</td>
      <td>0.283184</td>
      <td>0.259674</td>
      <td>0.544265</td>
      <td>0.111179</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



## 위 상관계수 시각적으로 보기


```python
train[['genre','num_staff','box_off_num']].groupby('genre').mean().sort_values('box_off_num')
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
      <th>num_staff</th>
      <th>box_off_num</th>
    </tr>
    <tr>
      <th>genre</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>뮤지컬</th>
      <td>2.200000</td>
      <td>6.627000e+03</td>
    </tr>
    <tr>
      <th>다큐멘터리</th>
      <td>17.849462</td>
      <td>6.717226e+04</td>
    </tr>
    <tr>
      <th>서스펜스</th>
      <td>111.000000</td>
      <td>8.261100e+04</td>
    </tr>
    <tr>
      <th>애니메이션</th>
      <td>44.619048</td>
      <td>1.819267e+05</td>
    </tr>
    <tr>
      <th>멜로/로맨스</th>
      <td>135.782051</td>
      <td>4.259680e+05</td>
    </tr>
    <tr>
      <th>미스터리</th>
      <td>117.352941</td>
      <td>5.275482e+05</td>
    </tr>
    <tr>
      <th>공포</th>
      <td>176.380952</td>
      <td>5.908325e+05</td>
    </tr>
    <tr>
      <th>드라마</th>
      <td>164.484163</td>
      <td>6.256898e+05</td>
    </tr>
    <tr>
      <th>코미디</th>
      <td>209.075472</td>
      <td>1.193914e+06</td>
    </tr>
    <tr>
      <th>SF</th>
      <td>197.307692</td>
      <td>1.788346e+06</td>
    </tr>
    <tr>
      <th>액션</th>
      <td>337.535714</td>
      <td>2.203974e+06</td>
    </tr>
    <tr>
      <th>느와르</th>
      <td>311.074074</td>
      <td>2.263695e+06</td>
    </tr>
  </tbody>
</table>
</div>




```python
train_genre = train[['genre', 'box_off_num']].groupby('genre').mean().sort_values('box_off_num')
```


```python
train_genre
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
      <th>box_off_num</th>
    </tr>
    <tr>
      <th>genre</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>뮤지컬</th>
      <td>6.627000e+03</td>
    </tr>
    <tr>
      <th>다큐멘터리</th>
      <td>6.717226e+04</td>
    </tr>
    <tr>
      <th>서스펜스</th>
      <td>8.261100e+04</td>
    </tr>
    <tr>
      <th>애니메이션</th>
      <td>1.819267e+05</td>
    </tr>
    <tr>
      <th>멜로/로맨스</th>
      <td>4.259680e+05</td>
    </tr>
    <tr>
      <th>미스터리</th>
      <td>5.275482e+05</td>
    </tr>
    <tr>
      <th>공포</th>
      <td>5.908325e+05</td>
    </tr>
    <tr>
      <th>드라마</th>
      <td>6.256898e+05</td>
    </tr>
    <tr>
      <th>코미디</th>
      <td>1.193914e+06</td>
    </tr>
    <tr>
      <th>SF</th>
      <td>1.788346e+06</td>
    </tr>
    <tr>
      <th>액션</th>
      <td>2.203974e+06</td>
    </tr>
    <tr>
      <th>느와르</th>
      <td>2.263695e+06</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 히스토그램
import matplotlib.pyplot as plt
%matplotlib inline

train.hist(bins = 50, figsize = (20,15))
plt.show()
```


    
![output_40_0](https://user-images.githubusercontent.com/113446739/216802941-8550df6c-5422-4d2e-817a-a0fd435094c9.png)
    



```python
sns.heatmap(train.corr(), annot = True)
```




    <AxesSubplot:>




    
![output_41_1](https://user-images.githubusercontent.com/113446739/216802942-15595a5f-eecd-4bc3-be98-28474540bfd5.png)
    


### 배급사와 관객수 관계

유명한 배급사는 광고를 많이해서 관객수에 영향을 줄거라는 생각


```python
train['distributor'] = train.distributor.str.replace('(주)', '')
test['distributor'] = test.distributor.str.replace('(주)', '')
```

    C:\Users\user\AppData\Local\Temp\ipykernel_14156\3846023154.py:1: FutureWarning: The default value of regex will change from True to False in a future version.
      train['distributor'] = train.distributor.str.replace('(주)', '')
    C:\Users\user\AppData\Local\Temp\ipykernel_14156\3846023154.py:2: FutureWarning: The default value of regex will change from True to False in a future version.
      test['distributor'] = test.distributor.str.replace('(주)', '')
    

#### 특수문자 제거


```python
train['distributor'] = [re.sub(r'[^0-9a-zA-Z가-힣]', '', x) for x in train.distributor]
test['distributor'] = [re.sub(r'[^0-9a-zA-Z가-힣]', '', x) for x in test.distributor]
```


```python
def get_dis(x) :
    if 'CJ' in x or 'CGV' in x :
        return 'CJ'
    elif '쇼박스' in x :
        return '쇼박스'
    elif 'SK' in x :
        return 'SK'
    elif '리틀빅픽' in x :
        return '리틀빅픽처스'
    elif '스폰지' in x :
        return '스폰지'
    elif '싸이더스' in x :
        return '싸이더스'
    elif '에이원' in x :
        return '에이원'
    elif '마인스' in x :
        return '마인스'
    elif '마운틴픽' in x :
        return '마운틴픽처스'
    elif '디씨드' in x :
        return '디씨드'
    elif '드림팩트' in x :
        return '드림팩트'
    elif '메가박스' in x :
        return '메가박스'
    elif '마운틴' in x :
        return '마운틴'
    else :
        return x
```


```python
train['distributor'] = train.distributor.apply(get_dis)
test['distributor'] = test.distributor.apply(get_dis)
```


```python
train.groupby('genre').box_off_num.mean().sort_values()
```




    genre
    뮤지컬       6.627000e+03
    다큐멘터리     6.717226e+04
    서스펜스      8.261100e+04
    애니메이션     1.819267e+05
    멜로/로맨스    4.259680e+05
    미스터리      5.275482e+05
    공포        5.908325e+05
    드라마       6.256898e+05
    코미디       1.193914e+06
    SF        1.788346e+06
    액션        2.203974e+06
    느와르       2.263695e+06
    Name: box_off_num, dtype: float64




```python
train['genre_rank'] = train.genre.map({'뮤지컬' : 1, '다큐멘터리' : 2, '서스펜스' : 3, '애니메이션' : 4, '멜로/로맨스' : 5,
                                      '미스터리' : 6, '공포' : 7, '드라마' : 8, '코미디' : 9, 'SF' : 10, '액션' : 11, '느와르' : 12})
test['genre_rank'] = test.genre.map({'뮤지컬' : 1, '다큐멘터리' : 2, '서스펜스' : 3, '애니메이션' : 4, '멜로/로맨스' : 5,
                                      '미스터리' : 6, '공포' : 7, '드라마' : 8, '코미디' : 9, 'SF' : 10, '액션' : 11, '느와르' : 12})
```


```python
train['distributor'].unique()
```




    array(['롯데엔터테인먼트', '쇼박스', 'NEW', '백두대간', '유니버설픽쳐스인터내셔널코리아', '두타연',
           '케이알씨지', '콘텐츠윙', '키노아이', '팝파트너스', 'CJ', '영화제작전원사', '리틀빅픽처스', '스폰지',
           '조이앤시네마', '인디플러그', '콘텐츠판다', '인디스토리', '팝엔터테인먼트', '시네마서비스', '웃기씨네',
           '영화사진진', '레인보우팩토리', '김기덕필름', '동국대학교충무로영상제작센터', 'BoXoo엔터테인먼트',
           '마운틴픽처스', '메가박스', '골든타이드픽처스', '파이오니아21', '디씨드', '드림팩트', '시너지',
           '디마엔터테인먼트', '판다미디어', '스톰픽쳐스코리아', '예지림엔터테인먼트', '영화사조제', '보람엔터테인먼트',
           '시네마달', '노바엔터테인먼트', '패스파인더씨앤씨', '대명문화공장', '온비즈넷', 'KTG상상마당',
           '무비꼴라쥬', '인벤트디', '씨네그루키다리이엔티', '스튜디오후크', '나이너스엔터테인먼트', 'THE픽쳐스',
           '영구아트무비', '어뮤즈', '이모션픽처스', '이스트스카이필름', '필라멘트픽쳐스', '조이앤컨텐츠그룹',
           '타임스토리그룹', '휘엔터테인먼트', '이십세기폭스코리아', '피터팬픽쳐스', '에스와이코마드', '더픽쳐스',
           '오퍼스픽쳐스', '고앤고필름', '사람과사람들', 'JK필름', '씨너스엔터테인먼트', 'KT', '싸이더스',
           '프레인글로벌', '나우콘텐츠', '홀리가든', '브릿지웍스', '엣나인필름', '위더스필름', '에이원',
           'OAL올', '전망좋은영화사', '스토리셋', '이상우필름', '씨네굿필름', '영희야놀자', '찬란', '어썸피플',
           '아방가르드필름', '스크린조이', '와이드릴리즈', 'tvN', '액티버스엔터테인먼트', '제나두엔터테인먼트',
           '아이필름코퍼레이션', '쟈비스미디어', '트리필름', '에스피엠', '건시네마', '키노엔터테인먼트',
           '아우라픽처스', '에이블엔터테인먼트', '드림로드', '인피니티엔터테인먼트', '새인컴퍼니', '스튜디오느림보',
           '필름라인', 'M2픽처스', '고구마공작소', '미디어데이', '마노엔터테인먼트', '화앤담이엔티', '스마일이엔티',
           '패뷸러스', '영화사조아', '판씨네마', '두엔터테인먼트', '마인스', '전국제영화제', '상구네필름',
           '케이엠스타', '유비네트워크', '한국YWCA연합회', 'KBS미디어', '더피플', '위드시네마',
           '팜코리아미디어', '씨엠닉스', 'SBS콘텐츠허브', '인터콘미디어', '유비콘텐츠', '프로젝트엠피', '하준사',
           '노버스엔터테인먼트', '머니필름', '롤러코스터프로덕션', 'SK', '서울독립영화제', '스튜디오블루',
           '랠리버튼', '머스트씨무비', '마법사필름', '로드하우스', '미라클필름', '프리비젼엔터테인먼트', '영화사',
           '크리에이티브컴즈', 'ysfilm', '이달투', '퍼스트런'], dtype=object)




```python
tr_nm_rank = train.groupby('distributor').box_off_num.median().reset_index(name = 'num_rank').sort_values(by = 'num_rank')
tr_nm_rank
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
      <th>distributor</th>
      <th>num_rank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>110</th>
      <td>인피니티엔터테인먼트</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>고구마공작소</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>52</th>
      <td>사람과사람들</td>
      <td>42.0</td>
    </tr>
    <tr>
      <th>97</th>
      <td>위드시네마</td>
      <td>46.0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>나우콘텐츠</td>
      <td>54.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>113</th>
      <td>전망좋은영화사</td>
      <td>1214237.0</td>
    </tr>
    <tr>
      <th>105</th>
      <td>이십세기폭스코리아</td>
      <td>1422844.0</td>
    </tr>
    <tr>
      <th>56</th>
      <td>쇼박스</td>
      <td>2138560.0</td>
    </tr>
    <tr>
      <th>84</th>
      <td>영구아트무비</td>
      <td>2541603.0</td>
    </tr>
    <tr>
      <th>75</th>
      <td>아이필름코퍼레이션</td>
      <td>3117859.0</td>
    </tr>
  </tbody>
</table>
<p>147 rows × 2 columns</p>
</div>




```python
tr_nm_rank['num_rank'] = [i + 1 for i in range(tr_nm_rank.shape[0])]
```


```python
tr_nm_rank
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
      <th>distributor</th>
      <th>num_rank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>110</th>
      <td>인피니티엔터테인먼트</td>
      <td>1</td>
    </tr>
    <tr>
      <th>15</th>
      <td>고구마공작소</td>
      <td>2</td>
    </tr>
    <tr>
      <th>52</th>
      <td>사람과사람들</td>
      <td>3</td>
    </tr>
    <tr>
      <th>97</th>
      <td>위드시네마</td>
      <td>4</td>
    </tr>
    <tr>
      <th>19</th>
      <td>나우콘텐츠</td>
      <td>5</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>113</th>
      <td>전망좋은영화사</td>
      <td>143</td>
    </tr>
    <tr>
      <th>105</th>
      <td>이십세기폭스코리아</td>
      <td>144</td>
    </tr>
    <tr>
      <th>56</th>
      <td>쇼박스</td>
      <td>145</td>
    </tr>
    <tr>
      <th>84</th>
      <td>영구아트무비</td>
      <td>146</td>
    </tr>
    <tr>
      <th>75</th>
      <td>아이필름코퍼레이션</td>
      <td>147</td>
    </tr>
  </tbody>
</table>
<p>147 rows × 2 columns</p>
</div>



#### 데이터 병합


```python
train = pd.merge(train, tr_nm_rank, on = 'distributor', how = 'left')
test = pd.merge(test, tr_nm_rank, on = 'distributor', how = 'left')
```


```python
train.head()
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
      <th>title</th>
      <th>distributor</th>
      <th>genre</th>
      <th>release_time</th>
      <th>time</th>
      <th>screening_rat</th>
      <th>director</th>
      <th>dir_prev_bfnum</th>
      <th>dir_prev_num</th>
      <th>num_staff</th>
      <th>num_actor</th>
      <th>box_off_num</th>
      <th>genre_rank</th>
      <th>num_rank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>개들의 전쟁</td>
      <td>롯데엔터테인먼트</td>
      <td>액션</td>
      <td>2012-11-22</td>
      <td>96</td>
      <td>청소년 관람불가</td>
      <td>조병옥</td>
      <td>NaN</td>
      <td>0</td>
      <td>91</td>
      <td>2</td>
      <td>23398</td>
      <td>11</td>
      <td>134</td>
    </tr>
    <tr>
      <th>1</th>
      <td>내부자들</td>
      <td>쇼박스</td>
      <td>느와르</td>
      <td>2015-11-19</td>
      <td>130</td>
      <td>청소년 관람불가</td>
      <td>우민호</td>
      <td>1161602.50</td>
      <td>2</td>
      <td>387</td>
      <td>3</td>
      <td>7072501</td>
      <td>12</td>
      <td>145</td>
    </tr>
    <tr>
      <th>2</th>
      <td>은밀하게 위대하게</td>
      <td>쇼박스</td>
      <td>액션</td>
      <td>2013-06-05</td>
      <td>123</td>
      <td>15세 관람가</td>
      <td>장철수</td>
      <td>220775.25</td>
      <td>4</td>
      <td>343</td>
      <td>4</td>
      <td>6959083</td>
      <td>11</td>
      <td>145</td>
    </tr>
    <tr>
      <th>3</th>
      <td>나는 공무원이다</td>
      <td>NEW</td>
      <td>코미디</td>
      <td>2012-07-12</td>
      <td>101</td>
      <td>전체 관람가</td>
      <td>구자홍</td>
      <td>23894.00</td>
      <td>2</td>
      <td>20</td>
      <td>6</td>
      <td>217866</td>
      <td>9</td>
      <td>140</td>
    </tr>
    <tr>
      <th>4</th>
      <td>불량남녀</td>
      <td>쇼박스</td>
      <td>코미디</td>
      <td>2010-11-04</td>
      <td>108</td>
      <td>15세 관람가</td>
      <td>신근호</td>
      <td>1.00</td>
      <td>1</td>
      <td>251</td>
      <td>2</td>
      <td>483387</td>
      <td>9</td>
      <td>145</td>
    </tr>
  </tbody>
</table>
</div>




```python
test
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
      <th>title</th>
      <th>distributor</th>
      <th>genre</th>
      <th>release_time</th>
      <th>time</th>
      <th>screening_rat</th>
      <th>director</th>
      <th>dir_prev_bfnum</th>
      <th>dir_prev_num</th>
      <th>num_staff</th>
      <th>num_actor</th>
      <th>genre_rank</th>
      <th>num_rank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>용서는 없다</td>
      <td>시네마서비스</td>
      <td>느와르</td>
      <td>2010-01-07</td>
      <td>125</td>
      <td>청소년 관람불가</td>
      <td>김형준</td>
      <td>3.005290e+05</td>
      <td>2</td>
      <td>304</td>
      <td>3</td>
      <td>12</td>
      <td>132.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>아빠가 여자를 좋아해</td>
      <td>쇼박스</td>
      <td>멜로/로맨스</td>
      <td>2010-01-14</td>
      <td>113</td>
      <td>12세 관람가</td>
      <td>이광재</td>
      <td>3.427002e+05</td>
      <td>4</td>
      <td>275</td>
      <td>3</td>
      <td>5</td>
      <td>145.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>하모니</td>
      <td>CJ</td>
      <td>드라마</td>
      <td>2010-01-28</td>
      <td>115</td>
      <td>12세 관람가</td>
      <td>강대규</td>
      <td>4.206611e+06</td>
      <td>3</td>
      <td>419</td>
      <td>7</td>
      <td>8</td>
      <td>141.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>의형제</td>
      <td>쇼박스</td>
      <td>액션</td>
      <td>2010-02-04</td>
      <td>116</td>
      <td>15세 관람가</td>
      <td>장훈</td>
      <td>6.913420e+05</td>
      <td>2</td>
      <td>408</td>
      <td>2</td>
      <td>11</td>
      <td>145.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>평행 이론</td>
      <td>CJ</td>
      <td>공포</td>
      <td>2010-02-18</td>
      <td>110</td>
      <td>15세 관람가</td>
      <td>권호영</td>
      <td>3.173800e+04</td>
      <td>1</td>
      <td>380</td>
      <td>1</td>
      <td>7</td>
      <td>141.0</td>
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
    </tr>
    <tr>
      <th>238</th>
      <td>해에게서 소년에게</td>
      <td>디씨드</td>
      <td>드라마</td>
      <td>2015-11-19</td>
      <td>78</td>
      <td>15세 관람가</td>
      <td>안슬기</td>
      <td>2.590000e+03</td>
      <td>1</td>
      <td>4</td>
      <td>4</td>
      <td>8</td>
      <td>112.0</td>
    </tr>
    <tr>
      <th>239</th>
      <td>울보 권투부</td>
      <td>인디스토리</td>
      <td>다큐멘터리</td>
      <td>2015-10-29</td>
      <td>86</td>
      <td>12세 관람가</td>
      <td>이일하</td>
      <td>NaN</td>
      <td>0</td>
      <td>18</td>
      <td>2</td>
      <td>2</td>
      <td>69.0</td>
    </tr>
    <tr>
      <th>240</th>
      <td>어떤살인</td>
      <td>컨텐츠온미디어</td>
      <td>느와르</td>
      <td>2015-10-28</td>
      <td>107</td>
      <td>청소년 관람불가</td>
      <td>안용훈</td>
      <td>NaN</td>
      <td>0</td>
      <td>224</td>
      <td>4</td>
      <td>12</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>241</th>
      <td>말하지 못한 비밀</td>
      <td>마운틴픽처스</td>
      <td>드라마</td>
      <td>2015-10-22</td>
      <td>102</td>
      <td>청소년 관람불가</td>
      <td>송동윤</td>
      <td>5.069900e+04</td>
      <td>1</td>
      <td>68</td>
      <td>7</td>
      <td>8</td>
      <td>46.0</td>
    </tr>
    <tr>
      <th>242</th>
      <td>조선안방 스캔들-칠거지악 2</td>
      <td>케이알씨지</td>
      <td>멜로/로맨스</td>
      <td>2015-10-22</td>
      <td>76</td>
      <td>청소년 관람불가</td>
      <td>이전</td>
      <td>NaN</td>
      <td>0</td>
      <td>10</td>
      <td>4</td>
      <td>5</td>
      <td>29.0</td>
    </tr>
  </tbody>
</table>
<p>243 rows × 13 columns</p>
</div>



## 데이터 전처리

### isna함수는 결측치 여부를 확인해준다. 결측치면 True, 아니면 False로 확인된다.


```python
train.isna().head()
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
      <th>title</th>
      <th>distributor</th>
      <th>genre</th>
      <th>release_time</th>
      <th>time</th>
      <th>screening_rat</th>
      <th>director</th>
      <th>dir_prev_bfnum</th>
      <th>dir_prev_num</th>
      <th>num_staff</th>
      <th>num_actor</th>
      <th>box_off_num</th>
      <th>genre_rank</th>
      <th>num_rank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



이렇게 보면 결측치를 확인하기 어렵다. 여기서 sum함수를 이용해 확인해보자.


```python
train.isna().sum()
```




    title               0
    distributor         0
    genre               0
    release_time        0
    time                0
    screening_rat       0
    director            0
    dir_prev_bfnum    330
    dir_prev_num        0
    num_staff           0
    num_actor           0
    box_off_num         0
    genre_rank          0
    num_rank            0
    dtype: int64




```python
test.isna().sum()
```




    title               0
    distributor         0
    genre               0
    release_time        0
    time                0
    screening_rat       0
    director            0
    dir_prev_bfnum    136
    dir_prev_num        0
    num_staff           0
    num_actor           0
    genre_rank          0
    num_rank           31
    dtype: int64




```python
test
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
      <th>title</th>
      <th>distributor</th>
      <th>genre</th>
      <th>release_time</th>
      <th>time</th>
      <th>screening_rat</th>
      <th>director</th>
      <th>dir_prev_bfnum</th>
      <th>dir_prev_num</th>
      <th>num_staff</th>
      <th>num_actor</th>
      <th>genre_rank</th>
      <th>num_rank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>용서는 없다</td>
      <td>시네마서비스</td>
      <td>느와르</td>
      <td>2010-01-07</td>
      <td>125</td>
      <td>청소년 관람불가</td>
      <td>김형준</td>
      <td>3.005290e+05</td>
      <td>2</td>
      <td>304</td>
      <td>3</td>
      <td>12</td>
      <td>132.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>아빠가 여자를 좋아해</td>
      <td>쇼박스</td>
      <td>멜로/로맨스</td>
      <td>2010-01-14</td>
      <td>113</td>
      <td>12세 관람가</td>
      <td>이광재</td>
      <td>3.427002e+05</td>
      <td>4</td>
      <td>275</td>
      <td>3</td>
      <td>5</td>
      <td>145.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>하모니</td>
      <td>CJ</td>
      <td>드라마</td>
      <td>2010-01-28</td>
      <td>115</td>
      <td>12세 관람가</td>
      <td>강대규</td>
      <td>4.206611e+06</td>
      <td>3</td>
      <td>419</td>
      <td>7</td>
      <td>8</td>
      <td>141.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>의형제</td>
      <td>쇼박스</td>
      <td>액션</td>
      <td>2010-02-04</td>
      <td>116</td>
      <td>15세 관람가</td>
      <td>장훈</td>
      <td>6.913420e+05</td>
      <td>2</td>
      <td>408</td>
      <td>2</td>
      <td>11</td>
      <td>145.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>평행 이론</td>
      <td>CJ</td>
      <td>공포</td>
      <td>2010-02-18</td>
      <td>110</td>
      <td>15세 관람가</td>
      <td>권호영</td>
      <td>3.173800e+04</td>
      <td>1</td>
      <td>380</td>
      <td>1</td>
      <td>7</td>
      <td>141.0</td>
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
    </tr>
    <tr>
      <th>238</th>
      <td>해에게서 소년에게</td>
      <td>디씨드</td>
      <td>드라마</td>
      <td>2015-11-19</td>
      <td>78</td>
      <td>15세 관람가</td>
      <td>안슬기</td>
      <td>2.590000e+03</td>
      <td>1</td>
      <td>4</td>
      <td>4</td>
      <td>8</td>
      <td>112.0</td>
    </tr>
    <tr>
      <th>239</th>
      <td>울보 권투부</td>
      <td>인디스토리</td>
      <td>다큐멘터리</td>
      <td>2015-10-29</td>
      <td>86</td>
      <td>12세 관람가</td>
      <td>이일하</td>
      <td>NaN</td>
      <td>0</td>
      <td>18</td>
      <td>2</td>
      <td>2</td>
      <td>69.0</td>
    </tr>
    <tr>
      <th>240</th>
      <td>어떤살인</td>
      <td>컨텐츠온미디어</td>
      <td>느와르</td>
      <td>2015-10-28</td>
      <td>107</td>
      <td>청소년 관람불가</td>
      <td>안용훈</td>
      <td>NaN</td>
      <td>0</td>
      <td>224</td>
      <td>4</td>
      <td>12</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>241</th>
      <td>말하지 못한 비밀</td>
      <td>마운틴픽처스</td>
      <td>드라마</td>
      <td>2015-10-22</td>
      <td>102</td>
      <td>청소년 관람불가</td>
      <td>송동윤</td>
      <td>5.069900e+04</td>
      <td>1</td>
      <td>68</td>
      <td>7</td>
      <td>8</td>
      <td>46.0</td>
    </tr>
    <tr>
      <th>242</th>
      <td>조선안방 스캔들-칠거지악 2</td>
      <td>케이알씨지</td>
      <td>멜로/로맨스</td>
      <td>2015-10-22</td>
      <td>76</td>
      <td>청소년 관람불가</td>
      <td>이전</td>
      <td>NaN</td>
      <td>0</td>
      <td>10</td>
      <td>4</td>
      <td>5</td>
      <td>29.0</td>
    </tr>
  </tbody>
</table>
<p>243 rows × 13 columns</p>
</div>



test데이터도 마찬가지로 확인해보면 136개의 결측값을 확인 할 수있다.

### fillna함수 결측치를 채울수 있는 함수.


```python
train['dir_prev_bfnum'].fillna(0, inplace = True)
```


```python
train.isna().sum()
```




    title             0
    distributor       0
    genre             0
    release_time      0
    time              0
    screening_rat     0
    director          0
    dir_prev_bfnum    0
    dir_prev_num      0
    num_staff         0
    num_actor         0
    box_off_num       0
    genre_rank        0
    num_rank          0
    dtype: int64




```python
test['dir_prev_bfnum'].fillna(0, inplace = True)
test['num_rank'].fillna(0, inplace = True)
```


```python
test.isna().sum()
```




    title             0
    distributor       0
    genre             0
    release_time      0
    time              0
    screening_rat     0
    director          0
    dir_prev_bfnum    0
    dir_prev_num      0
    num_staff         0
    num_actor         0
    genre_rank        0
    num_rank          0
    dtype: int64



## 변수 선택 및 모델 구축
## Feature Engineering & Initial Modeling  


```python
model = lgb.LGBMRegressor(random_state=123, n_estimators=1000)
```

n_estimators은 순차적으로 만드는 모델을 1000번 반복해서 만들겟다는 뜻이라 생각하자.


```python
features = ['num_rank', 'time', 'dir_prev_num', 'num_staff', 'num_actor', 'genre_rank']
target = ['box_off_num']
```


```python
X = train[features]
y = train[target]
```


```python
from sklearn.model_selection import train_test_split

random_state = 123

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, shuffle = True, random_state = random_state)
```


```python
len(X_train), len(X_test)
```




    (480, 120)




```python
model.fit(X_train, y_train)
```




    LGBMRegressor(n_estimators=1000, random_state=123)




```python
y_pred = model.predict(X_test)
```

예측모델 RMSE 값


```python
import numpy as np
from sklearn.metrics import mean_squared_error

MSE = mean_squared_error(y_test, y_pred)
np.sqrt(MSE) 
```




    2146951.0650640577



모델 성능 나쁘다는 것을 알 수 있다.

## k-fold 교차검증 (k-fold cross validation)


```python
from sklearn.model_selection import KFold
```


```python
k_fold = KFold(n_splits=5, shuffle=True, random_state=123)
```

n_splits = 5은 몇 등분 할지 설정후 shuffle = True는 섞어서 5등분을 시킨다는 의미이다. 


```python
model = lgb.LGBMRegressor(random_state=123, n_estimators=1000)

models = [] 

for train_idx, test_idx in k_fold.split(X, y):
    X_train,X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx],y.iloc[test_idx]
    
    models.append(model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=100, verbose = 50))
```

    [50]	valid_0's l2: 2.94788e+12
    [100]	valid_0's l2: 3.29882e+12
    [50]	valid_0's l2: 1.99173e+12
    [100]	valid_0's l2: 2.42672e+12
    [50]	valid_0's l2: 9.95996e+11
    [100]	valid_0's l2: 1.08187e+12
    [50]	valid_0's l2: 2.0035e+12
    [100]	valid_0's l2: 2.02618e+12
    [50]	valid_0's l2: 3.25215e+12
    [100]	valid_0's l2: 3.22874e+12
    [150]	valid_0's l2: 3.33672e+12
    [200]	valid_0's l2: 3.4445e+12
    

    C:\Users\user\AppData\Roaming\Python\Python39\site-packages\lightgbm\sklearn.py:726: UserWarning: 'early_stopping_rounds' argument is deprecated and will be removed in a future release of LightGBM. Pass 'early_stopping()' callback via 'callbacks' argument instead.
      _log_warning("'early_stopping_rounds' argument is deprecated and will be removed in a future release of LightGBM. "
    C:\Users\user\AppData\Roaming\Python\Python39\site-packages\lightgbm\sklearn.py:736: UserWarning: 'verbose' argument is deprecated and will be removed in a future release of LightGBM. Pass 'log_evaluation()' callback via 'callbacks' argument instead.
      _log_warning("'verbose' argument is deprecated and will be removed in a future release of LightGBM. "
    

early_stopping_rounds=100은 100번의 과정동안 오차율이 감소되지 않으면 멈춰라라는 명령어이다. 그이상하면 과적합이 발생하기 때문에 설정해 준다고 생각하자.
verbose = 50 은 우리가 1000번 학습을 하는데 50번째 모델마다 출력값을 산출해 달라라는 뜻이다.


```python
models
```




    [LGBMRegressor(n_estimators=1000, random_state=123),
     LGBMRegressor(n_estimators=1000, random_state=123),
     LGBMRegressor(n_estimators=1000, random_state=123),
     LGBMRegressor(n_estimators=1000, random_state=123),
     LGBMRegressor(n_estimators=1000, random_state=123)]




```python
preds = []
for model in models:
    preds.append(model.predict(X_test))
len(preds)
```




    5



preds 배열을 만들어서 만들어두었던 모델들의 예측값을 저장해준다


```python
preds
```




    [array([ 4.50582365e+06,  5.52108543e+05,  6.91053938e+04,  4.50321529e+06,
             4.90019161e+05,  3.69946931e+06,  7.03902493e+05,  2.67623960e+06,
             3.53204848e+05,  3.08759019e+05,  6.60376890e+05,  1.43478778e+06,
            -2.81200472e+04, -2.90339406e+04,  1.98932869e+06, -4.01192683e+04,
             3.06261921e+04,  2.66593690e+05,  1.29635510e+05,  1.56763011e+05,
            -2.31581654e+04,  8.78765299e+04,  6.60929096e+04,  1.90603844e+05,
             1.53110737e+05,  1.77446836e+06, -3.08696313e+05,  5.39464703e+06,
             7.70463940e+05,  2.23829212e+05,  4.59368674e+04, -4.84501454e+04,
            -2.22731699e+04,  4.80125554e+05,  9.83746339e+05, -1.38454516e+04,
            -8.93701713e+04,  3.69857506e+05,  1.69673115e+04,  1.93336257e+06,
             1.10665071e+04,  2.69030593e+04,  2.85695231e+06,  3.27550636e+05,
             2.74459865e+05,  5.44421286e+05, -3.06590674e+03,  1.07491170e+05,
             6.57373429e+03, -2.87820815e+04,  4.55673940e+04,  1.14045801e+05,
             1.42966105e+04,  1.35545673e+06,  3.62704649e+05,  1.44183242e+05,
             1.35405667e+05,  9.33504063e+03,  2.29734176e+05,  2.98486599e+04,
             1.58376818e+05,  2.62901075e+06, -1.52044550e+04,  6.50135813e+04,
             8.62541409e+03, -6.25877065e+04,  5.12181045e+04,  1.02044374e+04,
             1.20368052e+05,  2.65576689e+05,  2.40964215e+06,  1.14189080e+05,
             2.88535624e+06, -3.32355574e+05,  2.92291969e+05,  3.72113227e+06,
             8.10795673e+04,  1.03433353e+06, -2.35844165e+05,  5.20791175e+04,
             9.66512479e+04,  1.02385632e+05,  4.38728153e+03,  2.61672382e+06,
             1.66516926e+04, -8.39471441e+04, -1.39767082e+04, -4.99159854e+04,
            -1.82173180e+04,  1.20193615e+06,  3.19888540e+05,  1.87234705e+06,
            -4.27641969e+05, -6.36310364e+04,  1.59523911e+06, -1.55615345e+05,
             6.45935313e+05,  2.08286109e+06,  4.82135358e+04,  4.13954830e+06,
             1.38584720e+06,  2.81989882e+03, -2.11377035e+04,  1.31063779e+04,
             1.80200833e+06,  3.67190456e+04, -1.40980975e+05,  5.99407471e+06,
             6.10935249e+05,  6.38769882e+03, -2.49858693e+04,  2.68950817e+06,
            -1.07178233e+05, -2.10074098e+04,  1.59916726e+03, -6.24119387e+04,
             5.60765027e+05, -4.96809248e+04,  3.80605853e+06,  1.49413490e+06]),
     array([ 4.50582365e+06,  5.52108543e+05,  6.91053938e+04,  4.50321529e+06,
             4.90019161e+05,  3.69946931e+06,  7.03902493e+05,  2.67623960e+06,
             3.53204848e+05,  3.08759019e+05,  6.60376890e+05,  1.43478778e+06,
            -2.81200472e+04, -2.90339406e+04,  1.98932869e+06, -4.01192683e+04,
             3.06261921e+04,  2.66593690e+05,  1.29635510e+05,  1.56763011e+05,
            -2.31581654e+04,  8.78765299e+04,  6.60929096e+04,  1.90603844e+05,
             1.53110737e+05,  1.77446836e+06, -3.08696313e+05,  5.39464703e+06,
             7.70463940e+05,  2.23829212e+05,  4.59368674e+04, -4.84501454e+04,
            -2.22731699e+04,  4.80125554e+05,  9.83746339e+05, -1.38454516e+04,
            -8.93701713e+04,  3.69857506e+05,  1.69673115e+04,  1.93336257e+06,
             1.10665071e+04,  2.69030593e+04,  2.85695231e+06,  3.27550636e+05,
             2.74459865e+05,  5.44421286e+05, -3.06590674e+03,  1.07491170e+05,
             6.57373429e+03, -2.87820815e+04,  4.55673940e+04,  1.14045801e+05,
             1.42966105e+04,  1.35545673e+06,  3.62704649e+05,  1.44183242e+05,
             1.35405667e+05,  9.33504063e+03,  2.29734176e+05,  2.98486599e+04,
             1.58376818e+05,  2.62901075e+06, -1.52044550e+04,  6.50135813e+04,
             8.62541409e+03, -6.25877065e+04,  5.12181045e+04,  1.02044374e+04,
             1.20368052e+05,  2.65576689e+05,  2.40964215e+06,  1.14189080e+05,
             2.88535624e+06, -3.32355574e+05,  2.92291969e+05,  3.72113227e+06,
             8.10795673e+04,  1.03433353e+06, -2.35844165e+05,  5.20791175e+04,
             9.66512479e+04,  1.02385632e+05,  4.38728153e+03,  2.61672382e+06,
             1.66516926e+04, -8.39471441e+04, -1.39767082e+04, -4.99159854e+04,
            -1.82173180e+04,  1.20193615e+06,  3.19888540e+05,  1.87234705e+06,
            -4.27641969e+05, -6.36310364e+04,  1.59523911e+06, -1.55615345e+05,
             6.45935313e+05,  2.08286109e+06,  4.82135358e+04,  4.13954830e+06,
             1.38584720e+06,  2.81989882e+03, -2.11377035e+04,  1.31063779e+04,
             1.80200833e+06,  3.67190456e+04, -1.40980975e+05,  5.99407471e+06,
             6.10935249e+05,  6.38769882e+03, -2.49858693e+04,  2.68950817e+06,
            -1.07178233e+05, -2.10074098e+04,  1.59916726e+03, -6.24119387e+04,
             5.60765027e+05, -4.96809248e+04,  3.80605853e+06,  1.49413490e+06]),
     array([ 4.50582365e+06,  5.52108543e+05,  6.91053938e+04,  4.50321529e+06,
             4.90019161e+05,  3.69946931e+06,  7.03902493e+05,  2.67623960e+06,
             3.53204848e+05,  3.08759019e+05,  6.60376890e+05,  1.43478778e+06,
            -2.81200472e+04, -2.90339406e+04,  1.98932869e+06, -4.01192683e+04,
             3.06261921e+04,  2.66593690e+05,  1.29635510e+05,  1.56763011e+05,
            -2.31581654e+04,  8.78765299e+04,  6.60929096e+04,  1.90603844e+05,
             1.53110737e+05,  1.77446836e+06, -3.08696313e+05,  5.39464703e+06,
             7.70463940e+05,  2.23829212e+05,  4.59368674e+04, -4.84501454e+04,
            -2.22731699e+04,  4.80125554e+05,  9.83746339e+05, -1.38454516e+04,
            -8.93701713e+04,  3.69857506e+05,  1.69673115e+04,  1.93336257e+06,
             1.10665071e+04,  2.69030593e+04,  2.85695231e+06,  3.27550636e+05,
             2.74459865e+05,  5.44421286e+05, -3.06590674e+03,  1.07491170e+05,
             6.57373429e+03, -2.87820815e+04,  4.55673940e+04,  1.14045801e+05,
             1.42966105e+04,  1.35545673e+06,  3.62704649e+05,  1.44183242e+05,
             1.35405667e+05,  9.33504063e+03,  2.29734176e+05,  2.98486599e+04,
             1.58376818e+05,  2.62901075e+06, -1.52044550e+04,  6.50135813e+04,
             8.62541409e+03, -6.25877065e+04,  5.12181045e+04,  1.02044374e+04,
             1.20368052e+05,  2.65576689e+05,  2.40964215e+06,  1.14189080e+05,
             2.88535624e+06, -3.32355574e+05,  2.92291969e+05,  3.72113227e+06,
             8.10795673e+04,  1.03433353e+06, -2.35844165e+05,  5.20791175e+04,
             9.66512479e+04,  1.02385632e+05,  4.38728153e+03,  2.61672382e+06,
             1.66516926e+04, -8.39471441e+04, -1.39767082e+04, -4.99159854e+04,
            -1.82173180e+04,  1.20193615e+06,  3.19888540e+05,  1.87234705e+06,
            -4.27641969e+05, -6.36310364e+04,  1.59523911e+06, -1.55615345e+05,
             6.45935313e+05,  2.08286109e+06,  4.82135358e+04,  4.13954830e+06,
             1.38584720e+06,  2.81989882e+03, -2.11377035e+04,  1.31063779e+04,
             1.80200833e+06,  3.67190456e+04, -1.40980975e+05,  5.99407471e+06,
             6.10935249e+05,  6.38769882e+03, -2.49858693e+04,  2.68950817e+06,
            -1.07178233e+05, -2.10074098e+04,  1.59916726e+03, -6.24119387e+04,
             5.60765027e+05, -4.96809248e+04,  3.80605853e+06,  1.49413490e+06]),
     array([ 4.50582365e+06,  5.52108543e+05,  6.91053938e+04,  4.50321529e+06,
             4.90019161e+05,  3.69946931e+06,  7.03902493e+05,  2.67623960e+06,
             3.53204848e+05,  3.08759019e+05,  6.60376890e+05,  1.43478778e+06,
            -2.81200472e+04, -2.90339406e+04,  1.98932869e+06, -4.01192683e+04,
             3.06261921e+04,  2.66593690e+05,  1.29635510e+05,  1.56763011e+05,
            -2.31581654e+04,  8.78765299e+04,  6.60929096e+04,  1.90603844e+05,
             1.53110737e+05,  1.77446836e+06, -3.08696313e+05,  5.39464703e+06,
             7.70463940e+05,  2.23829212e+05,  4.59368674e+04, -4.84501454e+04,
            -2.22731699e+04,  4.80125554e+05,  9.83746339e+05, -1.38454516e+04,
            -8.93701713e+04,  3.69857506e+05,  1.69673115e+04,  1.93336257e+06,
             1.10665071e+04,  2.69030593e+04,  2.85695231e+06,  3.27550636e+05,
             2.74459865e+05,  5.44421286e+05, -3.06590674e+03,  1.07491170e+05,
             6.57373429e+03, -2.87820815e+04,  4.55673940e+04,  1.14045801e+05,
             1.42966105e+04,  1.35545673e+06,  3.62704649e+05,  1.44183242e+05,
             1.35405667e+05,  9.33504063e+03,  2.29734176e+05,  2.98486599e+04,
             1.58376818e+05,  2.62901075e+06, -1.52044550e+04,  6.50135813e+04,
             8.62541409e+03, -6.25877065e+04,  5.12181045e+04,  1.02044374e+04,
             1.20368052e+05,  2.65576689e+05,  2.40964215e+06,  1.14189080e+05,
             2.88535624e+06, -3.32355574e+05,  2.92291969e+05,  3.72113227e+06,
             8.10795673e+04,  1.03433353e+06, -2.35844165e+05,  5.20791175e+04,
             9.66512479e+04,  1.02385632e+05,  4.38728153e+03,  2.61672382e+06,
             1.66516926e+04, -8.39471441e+04, -1.39767082e+04, -4.99159854e+04,
            -1.82173180e+04,  1.20193615e+06,  3.19888540e+05,  1.87234705e+06,
            -4.27641969e+05, -6.36310364e+04,  1.59523911e+06, -1.55615345e+05,
             6.45935313e+05,  2.08286109e+06,  4.82135358e+04,  4.13954830e+06,
             1.38584720e+06,  2.81989882e+03, -2.11377035e+04,  1.31063779e+04,
             1.80200833e+06,  3.67190456e+04, -1.40980975e+05,  5.99407471e+06,
             6.10935249e+05,  6.38769882e+03, -2.49858693e+04,  2.68950817e+06,
            -1.07178233e+05, -2.10074098e+04,  1.59916726e+03, -6.24119387e+04,
             5.60765027e+05, -4.96809248e+04,  3.80605853e+06,  1.49413490e+06]),
     array([ 4.50582365e+06,  5.52108543e+05,  6.91053938e+04,  4.50321529e+06,
             4.90019161e+05,  3.69946931e+06,  7.03902493e+05,  2.67623960e+06,
             3.53204848e+05,  3.08759019e+05,  6.60376890e+05,  1.43478778e+06,
            -2.81200472e+04, -2.90339406e+04,  1.98932869e+06, -4.01192683e+04,
             3.06261921e+04,  2.66593690e+05,  1.29635510e+05,  1.56763011e+05,
            -2.31581654e+04,  8.78765299e+04,  6.60929096e+04,  1.90603844e+05,
             1.53110737e+05,  1.77446836e+06, -3.08696313e+05,  5.39464703e+06,
             7.70463940e+05,  2.23829212e+05,  4.59368674e+04, -4.84501454e+04,
            -2.22731699e+04,  4.80125554e+05,  9.83746339e+05, -1.38454516e+04,
            -8.93701713e+04,  3.69857506e+05,  1.69673115e+04,  1.93336257e+06,
             1.10665071e+04,  2.69030593e+04,  2.85695231e+06,  3.27550636e+05,
             2.74459865e+05,  5.44421286e+05, -3.06590674e+03,  1.07491170e+05,
             6.57373429e+03, -2.87820815e+04,  4.55673940e+04,  1.14045801e+05,
             1.42966105e+04,  1.35545673e+06,  3.62704649e+05,  1.44183242e+05,
             1.35405667e+05,  9.33504063e+03,  2.29734176e+05,  2.98486599e+04,
             1.58376818e+05,  2.62901075e+06, -1.52044550e+04,  6.50135813e+04,
             8.62541409e+03, -6.25877065e+04,  5.12181045e+04,  1.02044374e+04,
             1.20368052e+05,  2.65576689e+05,  2.40964215e+06,  1.14189080e+05,
             2.88535624e+06, -3.32355574e+05,  2.92291969e+05,  3.72113227e+06,
             8.10795673e+04,  1.03433353e+06, -2.35844165e+05,  5.20791175e+04,
             9.66512479e+04,  1.02385632e+05,  4.38728153e+03,  2.61672382e+06,
             1.66516926e+04, -8.39471441e+04, -1.39767082e+04, -4.99159854e+04,
            -1.82173180e+04,  1.20193615e+06,  3.19888540e+05,  1.87234705e+06,
            -4.27641969e+05, -6.36310364e+04,  1.59523911e+06, -1.55615345e+05,
             6.45935313e+05,  2.08286109e+06,  4.82135358e+04,  4.13954830e+06,
             1.38584720e+06,  2.81989882e+03, -2.11377035e+04,  1.31063779e+04,
             1.80200833e+06,  3.67190456e+04, -1.40980975e+05,  5.99407471e+06,
             6.10935249e+05,  6.38769882e+03, -2.49858693e+04,  2.68950817e+06,
            -1.07178233e+05, -2.10074098e+04,  1.59916726e+03, -6.24119387e+04,
             5.60765027e+05, -4.96809248e+04,  3.80605853e+06,  1.49413490e+06])]



이렇게 예측값들이 5세트가 있는것을 볼 수 있다.


```python
y_pred=np.mean(preds, axis = 0)
```


```python
MSE = mean_squared_error(y_test, y_pred)
np.sqrt(MSE)
```




    1791255.1889112082



전보다는 모델 성능이 조금 더 좋아졌음을 알 수 있다.

## Grid search CV 모델 튜닝


```python
from sklearn.model_selection import GridSearchCV
```


```python
model = lgb.LGBMRegressor(random_state=123, n_estimators=1000)
```


```python
params = {
    'learning_rate': [0.1, 0.01, 0.003],
    'min_child_samples': [20, 30]}

gs = GridSearchCV(estimator=model,
                 param_grid=params,
                  scoring= 'neg_mean_squared_error',
                  cv = k_fold)
```


```python
gs.fit(X_train, y_train)
```




    GridSearchCV(cv=KFold(n_splits=5, random_state=123, shuffle=True),
                 estimator=LGBMRegressor(n_estimators=1000, random_state=123),
                 param_grid={'learning_rate': [0.1, 0.01, 0.003],
                             'min_child_samples': [20, 30]},
                 scoring='neg_mean_squared_error')




```python
gs.best_params_
```




    {'learning_rate': 0.003, 'min_child_samples': 30}




```python
model = lgb.LGBMRegressor(random_state=123, n_estimators=1000, learning_rate=0.003, min_child_samples=30)

models = [] 

for train_idx, test_idx in k_fold.split(X, y):
    X_train,X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx],y.iloc[test_idx]
    
    models.append(model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=100, verbose = 50))
```

    C:\Users\user\AppData\Roaming\Python\Python39\site-packages\lightgbm\sklearn.py:726: UserWarning: 'early_stopping_rounds' argument is deprecated and will be removed in a future release of LightGBM. Pass 'early_stopping()' callback via 'callbacks' argument instead.
      _log_warning("'early_stopping_rounds' argument is deprecated and will be removed in a future release of LightGBM. "
    C:\Users\user\AppData\Roaming\Python\Python39\site-packages\lightgbm\sklearn.py:736: UserWarning: 'verbose' argument is deprecated and will be removed in a future release of LightGBM. Pass 'log_evaluation()' callback via 'callbacks' argument instead.
      _log_warning("'verbose' argument is deprecated and will be removed in a future release of LightGBM. "
    

    [50]	valid_0's l2: 4.23106e+12
    [100]	valid_0's l2: 3.89784e+12
    [150]	valid_0's l2: 3.63044e+12
    [200]	valid_0's l2: 3.41848e+12
    [250]	valid_0's l2: 3.25076e+12
    [300]	valid_0's l2: 3.11561e+12
    [350]	valid_0's l2: 3.01411e+12
    [400]	valid_0's l2: 2.95362e+12
    [450]	valid_0's l2: 2.90652e+12
    [500]	valid_0's l2: 2.86775e+12
    [550]	valid_0's l2: 2.83672e+12
    [600]	valid_0's l2: 2.81448e+12
    [650]	valid_0's l2: 2.79625e+12
    [700]	valid_0's l2: 2.77695e+12
    [750]	valid_0's l2: 2.75733e+12
    [800]	valid_0's l2: 2.7472e+12
    [850]	valid_0's l2: 2.72458e+12
    [900]	valid_0's l2: 2.70558e+12
    [950]	valid_0's l2: 2.69246e+12
    [1000]	valid_0's l2: 2.6804e+12
    [50]	valid_0's l2: 1.98e+12
    [100]	valid_0's l2: 1.81065e+12
    [150]	valid_0's l2: 1.70051e+12
    [200]	valid_0's l2: 1.63834e+12
    [250]	valid_0's l2: 1.61087e+12
    [300]	valid_0's l2: 1.59063e+12
    [350]	valid_0's l2: 1.57955e+12
    [400]	valid_0's l2: 1.57547e+12
    [450]	valid_0's l2: 1.5751e+12
    [500]	valid_0's l2: 1.58341e+12
    [50]	valid_0's l2: 1.19405e+12
    [100]	valid_0's l2: 1.07143e+12
    [150]	valid_0's l2: 9.80857e+11
    [200]	valid_0's l2: 9.16734e+11
    [250]	valid_0's l2: 8.70518e+11
    [300]	valid_0's l2: 8.41298e+11
    [350]	valid_0's l2: 8.26252e+11
    [400]	valid_0's l2: 8.20997e+11
    [450]	valid_0's l2: 8.21185e+11
    [500]	valid_0's l2: 8.24422e+11
    [50]	valid_0's l2: 2.49065e+12
    [100]	valid_0's l2: 2.27493e+12
    [150]	valid_0's l2: 2.122e+12
    [200]	valid_0's l2: 1.98767e+12
    [250]	valid_0's l2: 1.90344e+12
    [300]	valid_0's l2: 1.84926e+12
    [350]	valid_0's l2: 1.81487e+12
    [400]	valid_0's l2: 1.80022e+12
    [450]	valid_0's l2: 1.79208e+12
    [500]	valid_0's l2: 1.79388e+12
    [550]	valid_0's l2: 1.80207e+12
    [50]	valid_0's l2: 5.25464e+12
    [100]	valid_0's l2: 4.88747e+12
    [150]	valid_0's l2: 4.55827e+12
    [200]	valid_0's l2: 4.27996e+12
    [250]	valid_0's l2: 4.10147e+12
    [300]	valid_0's l2: 3.95111e+12
    [350]	valid_0's l2: 3.83281e+12
    [400]	valid_0's l2: 3.74671e+12
    [450]	valid_0's l2: 3.68203e+12
    [500]	valid_0's l2: 3.62971e+12
    [550]	valid_0's l2: 3.58539e+12
    [600]	valid_0's l2: 3.55124e+12
    [650]	valid_0's l2: 3.53131e+12
    [700]	valid_0's l2: 3.51644e+12
    [750]	valid_0's l2: 3.49808e+12
    [800]	valid_0's l2: 3.48215e+12
    [850]	valid_0's l2: 3.46962e+12
    [900]	valid_0's l2: 3.45686e+12
    [950]	valid_0's l2: 3.44662e+12
    [1000]	valid_0's l2: 3.43807e+12
    


```python
preds = []

for model in models:
    preds.append(model.predict(X_test))
```


```python
preds
```




    [array([3707090.72747201,  618489.82621842,  499741.94158279,
            3175439.07212819,  510565.20188139, 2795072.58419054,
            1063651.22746435, 1630246.46355174,  718309.9823607 ,
             -28915.94734822,  674795.70436821, 1462546.17508465,
              64696.8376484 ,   42886.67372907, 1668587.4937159 ,
              50256.94207378,  299686.57314251,  235283.84863858,
              66517.5772929 ,  190184.18273191,   40089.41139856,
             485097.84705045,   60803.88097369,  860374.70633705,
              82389.32942024, 1526958.89638823,  208339.88513953,
            3018788.16705013,  506897.44227779,  702908.60018576,
              31673.92690816,   50254.84673294,   29521.51703944,
             831663.65397235,  521100.36330269,   98538.14108199,
             288509.40082172,  710595.0346185 ,  296356.73568712,
            1799647.61252069,   29675.16466354,   44548.87832803,
            2467691.57073234,  126112.84413409,  968291.88397411,
             499588.99098028,   50683.1159181 ,   82247.24657888,
              39521.50580163,   84983.81815567,   27379.11196385,
              87233.74696927,   31207.87946446,  813968.10889253,
             144984.72938574,  366839.34404298,   68493.18999547,
              90900.96769107,  115998.74187255,  296854.95637646,
             148285.82867353, 2736463.46774932,   25056.45679944,
              57529.69892476,   65285.36334683,  403764.27745783,
              63912.22512017,   35921.16027256,   61442.75006922,
             257004.04151597, 2215543.37756749,   66514.96059313,
            2815335.87573758,  539669.10172766,   24044.70214323,
            2893754.00526755,  680874.10808372, 1299824.73829306,
             -26436.15183252,  157503.73902103,   68203.35464698,
             126288.18322987,   83270.98261726, 2268448.66296715,
              73032.35143128,   42092.95189384,  -31316.67394652,
              38971.03185366,   27025.55726587, 1195160.06482087,
             673756.07068498, 2956260.0022415 ,  593510.64040932,
              44305.60255564, 2763066.69846932,  209995.66401069,
             618152.33645849, 2335753.50191399,   55707.76481429,
            2972626.41325582, 1518544.48063777,   34308.54429598,
             -49265.90746961,   36762.6839318 , 2027802.21554158,
             298706.27132187,   13699.78222336, 3581939.6616339 ,
             557363.40191383,   38181.14179847,   27308.33668333,
            2630589.48818889,    9088.78398261,  -58451.49372293,
              58715.86204659,   42074.53617625,  693207.52725733,
              27633.79940328, 2944025.8388212 , 1676757.06154456]),
     array([3707090.72747201,  618489.82621842,  499741.94158279,
            3175439.07212819,  510565.20188139, 2795072.58419054,
            1063651.22746435, 1630246.46355174,  718309.9823607 ,
             -28915.94734822,  674795.70436821, 1462546.17508465,
              64696.8376484 ,   42886.67372907, 1668587.4937159 ,
              50256.94207378,  299686.57314251,  235283.84863858,
              66517.5772929 ,  190184.18273191,   40089.41139856,
             485097.84705045,   60803.88097369,  860374.70633705,
              82389.32942024, 1526958.89638823,  208339.88513953,
            3018788.16705013,  506897.44227779,  702908.60018576,
              31673.92690816,   50254.84673294,   29521.51703944,
             831663.65397235,  521100.36330269,   98538.14108199,
             288509.40082172,  710595.0346185 ,  296356.73568712,
            1799647.61252069,   29675.16466354,   44548.87832803,
            2467691.57073234,  126112.84413409,  968291.88397411,
             499588.99098028,   50683.1159181 ,   82247.24657888,
              39521.50580163,   84983.81815567,   27379.11196385,
              87233.74696927,   31207.87946446,  813968.10889253,
             144984.72938574,  366839.34404298,   68493.18999547,
              90900.96769107,  115998.74187255,  296854.95637646,
             148285.82867353, 2736463.46774932,   25056.45679944,
              57529.69892476,   65285.36334683,  403764.27745783,
              63912.22512017,   35921.16027256,   61442.75006922,
             257004.04151597, 2215543.37756749,   66514.96059313,
            2815335.87573758,  539669.10172766,   24044.70214323,
            2893754.00526755,  680874.10808372, 1299824.73829306,
             -26436.15183252,  157503.73902103,   68203.35464698,
             126288.18322987,   83270.98261726, 2268448.66296715,
              73032.35143128,   42092.95189384,  -31316.67394652,
              38971.03185366,   27025.55726587, 1195160.06482087,
             673756.07068498, 2956260.0022415 ,  593510.64040932,
              44305.60255564, 2763066.69846932,  209995.66401069,
             618152.33645849, 2335753.50191399,   55707.76481429,
            2972626.41325582, 1518544.48063777,   34308.54429598,
             -49265.90746961,   36762.6839318 , 2027802.21554158,
             298706.27132187,   13699.78222336, 3581939.6616339 ,
             557363.40191383,   38181.14179847,   27308.33668333,
            2630589.48818889,    9088.78398261,  -58451.49372293,
              58715.86204659,   42074.53617625,  693207.52725733,
              27633.79940328, 2944025.8388212 , 1676757.06154456]),
     array([3707090.72747201,  618489.82621842,  499741.94158279,
            3175439.07212819,  510565.20188139, 2795072.58419054,
            1063651.22746435, 1630246.46355174,  718309.9823607 ,
             -28915.94734822,  674795.70436821, 1462546.17508465,
              64696.8376484 ,   42886.67372907, 1668587.4937159 ,
              50256.94207378,  299686.57314251,  235283.84863858,
              66517.5772929 ,  190184.18273191,   40089.41139856,
             485097.84705045,   60803.88097369,  860374.70633705,
              82389.32942024, 1526958.89638823,  208339.88513953,
            3018788.16705013,  506897.44227779,  702908.60018576,
              31673.92690816,   50254.84673294,   29521.51703944,
             831663.65397235,  521100.36330269,   98538.14108199,
             288509.40082172,  710595.0346185 ,  296356.73568712,
            1799647.61252069,   29675.16466354,   44548.87832803,
            2467691.57073234,  126112.84413409,  968291.88397411,
             499588.99098028,   50683.1159181 ,   82247.24657888,
              39521.50580163,   84983.81815567,   27379.11196385,
              87233.74696927,   31207.87946446,  813968.10889253,
             144984.72938574,  366839.34404298,   68493.18999547,
              90900.96769107,  115998.74187255,  296854.95637646,
             148285.82867353, 2736463.46774932,   25056.45679944,
              57529.69892476,   65285.36334683,  403764.27745783,
              63912.22512017,   35921.16027256,   61442.75006922,
             257004.04151597, 2215543.37756749,   66514.96059313,
            2815335.87573758,  539669.10172766,   24044.70214323,
            2893754.00526755,  680874.10808372, 1299824.73829306,
             -26436.15183252,  157503.73902103,   68203.35464698,
             126288.18322987,   83270.98261726, 2268448.66296715,
              73032.35143128,   42092.95189384,  -31316.67394652,
              38971.03185366,   27025.55726587, 1195160.06482087,
             673756.07068498, 2956260.0022415 ,  593510.64040932,
              44305.60255564, 2763066.69846932,  209995.66401069,
             618152.33645849, 2335753.50191399,   55707.76481429,
            2972626.41325582, 1518544.48063777,   34308.54429598,
             -49265.90746961,   36762.6839318 , 2027802.21554158,
             298706.27132187,   13699.78222336, 3581939.6616339 ,
             557363.40191383,   38181.14179847,   27308.33668333,
            2630589.48818889,    9088.78398261,  -58451.49372293,
              58715.86204659,   42074.53617625,  693207.52725733,
              27633.79940328, 2944025.8388212 , 1676757.06154456]),
     array([3707090.72747201,  618489.82621842,  499741.94158279,
            3175439.07212819,  510565.20188139, 2795072.58419054,
            1063651.22746435, 1630246.46355174,  718309.9823607 ,
             -28915.94734822,  674795.70436821, 1462546.17508465,
              64696.8376484 ,   42886.67372907, 1668587.4937159 ,
              50256.94207378,  299686.57314251,  235283.84863858,
              66517.5772929 ,  190184.18273191,   40089.41139856,
             485097.84705045,   60803.88097369,  860374.70633705,
              82389.32942024, 1526958.89638823,  208339.88513953,
            3018788.16705013,  506897.44227779,  702908.60018576,
              31673.92690816,   50254.84673294,   29521.51703944,
             831663.65397235,  521100.36330269,   98538.14108199,
             288509.40082172,  710595.0346185 ,  296356.73568712,
            1799647.61252069,   29675.16466354,   44548.87832803,
            2467691.57073234,  126112.84413409,  968291.88397411,
             499588.99098028,   50683.1159181 ,   82247.24657888,
              39521.50580163,   84983.81815567,   27379.11196385,
              87233.74696927,   31207.87946446,  813968.10889253,
             144984.72938574,  366839.34404298,   68493.18999547,
              90900.96769107,  115998.74187255,  296854.95637646,
             148285.82867353, 2736463.46774932,   25056.45679944,
              57529.69892476,   65285.36334683,  403764.27745783,
              63912.22512017,   35921.16027256,   61442.75006922,
             257004.04151597, 2215543.37756749,   66514.96059313,
            2815335.87573758,  539669.10172766,   24044.70214323,
            2893754.00526755,  680874.10808372, 1299824.73829306,
             -26436.15183252,  157503.73902103,   68203.35464698,
             126288.18322987,   83270.98261726, 2268448.66296715,
              73032.35143128,   42092.95189384,  -31316.67394652,
              38971.03185366,   27025.55726587, 1195160.06482087,
             673756.07068498, 2956260.0022415 ,  593510.64040932,
              44305.60255564, 2763066.69846932,  209995.66401069,
             618152.33645849, 2335753.50191399,   55707.76481429,
            2972626.41325582, 1518544.48063777,   34308.54429598,
             -49265.90746961,   36762.6839318 , 2027802.21554158,
             298706.27132187,   13699.78222336, 3581939.6616339 ,
             557363.40191383,   38181.14179847,   27308.33668333,
            2630589.48818889,    9088.78398261,  -58451.49372293,
              58715.86204659,   42074.53617625,  693207.52725733,
              27633.79940328, 2944025.8388212 , 1676757.06154456]),
     array([3707090.72747201,  618489.82621842,  499741.94158279,
            3175439.07212819,  510565.20188139, 2795072.58419054,
            1063651.22746435, 1630246.46355174,  718309.9823607 ,
             -28915.94734822,  674795.70436821, 1462546.17508465,
              64696.8376484 ,   42886.67372907, 1668587.4937159 ,
              50256.94207378,  299686.57314251,  235283.84863858,
              66517.5772929 ,  190184.18273191,   40089.41139856,
             485097.84705045,   60803.88097369,  860374.70633705,
              82389.32942024, 1526958.89638823,  208339.88513953,
            3018788.16705013,  506897.44227779,  702908.60018576,
              31673.92690816,   50254.84673294,   29521.51703944,
             831663.65397235,  521100.36330269,   98538.14108199,
             288509.40082172,  710595.0346185 ,  296356.73568712,
            1799647.61252069,   29675.16466354,   44548.87832803,
            2467691.57073234,  126112.84413409,  968291.88397411,
             499588.99098028,   50683.1159181 ,   82247.24657888,
              39521.50580163,   84983.81815567,   27379.11196385,
              87233.74696927,   31207.87946446,  813968.10889253,
             144984.72938574,  366839.34404298,   68493.18999547,
              90900.96769107,  115998.74187255,  296854.95637646,
             148285.82867353, 2736463.46774932,   25056.45679944,
              57529.69892476,   65285.36334683,  403764.27745783,
              63912.22512017,   35921.16027256,   61442.75006922,
             257004.04151597, 2215543.37756749,   66514.96059313,
            2815335.87573758,  539669.10172766,   24044.70214323,
            2893754.00526755,  680874.10808372, 1299824.73829306,
             -26436.15183252,  157503.73902103,   68203.35464698,
             126288.18322987,   83270.98261726, 2268448.66296715,
              73032.35143128,   42092.95189384,  -31316.67394652,
              38971.03185366,   27025.55726587, 1195160.06482087,
             673756.07068498, 2956260.0022415 ,  593510.64040932,
              44305.60255564, 2763066.69846932,  209995.66401069,
             618152.33645849, 2335753.50191399,   55707.76481429,
            2972626.41325582, 1518544.48063777,   34308.54429598,
             -49265.90746961,   36762.6839318 , 2027802.21554158,
             298706.27132187,   13699.78222336, 3581939.6616339 ,
             557363.40191383,   38181.14179847,   27308.33668333,
            2630589.48818889,    9088.78398261,  -58451.49372293,
              58715.86204659,   42074.53617625,  693207.52725733,
              27633.79940328, 2944025.8388212 , 1676757.06154456])]




```python
y_pred=np.mean(preds, axis=0)
```


```python
MSE = mean_squared_error(y_test, y_pred)
np.sqrt(MSE)
```




    1854144.9752521503


