---
layout: post
title: Diabetes Classification and Analysis
authors: [Seoyeon Kang]
categories: [1기 AI/SW developers(개인 프로젝트)]
---

```python
# data manipulation
import pandas as pd
import numpy as np
import scipy.stats as stats

# visualisation
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('darkgrid')
import warnings 
warnings.filterwarnings('ignore')

# sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
```

# Introduction

Feature Engineering을 수행하는 것을 매우 좋지만 때로는 실제로 전혀 도움이 되지 않고 오히려 시간이 많이 걸립니다. 그래서 데이터셋을 그대로 두고 사람이 당뇨병인지 아닌지를 예측하는 모델을 생성하고, 데이터셋을 분석하고 적절한 곳에서 Feature Engineering을 수행한 후 생성된 모델과 비교할 겁니다.


```python
# reading file
data = pd.read_csv('diabetes.csv')
```

# Version One : without Feature Engineering


```python
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 768 entries, 0 to 767
    Data columns (total 9 columns):
     #   Column                    Non-Null Count  Dtype  
    ---  ------                    --------------  -----  
     0   Pregnancies               768 non-null    int64  
     1   Glucose                   768 non-null    int64  
     2   BloodPressure             768 non-null    int64  
     3   SkinThickness             768 non-null    int64  
     4   Insulin                   768 non-null    int64  
     5   BMI                       768 non-null    float64
     6   DiabetesPedigreeFunction  768 non-null    float64
     7   Age                       768 non-null    int64  
     8   Outcome                   768 non-null    int64  
    dtypes: float64(2), int64(7)
    memory usage: 54.1 KB
    


```python
data.isnull().sum()
```




    Pregnancies                 0
    Glucose                     0
    BloodPressure               0
    SkinThickness               0
    Insulin                     0
    BMI                         0
    DiabetesPedigreeFunction    0
    Age                         0
    Outcome                     0
    dtype: int64




```python
!pip install lazypredict
```

    Requirement already satisfied: lazypredict in c:\users\seoyeon\anaconda3\lib\site-packages (0.2.12)
    Requirement already satisfied: scikit-learn in c:\users\seoyeon\anaconda3\lib\site-packages (from lazypredict) (1.0.2)
    Requirement already satisfied: joblib in c:\users\seoyeon\anaconda3\lib\site-packages (from lazypredict) (1.1.0)
    Requirement already satisfied: xgboost in c:\users\seoyeon\anaconda3\lib\site-packages (from lazypredict) (1.6.1)
    Requirement already satisfied: tqdm in c:\users\seoyeon\anaconda3\lib\site-packages (from lazypredict) (4.64.0)
    Requirement already satisfied: click in c:\users\seoyeon\anaconda3\lib\site-packages (from lazypredict) (8.0.4)
    Requirement already satisfied: lightgbm in c:\users\seoyeon\anaconda3\lib\site-packages (from lazypredict) (3.3.2)
    Requirement already satisfied: pandas in c:\users\seoyeon\anaconda3\lib\site-packages (from lazypredict) (1.4.2)
    Requirement already satisfied: colorama in c:\users\seoyeon\anaconda3\lib\site-packages (from click->lazypredict) (0.4.4)
    Requirement already satisfied: scipy in c:\users\seoyeon\anaconda3\lib\site-packages (from lightgbm->lazypredict) (1.7.3)
    Requirement already satisfied: numpy in c:\users\seoyeon\anaconda3\lib\site-packages (from lightgbm->lazypredict) (1.21.5)
    Requirement already satisfied: wheel in c:\users\seoyeon\anaconda3\lib\site-packages (from lightgbm->lazypredict) (0.37.1)
    Requirement already satisfied: threadpoolctl>=2.0.0 in c:\users\seoyeon\anaconda3\lib\site-packages (from scikit-learn->lazypredict) (2.2.0)
    Requirement already satisfied: pytz>=2020.1 in c:\users\seoyeon\anaconda3\lib\site-packages (from pandas->lazypredict) (2021.3)
    Requirement already satisfied: python-dateutil>=2.8.1 in c:\users\seoyeon\anaconda3\lib\site-packages (from pandas->lazypredict) (2.8.2)
    Requirement already satisfied: six>=1.5 in c:\users\seoyeon\anaconda3\lib\site-packages (from python-dateutil>=2.8.1->pandas->lazypredict) (1.16.0)
    


```python
from lazypredict.Supervised import LazyClassifier

X_train, X_test, y_train, y_test = train_test_split(data.drop('Outcome',axis=1), 
                                                    data['Outcome'], test_size=0.2, 
                                                    random_state= 42)

clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
models,predictions = clf.fit(X_train, X_test, y_train, y_test)
```

    100%|██████████████████████████████████████████████████████████████████████████████████| 29/29 [00:01<00:00, 16.89it/s]
    


```python
models.sort_values('Accuracy',ascending =False)
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
      <th>Accuracy</th>
      <th>Balanced Accuracy</th>
      <th>ROC AUC</th>
      <th>F1 Score</th>
      <th>Time Taken</th>
    </tr>
    <tr>
      <th>Model</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>QuadraticDiscriminantAnalysis</th>
      <td>0.78</td>
      <td>0.76</td>
      <td>0.76</td>
      <td>0.78</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>RidgeClassifierCV</th>
      <td>0.77</td>
      <td>0.75</td>
      <td>0.75</td>
      <td>0.77</td>
      <td>0.02</td>
    </tr>
    <tr>
      <th>GaussianNB</th>
      <td>0.77</td>
      <td>0.75</td>
      <td>0.75</td>
      <td>0.77</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>LinearDiscriminantAnalysis</th>
      <td>0.76</td>
      <td>0.74</td>
      <td>0.74</td>
      <td>0.76</td>
      <td>0.05</td>
    </tr>
    <tr>
      <th>RidgeClassifier</th>
      <td>0.76</td>
      <td>0.74</td>
      <td>0.74</td>
      <td>0.76</td>
      <td>0.02</td>
    </tr>
    <tr>
      <th>LinearSVC</th>
      <td>0.75</td>
      <td>0.74</td>
      <td>0.74</td>
      <td>0.75</td>
      <td>0.04</td>
    </tr>
    <tr>
      <th>LogisticRegression</th>
      <td>0.75</td>
      <td>0.74</td>
      <td>0.74</td>
      <td>0.75</td>
      <td>0.02</td>
    </tr>
    <tr>
      <th>CalibratedClassifierCV</th>
      <td>0.75</td>
      <td>0.73</td>
      <td>0.73</td>
      <td>0.75</td>
      <td>0.15</td>
    </tr>
    <tr>
      <th>DecisionTreeClassifier</th>
      <td>0.75</td>
      <td>0.74</td>
      <td>0.74</td>
      <td>0.75</td>
      <td>0.02</td>
    </tr>
    <tr>
      <th>ExtraTreesClassifier</th>
      <td>0.74</td>
      <td>0.71</td>
      <td>0.71</td>
      <td>0.74</td>
      <td>0.16</td>
    </tr>
    <tr>
      <th>AdaBoostClassifier</th>
      <td>0.73</td>
      <td>0.71</td>
      <td>0.71</td>
      <td>0.73</td>
      <td>0.16</td>
    </tr>
    <tr>
      <th>SVC</th>
      <td>0.73</td>
      <td>0.70</td>
      <td>0.70</td>
      <td>0.73</td>
      <td>0.04</td>
    </tr>
    <tr>
      <th>Perceptron</th>
      <td>0.73</td>
      <td>0.72</td>
      <td>0.72</td>
      <td>0.73</td>
      <td>0.02</td>
    </tr>
    <tr>
      <th>PassiveAggressiveClassifier</th>
      <td>0.73</td>
      <td>0.72</td>
      <td>0.72</td>
      <td>0.73</td>
      <td>0.02</td>
    </tr>
    <tr>
      <th>RandomForestClassifier</th>
      <td>0.72</td>
      <td>0.70</td>
      <td>0.70</td>
      <td>0.72</td>
      <td>0.29</td>
    </tr>
    <tr>
      <th>NuSVC</th>
      <td>0.72</td>
      <td>0.68</td>
      <td>0.68</td>
      <td>0.72</td>
      <td>0.04</td>
    </tr>
    <tr>
      <th>LGBMClassifier</th>
      <td>0.71</td>
      <td>0.70</td>
      <td>0.70</td>
      <td>0.71</td>
      <td>0.12</td>
    </tr>
    <tr>
      <th>BaggingClassifier</th>
      <td>0.70</td>
      <td>0.67</td>
      <td>0.67</td>
      <td>0.70</td>
      <td>0.07</td>
    </tr>
    <tr>
      <th>NearestCentroid</th>
      <td>0.70</td>
      <td>0.71</td>
      <td>0.71</td>
      <td>0.71</td>
      <td>0.02</td>
    </tr>
    <tr>
      <th>SGDClassifier</th>
      <td>0.69</td>
      <td>0.66</td>
      <td>0.66</td>
      <td>0.69</td>
      <td>0.02</td>
    </tr>
    <tr>
      <th>KNeighborsClassifier</th>
      <td>0.69</td>
      <td>0.65</td>
      <td>0.65</td>
      <td>0.69</td>
      <td>0.02</td>
    </tr>
    <tr>
      <th>XGBClassifier</th>
      <td>0.69</td>
      <td>0.68</td>
      <td>0.68</td>
      <td>0.69</td>
      <td>0.21</td>
    </tr>
    <tr>
      <th>ExtraTreeClassifier</th>
      <td>0.66</td>
      <td>0.65</td>
      <td>0.65</td>
      <td>0.67</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>BernoulliNB</th>
      <td>0.66</td>
      <td>0.64</td>
      <td>0.64</td>
      <td>0.66</td>
      <td>0.06</td>
    </tr>
    <tr>
      <th>DummyClassifier</th>
      <td>0.64</td>
      <td>0.50</td>
      <td>0.50</td>
      <td>0.50</td>
      <td>0.02</td>
    </tr>
    <tr>
      <th>LabelSpreading</th>
      <td>0.64</td>
      <td>0.60</td>
      <td>0.60</td>
      <td>0.64</td>
      <td>0.04</td>
    </tr>
    <tr>
      <th>LabelPropagation</th>
      <td>0.63</td>
      <td>0.60</td>
      <td>0.60</td>
      <td>0.63</td>
      <td>0.03</td>
    </tr>
  </tbody>
</table>
</div>



# Version Two : with Feature Engineering


```python
data.head()
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
      <th>Pregnancies</th>
      <th>Glucose</th>
      <th>BloodPressure</th>
      <th>SkinThickness</th>
      <th>Insulin</th>
      <th>BMI</th>
      <th>DiabetesPedigreeFunction</th>
      <th>Age</th>
      <th>Outcome</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>148</td>
      <td>72</td>
      <td>35</td>
      <td>0</td>
      <td>33.60</td>
      <td>0.63</td>
      <td>50</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>85</td>
      <td>66</td>
      <td>29</td>
      <td>0</td>
      <td>26.60</td>
      <td>0.35</td>
      <td>31</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>183</td>
      <td>64</td>
      <td>0</td>
      <td>0</td>
      <td>23.30</td>
      <td>0.67</td>
      <td>32</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>89</td>
      <td>66</td>
      <td>23</td>
      <td>94</td>
      <td>28.10</td>
      <td>0.17</td>
      <td>21</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>137</td>
      <td>40</td>
      <td>35</td>
      <td>168</td>
      <td>43.10</td>
      <td>2.29</td>
      <td>33</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.describe(include='all').head()
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
      <th>Pregnancies</th>
      <th>Glucose</th>
      <th>BloodPressure</th>
      <th>SkinThickness</th>
      <th>Insulin</th>
      <th>BMI</th>
      <th>DiabetesPedigreeFunction</th>
      <th>Age</th>
      <th>Outcome</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>768.00</td>
      <td>768.00</td>
      <td>768.00</td>
      <td>768.00</td>
      <td>768.00</td>
      <td>768.00</td>
      <td>768.00</td>
      <td>768.00</td>
      <td>768.00</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.85</td>
      <td>120.89</td>
      <td>69.11</td>
      <td>20.54</td>
      <td>79.80</td>
      <td>31.99</td>
      <td>0.47</td>
      <td>33.24</td>
      <td>0.35</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3.37</td>
      <td>31.97</td>
      <td>19.36</td>
      <td>15.95</td>
      <td>115.24</td>
      <td>7.88</td>
      <td>0.33</td>
      <td>11.76</td>
      <td>0.48</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.08</td>
      <td>21.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.00</td>
      <td>99.00</td>
      <td>62.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>27.30</td>
      <td>0.24</td>
      <td>24.00</td>
      <td>0.00</td>
    </tr>
  </tbody>
</table>
</div>



# Pregnancies


```python
plt.figure(figsize=(20,5),dpi=200)
ax = sns.histplot(data=data, x="Pregnancies", kde=True)
ax.set_xticks(range(18))
plt.show()
```


    
![png](output_13_0.png)
    



```python
plt.figure(figsize=(20,5),dpi=200)
ax = sns.kdeplot(data=data, x="Pregnancies", hue='Outcome')
ax.set_xticks(range(18))
plt.show()
```


    
![png](output_14_0.png)
    


임신의 전체 분포는 오른쪽으로 치우친 것으로 볼 수 있습니다. 아래 상자 그림에서 볼 수 있듯이 이 열에는 확실히 이상치가 있습니다. 접근 방식은 이 열을 완전히 제거하는 것보다 이진 결과가 1과 0인 Children이라는 새 열을 생성하는 것입니다. 이 추가 기능을 통해 모델 성능이 향상될 수 있기를 바랍니다.

이상값을 처리하기 위해 이상값 감지의 가장 기본적인 형태인 **극단값 분석**이라는 접근 방식을 사용합니다. 이상치는 변수가 정규 분포(가우스)인 경우 변수의 표준 편차의 3배보다 크거나 작은 평균을 벗어나는 값입니다.
outlier = mean +/- 3* std.

그러나 변수가 치우쳐 있으므로 일반적인 통계적 접근 방식은 분위수를 계산한 다음 사분위수 범위를 계산하는 것입니다.
- IQR = 75th quantile - 25th quanitile.

이상값은 다음 상한 및 하한 경계 밖에 있습니다.:

- UPPER : 75th quantile + (IQR * 1.5)

- LOWER : 25th quanitile - (IQR * 1.5)


```python
def plots(data, variable):

    plt.figure(figsize=(16, 4), dpi=200)
    
    # histogram
    ax = plt.subplot(1, 3, 1)
    sns.kdeplot(data[variable], color='Red', fill=True)
    ax.set_title("Density Plot", fontsize=15, fontweight='normal', fontfamily='serif')
    plt.ylabel('Density', fontfamily='serif')
    plt.xlabel(variable, fontfamily='serif')
    
    for s in ['top', 'left', 'right']:
        ax.spines[s].set_visible(False)  
    
    # Q-Q plot
    ax = plt.subplot(1, 3, 2)
    stats.probplot(data[variable], dist="norm", plot=plt)
    ax.get_lines()[0].set_markerfacecolor('r')
    ax.set_title("Probability Plot", fontsize=15, fontweight='normal', fontfamily='serif')
    plt.ylabel('Quantiles', fontfamily='serif')
    plt.xlabel('Theoretical Quantiles', fontfamily='serif')
    
    for s in ['top', 'left', 'right']:
        ax.spines[s].set_visible(False)

    # boxplot
    ax = plt.subplot(1, 3, 3)
    sns.boxplot(x=data[variable], color='Lightblue')
    plt.title('Boxplot')
    ax.set_title("Boxplot", fontsize=15, fontweight='normal', fontfamily='serif')
    plt.xlabel(variable, fontfamily='serif')
    
    for s in ['top', 'left', 'right']:
        ax.spines[s].set_visible(False)    
    
    plt.show()
    
# plotting
plots(data, 'Pregnancies')
```


    
![png](output_16_0.png)
    



```python
def find_skewed_boundaries(data, variable, distance):

    IQR = data[variable].quantile(0.75) - data[variable].quantile(0.25)

    lower_boundary = data[variable].quantile(0.25) - (IQR * distance)
    upper_boundary = data[variable].quantile(0.75) + (IQR * distance)

    return upper_boundary, lower_boundary

# finding upper, lower boundary for pregnancies 
upper_boundary, lower_boundary = find_skewed_boundaries(data, 'Pregnancies', 1.5)
print(f"UPPER: {upper_boundary} & LOWER: {lower_boundary}")

# updating all values greater than upper to upper value
data.loc[data['Pregnancies'] > upper_boundary, 'Pregnancies'] = upper_boundary
```

    UPPER: 13.5 & LOWER: -6.5
    


```python
plots(data, 'Pregnancies')
```


    
![png](output_18_0.png)
    


#### Now creating a new column called More than 1 child (1/0)


```python
children = [1 if i > 0 else 0 for i in data['Pregnancies']]
data['More than 1 child?'] = children
```


```python
sns.countplot(data=data, x='More than 1 child?',
              hue='Outcome')
```




    <AxesSubplot:xlabel='More than 1 child?', ylabel='count'>




    
![png](output_21_1.png)
    


# Glucose


```python
plt.figure(figsize=(20,5))
sns.kdeplot('Glucose', hue='Outcome',data=data, palette='Set2')
plt.show()
```


    
![png](output_23_0.png)
    



```python
plots(data, 'Glucose')
```


    
![png](output_24_0.png)
    



```python
# finding upper, lower boundary for glucose 
upper_boundary, lower_boundary = find_skewed_boundaries(data, 'Glucose', 1.5)
print(f"UPPER: {upper_boundary} & LOWER: {lower_boundary}")

# updating all values greater than upper to upper value
data.loc[data['Glucose'] > upper_boundary, 'Glucose'] = upper_boundary
data.loc[data['Glucose'] < lower_boundary, 'Glucose'] = lower_boundary
```

    UPPER: 202.125 & LOWER: 37.125
    


```python
plots(data, 'Glucose')
```


    
![png](output_26_0.png)
    


# BloodPressure


```python
plt.figure(figsize=(20,5))
sns.kdeplot('BloodPressure', hue='Outcome',data=data, palette='Accent')
plt.show()
```


    
![png](output_28_0.png)
    



```python
plots(data, 'BloodPressure')
```


    
![png](output_29_0.png)
    



```python
data[data['BloodPressure']<=0].sort_values('Outcome', ascending=False)
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
      <th>Pregnancies</th>
      <th>Glucose</th>
      <th>BloodPressure</th>
      <th>SkinThickness</th>
      <th>Insulin</th>
      <th>BMI</th>
      <th>DiabetesPedigreeFunction</th>
      <th>Age</th>
      <th>Outcome</th>
      <th>More than 1 child?</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>706</th>
      <td>10.00</td>
      <td>115.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.00</td>
      <td>0.26</td>
      <td>30</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>266</th>
      <td>0.00</td>
      <td>138.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>36.30</td>
      <td>0.93</td>
      <td>25</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>468</th>
      <td>8.00</td>
      <td>120.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>30.00</td>
      <td>0.18</td>
      <td>38</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>435</th>
      <td>0.00</td>
      <td>141.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>42.40</td>
      <td>0.20</td>
      <td>29</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>535</th>
      <td>4.00</td>
      <td>132.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>32.90</td>
      <td>0.30</td>
      <td>23</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>15</th>
      <td>7.00</td>
      <td>100.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>30.00</td>
      <td>0.48</td>
      <td>32</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>357</th>
      <td>13.00</td>
      <td>129.00</td>
      <td>0</td>
      <td>30</td>
      <td>0</td>
      <td>39.90</td>
      <td>0.57</td>
      <td>44</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>332</th>
      <td>1.00</td>
      <td>180.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>43.30</td>
      <td>0.28</td>
      <td>41</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>300</th>
      <td>0.00</td>
      <td>167.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>32.30</td>
      <td>0.84</td>
      <td>30</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>269</th>
      <td>2.00</td>
      <td>146.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>27.50</td>
      <td>0.24</td>
      <td>28</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>261</th>
      <td>3.00</td>
      <td>141.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>30.00</td>
      <td>0.76</td>
      <td>27</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>604</th>
      <td>4.00</td>
      <td>183.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>28.40</td>
      <td>0.21</td>
      <td>36</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>193</th>
      <td>11.00</td>
      <td>135.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>52.30</td>
      <td>0.58</td>
      <td>40</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>619</th>
      <td>0.00</td>
      <td>119.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>32.40</td>
      <td>0.14</td>
      <td>24</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>78</th>
      <td>0.00</td>
      <td>131.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>43.20</td>
      <td>0.27</td>
      <td>26</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>484</th>
      <td>0.00</td>
      <td>145.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>44.20</td>
      <td>0.63</td>
      <td>31</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>522</th>
      <td>6.00</td>
      <td>114.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.00</td>
      <td>0.19</td>
      <td>26</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>601</th>
      <td>6.00</td>
      <td>96.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>23.70</td>
      <td>0.19</td>
      <td>28</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>589</th>
      <td>0.00</td>
      <td>73.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>21.10</td>
      <td>0.34</td>
      <td>25</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>643</th>
      <td>4.00</td>
      <td>90.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>28.00</td>
      <td>0.61</td>
      <td>31</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>697</th>
      <td>0.00</td>
      <td>99.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>25.00</td>
      <td>0.25</td>
      <td>22</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>533</th>
      <td>6.00</td>
      <td>91.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>29.80</td>
      <td>0.50</td>
      <td>31</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>703</th>
      <td>2.00</td>
      <td>129.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>38.50</td>
      <td>0.30</td>
      <td>41</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>10.00</td>
      <td>115.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>35.30</td>
      <td>0.13</td>
      <td>29</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>494</th>
      <td>3.00</td>
      <td>80.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.00</td>
      <td>0.17</td>
      <td>22</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>453</th>
      <td>2.00</td>
      <td>119.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>19.60</td>
      <td>0.83</td>
      <td>72</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>430</th>
      <td>2.00</td>
      <td>99.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>22.20</td>
      <td>0.11</td>
      <td>23</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>347</th>
      <td>3.00</td>
      <td>116.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>23.50</td>
      <td>0.19</td>
      <td>23</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>336</th>
      <td>0.00</td>
      <td>117.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>33.80</td>
      <td>0.93</td>
      <td>44</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>222</th>
      <td>7.00</td>
      <td>119.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>25.20</td>
      <td>0.21</td>
      <td>37</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>172</th>
      <td>2.00</td>
      <td>87.00</td>
      <td>0</td>
      <td>23</td>
      <td>0</td>
      <td>28.90</td>
      <td>0.77</td>
      <td>25</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>81</th>
      <td>2.00</td>
      <td>74.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.00</td>
      <td>0.10</td>
      <td>22</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>60</th>
      <td>2.00</td>
      <td>84.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.00</td>
      <td>0.30</td>
      <td>21</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>49</th>
      <td>7.00</td>
      <td>105.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.00</td>
      <td>0.30</td>
      <td>24</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>426</th>
      <td>0.00</td>
      <td>94.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.00</td>
      <td>0.26</td>
      <td>25</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



혈압 판독값이 0일 수는 없지만 상한선에 일부 이상값이 있지만 BloodPressure 열의 하한선만 확인하겠습니다.


```python
# finding upper, lower boundary for blood pressure 
upper_boundary, lower_boundary = find_skewed_boundaries(data, 'BloodPressure', 1.5)
print(f"UPPER: {upper_boundary} & LOWER: {lower_boundary}")

data.loc[data['BloodPressure'] < lower_boundary, 'BloodPressure'] = lower_boundary
```

    UPPER: 107.0 & LOWER: 35.0
    


```python
plots(data, 'BloodPressure')
```


    
![png](output_33_0.png)
    


# SkinThickness


```python
plt.figure(figsize=(20,5))
sns.kdeplot('SkinThickness', hue='Outcome',data=data, palette='Accent')
plt.show()
```


    
![png](output_35_0.png)
    



```python
plots(data, 'SkinThickness')
```


    
![png](output_36_0.png)
    


SkinThicknessRating이라는 다른 열을 만들고 SkinThickness 열을 유지하겠습니다.
- 0 = normal thickness
- 1 = mild thickness
- 2 = moderate thickness

제 생각에는 이것이 제로 측정의 양과 이것이 무엇을 나타내는지 전혀 모른다는 사실을 고려할 때 이것이 가장 적절할 것입니다. 이것이 잘못 측정되었거나 처음에는 실제로 null이었다고 가정하지만 확신할 수는 없습니다.

25번째, 중앙값 및 75번째 분위수를 사용하여 새로운 기능을 생성할 수 있습니다. 초기 열은 그대로 유지하지만 나중에 제거할지 여부를 확인합니다.:
- improves our model or;
- decreases the models optimisation.


```python
# creating values with quantiles
normal   = data['SkinThickness'].quantile(0.25)
mild     = data['SkinThickness'].quantile(0.50)
moderate = data['SkinThickness'].quantile(0.75)

# creating new column
thickness = []

for i in data['SkinThickness']:
    if i >= normal and i < mild:
        thickness.append(0)
    elif i >= mild and i < moderate:
        thickness.append(1)
    else:
        thickness.append(2)
        
data['SkinThicknessRating'] = thickness
```


```python
sns.countplot(data=data, x='SkinThicknessRating',
              hue='Outcome')
```




    <AxesSubplot:xlabel='SkinThicknessRating', ylabel='count'>




    
![png](output_39_1.png)
    



```python
data.columns
```




    Index(['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
           'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome',
           'More than 1 child?', 'SkinThicknessRating'],
          dtype='object')



# Insulin


```python
plt.figure(figsize=(20,5))
sns.kdeplot('Insulin', hue='Outcome',data=data, palette='Accent')
plt.show()
```


    
![png](output_42_0.png)
    



```python
plots(data, 'Insulin')
```


    
![png](output_43_0.png)
    



```python
# finding upper, lower boundary for insulin 
upper_boundary, lower_boundary = find_skewed_boundaries(data, 'Insulin', 1.5)
print(f"UPPER: {upper_boundary} & LOWER: {lower_boundary}")

# updating all values greater than upper to upper value
data.loc[data['Insulin'] > upper_boundary, 'Insulin'] = upper_boundary
```

    UPPER: 318.125 & LOWER: -190.875
    


```python
plots(data, 'Insulin')
```


    
![png](output_45_0.png)
    


> 

# BMI


```python
plt.figure(figsize=(20,5))
sns.kdeplot('BMI', hue='Outcome',data=data, palette='Accent')
plt.show()
```


    
![png](output_48_0.png)
    



```python
plots(data, 'BMI')
```


    
![png](output_49_0.png)
    



```python
# as BMI is normally distributed
def find_normal_boundaries(data, variable):

    upper_boundary_normal = data[variable].mean() + 3 * data[variable].std()
    lower_boundary_normal = data[variable].mean() - 3 * data[variable].std()

    return upper_boundary_normal, lower_boundary_normal

# finding upper, lower boundary for insulin 
upper_boundary, lower_boundary = find_normal_boundaries(data, 'BMI')
print(f"UPPER: {upper_boundary} & LOWER: {lower_boundary}")

# updating all values greater than upper to upper value
data.loc[data['BMI'] > upper_boundary, 'BMI'] = upper_boundary
data.loc[data['BMI'] < lower_boundary, 'BMI'] = lower_boundary
```

    UPPER: 55.645059086126295 & LOWER: 8.340097163873654
    


```python
plots(data, 'BMI')
```


    
![png](output_51_0.png)
    



```python
data.columns
```




    Index(['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
           'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome',
           'More than 1 child?', 'SkinThicknessRating'],
          dtype='object')



# DiabetesPedigreeFunction


```python
plt.figure(figsize=(20,5))
sns.kdeplot('DiabetesPedigreeFunction', hue='Outcome',data=data, palette='Accent')
plt.show()
```


    
![png](output_54_0.png)
    



```python
plots(data, 'DiabetesPedigreeFunction')
```


    
![png](output_55_0.png)
    



```python
# finding upper, lower boundary for insulin 
upper_boundary, lower_boundary = find_skewed_boundaries(data, 'DiabetesPedigreeFunction', 1.5)
print(f"UPPER: {upper_boundary} & LOWER: {lower_boundary}")

# updating all values greater than upper to upper value
data.loc[data['DiabetesPedigreeFunction'] > upper_boundary, 'DiabetesPedigreeFunction'] = upper_boundary
```

    UPPER: 1.2 & LOWER: -0.32999999999999996
    

# Age


```python
plt.figure(figsize=(20,5))
sns.kdeplot('Age', hue='Outcome',data=data, palette='Accent')
plt.show()
```


    
![png](output_58_0.png)
    



```python
plots(data, 'Age')
```


    
![png](output_59_0.png)
    



```python
# finding upper, lower boundary for insulin 
upper_boundary, lower_boundary = find_skewed_boundaries(data, 'Age', 1.5)
print(f"UPPER: {upper_boundary} & LOWER: {lower_boundary}")

# updating all values greater than upper to upper value
data.loc[data['Age'] > upper_boundary, 'Age'] = upper_boundary
```

    UPPER: 66.5 & LOWER: -1.5
    

# Second Model


```python
# splitting to train / test
X_train, X_test, y_train, y_test = train_test_split(data.drop('Outcome',axis=1), 
                                                    data['Outcome'], test_size=0.2, 
                                                    random_state= 42)

clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
models,predictions = clf.fit(X_train, X_test, y_train, y_test)
```

    100%|██████████████████████████████████████████████████████████████████████████████████| 29/29 [00:01<00:00, 18.23it/s]
    


```python
models.sort_values('Accuracy',ascending =False)
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
      <th>Accuracy</th>
      <th>Balanced Accuracy</th>
      <th>ROC AUC</th>
      <th>F1 Score</th>
      <th>Time Taken</th>
    </tr>
    <tr>
      <th>Model</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>QuadraticDiscriminantAnalysis</th>
      <td>0.79</td>
      <td>0.78</td>
      <td>0.78</td>
      <td>0.79</td>
      <td>0.02</td>
    </tr>
    <tr>
      <th>CalibratedClassifierCV</th>
      <td>0.78</td>
      <td>0.75</td>
      <td>0.75</td>
      <td>0.78</td>
      <td>0.12</td>
    </tr>
    <tr>
      <th>LinearSVC</th>
      <td>0.77</td>
      <td>0.75</td>
      <td>0.75</td>
      <td>0.77</td>
      <td>0.03</td>
    </tr>
    <tr>
      <th>LinearDiscriminantAnalysis</th>
      <td>0.77</td>
      <td>0.75</td>
      <td>0.75</td>
      <td>0.77</td>
      <td>0.02</td>
    </tr>
    <tr>
      <th>RidgeClassifierCV</th>
      <td>0.77</td>
      <td>0.75</td>
      <td>0.75</td>
      <td>0.77</td>
      <td>0.02</td>
    </tr>
    <tr>
      <th>RidgeClassifier</th>
      <td>0.77</td>
      <td>0.75</td>
      <td>0.75</td>
      <td>0.77</td>
      <td>0.02</td>
    </tr>
    <tr>
      <th>LogisticRegression</th>
      <td>0.76</td>
      <td>0.74</td>
      <td>0.74</td>
      <td>0.76</td>
      <td>0.02</td>
    </tr>
    <tr>
      <th>ExtraTreesClassifier</th>
      <td>0.75</td>
      <td>0.72</td>
      <td>0.72</td>
      <td>0.75</td>
      <td>0.21</td>
    </tr>
    <tr>
      <th>GaussianNB</th>
      <td>0.75</td>
      <td>0.75</td>
      <td>0.75</td>
      <td>0.76</td>
      <td>0.02</td>
    </tr>
    <tr>
      <th>AdaBoostClassifier</th>
      <td>0.74</td>
      <td>0.72</td>
      <td>0.72</td>
      <td>0.74</td>
      <td>0.15</td>
    </tr>
    <tr>
      <th>RandomForestClassifier</th>
      <td>0.74</td>
      <td>0.72</td>
      <td>0.72</td>
      <td>0.74</td>
      <td>0.30</td>
    </tr>
    <tr>
      <th>DecisionTreeClassifier</th>
      <td>0.74</td>
      <td>0.73</td>
      <td>0.73</td>
      <td>0.74</td>
      <td>0.02</td>
    </tr>
    <tr>
      <th>SVC</th>
      <td>0.73</td>
      <td>0.70</td>
      <td>0.70</td>
      <td>0.73</td>
      <td>0.04</td>
    </tr>
    <tr>
      <th>SGDClassifier</th>
      <td>0.73</td>
      <td>0.70</td>
      <td>0.70</td>
      <td>0.73</td>
      <td>0.02</td>
    </tr>
    <tr>
      <th>NuSVC</th>
      <td>0.72</td>
      <td>0.70</td>
      <td>0.70</td>
      <td>0.72</td>
      <td>0.04</td>
    </tr>
    <tr>
      <th>KNeighborsClassifier</th>
      <td>0.72</td>
      <td>0.69</td>
      <td>0.69</td>
      <td>0.72</td>
      <td>0.02</td>
    </tr>
    <tr>
      <th>LGBMClassifier</th>
      <td>0.71</td>
      <td>0.71</td>
      <td>0.71</td>
      <td>0.72</td>
      <td>0.10</td>
    </tr>
    <tr>
      <th>XGBClassifier</th>
      <td>0.71</td>
      <td>0.69</td>
      <td>0.69</td>
      <td>0.71</td>
      <td>0.18</td>
    </tr>
    <tr>
      <th>Perceptron</th>
      <td>0.71</td>
      <td>0.67</td>
      <td>0.67</td>
      <td>0.70</td>
      <td>0.02</td>
    </tr>
    <tr>
      <th>PassiveAggressiveClassifier</th>
      <td>0.70</td>
      <td>0.68</td>
      <td>0.68</td>
      <td>0.70</td>
      <td>0.02</td>
    </tr>
    <tr>
      <th>BaggingClassifier</th>
      <td>0.69</td>
      <td>0.66</td>
      <td>0.66</td>
      <td>0.69</td>
      <td>0.06</td>
    </tr>
    <tr>
      <th>NearestCentroid</th>
      <td>0.69</td>
      <td>0.70</td>
      <td>0.70</td>
      <td>0.69</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>BernoulliNB</th>
      <td>0.68</td>
      <td>0.66</td>
      <td>0.66</td>
      <td>0.68</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>LabelSpreading</th>
      <td>0.66</td>
      <td>0.63</td>
      <td>0.63</td>
      <td>0.66</td>
      <td>0.04</td>
    </tr>
    <tr>
      <th>LabelPropagation</th>
      <td>0.66</td>
      <td>0.63</td>
      <td>0.63</td>
      <td>0.66</td>
      <td>0.02</td>
    </tr>
    <tr>
      <th>DummyClassifier</th>
      <td>0.64</td>
      <td>0.50</td>
      <td>0.50</td>
      <td>0.50</td>
      <td>0.02</td>
    </tr>
    <tr>
      <th>ExtraTreeClassifier</th>
      <td>0.64</td>
      <td>0.61</td>
      <td>0.61</td>
      <td>0.64</td>
      <td>0.02</td>
    </tr>
  </tbody>
</table>
</div>



# Conclusion

일부 기능 엔지니어링을 수행한 후 모델이 약간 개선되었습니다. 그러나 전혀 0이 아니어야 하는 0에 값이 있기 때문에 데이터 세트에 대해 약간 확신할 수 없습니다. 따라서 모델은 부정확한 데이터에 대해 훈련됩니다. 이 데이터 세트를 더 잘 이해했다면 0이 실제로 null인지 파악할 수 있었을까요? 아니면 잘못 입력한 걸까요?
