---
layout: post
title: Consumer Complaints Analysis
authors: [Sunwoo Lee]
categories: [1기 AI/SW developers(개인 프로젝트)]
---


# import


```python
import numpy as np 
import pandas as pd 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
```


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings("ignore")
import matplotlib.style
```

# Load Data

##### kaggle의 consumer_complaints.csv 파일 사용


```python
RawDf = pd.read_csv('/content/drive/MyDrive/Consumer_Complaints.csv')
RawDf.head()
```





  <div id="df-6d1f5565-46f5-4d02-8f0a-6f1fe3e60a75">
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
      <th>Date received</th>
      <th>Product</th>
      <th>Sub-product</th>
      <th>Issue</th>
      <th>Sub-issue</th>
      <th>Consumer complaint narrative</th>
      <th>Company public response</th>
      <th>Company</th>
      <th>State</th>
      <th>ZIP code</th>
      <th>Tags</th>
      <th>Consumer consent provided?</th>
      <th>Submitted via</th>
      <th>Date sent to company</th>
      <th>Company response to consumer</th>
      <th>Timely response?</th>
      <th>Consumer disputed?</th>
      <th>Complaint ID</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>03/21/2017</td>
      <td>Credit reporting</td>
      <td>NaN</td>
      <td>Incorrect information on credit report</td>
      <td>Information is not mine</td>
      <td>NaN</td>
      <td>Company has responded to the consumer and the ...</td>
      <td>EXPERIAN DELAWARE GP</td>
      <td>TX</td>
      <td>77075</td>
      <td>Older American</td>
      <td>NaN</td>
      <td>Phone</td>
      <td>03/21/2017</td>
      <td>Closed with non-monetary relief</td>
      <td>Yes</td>
      <td>No</td>
      <td>2397100</td>
    </tr>
    <tr>
      <th>1</th>
      <td>04/19/2017</td>
      <td>Debt collection</td>
      <td>Other (i.e. phone, health club, etc.)</td>
      <td>Disclosure verification of debt</td>
      <td>Not disclosed as an attempt to collect</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Security Credit Services, LLC</td>
      <td>IL</td>
      <td>60643</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Web</td>
      <td>04/20/2017</td>
      <td>Closed with explanation</td>
      <td>Yes</td>
      <td>No</td>
      <td>2441777</td>
    </tr>
    <tr>
      <th>2</th>
      <td>04/19/2017</td>
      <td>Credit card</td>
      <td>NaN</td>
      <td>Other</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Company has responded to the consumer and the ...</td>
      <td>CITIBANK, N.A.</td>
      <td>IL</td>
      <td>62025</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Referral</td>
      <td>04/20/2017</td>
      <td>Closed with explanation</td>
      <td>Yes</td>
      <td>No</td>
      <td>2441830</td>
    </tr>
    <tr>
      <th>3</th>
      <td>04/14/2017</td>
      <td>Mortgage</td>
      <td>Other mortgage</td>
      <td>Loan modification,collection,foreclosure</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Company believes it acted appropriately as aut...</td>
      <td>Shellpoint Partners, LLC</td>
      <td>CA</td>
      <td>90305</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Referral</td>
      <td>04/14/2017</td>
      <td>Closed with explanation</td>
      <td>Yes</td>
      <td>No</td>
      <td>2436165</td>
    </tr>
    <tr>
      <th>4</th>
      <td>04/19/2017</td>
      <td>Credit card</td>
      <td>NaN</td>
      <td>Credit determination</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Company has responded to the consumer and the ...</td>
      <td>U.S. BANCORP</td>
      <td>LA</td>
      <td>70571</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Postal mail</td>
      <td>04/21/2017</td>
      <td>Closed with explanation</td>
      <td>Yes</td>
      <td>No</td>
      <td>2441726</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-6d1f5565-46f5-4d02-8f0a-6f1fe3e60a75')"
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
          document.querySelector('#df-6d1f5565-46f5-4d02-8f0a-6f1fe3e60a75 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-6d1f5565-46f5-4d02-8f0a-6f1fe3e60a75');
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
# shape of data
RawDf.shape
```




    (777959, 18)




```python
RawDf.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 777959 entries, 0 to 777958
    Data columns (total 18 columns):
     #   Column                        Non-Null Count   Dtype 
    ---  ------                        --------------   ----- 
     0   Date received                 777959 non-null  object
     1   Product                       777959 non-null  object
     2   Sub-product                   542822 non-null  object
     3   Issue                         777959 non-null  object
     4   Sub-issue                     320986 non-null  object
     5   Consumer complaint narrative  157865 non-null  object
     6   Company public response       197884 non-null  object
     7   Company                       777959 non-null  object
     8   State                         772056 non-null  object
     9   ZIP code                      772001 non-null  object
     10  Tags                          109264 non-null  object
     11  Consumer consent provided?    288311 non-null  object
     12  Submitted via                 777959 non-null  object
     13  Date sent to company          777959 non-null  object
     14  Company response to consumer  777959 non-null  object
     15  Timely response?              777959 non-null  object
     16  Consumer disputed?            768414 non-null  object
     17  Complaint ID                  777959 non-null  int64 
    dtypes: int64(1), object(17)
    memory usage: 106.8+ MB
    

# Handeling Missing Values


```python
# Percentage of missing values
plt.figure(figsize = (15,12))
plt.style.use('ggplot')
bar_plot = RawDf.isnull().sum().sort_values(ascending=False)*100/len(RawDf)
plt.title("Missing Values of columns By Percentage")
plt.xlabel("Percentage")
plt.ylabel("Columns")
sns.barplot(y = bar_plot.keys(), x = bar_plot.values, orient="h")
plt.show()
```


    
![png](output_9_0.png)
    




> 결측값이 있는 열들이 많이 존재하기 때문에 결측값이 10% 이상인 열들은 삭제했다.




```python
col_need_to_drop = RawDf.isnull().sum().sort_values(ascending=False
                                )[RawDf.isnull().sum().sort_values(ascending=False)*100/len(RawDf) > 10].keys()

RawDf1 = RawDf.drop(col_need_to_drop, axis=1)
RawDf1.isnull().sum().sort_values(ascending=False)*100/len(RawDf1)
```




    Consumer disputed?              1.226928
    ZIP code                        0.765850
    State                           0.758780
    Date received                   0.000000
    Product                         0.000000
    Issue                           0.000000
    Company                         0.000000
    Submitted via                   0.000000
    Date sent to company            0.000000
    Company response to consumer    0.000000
    Timely response?                0.000000
    Complaint ID                    0.000000
    dtype: float64




```python
RawDf3 = RawDf1.dropna()
RawDf3.isnull().sum()
```




    Date received                   0
    Product                         0
    Issue                           0
    Company                         0
    State                           0
    ZIP code                        0
    Submitted via                   0
    Date sent to company            0
    Company response to consumer    0
    Timely response?                0
    Consumer disputed?              0
    Complaint ID                    0
    dtype: int64





>이제 데이터 세트에 결측값이 없음을 알 수 있다.



# EDA



> 받은 날짜 및 회사로 보낸 날짜 유형은 DateTime 열이고, "Zip Code" 유형은 int64 열이어야 함




```python
RawDf3['Date received'] = pd.to_datetime(RawDf3['Date received'], format="%m/%d/%Y")
RawDf3['Date sent to company'] = pd.to_datetime(RawDf3['Date sent to company'], format="%m/%d/%Y")
```


```python
RawDf3.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 762734 entries, 0 to 777958
    Data columns (total 12 columns):
     #   Column                        Non-Null Count   Dtype         
    ---  ------                        --------------   -----         
     0   Date received                 762734 non-null  datetime64[ns]
     1   Product                       762734 non-null  object        
     2   Issue                         762734 non-null  object        
     3   Company                       762734 non-null  object        
     4   State                         762734 non-null  object        
     5   ZIP code                      762734 non-null  object        
     6   Submitted via                 762734 non-null  object        
     7   Date sent to company          762734 non-null  datetime64[ns]
     8   Company response to consumer  762734 non-null  object        
     9   Timely response?              762734 non-null  object        
     10  Consumer disputed?            762734 non-null  object        
     11  Complaint ID                  762734 non-null  int64         
    dtypes: datetime64[ns](2), int64(1), object(9)
    memory usage: 75.6+ MB
    

##### 기간별 접수된 컴플레인


```python
# Handeling Missing Valuesplt.style.use('ggplot')
sns.histplot(RawDf3['Date received'], kde=True, bins=65,element= 'poly')
plt.xlabel("Year wise trends of complaints")
plt.show()
```


    
![png](output_19_0.png)
    




> 점점 증가하는 것을 볼 수 있음



##### 컴플레인이 발생하는 회사에 컴플레인이 전달되는 시각


```python
Responce_day = RawDf3['Date sent to company']-RawDf3['Date received']
responce_day = Responce_day.value_counts()/len(Responce_day)*100
print(responce_day[:20])
print("Within 20 days " , str(responce_day[:20].sum())[:4], "% of complained solved")
```

    0 days     48.245391
    1 days     11.137828
    2 days      7.629003
    3 days      5.994226
    4 days      5.683371
    5 days      5.071231
    6 days      3.661958
    7 days      2.286774
    8 days      0.946070
    -1 days     0.923389
    9 days      0.535180
    10 days     0.465955
    13 days     0.394764
    14 days     0.391486
    11 days     0.389651
    12 days     0.373525
    15 days     0.354121
    20 days     0.295123
    21 days     0.272834
    19 days     0.259592
    dtype: float64
    Within 20 days  95.3 % of complained solved
    


```python
x_plot = responce_day[:20].keys()
y_plot = responce_day[:20].values
plt.figure(figsize = (10,10))
plt.pie(y_plot, labels=x_plot, autopct='%.2f')
plt.show()
```


    
![png](output_23_0.png)
    




> 결론: 불만사항 50%가 회사로 전송됨, 같은 날 불만사항 11%가 하루 후 회사로 전송됨, 8%가 이틀 후 회사로 전송됨





##### 어떤 제품에 불만이 많은지


```python
Product_groupby_series =  RawDf3.Product.value_counts()
Product_groupby_series
```




    Mortgage                   225394
    Debt collection            145071
    Credit reporting           139929
    Credit card                 88471
    Bank account or service     84643
    Student loan                32315
    Consumer Loan               31411
    Payday loan                  5523
    Money transfers              5155
    Prepaid card                 3774
    Other financial service      1031
    Virtual currency               17
    Name: Product, dtype: int64




```python
plt.style.use('ggplot')
plt.title("Which product have more complain")
plt.xlabel("No of complains")
plt.ylabel("Products")
sns.barplot(y = Product_groupby_series.keys(), x = Product_groupby_series.values)
plt.show()
```


    
![png](output_27_0.png)
    




> 대출에 관련된 불만사항이 가장 많은 것을 볼 수 있음





*   Analysis if mortgage




```python
mortgage_df = RawDf3[RawDf3['Product'] == 'Mortgage']
mortgage_issue = mortgage_df['Issue'].value_counts()
plt.style.use('ggplot')
sns.barplot(mortgage_issue.values, mortgage_issue.keys(), orient='h')
plt.show()
```


    
![png](output_30_0.png)
    




> 대출 중 수정, 회수, 압류에 관한 이슈가 가장 많은 것을 볼 수 있음



##### 가장 높은 불만을 받은 상위 20개 회사


```python
plt.style.use('ggplot')
Top_20_company = RawDf3['Company'].value_counts()[:20]
sns.barplot(Top_20_company.values, Top_20_company.keys(), orient='h')
plt.show()
```


    
![png](output_33_0.png)
    



```python
top_20_company_names = RawDf3['Company'].value_counts().keys()[:20]

group_df_name = []
for i in range(len(top_20_company_names)):
    group_df_name.append('top_'+ str(i+1))
company_group_by = RawDf3.groupby(by=RawDf3['Company'])

# 회사별로 별도의 데이터 프레임 만들기
for i in range(len(top_20_company_names)):
    group_df_name[i] = company_group_by.get_group(top_20_company_names[i])
    
# 분석을 위해'Company','Timely response?'열만 선택했습니다
for i in range(len(group_df_name)):
    group_df_name[i] = group_df_name[i][['Company','Timely response?']]
    
#making data frame for plotting
company = []
yes = []
no = []
for i in range(len(group_df_name)):
    company.append(group_df_name[i].value_counts().keys()[1][0])
    yes.append(group_df_name[i].value_counts().values[0])
    no.append(group_df_name[i].value_counts().values[1])
com = np.array([company]).reshape(20,1)
y = np.array([yes]).reshape(20,1)
n = np.array([no]).reshape(20,1)
plot_data = pd.DataFrame(data=com, columns=["Company"])
plot_data["yes"] = y
plot_data["no"] = n
plot_data['percentage_of timely_responce'] = plot_data["yes"]*100/(plot_data["yes"] + plot_data["no"])
plot_data
```





  <div id="df-e21d61d9-11b0-4e6b-8a96-532aafa7bc50">
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
      <th>Company</th>
      <th>yes</th>
      <th>no</th>
      <th>percentage_of timely_responce</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>BANK OF AMERICA, NATIONAL ASSOCIATION</td>
      <td>63763</td>
      <td>1552</td>
      <td>97.623823</td>
    </tr>
    <tr>
      <th>1</th>
      <td>WELLS FARGO BANK, NATIONAL ASSOCIATION</td>
      <td>50510</td>
      <td>2594</td>
      <td>95.115246</td>
    </tr>
    <tr>
      <th>2</th>
      <td>EQUIFAX, INC.</td>
      <td>48209</td>
      <td>1</td>
      <td>99.997926</td>
    </tr>
    <tr>
      <th>3</th>
      <td>EXPERIAN DELAWARE GP</td>
      <td>45467</td>
      <td>7</td>
      <td>99.984607</td>
    </tr>
    <tr>
      <th>4</th>
      <td>JPMORGAN CHASE &amp; CO.</td>
      <td>42070</td>
      <td>86</td>
      <td>99.795996</td>
    </tr>
    <tr>
      <th>5</th>
      <td>TRANSUNION INTERMEDIATE HOLDINGS, INC.</td>
      <td>39854</td>
      <td>8</td>
      <td>99.979931</td>
    </tr>
    <tr>
      <th>6</th>
      <td>CITIBANK, N.A.</td>
      <td>34057</td>
      <td>339</td>
      <td>99.014420</td>
    </tr>
    <tr>
      <th>7</th>
      <td>OCWEN LOAN SERVICING LLC</td>
      <td>23428</td>
      <td>535</td>
      <td>97.767391</td>
    </tr>
    <tr>
      <th>8</th>
      <td>CAPITAL ONE FINANCIAL CORPORATION</td>
      <td>20080</td>
      <td>65</td>
      <td>99.677339</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Navient Solutions, LLC.</td>
      <td>17874</td>
      <td>1</td>
      <td>99.994406</td>
    </tr>
    <tr>
      <th>10</th>
      <td>NATIONSTAR MORTGAGE</td>
      <td>16006</td>
      <td>125</td>
      <td>99.225095</td>
    </tr>
    <tr>
      <th>11</th>
      <td>SYNCHRONY BANK</td>
      <td>12905</td>
      <td>17</td>
      <td>99.868441</td>
    </tr>
    <tr>
      <th>12</th>
      <td>U.S. BANCORP</td>
      <td>12151</td>
      <td>91</td>
      <td>99.256657</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Ditech Financial LLC</td>
      <td>11384</td>
      <td>27</td>
      <td>99.763386</td>
    </tr>
    <tr>
      <th>14</th>
      <td>PNC Bank N.A.</td>
      <td>8486</td>
      <td>108</td>
      <td>98.743309</td>
    </tr>
    <tr>
      <th>15</th>
      <td>AMERICAN EXPRESS CENTURION BANK</td>
      <td>8218</td>
      <td>2</td>
      <td>99.975669</td>
    </tr>
    <tr>
      <th>16</th>
      <td>ENCORE CAPITAL GROUP INC.</td>
      <td>7780</td>
      <td>18</td>
      <td>99.769172</td>
    </tr>
    <tr>
      <th>17</th>
      <td>HSBC NORTH AMERICA HOLDINGS INC.</td>
      <td>6943</td>
      <td>152</td>
      <td>97.857646</td>
    </tr>
    <tr>
      <th>18</th>
      <td>DISCOVER BANK</td>
      <td>6386</td>
      <td>15</td>
      <td>99.765662</td>
    </tr>
    <tr>
      <th>19</th>
      <td>SUNTRUST BANKS, INC.</td>
      <td>6124</td>
      <td>6</td>
      <td>99.902121</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-e21d61d9-11b0-4e6b-8a96-532aafa7bc50')"
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
          document.querySelector('#df-e21d61d9-11b0-4e6b-8a96-532aafa7bc50 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-e21d61d9-11b0-4e6b-8a96-532aafa7bc50');
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




##### 대부분의 불만이 어느 주(state)에서 발생하고 있는지


```python
plt.style.use('ggplot')
plt.figure(figsize=(10,5))
sns.barplot(x=RawDf3['State'].value_counts().values[:20], y=RawDf3['State'].value_counts().keys()[:20])
plt.show()
```


    
![png](output_36_0.png)
    




> 이는 제품의 82% 이상이 해당 주에서 판매된다는 것을 의미





*  국가별 회사의 Top complain




```python
states_list = RawDf3['State'].value_counts().keys()
company_state_wise = []
for i in range(len(states_list)):
    company_state_wise.append(RawDf3.groupby(by=RawDf3['State']).get_group(states_list[i]
                                                                              )['Company'].value_counts().keys()[0])
    
top_company_state_df = pd.DataFrame({'States':states_list})
top_company_state_df['Company'] = np.array(company_state_wise) 
plt.style.use('ggplot')
sns.barplot(x=top_company_state_df['Company'].value_counts().values
           , y=top_company_state_df['Company'].value_counts().keys(), orient='h')
plt.show()
```


    
![png](output_39_0.png)
    




> Bank of america는 24개 주와 우물에서 1위를 차지함



##### 회사별 답변 제출방법


```python
x = RawDf3['Submitted via'].value_counts().keys()
y = RawDf3['Submitted via'].value_counts().values
sns.barplot(x,y)
plt.plot()
```




    []




    
![png](output_42_1.png)
    




*   제출된 데이터에 적시에 응답할 수 있는 방법




```python
submited_medium = RawDf3['Submitted via'].value_counts().keys()
disputed_complains= RawDf3.groupby(by= RawDf3['Submitted via']).get_group('Web')['Timely response?'].value_counts().values[0]
non_disputed_complains = RawDf3.groupby(by= RawDf3['Submitted via']).get_group(submited_medium[0])['Timely response?'].value_counts().values[1]
total = disputed_complains + non_disputed_complains
pct_complain_disputed = disputed_complains*100/total
pct_complain_disputed
submited_medium = RawDf3['Submitted via'].value_counts().keys()
pct_complain_disputed = []
for i in range(len(submited_medium)):
    disputed_complains= RawDf3.groupby(by= RawDf3['Submitted via']).get_group(submited_medium[i])['Timely response?'].value_counts().values[0]
    non_disputed_complains = RawDf3.groupby(by= RawDf3['Submitted via']).get_group(submited_medium[i])['Timely response?'].value_counts().values[1]
    total = disputed_complains + non_disputed_complains
    pct = disputed_complains*100/total
    pct_complain_disputed.append(pct)


plot_data_frame = pd.DataFrame([np.array(submited_medium), np.array(pct_complain_disputed)]).T
plot_data_frame.rename(columns = {0:'medium'}, inplace = True)
plot_data_frame.rename(columns = {1:'pct_timely_responce'}, inplace = True)

plt.title("Timely responced percentage submitted medium wise")
sns.barplot(data=plot_data_frame, x='medium', y = 'pct_timely_responce')
plt.ylim(90,101)
```




    (90.0, 101.0)




    
![png](output_44_1.png)
    




> Postal mail로 제출된 불만 사항은 가장 시기적절한 응답이지만 다른 모드도 96% 이상의 시기적절한 응답을 받음





*   어떤 Complain이 회사로 더 빨리 전달되는지




```python
RawDf3['time_taken_to_transfer_complains_to_company'] = RawDf3['Date sent to company']- RawDf3['Date received']
required_df = RawDf3[['Submitted via','time_taken_to_transfer_complains_to_company']]
required_df.value_counts()[:10]
```




    Submitted via  time_taken_to_transfer_complains_to_company
    Web            0 days                                         337467
                   1 days                                          39251
                   2 days                                          26834
    Referral       1 days                                          25092
    Web            3 days                                          22224
                   4 days                                          19511
    Referral       2 days                                          18123
    Web            5 days                                          16230
    Referral       4 days                                          14995
                   5 days                                          14026
    dtype: int64



# 고객 이탈 가능성

 

*   소비자의 불만사항을 제기하면서 회사 응답 시간이 4일이상이고, 적시에 답변하지 않으면, 고객이탈의 가능성이있다고 가정






> column은 'Date received','Date sent to company','Timely response?','Consumer disputed?'만 사용




```python
df_com = RawDf.loc[RawDf['Consumer disputed?']=="Yes",['Date received','Date sent to company','Timely response?','Consumer disputed?']]
```


```python
df_com.columns = ['Date_received','Date_sent_to_company','Timely_response','Consumer_disputed']
df_com.head()
```





  <div id="df-d71b6b37-9905-47d4-abec-548e271186e6">
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
      <th>Date_received</th>
      <th>Date_sent_to_company</th>
      <th>Timely_response</th>
      <th>Consumer_disputed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>35</th>
      <td>03/04/2017</td>
      <td>03/04/2017</td>
      <td>Yes</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>87</th>
      <td>03/04/2017</td>
      <td>03/04/2017</td>
      <td>Yes</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>104</th>
      <td>07/06/2014</td>
      <td>07/06/2014</td>
      <td>Yes</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>107</th>
      <td>03/04/2017</td>
      <td>03/04/2017</td>
      <td>Yes</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>167</th>
      <td>03/01/2017</td>
      <td>03/04/2017</td>
      <td>Yes</td>
      <td>Yes</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-d71b6b37-9905-47d4-abec-548e271186e6')"
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
          document.querySelector('#df-d71b6b37-9905-47d4-abec-548e271186e6 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-d71b6b37-9905-47d4-abec-548e271186e6');
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
df_com['start'] = df_com.Date_received.str.split('/').str[1]
df_com['end'] = df_com.Date_sent_to_company.str.split('/').str[1]
df_com
```





  <div id="df-9f2415d7-2ba3-4110-a00b-39764986a639">
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
      <th>Date_received</th>
      <th>Date_sent_to_company</th>
      <th>Timely_response</th>
      <th>Consumer_disputed</th>
      <th>start</th>
      <th>end</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>35</th>
      <td>03/04/2017</td>
      <td>03/04/2017</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>04</td>
      <td>04</td>
    </tr>
    <tr>
      <th>87</th>
      <td>03/04/2017</td>
      <td>03/04/2017</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>04</td>
      <td>04</td>
    </tr>
    <tr>
      <th>104</th>
      <td>07/06/2014</td>
      <td>07/06/2014</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>06</td>
      <td>06</td>
    </tr>
    <tr>
      <th>107</th>
      <td>03/04/2017</td>
      <td>03/04/2017</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>04</td>
      <td>04</td>
    </tr>
    <tr>
      <th>167</th>
      <td>03/01/2017</td>
      <td>03/04/2017</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>01</td>
      <td>04</td>
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
      <th>777936</th>
      <td>12/15/2016</td>
      <td>12/15/2016</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>15</td>
      <td>15</td>
    </tr>
    <tr>
      <th>777940</th>
      <td>08/27/2013</td>
      <td>08/28/2013</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>27</td>
      <td>28</td>
    </tr>
    <tr>
      <th>777947</th>
      <td>03/08/2016</td>
      <td>03/09/2016</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>08</td>
      <td>09</td>
    </tr>
    <tr>
      <th>777948</th>
      <td>07/01/2012</td>
      <td>07/06/2012</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>01</td>
      <td>06</td>
    </tr>
    <tr>
      <th>777957</th>
      <td>09/15/2014</td>
      <td>09/19/2014</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>15</td>
      <td>19</td>
    </tr>
  </tbody>
</table>
<p>148378 rows × 6 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-9f2415d7-2ba3-4110-a00b-39764986a639')"
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
          document.querySelector('#df-9f2415d7-2ba3-4110-a00b-39764986a639 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-9f2415d7-2ba3-4110-a00b-39764986a639');
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
df_com['Difference']=pd.to_numeric(df_com['end'])-pd.to_numeric(df_com['start'])
```


```python
df_complain = df_com.drop(['start','end'],axis=1)
df_complain.head()
```





  <div id="df-55a47c41-e4a0-4bca-a509-260939f46021">
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
      <th>Date_received</th>
      <th>Date_sent_to_company</th>
      <th>Timely_response</th>
      <th>Consumer_disputed</th>
      <th>Difference</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>35</th>
      <td>03/04/2017</td>
      <td>03/04/2017</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>0</td>
    </tr>
    <tr>
      <th>87</th>
      <td>03/04/2017</td>
      <td>03/04/2017</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>0</td>
    </tr>
    <tr>
      <th>104</th>
      <td>07/06/2014</td>
      <td>07/06/2014</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>0</td>
    </tr>
    <tr>
      <th>107</th>
      <td>03/04/2017</td>
      <td>03/04/2017</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>0</td>
    </tr>
    <tr>
      <th>167</th>
      <td>03/01/2017</td>
      <td>03/04/2017</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-55a47c41-e4a0-4bca-a509-260939f46021')"
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
          document.querySelector('#df-55a47c41-e4a0-4bca-a509-260939f46021 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-55a47c41-e4a0-4bca-a509-260939f46021');
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
df_out = df_complain[(df_complain['Difference']>=4)&(df_complain['Timely_response']=="No")]
df_out.head()
```





  <div id="df-0b0c6458-19e7-494e-90bf-c243ada0cae7">
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
      <th>Date_received</th>
      <th>Date_sent_to_company</th>
      <th>Timely_response</th>
      <th>Consumer_disputed</th>
      <th>Difference</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>11266</th>
      <td>03/08/2017</td>
      <td>03/13/2017</td>
      <td>No</td>
      <td>Yes</td>
      <td>5</td>
    </tr>
    <tr>
      <th>18675</th>
      <td>01/07/2012</td>
      <td>01/19/2012</td>
      <td>No</td>
      <td>Yes</td>
      <td>12</td>
    </tr>
    <tr>
      <th>22453</th>
      <td>01/09/2015</td>
      <td>01/16/2015</td>
      <td>No</td>
      <td>Yes</td>
      <td>7</td>
    </tr>
    <tr>
      <th>23976</th>
      <td>07/10/2015</td>
      <td>07/20/2015</td>
      <td>No</td>
      <td>Yes</td>
      <td>10</td>
    </tr>
    <tr>
      <th>24685</th>
      <td>02/04/2015</td>
      <td>02/10/2015</td>
      <td>No</td>
      <td>Yes</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-0b0c6458-19e7-494e-90bf-c243ada0cae7')"
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
          document.querySelector('#df-0b0c6458-19e7-494e-90bf-c243ada0cae7 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-0b0c6458-19e7-494e-90bf-c243ada0cae7');
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
out_per = len(df_out)/len(df_complain)*100
out_per
```




    0.3430427691436736





> 0.34%의 고객은 이탈가능성이 있으므로, 회사에서 더 주의를 기울여야함


