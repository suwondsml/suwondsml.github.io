---
layout: post
title: dataon-project
authors: [Jinah Kim,Hyungjun Seo, Woojin Jo]
categories: [1기 AI/SW developers(개인 프로젝트)]
---


# Summary
  - 대회 명 : 2022 연구데이터 활용분석 경진대회
  - 대회 사이트 : [http://dataon-con.kr/](http://dataon-con.kr/)
  - 팀 명 : 데사증후군 (Data Science Syndrom)
  - `데사증후군` github 링크 : [https://github.com/Data-analysis-utilization-contest](https://github.com/Data-analysis-utilization-contest)

<br/>

# Description
  2022년 7월부터 9월까지 진행한 연구 데이터 활용분석 경진대회 관련 글입니다.

  제가 맡은 주요한 과제는 국민건강영양조사 데이터의 **메타데이터**를 연도별로 구축하는 것입니다.

  메타데이터는 각 년도별 국민건강영양조사 원시자료 이용지침서를 기준으로, 
  
  1998, 2001, 2005, 2007-2009, 2010-2012, 2013,2015, 2014, 2016-2018, 2019-2020으로 나눠 구축되었습니다.

<br/>

# workflow
  1. `각 연도별 원시자료 이용지침서를 바탕으로 엑셀파일에 "변수명, 변수설명, 내용"컬럼의 내용들을 복사하여 붙여줍니다.`

        (~~맥북은 한글파일 작업을 못해 한글파일 작업하기 번거롭다..~~)

        이용지침서에 없는 변수명을 다른 년도의 이용지침서를 참고하여 보강했습니다.

  2. `1차로 필요 데이터를 가공합니다.`
    
        "변수명, 변수설명, 내용" 컬럼 데이터 중 데이터분석에 중요하지 않은 내용들을 etc라는 컬럼에 저장합니다.  
        (~~"내용" 컬럼 데이터에 "\xa0" 나 여러 자세한 설명들이 기재되어 있어 코드가 점점 늘어납니다.~~)

        etc

  3. `2차로 선택지 설명 컬럼 데이터를 가공합니다.`

        "내용" 컬럼을 "선택지_설명"으로 이름을 바꿔줍니다.
        
        각 데이터가 없을 경우 `nan`, `NaN`으로 같은 nan값이 다른 모습으로 나타나 있어 통일시켜 주고, 각 선택지를 `,`에서 `|`로 구분합니다.

  4. `최종 데이터를 병합하여 엑셀파일로 다시 저장해줍니다.`

<br/>

---

  *다른 년도의 메타데이터는 코드내용이 각각 달라, 주로 활용한 2019-2020년도 메타데이터 구축 코드 첨부하였습니다.*

<br/>

# Contents

## 필요 모듈 임포트
  


```python
import pandas as pd
import numpy as np
```

## 필요 데이터 가공


```python
df=pd.read_excel('/Users/i/Downloads/variable.xlsx',sheet_name='2019-2020')
variable=pd.DataFrame({'변수명':[],'변수설명':[],'내용':[],'etc':[]})
df2=df[['변수명','변수설명','내용']]
df2
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
      <th>변수명</th>
      <th>변수설명</th>
      <th>내용</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>mod_d</td>
      <td>최종 DB 수정일</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ID</td>
      <td>개인 아이디</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ID_fam</td>
      <td>가구 아이디</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>year</td>
      <td>조사연도</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>region</td>
      <td>17개 시도</td>
      <td>1. 서울</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3211</th>
      <td>HE_prg</td>
      <td>임신여부</td>
      <td>0. 아니오</td>
    </tr>
    <tr>
      <th>3212</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>1. 예</td>
    </tr>
    <tr>
      <th>3213</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>8. 비해당(남자)</td>
    </tr>
    <tr>
      <th>3214</th>
      <td>HE_dprg</td>
      <td>임신개월수</td>
      <td>□□ 개월</td>
    </tr>
    <tr>
      <th>3215</th>
      <td>wt_oent</td>
      <td>(건강설문(건강면접/건강행태) OR 검진) AND 영양 연관성분석 가중치</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>3216 rows × 3 columns</p>
</div>



## 1차 데이터 생성 ( 데이터 전처리 )


```python
lis=[]
col=[]
for j in range(len(df2)):
    data1 = df2.loc[j][0]
    data2 = df2.loc[j][1]
    data3 = df2.loc[j][2]
    if j==0:
        lis.append(data1)
        lis.append(data2)
        lis.append(data3)
    else:
        if df2.loc[j][0] == data1: #변수명에 값이 있다면  #변수명, 변수설명 추가 
            variable = pd.concat([variable,pd.DataFrame({'변수명': [lis[0]],'변수설명':[lis[1]],'내용':[lis[2:]],'etc':[col]})])
            lis=[] #변수명, 변수설명, 내용
            col=[] #기타
            lis.append(data1) #변수명 
            lis.append(data2) #변수설명
            if df2.loc[j][2] != data3: #내용이 없다면
                continue #계속 
            elif df2.loc[j][2] == data3: #있다면
                if (data3[0].isdigit()==True and not data3.startswith('1일') and '.' in data3) or (data3.startswith(' ')):
                    data3=data3.replace(". "," : ")
                    data3=data3.replace("\xa0","")
                    lis.append(data3) #리스트에 일단 저장
                elif data3.startswith('(청'):
                    lis[-1]+=data3
                elif data3.startswith('1일'):
                    col.append(data3)
                else:
                    col.append(data3)

        elif df2.loc[j][0] != data1: #변수명에 값이 없다면
            if type(data3)==str:
                if (data3[0].isdigit()==True and '.' in data3) or (data3.startswith(' ')):
                    data3=data3.replace(". "," : ")
                    data3=data3.replace("\xa0","")
                    lis.append(data3)
                elif data3.startswith('(청'):
                    lis[-1]+=data3
                else:
                     col.append(data3)
```


```python
len(variable)
```



```python
    936
```



```python
variable
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
      <th>변수명</th>
      <th>변수설명</th>
      <th>내용</th>
      <th>etc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>mod_d</td>
      <td>최종 DB 수정일</td>
      <td>[nan]</td>
      <td>[]</td>
    </tr>
    <tr>
      <th>0</th>
      <td>ID</td>
      <td>개인 아이디</td>
      <td>[]</td>
      <td>[]</td>
    </tr>
    <tr>
      <th>0</th>
      <td>ID_fam</td>
      <td>가구 아이디</td>
      <td>[]</td>
      <td>[]</td>
    </tr>
    <tr>
      <th>0</th>
      <td>year</td>
      <td>조사연도</td>
      <td>[]</td>
      <td>[]</td>
    </tr>
    <tr>
      <th>0</th>
      <td>region</td>
      <td>17개 시도</td>
      <td>[1 : 서울, 2 : 부산, 3 : 대구, 4 : 인천, 5 : 광주, 6 : 대...</td>
      <td>[]</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>0</th>
      <td>HE_DMdg</td>
      <td>당뇨병 의사진단 여부</td>
      <td>[0 : 아니오, 1 : 예]</td>
      <td>[]</td>
    </tr>
    <tr>
      <th>0</th>
      <td>HE_DMdr</td>
      <td>검진당일 당뇨병 약 복용 여부</td>
      <td>[0 : 아니오, 1 : 예]</td>
      <td>[]</td>
    </tr>
    <tr>
      <th>0</th>
      <td>HE_mens</td>
      <td>생리여부</td>
      <td>[0 : 아니오, 1 : 예, 8 : 비해당(남자)]</td>
      <td>[]</td>
    </tr>
    <tr>
      <th>0</th>
      <td>HE_prg</td>
      <td>임신여부</td>
      <td>[0 : 아니오, 1 : 예, 8 : 비해당(남자)]</td>
      <td>[]</td>
    </tr>
    <tr>
      <th>0</th>
      <td>HE_dprg</td>
      <td>임신개월수</td>
      <td>[]</td>
      <td>[□□ 개월]</td>
    </tr>
  </tbody>
</table>
<p>936 rows × 4 columns</p>
</div>



## 2차 데이터 생성 ( 데이터 병합 )


```python
tt=pd.DataFrame({'선택지_설명':[]})
for data in variable['내용']:
    if len(data)==0:
        data=np.nan
        tt=pd.concat([tt,pd.DataFrame({'선택지_설명':[data]})])
    elif np.nan in data:
        data=np.nan
        tt=pd.concat([tt,pd.DataFrame({'선택지_설명':[data]})])
    else:
        data=" ".join(j+" |" for j in data)
        tt=pd.concat([tt,pd.DataFrame({'선택지_설명':[data]})])
tt
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
      <th>선택지_설명</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
    </tr>
    <tr>
      <th>0</th>
      <td>NaN</td>
    </tr>
    <tr>
      <th>0</th>
      <td>NaN</td>
    </tr>
    <tr>
      <th>0</th>
      <td>NaN</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1 : 서울 | 2 : 부산 | 3 : 대구 | 4 : 인천 | 5 : 광주 | 6...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0 : 아니오 | 1 : 예 |</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0 : 아니오 | 1 : 예 |</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0 : 아니오 | 1 : 예 | 8 : 비해당(남자) |</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0 : 아니오 | 1 : 예 | 8 : 비해당(남자) |</td>
    </tr>
    <tr>
      <th>0</th>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>936 rows × 1 columns</p>
</div>



## 최종 결과 데이터 생성 ( 1, 2차 데이터 병합 )


```python
variable_total=pd.concat([variable,tt],axis=1)
variable_total=variable_total[['변수명','변수설명','선택지_설명','etc']]
variable_total
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
      <th>변수명</th>
      <th>변수설명</th>
      <th>선택지_설명</th>
      <th>etc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>mod_d</td>
      <td>최종 DB 수정일</td>
      <td>NaN</td>
      <td>[]</td>
    </tr>
    <tr>
      <th>0</th>
      <td>ID</td>
      <td>개인 아이디</td>
      <td>NaN</td>
      <td>[]</td>
    </tr>
    <tr>
      <th>0</th>
      <td>ID_fam</td>
      <td>가구 아이디</td>
      <td>NaN</td>
      <td>[]</td>
    </tr>
    <tr>
      <th>0</th>
      <td>year</td>
      <td>조사연도</td>
      <td>NaN</td>
      <td>[]</td>
    </tr>
    <tr>
      <th>0</th>
      <td>region</td>
      <td>17개 시도</td>
      <td>1 : 서울 | 2 : 부산 | 3 : 대구 | 4 : 인천 | 5 : 광주 | 6...</td>
      <td>[]</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>0</th>
      <td>HE_DMdg</td>
      <td>당뇨병 의사진단 여부</td>
      <td>0 : 아니오 | 1 : 예 |</td>
      <td>[]</td>
    </tr>
    <tr>
      <th>0</th>
      <td>HE_DMdr</td>
      <td>검진당일 당뇨병 약 복용 여부</td>
      <td>0 : 아니오 | 1 : 예 |</td>
      <td>[]</td>
    </tr>
    <tr>
      <th>0</th>
      <td>HE_mens</td>
      <td>생리여부</td>
      <td>0 : 아니오 | 1 : 예 | 8 : 비해당(남자) |</td>
      <td>[]</td>
    </tr>
    <tr>
      <th>0</th>
      <td>HE_prg</td>
      <td>임신여부</td>
      <td>0 : 아니오 | 1 : 예 | 8 : 비해당(남자) |</td>
      <td>[]</td>
    </tr>
    <tr>
      <th>0</th>
      <td>HE_dprg</td>
      <td>임신개월수</td>
      <td>NaN</td>
      <td>[□□ 개월]</td>
    </tr>
  </tbody>
</table>
<p>936 rows × 4 columns</p>
</div>



## 최종 결과 데이터 csv파일로 변환


```python
variable_total.to_csv('hn_19_variable.csv',index=False,encoding='utf-8')
variable_total.to_csv('hn_20_variable.csv',index=False,encoding='utf-8')
```

<br/>

# 결과
  장려상

<br/>


