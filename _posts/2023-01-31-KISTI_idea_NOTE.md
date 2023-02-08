---
layout: post
title: KISTI_idea_NOTE
authors: [HyungJun Seo]
categories: [1기 AI/SW developers(개인 프로젝트)]
---

 데이터셋은 보통 분석에 적합하게 구성되어 있지 않다. 해당 프로젝트는 600여개나 되는 열의 모든 정보를 트리기반의 모델에서 원핫인코딩을 적용하여 **feature_importance**를 추출하는 것이 목적이었고, 그에 따라 **수치형 자료형**들을 **범주화** 시켜줄 필요가 있었다. 

## 원시데이터 데이터 타입 분류 문제

범주화 이전에 어떤 column이 어떤 데이터타입을 가지고 있는지를 알아야했다.

- 처음에는 value_count의 길이를 통해 분류하고자 했다
    
    ```python
    for i in range(0,len(c_li)):    
        val_c[i] = len(data20.loc[:,c_li[i]].value_counts())
    ```
    
    하지만 명확히 나뉘는 기준이 없었고 개중에 범주형 자료형인데 수치형보다 많은 종류를 가진 것 도있었다. 
    
- 결국 복붙한 메타데이터 기반으로 규칙을 만들고, 손으로 예외처리했다.
    
    이것도 그렇고 PPoA(Predict Price of Agriculture) 문제도 그렇고 괜히 규칙 찾겠다고 덤비는 것보다 그냥 손으로 데이터 파일을 수정하는게 더 빠른 것 같다.
    

## 빈도 분위 범주화(Frequency Quantile)

이런 구린 메소드를 다시 사용하게 될지는 모르겠지만.. 대체 방법론을 아는 것도 없다. 지시받은 것은 **빈도 순의로 4분위**로 수치형 자료형을 나눌 기준을 만들라는 지시였다. 

```python
li = []
for i in range(0,len(c_li)):
    if df["데이터_종류"][i] == "numeric":
        if "8" in str(df["map_해당없음"][i]) or "9" in str(df["map_모름"][i]):
            No_ad = float(df["map_해당없음"][i])
            No_re = float(df["map_모름"][i])
            li.append([i,No_ad,No_re])
        else:
            li.append([i,float("nan"),float("nan")])
```

```
out.
[[3, nan, nan],
 [10, nan, nan],
 [36, nan, 999999.0],
```

 

 데이터 타입 분류 문제가 해결된 이후 나는 2차원 배열을 통해 수치형 자료들을 범주화 하기 위한 인덱스를 저장했다. 본디 실수형 자료형이라면 저런 어이 없는 `해당없음` ,`모름` 에 대한 정보는 없었어야 하나, 수치형 자료 중에서도 연령을 포함한 정보나 소득분위와 같은 정보들은 저런 outlier를 포함하고 있었기에 저장해 주었다.

```python
def Gen_bins(index, drop_8, drop_9):
    
    if drop_8 is not False or drop_9 is not False:
        idx = []
        for i in range(0,len(data19)):
            if data19.iloc[:,index][i] == drop_8:
                idx.append(i)
            if data19.iloc[:,index][i] == drop_9:
                idx.append(i)
        data_tmp = data19.iloc[:,index].drop(idx).dropna()
    
    else:
        data_tmp = data19.iloc[:,index].dropna()
        
    cnt, bins = np.histogram(data_tmp,bins=4)
    return index, binsß

Gen_bins(41, 88.0, 99.0)
```

```
out.
(41, array([10., 15., 20., 25., 30.]))
```

`np.histogram` 메소드를 사용하여 빈도 분위로 나누어 주었는데 사실 이것은 목적에 맞지 않다

![img1](https://raw.githubusercontent.com/suwondsml/suwondsml.github.io/main/data/2023-02-08-DiseaseConqueror/img1.png)

![img2](https://raw.githubusercontent.com/suwondsml/suwondsml.github.io/main/data/2023-02-08-DiseaseConqueror/img2.png)

 왼쪽 그림이 88,99를 제거한 데이터 프레임의 히스토, 오른쪽이 `Gen_bins`의 bins를 통해 `np.digitize` 한 데이터 프레임의 히스토이다.

그렇다. `np.histogram` 메소드는 빈도 기준으로 bins를 생성하지 않는다. 그냥 min-max를 통해 나눌 뿐이다. 하지만 메타 데이터 파일에 포함한 데이터였던 만큼 그누구도 눈치채지 못했다..

 빈도 순으로 bins를 나누는 로직은 간단하다. 결측값들이 제거된 데이터 프레임의 길이에서 4로 나누고 크기 순으로 정렬된 데이터의 구간을 나누어주면 된다. 왜 당시에 이렇게 단순한 함수를 짜지않고 패키지함수를 찾아 해맸는지 이해하기 힘들다.
