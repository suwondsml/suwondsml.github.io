---
layout: post
title: About DiseaseConqueror
authors: [HyungJun Seo]
categories: [1기 AI/SW developers(팀 프로젝트)]
---

# About DiseaseConqueror


<aside>
☝️  프로젝트에 있어, 유의한 인사이트를 발견하는 것은 **삼류**에 불과하다. **이류**는 그것을 통해 사용할 수 있는 제품을 만들어내는 것이고, 그 제품을 통해 사용자를 기여자로 만드는 것만이 비로소 **일류다.**

</aside>

# 문서의 의도

---

좋은 프로젝트는 다음과 같은 조건을 완수해야 한다.

- 도출해낸 결론이 **세상이 필요**로 하는 것인가?
- **사용자가** 해당 프로젝트에 **바로 기여할 수 있도록** 결론까지의 과정이 잘 기록되고 관리되었는가?
- 도출한 인사이트를 **중학생도 이해할 수 있는 레벨**로 설득할 수 있는가?

 해당 프로젝트와 대회를 이끌어가고 희생적인 위치에서도 최선을 다해준 팀원들에게 실례가 될지라도. 냉정하게 평가했을 때, 이 프로젝트는 상기한 조건들을 **모두 만족하지 못한** 프로젝트 였다. 하지만 **결과**에 있어서까지 실패는 아니었으므로, 해당 프로젝트와 대회에 관심을 가지고 있는 사용자들에게 더 높은 경험을 제공하고 싶어졌다. 

 프로젝트는 그 목적성(이를테면 컨퍼런스 혹은 공모전 같은)을 상실했다고 그 수명이 끝나는 것이 아니다. 미완성된 가치라도 잘 기록되어 전파된다면 후배들이 하게 될 시행착오를 줄여 그들이 만들어낼 더 높은 가치로 이끌수 있다. 이것이 DNA 구성원 전체가 공감하(게끔 하)는**“정보와 기술의 보존 및 전달”**이 가지는 가치라 확신한다.

 지금 이 문서는 수상한 프로젝트에 대한 기록이 아닌, **가치의 재구성**이다. 마감에 쫓기던 개발자가 아닌, **사용자의 시선**에서 우리가 개발한 소프트웨어에 접근하는 것을 가정한다. 그것을 통해 설계에서의 취약성 개선을 목표한다.

 **P.S.** 우리는 수상작 사이에서도 높은 수준의 **프로그램 메뉴얼**을 완성시켰다. 프로그램을 다루는 메뉴얼을 설계하는데 좋은 레퍼런스가 될 것이라 자부한다. 또한 팀 내부 최고 전문가가 설계한 **발표자료**들 또한 모두 DNA Project 페이지에서 공유되고 있으니 해당 프로젝트에 관심이 있는 사용자라면 **반드시** 해당 정보에 대한 접근을 하도록 권장한다.

# 설계

---

 **DiseaseConqueror의 사용자는** 영양·보건분야 **Data-inspired** 인사이트를 필요로 하는 머신러닝 비전문가이기에 소프트웨어는 다음과 같은 요소를 가져야 할 필요가 있다.

- 소프트웨어 역량이 없이도 사용 할 수 있는 **통합 머신러닝 분석 프로세스**
- 사용자 목적에 맞는 변형이 가능하도록 하는 **모듈화**

![Disease Conqueror_PPT.png](https://github.com/suwondsml/suwondsml.github.io/blob/main/data/2023-02-08-DiseaseConqueror/Disease_Conqueror_PPT.png)

우리는 데이터에 대한 접근 경험 강화를 위해 데이터 분석 프로세스를 **4가지의 모듈**로 분해했다.

- **Data_Load 모듈**은 관련된 라이브러리 및 메타데이터등의 **Requirements**를 통합해서 import하는 기능을 한다.
- **Preprocessing 모듈**은 데이터를 처리할 **MetaData**를 기반으로 결측값 처리, 데이터타입 변환, 머신러닝에 부적합한 데이터를 제거하는 기능을한다.
- **Machine_learnig 모듈**은 **Feature Importace**방법론을 적용할 수 있는 GBM, AdaBoost, XGB 등의 (트리)앙상블 기법을 적용한 모델들을 적용 후 **K-fold CV**를 적용하여 6가지의 평가지표를 통해 모델 적합성을 평가한다.
- **Feature_importance 모듈**은 feature_importance를 통해 목표질병과 입력 변수 간의 상관관계에 대한 근거를 제공하는 기능을 한다.

# 분석 시나리오

---

 사용자가 **`DiseaseConqueror`**를 통해분석을 목표로 하는 조사연도는 **2020년**이며, 분석을 목표로 하는 질병은 `“고혈압”`이다.

```python
import DiseaseConqueror as dc
#data_load module

data, meta_data = dc.dataloader(datadir, = './data', year = 2020, target='고혈압')
```

 
**[output]** 
  총 인원: 5809명
  고혈압: 1690명
  정상:  3849명
  **고혈압 유병률: 33.74%** 

# 데이터 전처리 전략

---

 `**DieaseasConqueror**`를 개발하는데 있어 데이터 전처리의 목표는 **CART**(Classification and Regression Tree) 기반 모델에서의 **Feature Importance를 추론하는데 필요한 최소한의 리소스**였다. 이를 위해 필요한 전처리는 아래와 같다.

- 결측치 처리
- 모델에 부적합한 특징 제거
- **수치형 자료 범주화**

여기서 가장 핵심적인 기능은 **데이터 범주화**라고 할 수 있다.

- 프로젝트에 사용된 Scikit-learn에서는 **지니 중요도**를 이용해서 각 feature의 중요도를 측정한다. 이렇게 계산된 feature importance는 **연속형 변수 또는 카테고리 개수가 매우 많은 변수 (**high cardinality)’들의 중요도를 부풀릴 가능성이 높다.
- 그렇기에 연속형과 범주형 자료형이 혼재되어있는 데이터에서의 분석 균일성을 위해 연속형 데이터를 **[빈도 분위** frenqency quantile로 구성하는 메소드](https://www.notion.so/727bd7abc4c249b5a3ce68a1c7ca9d98)를 개발할 필요성이 있었다.

 `**DieaseasConqueror`** 에 포함된 하나의 매소드로 제작할 수 도 있었지만, 라이브러리에 필요한 리소스를 감소시키기 위해 위와 같은 정보를 `.xlsx` 파일로 저장하여 메타데이터를 생성하는 전략을 사용하였다.

| variable | variable description | option description | etc | data type | not applicable | unknown | variable bins |
| --- | --- | --- | --- | --- | --- | --- | --- |
| age | 만나이 | 1~79 : 1~79세 | 80 :  80세이상 | | ['세'] | numeric_age | None | None | 10:20:30:40:50:60:70:80 |

```python
# data_preprocessing module

data = dc.fill_exception(data, meta_data=meta_data) #결측치 처리
data = data.drop(drop_columns, axis=1) #부적합 특징 제거
data = dc.digitize(data, meta_data=meta_data) #수치형 자료 범주화
```

# 머신러닝 모듈의 의도

---

 `**DieaseasConqueror**`를 통해 분석을 시도하면 feature importance에 대한  정보뿐 아니라 **머신러닝 분류기**에서의 지표를 제공하고 있다. 선택된 모델에 대해 **K-fold CV를 통해 지표를 측정하고** 각 모델별 fold를 평균낸 값을 기반으로 **Barplot**으로 시각화하는 기능을 제공한다. 

 `“고혈압”` 을 분석목표로한 머신러닝 모듈의 결과는 아래와 같다.

```python
# machine_learning module
results = dc.modeling(data, target='고혈압', #K-fold CV 메소드
											models=['RandomForest', 'AdaBoost', 'GradientBoosting', 'XGBoost', 'LightGBM', 'CatBoost'], 
											one_hot_encoding=True, n_splits=5, test_size=0.33, random_state=42, save=True,prePath="./")

dc.metric_plot(results, target='고혈압', #지표 시각화 메소드
							 metrics=['Accuracy', 'Precision', 'Recall', 'F1-score', 'AUC', 'TrainingTime'], save=True,prePath="./")
```

**[OUT]**

![img1](https://github.com/suwondsml/suwondsml.github.io/blob/main/data/2023-02-08-Classify-Artist/image1.jpg?raw=true)
- 사진2 : Vincent van Gogh
![img1](https://github.com/suwondsml/suwondsml.github.io/blob/main/data/2023-02-08-Classify-Artist/image1.jpg?raw=true)
- 사진2 : Vincent van Gogh
![f1.png](About%20DiseaseConqueror%20ff8a94f2883841c292425c90cf4bd9f3/f1.png)

![recall.png](About%20DiseaseConqueror%20ff8a94f2883841c292425c90cf4bd9f3/recall.png)

![precision.png](About%20DiseaseConqueror%20ff8a94f2883841c292425c90cf4bd9f3/precision.png)

![time.png](About%20DiseaseConqueror%20ff8a94f2883841c292425c90cf4bd9f3/time.png)

- Accuracy
    - 맞춘 데이터 / 전체 데이터; 가장 직관성있는 평가방법임과 동시에 과적합을 확인하기에도 적합하다.
- Precision
    - 정밀도; 모델이 True라고 분류한 것 중에서 실제 True인 것의 비율
- Recall
    - 재현율; 실제 True인 것 중에서 모델이 True라고 예측한 것의 비율
- F1-score
    - Precision과 Recall의조화 평균으로 데이터 label이 불균형 구조일 때, 모델의 성능을 정확하게 평가할 수 있음
- AUC(Area Under Curve)
    - ROC곡선을 하단의 면적을 지표화 한 값으로 분류기 평가에 사용된다.  특정 모델이 AUC가 평균에서 크게 떨어진다면, 모델 선택에서 재고해야한다. (0.5 < AUC <1.0)
- TriningTime
    - 모델 학습까지 걸린 시간

 이 지표들을 통해 이러한 결론들에 도달할 수 있다.

> 분석목표 `“고혈압”` 에 대해서는 유의미한 지표차이가 없으니 모든 모델을 사용하여, 인사이트에 접근해도 되겠구나!
> 

**OR**

> 모델별 성능에 큰 차이가 없으니, 학습시간이 적게 걸리는 **LGBM, RF** 모델만을 선별적으로 적용해 볼 수 있겠다!
> 

# Conclusion

---

 우리가 도달하고자 했던 목표는 **머신러닝 방법론**을 통해 **질병과 독립변수 간의 상관관계**에 있어 새로운 기준을 제공하는 것으로 사용자의 의사결정에 기여하는 것이었다.

```python
# feature_importance module
factor = dc.factor_extraction(data, target='고혈압', models=['RandomForest', 'AdaBoost', 'Gradient Boosting', 'XGBoost', 'LightGBM'], one_hot_encoding=True, n=40, random_state=42, visualization=False)
```

![Untitled](About%20DiseaseConqueror%20ff8a94f2883841c292425c90cf4bd9f3/Untitled.png)

그렇기에 이렇게 모델별 FI를 평균낸 지표의 활용은 썩 **유의해 보였다.**

그렇다. **유의하게 보인 것이다.** 우리가 간과했던 것은 프로젝트의 논리적 완결성의 중추인 **CART 알고리즘의 FI 차트를 평균낸 방법**이 적용하기에 **합당**한 지에 대한 논의를 거치지 않았다. 

이것이 내가 발견한 프로젝트의 **가장 큰** **취약성**이다. 

그렇기에, 다음 글에서는 프로젝트에 사용된 지니계수 기반의 특징중요도를 사용할 시에 고려할 이론적인 내용과 함께, `**DieaseasConqueror**`를 사용해 도달한 결론의 적합성 평가를 다루고자 한다.

[DiseaseConqueror 적합성 평가](https://www.notion.so/DiseaseConqueror-626074aefb9a4ce5a23cc50896ef0e58)

# Ref.

---

[분류성능평가지표 - Precision(정밀도), Recall(재현율) and Accuracy(정확도)](https://sumniya.tistory.com/26)