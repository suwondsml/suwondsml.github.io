---
layout: post
title: Titanic Model Select
authors: [Hongryul Ahn]
categories: [1기 AI/SW developers( 프로젝트)]
---

# module import


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import transformers
from transformers import AutoTokenizer, AdamW, RobertaForSequenceClassification

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from tqdm.notebook import tqdm

import codecs
from bs4 import BeautifulSoup
from html.parser import HTMLParser

from sklearn.metrics import f1_score
```

# load HTML of Bookmarks


```python
f=codecs.open("./open/bookmarks_84.html", 'r')
soup = BeautifulSoup(f.read(), 'html.parser')
print(soup)
```

    <!DOCTYPE NETSCAPE-Bookmark-file-1>
    
    <!-- This is an automatically generated file.
         It will be read and overwritten.
         DO NOT EDIT! -->
    <meta content="text/html; charset=utf-8" http-equiv="Content-Type"/>
    <title>Bookmarks</title>
    <h1>Bookmarks</h1>
    <dt><h3 add_date="1629940719" last_modified="1671427858">신문 스크랩</h3>
    <dl><p>
    <dt><a add_date="1584968656" href="https://www.mk.co.kr/opinion/contributors/view/2016/08/618807/" icon="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAABPUlEQVQ4jaWTsUoDQRCGvz0OJWLEgEXSCCo2wVcQX8PGQsHCSgt9DBFJoYKp7QURH8FGJNgaFQsLEUXBC0d21+J273bXSxTdZmZ25v93ZnZGALxvzmj+cCb274T4EawBZWTf6NLoNYi8YDGEQBppdZm5Yze2utfNMG9PfLaWUc+PoCCammZs6wRRawDwsTKbE/gZ2EQmG1TWj2FkHDFapbLRzsEwJAP3RPV5KmstICKqz/lO24tBGciHTsbeXCJuLmZ3t9dFgO3DIIL04oD+1Xnx4OUZ6elRaQkFgfuZStNrbyPvb5DdDsnhDihdShDnYOUSgE4Skt1VkBqS3je/tUsJtAnQry/Fa65fgvCaqPwAocg6LR0Z+o2djXLq1BUCre5OoB3tBZtBCJa/ABOWIPEXRQZ3FhisnoD/rfMXtEaiNPk3uHYAAAAASUVORK5CYII=">통계는 과연 거짓말을 할까? - 매일경제</a>
    <dt><a add_date="1671427337" href="https://www.joongang.co.kr/article/25126744" icon="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAA4klEQVQ4jWP8//8/AymABV3gz2+Gv38YGBkZGBgYIEaxsUO5WDX8Pb7777r5/398Y2BgYGBkZFLVYUmpYOQVQKj4jwa+fv6za813B6lvusw/qxL+Pb77/99fZHkmdCdx8TDpmTPw8jMwMTGp6jDKKDEwoqjB0EAIDFcNf/8y/P3LwMjIwMFJlIZ/ty79f/eKUVCUSccUi3HIkfLv3es/u9Z8DzH8bifxe8mk/3//oEfr//+MyInv7+HtfzcuYJSQY3byZ9K3YGDGSGkMDCgaGP7+Zfj3l4GVlYGBEVMpNg1EAABdi4Sb+zSY9AAAAABJRU5ErkJggg==">故마라도나에 우승컵 바쳤다…‘메신’의 짜릿한 ‘카타르’시스 | 중앙일보</a>
    <dt><a add_date="1671427356" href="https://www.joongang.co.kr/article/25126791">文정부 ‘통계조작’ 공방 격화…與 “통계조작성장” 野 “모욕주기” | 중앙일보</a>
    <dt><a add_date="1671427361" href="https://www.joongang.co.kr/article/25126774">정청래, 박지원 복당 공개반대 "한 번 배신하면 또 배신" | 중앙일보</a>
    <dt><a add_date="1671427364" href="https://www.joongang.co.kr/article/25126755">尹지지율, 중도·20대가 쌍끌이로 올렸다…6월 이후 첫 40%대 [리얼미터] | 중앙일보</a>
    <dt><a add_date="1671427367" href="https://www.joongang.co.kr/article/25126692">[사진] 서초동 사저 찾아 작별 인사 | 중앙일보</a>
    <dt><a add_date="1671427371" href="https://www.joongang.co.kr/article/25126618">법인세보다 장관 '빅2'가 핵심...野 반대한 한동훈·이상민 예산은 | 중앙일보</a>
    <dt><a add_date="1671427376" href="https://www.joongang.co.kr/article/25126593">'게임체인저' 신입당원은 20만...국힘, 당원 포섭전쟁은 끝났다 | 중앙일보</a>
    <dt><a add_date="1671427381" href="https://www.joongang.co.kr/article/25126580" icon="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAA4klEQVQ4jWP8//8/AymABV3gz2+Gv38YGBkZGBgYIEaxsUO5WDX8Pb7777r5/398Y2BgYGBkZFLVYUmpYOQVQKj4jwa+fv6za813B6lvusw/qxL+Pb77/99fZHkmdCdx8TDpmTPw8jMwMTGp6jDKKDEwoqjB0EAIDFcNf/8y/P3LwMjIwMFJlIZ/ty79f/eKUVCUSccUi3HIkfLv3es/u9Z8DzH8bifxe8mk/3//oEfr//+MyInv7+HtfzcuYJSQY3byZ9K3YGDGSGkMDCgaGP7+Zfj3l4GVlYGBEVMpNg1EAABdi4Sb+zSY9AAAAABJRU5ErkJggg==">‘당심 100% 룰’ 김용태 “윤핵관 당헌·당규 손바닥 뒤집듯 바꿔…한심” | 중앙일보</a>
    <dt><a add_date="1671427389" href="https://www.joongang.co.kr/article/25126571">尹 '조용한 생일'…"직언 들어줘 감사하다" 참모들 축하 메시지 | 중앙일보</a>
    <dt><a add_date="1671427395" href="https://www.joongang.co.kr/article/25126550">[속보] 북, 탄도미사일 2발 발사…“일본 EEZ 밖 낙하 추정” | 중앙일보</a>
    <dt><a add_date="1671427399" href="https://www.joongang.co.kr/article/25126364" icon="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAA4klEQVQ4jWP8//8/AymABV3gz2+Gv38YGBkZGBgYIEaxsUO5WDX8Pb7777r5/398Y2BgYGBkZFLVYUmpYOQVQKj4jwa+fv6za813B6lvusw/qxL+Pb77/99fZHkmdCdx8TDpmTPw8jMwMTGp6jDKKDEwoqjB0EAIDFcNf/8y/P3LwMjIwMFJlIZ/ty79f/eKUVCUSccUi3HIkfLv3es/u9Z8DzH8bifxe8mk/3//oEfr//+MyInv7+HtfzcuYJSQY3byZ9K3YGDGSGkMDCgaGP7+Zfj3l4GVlYGBEVMpNg1EAABdi4Sb+zSY9AAAAABJRU5ErkJggg==">이정현 "당대표 나가든 자문을 하든 소극적으로 있지 않겠다" | 중앙일보</a>
    <dt><a add_date="1671427511" href="https://www.joongang.co.kr/article/25126336" icon="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAA4klEQVQ4jWP8//8/AymABV3gz2+Gv38YGBkZGBgYIEaxsUO5WDX8Pb7777r5/398Y2BgYGBkZFLVYUmpYOQVQKj4jwa+fv6za813B6lvusw/qxL+Pb77/99fZHkmdCdx8TDpmTPw8jMwMTGp6jDKKDEwoqjB0EAIDFcNf/8y/P3LwMjIwMFJlIZ/ty79f/eKUVCUSccUi3HIkfLv3es/u9Z8DzH8bifxe8mk/3//oEfr//+MyInv7+HtfzcuYJSQY3byZ9K3YGDGSGkMDCgaGP7+Zfj3l4GVlYGBEVMpNg1EAABdi4Sb+zSY9AAAAABJRU5ErkJggg==">유승민 여론조사 보니…친윤 "이게 당원투표 100% 이유" | 중앙일보</a>
    <dt><a add_date="1671427539" href="https://www.joongang.co.kr/article/25126817">정부가 눌러 5% 안넘은 근원물가…'공공요금 인상' 내년엔 | 중앙일보</a>
    <dt><a add_date="1671427543" href="https://www.joongang.co.kr/article/25126811">LG엔솔, 오창공장 배터리 라인 신·증설에 4조 투자…1800명 채용 | 중앙일보</a>
    <dt><a add_date="1671427547" href="https://www.joongang.co.kr/article/25126800">퇴직연금 중도인출 “집 사려고, 전세 보증금 때문에” 81.6% | 중앙일보</a>
    <dt><a add_date="1671427550" href="https://www.joongang.co.kr/article/25126770">기아, 카타르 월드컵에 차량 297대 투입…브랜드 홍보 | 중앙일보</a>
    <dt><a add_date="1671427553" href="https://www.joongang.co.kr/article/25126728">[알림] '펫(pet) 톡톡': 평생 간직하세요, 인생 사진 2탄 | 중앙일보</a>
    <dt><a add_date="1671427556" href="https://www.joongang.co.kr/article/25126677">금리↑ 집값↓…“2% 청약통장 깨서 7% 대출이자 갚는다” | 중앙일보</a>
    <dt><a add_date="1671427559" href="https://www.joongang.co.kr/article/25126674">[사진] 청약경쟁률 8년 만에 한 자릿수로 | 중앙일보</a>
    <dt><a add_date="1671427562" href="https://www.joongang.co.kr/article/25126670">SK그룹, CES 2023 무대로 넷제로 역량 선보인다 | 중앙일보</a>
    <dt><a add_date="1671427566" href="https://www.joongang.co.kr/article/25126664" icon="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAA4klEQVQ4jWP8//8/AymABV3gz2+Gv38YGBkZGBgYIEaxsUO5WDX8Pb7777r5/398Y2BgYGBkZFLVYUmpYOQVQKj4jwa+fv6za813B6lvusw/qxL+Pb77/99fZHkmdCdx8TDpmTPw8jMwMTGp6jDKKDEwoqjB0EAIDFcNf/8y/P3LwMjIwMFJlIZ/ty79f/eKUVCUSccUi3HIkfLv3es/u9Z8DzH8bifxe8mk/3//oEfr//+MyInv7+HtfzcuYJSQY3byZ9K3YGDGSGkMDCgaGP7+Zfj3l4GVlYGBEVMpNg1EAABdi4Sb+zSY9AAAAABJRU5ErkJggg==">조류인플루엔자 확산…달걀값 또다시 들썩 | 중앙일보</a>
    <dt><a add_date="1671427570" href="https://www.joongang.co.kr/article/25126621">[팩플] “더 오래, 더 젊게”…달라지는 카톡, 카카오의 노림수는 | 중앙일보</a>
    <dt><a add_date="1671427573" href="https://www.joongang.co.kr/article/25126583" icon="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAA4klEQVQ4jWP8//8/AymABV3gz2+Gv38YGBkZGBgYIEaxsUO5WDX8Pb7777r5/398Y2BgYGBkZFLVYUmpYOQVQKj4jwa+fv6za813B6lvusw/qxL+Pb77/99fZHkmdCdx8TDpmTPw8jMwMTGp6jDKKDEwoqjB0EAIDFcNf/8y/P3LwMjIwMFJlIZ/ty79f/eKUVCUSccUi3HIkfLv3es/u9Z8DzH8bifxe8mk/3//oEfr//+MyInv7+HtfzcuYJSQY3byZ9K3YGDGSGkMDCgaGP7+Zfj3l4GVlYGBEVMpNg1EAABdi4Sb+zSY9AAAAABJRU5ErkJggg==">1년 4번 그들은 점을 찍는다…美 Fed, 과학인가 예언인가 | 중앙일보</a>
    <dt><a add_date="1671427576" href="https://www.joongang.co.kr/article/25126570" icon="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAA4klEQVQ4jWP8//8/AymABV3gz2+Gv38YGBkZGBgYIEaxsUO5WDX8Pb7777r5/398Y2BgYGBkZFLVYUmpYOQVQKj4jwa+fv6za813B6lvusw/qxL+Pb77/99fZHkmdCdx8TDpmTPw8jMwMTGp6jDKKDEwoqjB0EAIDFcNf/8y/P3LwMjIwMFJlIZ/ty79f/eKUVCUSccUi3HIkfLv3es/u9Z8DzH8bifxe8mk/3//oEfr//+MyInv7+HtfzcuYJSQY3byZ9K3YGDGSGkMDCgaGP7+Zfj3l4GVlYGBEVMpNg1EAABdi4Sb+zSY9AAAAABJRU5ErkJggg==">“신곡 설령 좋아도 관심없다”…3000억 저작권 굴리는 29살 | 중앙일보</a>
    <dt><a add_date="1671427581" href="https://www.joongang.co.kr/article/25126732">[VIEW] 퍼펙트스톰 긴박감 사라졌지만, 수출ㆍ고용ㆍ부동산 위기 계속 | 중앙일보</a>
    <dt><a add_date="1671427591" href="https://www.joongang.co.kr/article/25126615">밤 7시 '구찌 매장' 불꺼졌다…쇼핑몰은 거대한 미술관이 됐다 | 중앙일보</a>
    <dt><a add_date="1671427597" href="https://www.joongang.co.kr/article/25126830">한국공대, 지역산업 연계 오픈랩 혁신기술 설명회 개최 | 중앙일보</a>
    <dt><a add_date="1671427600" href="https://www.joongang.co.kr/article/25126823">전문대교육협의회, ‘2022년 전문대학인상’ 시상 | 중앙일보</a>
    <dt><a add_date="1671427603" href="https://www.joongang.co.kr/article/25126793">'공무원' 인기는 뚝…중·고교생 희망직업으로 뜬 이 계열 | 중앙일보</a>
    <dt><a add_date="1671427606" href="https://www.joongang.co.kr/article/25126784">제주 항공기·여객선 운항 정상화…"서해안엔 계속 많은 눈 예상" | 중앙일보</a>
    <dt><a add_date="1671427609" href="https://www.joongang.co.kr/article/25126756">[THINK ENGLISH] 손흥민, 마스크 벗고 런던에서 훈련 재개 | 중앙일보</a>
    <dt><a add_date="1671427613" href="https://www.joongang.co.kr/article/25126745">[소년중앙] 어디 가야 먹을 수 있나요, 찬바람 불면 돌아오는 제철 붕어빵 | 중앙일보</a>
    <dt><a add_date="1671427617" href="https://www.joongang.co.kr/article/25126790">"다시 만나자" 안받아준 前연인…준비한 흉기로 살해한 50대男 | 중앙일보</a>
    <dt><a add_date="1671427620" href="https://www.joongang.co.kr/article/25126769">말다툼하다 아내 살해한 남편, 범행 후 그 옆에서 술 마셨다 | 중앙일보</a>
    <dt><a add_date="1671427628" href="https://www.joongang.co.kr/article/25126656">[오늘의 운세] 12월 19일 | 중앙일보</a>
    <dt><a add_date="1671427632" href="https://www.joongang.co.kr/article/25126787">용산역에만 17분 멈췄다…전장연 출근길 기습에 "좀 그만하라" | 중앙일보</a>
    <dt><a add_date="1671427640" href="https://www.joongang.co.kr/article/25126666">[우리말 바루기] ‘결제’를 할까, ‘결재’를 할까? | 중앙일보</a>
    <dt><a add_date="1671427644" href="https://www.joongang.co.kr/article/25126629">여객기 109편 결항…19일까지 많은 눈, 아침 -15도 강추위 | 중앙일보</a>
    <dt><a add_date="1671427651" href="https://www.joongang.co.kr/article/25126542">"돈 뺏길지 모른다""집주인 연락 안돼" 매일 고성 터지는 이곳 | 중앙일보</a>
    <dt><a add_date="1671427658" href="https://www.joongang.co.kr/article/25126818">미얀마 최대도시 양곤서 또 폭발 사고…페리 승객 11명 부상 | 중앙일보</a>
    <dt><a add_date="1671427662" href="https://www.joongang.co.kr/article/25126753">"메시 우승 자격 있다"…'축구의 신'에 축하 건넨 '축구 황제' | 중앙일보</a>
    <dt><a add_date="1671427665" href="https://www.joongang.co.kr/article/25126649">히잡시위 지지한 ‘이란 국민배우’ 체포 | 중앙일보</a>
    <dt><a add_date="1671427670" href="https://www.joongang.co.kr/article/25126545">"우크라, 미국 만류에도 러시아軍 최고 지휘관 암살 시도" | 중앙일보</a>
    <dt><a add_date="1671427673" href="https://www.joongang.co.kr/article/25126510">'건강이상설' 푸틴 군 사령관들 소집…"우크라 작전 제안하라" | 중앙일보</a>
    <dt><a add_date="1671427676" href="https://www.joongang.co.kr/article/25126489">코로나 지원금 100억 빼돌린 美목사…호화주택 사려다 덜미 | 중앙일보</a>
    <dt><a add_date="1671427680" href="https://www.joongang.co.kr/article/25126482">한 달 전 인니 지진 사망자, 300명대 아닌 602명…재집계 수치 발표 | 중앙일보</a>
    <dt><a add_date="1671427684" href="https://www.joongang.co.kr/article/25126355">美, 中 36개 기업 수출통제 추가…AI 등 첨단산업 육성 견제 | 중앙일보</a>
    <dt><a add_date="1671427689" href="https://www.joongang.co.kr/article/25126270">백지시위 물어뜯은 中대사 "외부사주 받은 색깔혁명 냄새난다" | 중앙일보</a>
    <dt><a add_date="1671427693" href="https://www.joongang.co.kr/article/25126320" icon="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAA4klEQVQ4jWP8//8/AymABV3gz2+Gv38YGBkZGBgYIEaxsUO5WDX8Pb7777r5/398Y2BgYGBkZFLVYUmpYOQVQKj4jwa+fv6za813B6lvusw/qxL+Pb77/99fZHkmdCdx8TDpmTPw8jMwMTGp6jDKKDEwoqjB0EAIDFcNf/8y/P3LwMjIwMFJlIZ/ty79f/eKUVCUSccUi3HIkfLv3es/u9Z8DzH8bifxe8mk/3//oEfr//+MyInv7+HtfzcuYJSQY3byZ9K3YGDGSGkMDCgaGP7+Zfj3l4GVlYGBEVMpNg1EAABdi4Sb+zSY9AAAAABJRU5ErkJggg==">“워뇽이 보면 기분 좋잖아”…‘여덕몰이상’은 따로 있다 | 중앙일보</a>
    <dt><a add_date="1671427697" href="https://www.joongang.co.kr/article/25126659">저렇게 어리숙해서야, 쯧쯧…근데 왜 부럽지? | 중앙일보</a>
    <dt><a add_date="1671427700" href="https://www.joongang.co.kr/article/25126693">“혐오·거짓에 돈·권력 주는 SNS, 민주주의 위기 낳았다” | 중앙일보</a>
    <dt><a add_date="1671427703" href="https://www.joongang.co.kr/article/25126572">“유리 위 걷듯 지나온 길…정형의 그릇에 무한한 이야기 담겠다” | 중앙일보</a>
    <dt><a add_date="1671427706" href="https://www.joongang.co.kr/article/25126367">우주선 닮은 동대문 DDP에서 열리는 연말 ‘빛의 예술’ 미디어 아트쇼… 무료관람 | 중앙일보</a>
    <dt><a add_date="1671427709" href="https://www.joongang.co.kr/article/25126323" icon="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAA4klEQVQ4jWP8//8/AymABV3gz2+Gv38YGBkZGBgYIEaxsUO5WDX8Pb7777r5/398Y2BgYGBkZFLVYUmpYOQVQKj4jwa+fv6za813B6lvusw/qxL+Pb77/99fZHkmdCdx8TDpmTPw8jMwMTGp6jDKKDEwoqjB0EAIDFcNf/8y/P3LwMjIwMFJlIZ/ty79f/eKUVCUSccUi3HIkfLv3es/u9Z8DzH8bifxe8mk/3//oEfr//+MyInv7+HtfzcuYJSQY3byZ9K3YGDGSGkMDCgaGP7+Zfj3l4GVlYGBEVMpNg1EAABdi4Sb+zSY9AAAAABJRU5ErkJggg==">‘예약률 100%’ 색다른 미식…창밖으로 나온 컬리의 초대 | 중앙일보</a>
    <dt><a add_date="1671427711" href="https://www.joongang.co.kr/article/25126314">근육이 말을 듣지 않는다, 로봇공학자의 대담한 도전은[BOOK 연말연시 읽을만한 책] | 중앙일보</a>
    <dt><a add_date="1671427716" href="https://www.joongang.co.kr/article/25126310">좌우명은 '더 멀리', 이 집안을 알면 유럽사·세계사가 보인다[BOOK 연말연시 읽을만한 책] | 중앙일보</a>
    <dt><a add_date="1671427720" href="https://www.joongang.co.kr/article/25126254" icon="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAA4klEQVQ4jWP8//8/AymABV3gz2+Gv38YGBkZGBgYIEaxsUO5WDX8Pb7777r5/398Y2BgYGBkZFLVYUmpYOQVQKj4jwa+fv6za813B6lvusw/qxL+Pb77/99fZHkmdCdx8TDpmTPw8jMwMTGp6jDKKDEwoqjB0EAIDFcNf/8y/P3LwMjIwMFJlIZ/ty79f/eKUVCUSccUi3HIkfLv3es/u9Z8DzH8bifxe8mk/3//oEfr//+MyInv7+HtfzcuYJSQY3byZ9K3YGDGSGkMDCgaGP7+Zfj3l4GVlYGBEVMpNg1EAABdi4Sb+zSY9AAAAABJRU5ErkJggg==">후크 "이승기에 54억 전액 지급…합의 못했지만 분쟁 종결" | 중앙일보</a>
    <dt><a add_date="1671427749" href="https://www.joongang.co.kr/article/25126086">BTS RM, 알고보니 기름 수저?…방송중 "SK에너지!" 외친 사연 | 중앙일보</a>
    <dt><a add_date="1671427753" href="https://www.joongang.co.kr/article/25126012" icon="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAA4klEQVQ4jWP8//8/AymABV3gz2+Gv38YGBkZGBgYIEaxsUO5WDX8Pb7777r5/398Y2BgYGBkZFLVYUmpYOQVQKj4jwa+fv6za813B6lvusw/qxL+Pb77/99fZHkmdCdx8TDpmTPw8jMwMTGp6jDKKDEwoqjB0EAIDFcNf/8y/P3LwMjIwMFJlIZ/ty79f/eKUVCUSccUi3HIkfLv3es/u9Z8DzH8bifxe8mk/3//oEfr//+MyInv7+HtfzcuYJSQY3byZ9K3YGDGSGkMDCgaGP7+Zfj3l4GVlYGBEVMpNg1EAABdi4Sb+zSY9AAAAABJRU5ErkJggg==">31만1200 vs 7만5000…‘걸그룹 천하’ 증명한 숫자 | 중앙일보</a>
    <dt><a add_date="1671427764" href="https://www.joongang.co.kr/article/25126726">메시, 골든볼 MVP에도 뽑혔다…월드컵 역대 첫 2회 수상 | 중앙일보</a>
    <dt><a add_date="1671427768" href="https://www.joongang.co.kr/article/25126667">[삼성화재배 AI와 함께하는 바둑 해설] 조용히 완성된 철갑 공격군 | 중앙일보</a>
    <dt><a add_date="1671427771" href="https://www.joongang.co.kr/article/25126624">황선우, 1분40초 벽 깼다…쇼트코스 자유형 200m '2연패' | 중앙일보</a>
    <dt><a add_date="1671427775" href="https://www.joongang.co.kr/article/25126483">메르스? 코로나? 결승 이틀 앞둔 프랑스 발칵, 5명이 아프다 | 중앙일보</a>
    <dt><a add_date="1671427779" href="https://www.joongang.co.kr/article/25126246">벤투 감독, 폴란드 사령탑 거론…레반도프스키 지휘하나 | 중앙일보</a>
    <dt><a add_date="1671427783" href="https://www.joongang.co.kr/article/25126081">벤투 사단 '비트박스 코치', 한국 이웃에 남긴 마지막 선물 | 중앙일보</a>
    <dt><a add_date="1671427787" href="https://www.joongang.co.kr/article/25125831">역시, 메시 | 중앙일보</a>
    <dt><a add_date="1671427791" href="https://www.joongang.co.kr/article/25125786">여자배구 조송화 계약해지 무효소송 패소… 잔여연봉 4억원 수령 어려워져 | 중앙일보</a>
    <dt><a add_date="1671427795" href="https://www.joongang.co.kr/article/25126376">황선우 앞세운 쇼트코스 계영 800ｍ 韓신기록...역대최고 4위 | 중앙일보</a>
    <dt><a add_date="1671427799" href="https://www.joongang.co.kr/article/25126500">난임, 이것도 원인이었어? 여성 70%가 걸리는 '은밀한 질환' [건강한 가족] | 중앙일보</a>
    <dt><a add_date="1671427802" href="https://www.joongang.co.kr/article/25126300">삼겹살·목살이면 충분해, 연말 홈파티 모두가 좋아할 요리 추천 [쿠킹] | 중앙일보</a>
    <dt><a add_date="1671427805" href="https://www.joongang.co.kr/article/25125708">"약혼반지 대신 이 팔찌를" 60년대에 젠더리스 컨셉, 뉴요커 홀린 까르띠에 [더 하이엔드] | 중앙일보</a>
    <dt><a add_date="1671427808" href="https://www.joongang.co.kr/article/25124732">"답 안하는 걸로 할게요"…조규성 3초 고민하게 만든 질문 | 중앙일보</a>
    <dt><a add_date="1671427812" href="https://www.joongang.co.kr/article/25124162" icon="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAA4klEQVQ4jWP8//8/AymABV3gz2+Gv38YGBkZGBgYIEaxsUO5WDX8Pb7777r5/398Y2BgYGBkZFLVYUmpYOQVQKj4jwa+fv6za813B6lvusw/qxL+Pb77/99fZHkmdCdx8TDpmTPw8jMwMTGp6jDKKDEwoqjB0EAIDFcNf/8y/P3LwMjIwMFJlIZ/ty79f/eKUVCUSccUi3HIkfLv3es/u9Z8DzH8bifxe8mk/3//oEfr//+MyInv7+HtfzcuYJSQY3byZ9K3YGDGSGkMDCgaGP7+Zfj3l4GVlYGBEVMpNg1EAABdi4Sb+zSY9AAAAABJRU5ErkJggg==">“이 사회 전체가 의자였구나” 샤워실 목욕의자의 재발견 | 중앙일보</a>
    <dt><a add_date="1671427816" href="https://www.joongang.co.kr/article/25122912">카페인 있어도 고혈압과 무관? 커피가 '두얼굴 헐크'인 까닭 [건강한 가족] | 중앙일보</a>
    <dt><a add_date="1671427819" href="https://www.joongang.co.kr/article/25122804">올가을 가뭄에 당도↑, 최근 비에 산도↓…역대급 당산비 감귤 [e즐펀한 토크] | 중앙일보</a>
    <dt><a add_date="1671427825" href="https://www.joongang.co.kr/article/25121510">예거 르쿨트르와 레터링 아티스트 알렉스 트로슈가 만드는 새로운 의미 [더 하이엔드] | 중앙일보</a>
    <dt><a add_date="1671427829" href="https://www.joongang.co.kr/article/25126648">[사랑방] 보훈처, 가수 이미자에 감사패 수여 | 중앙일보</a>
    <dt><a add_date="1671427837" href="https://www.joongang.co.kr/article/25125827">[사랑방] SK그룹, 이웃사랑성금 120억 | 중앙일보</a>
    <dt><a add_date="1671427840" href="https://www.joongang.co.kr/article/25125508">[사랑방] 권모세 회장 등 4명 ‘서울대AMP대상’ | 중앙일보</a>
    <dt><a add_date="1671427843" href="https://www.joongang.co.kr/article/25125189">[인사] 에스원 外 | 중앙일보</a>
    <dt><a add_date="1671427849" href="https://www.joongang.co.kr/article/25124850">[사진] 서울국제포럼-나카소네 세계평화연 ‘한·일 신시대를 향해’ 포럼 | 중앙일보</a>
    <dt><a add_date="1671427852" href="https://www.joongang.co.kr/article/25124840">[사랑방] 이덕로 한국행정학회장 취임 | 중앙일보</a>
    <dt><a add_date="1671427858" href="https://www.joongang.co.kr/article/25123714" icon="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAA4klEQVQ4jWP8//8/AymABV3gz2+Gv38YGBkZGBgYIEaxsUO5WDX8Pb7777r5/398Y2BgYGBkZFLVYUmpYOQVQKj4jwa+fv6za813B6lvusw/qxL+Pb77/99fZHkmdCdx8TDpmTPw8jMwMTGp6jDKKDEwoqjB0EAIDFcNf/8y/P3LwMjIwMFJlIZ/ty79f/eKUVCUSccUi3HIkfLv3es/u9Z8DzH8bifxe8mk/3//oEfr//+MyInv7+HtfzcuYJSQY3byZ9K3YGDGSGkMDCgaGP7+Zfj3l4GVlYGBEVMpNg1EAABdi4Sb+zSY9AAAAABJRU5ErkJggg==">[사진] 중앙광고대상 시상식 | 중앙일보</a>
    </dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></p></dl><p>
    <p>
    </p></p></dt>


# Dataframe

**기사 title은 a태그의 text로 존재함**

## tag dataframe

- a태그 전체를 column값으로 가짐


```python
# a 태그 전체 추출
tag_li = soup.find_all('a')
print(tag_li)
```

    [<a add_date="1584968656" href="https://www.mk.co.kr/opinion/contributors/view/2016/08/618807/" icon="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAABPUlEQVQ4jaWTsUoDQRCGvz0OJWLEgEXSCCo2wVcQX8PGQsHCSgt9DBFJoYKp7QURH8FGJNgaFQsLEUXBC0d21+J273bXSxTdZmZ25v93ZnZGALxvzmj+cCb274T4EawBZWTf6NLoNYi8YDGEQBppdZm5Yze2utfNMG9PfLaWUc+PoCCammZs6wRRawDwsTKbE/gZ2EQmG1TWj2FkHDFapbLRzsEwJAP3RPV5KmstICKqz/lO24tBGciHTsbeXCJuLmZ3t9dFgO3DIIL04oD+1Xnx4OUZ6elRaQkFgfuZStNrbyPvb5DdDsnhDihdShDnYOUSgE4Skt1VkBqS3je/tUsJtAnQry/Fa65fgvCaqPwAocg6LR0Z+o2djXLq1BUCre5OoB3tBZtBCJa/ABOWIPEXRQZ3FhisnoD/rfMXtEaiNPk3uHYAAAAASUVORK5CYII=">통계는 과연 거짓말을 할까? - 매일경제</a>, <a add_date="1671427337" href="https://www.joongang.co.kr/article/25126744" icon="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAA4klEQVQ4jWP8//8/AymABV3gz2+Gv38YGBkZGBgYIEaxsUO5WDX8Pb7777r5/398Y2BgYGBkZFLVYUmpYOQVQKj4jwa+fv6za813B6lvusw/qxL+Pb77/99fZHkmdCdx8TDpmTPw8jMwMTGp6jDKKDEwoqjB0EAIDFcNf/8y/P3LwMjIwMFJlIZ/ty79f/eKUVCUSccUi3HIkfLv3es/u9Z8DzH8bifxe8mk/3//oEfr//+MyInv7+HtfzcuYJSQY3byZ9K3YGDGSGkMDCgaGP7+Zfj3l4GVlYGBEVMpNg1EAABdi4Sb+zSY9AAAAABJRU5ErkJggg==">故마라도나에 우승컵 바쳤다…‘메신’의 짜릿한 ‘카타르’시스 | 중앙일보</a>, <a add_date="1671427356" href="https://www.joongang.co.kr/article/25126791">文정부 ‘통계조작’ 공방 격화…與 “통계조작성장” 野 “모욕주기” | 중앙일보</a>, <a add_date="1671427361" href="https://www.joongang.co.kr/article/25126774">정청래, 박지원 복당 공개반대 "한 번 배신하면 또 배신" | 중앙일보</a>, <a add_date="1671427364" href="https://www.joongang.co.kr/article/25126755">尹지지율, 중도·20대가 쌍끌이로 올렸다…6월 이후 첫 40%대 [리얼미터] | 중앙일보</a>, <a add_date="1671427367" href="https://www.joongang.co.kr/article/25126692">[사진] 서초동 사저 찾아 작별 인사 | 중앙일보</a>, <a add_date="1671427371" href="https://www.joongang.co.kr/article/25126618">법인세보다 장관 '빅2'가 핵심...野 반대한 한동훈·이상민 예산은 | 중앙일보</a>, <a add_date="1671427376" href="https://www.joongang.co.kr/article/25126593">'게임체인저' 신입당원은 20만...국힘, 당원 포섭전쟁은 끝났다 | 중앙일보</a>, <a add_date="1671427381" href="https://www.joongang.co.kr/article/25126580" icon="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAA4klEQVQ4jWP8//8/AymABV3gz2+Gv38YGBkZGBgYIEaxsUO5WDX8Pb7777r5/398Y2BgYGBkZFLVYUmpYOQVQKj4jwa+fv6za813B6lvusw/qxL+Pb77/99fZHkmdCdx8TDpmTPw8jMwMTGp6jDKKDEwoqjB0EAIDFcNf/8y/P3LwMjIwMFJlIZ/ty79f/eKUVCUSccUi3HIkfLv3es/u9Z8DzH8bifxe8mk/3//oEfr//+MyInv7+HtfzcuYJSQY3byZ9K3YGDGSGkMDCgaGP7+Zfj3l4GVlYGBEVMpNg1EAABdi4Sb+zSY9AAAAABJRU5ErkJggg==">‘당심 100% 룰’ 김용태 “윤핵관 당헌·당규 손바닥 뒤집듯 바꿔…한심” | 중앙일보</a>, <a add_date="1671427389" href="https://www.joongang.co.kr/article/25126571">尹 '조용한 생일'…"직언 들어줘 감사하다" 참모들 축하 메시지 | 중앙일보</a>, <a add_date="1671427395" href="https://www.joongang.co.kr/article/25126550">[속보] 북, 탄도미사일 2발 발사…“일본 EEZ 밖 낙하 추정” | 중앙일보</a>, <a add_date="1671427399" href="https://www.joongang.co.kr/article/25126364" icon="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAA4klEQVQ4jWP8//8/AymABV3gz2+Gv38YGBkZGBgYIEaxsUO5WDX8Pb7777r5/398Y2BgYGBkZFLVYUmpYOQVQKj4jwa+fv6za813B6lvusw/qxL+Pb77/99fZHkmdCdx8TDpmTPw8jMwMTGp6jDKKDEwoqjB0EAIDFcNf/8y/P3LwMjIwMFJlIZ/ty79f/eKUVCUSccUi3HIkfLv3es/u9Z8DzH8bifxe8mk/3//oEfr//+MyInv7+HtfzcuYJSQY3byZ9K3YGDGSGkMDCgaGP7+Zfj3l4GVlYGBEVMpNg1EAABdi4Sb+zSY9AAAAABJRU5ErkJggg==">이정현 "당대표 나가든 자문을 하든 소극적으로 있지 않겠다" | 중앙일보</a>, <a add_date="1671427511" href="https://www.joongang.co.kr/article/25126336" icon="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAA4klEQVQ4jWP8//8/AymABV3gz2+Gv38YGBkZGBgYIEaxsUO5WDX8Pb7777r5/398Y2BgYGBkZFLVYUmpYOQVQKj4jwa+fv6za813B6lvusw/qxL+Pb77/99fZHkmdCdx8TDpmTPw8jMwMTGp6jDKKDEwoqjB0EAIDFcNf/8y/P3LwMjIwMFJlIZ/ty79f/eKUVCUSccUi3HIkfLv3es/u9Z8DzH8bifxe8mk/3//oEfr//+MyInv7+HtfzcuYJSQY3byZ9K3YGDGSGkMDCgaGP7+Zfj3l4GVlYGBEVMpNg1EAABdi4Sb+zSY9AAAAABJRU5ErkJggg==">유승민 여론조사 보니…친윤 "이게 당원투표 100% 이유" | 중앙일보</a>, <a add_date="1671427539" href="https://www.joongang.co.kr/article/25126817">정부가 눌러 5% 안넘은 근원물가…'공공요금 인상' 내년엔 | 중앙일보</a>, <a add_date="1671427543" href="https://www.joongang.co.kr/article/25126811">LG엔솔, 오창공장 배터리 라인 신·증설에 4조 투자…1800명 채용 | 중앙일보</a>, <a add_date="1671427547" href="https://www.joongang.co.kr/article/25126800">퇴직연금 중도인출 “집 사려고, 전세 보증금 때문에” 81.6% | 중앙일보</a>, <a add_date="1671427550" href="https://www.joongang.co.kr/article/25126770">기아, 카타르 월드컵에 차량 297대 투입…브랜드 홍보 | 중앙일보</a>, <a add_date="1671427553" href="https://www.joongang.co.kr/article/25126728">[알림] '펫(pet) 톡톡': 평생 간직하세요, 인생 사진 2탄 | 중앙일보</a>, <a add_date="1671427556" href="https://www.joongang.co.kr/article/25126677">금리↑ 집값↓…“2% 청약통장 깨서 7% 대출이자 갚는다” | 중앙일보</a>, <a add_date="1671427559" href="https://www.joongang.co.kr/article/25126674">[사진] 청약경쟁률 8년 만에 한 자릿수로 | 중앙일보</a>, <a add_date="1671427562" href="https://www.joongang.co.kr/article/25126670">SK그룹, CES 2023 무대로 넷제로 역량 선보인다 | 중앙일보</a>, <a add_date="1671427566" href="https://www.joongang.co.kr/article/25126664" icon="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAA4klEQVQ4jWP8//8/AymABV3gz2+Gv38YGBkZGBgYIEaxsUO5WDX8Pb7777r5/398Y2BgYGBkZFLVYUmpYOQVQKj4jwa+fv6za813B6lvusw/qxL+Pb77/99fZHkmdCdx8TDpmTPw8jMwMTGp6jDKKDEwoqjB0EAIDFcNf/8y/P3LwMjIwMFJlIZ/ty79f/eKUVCUSccUi3HIkfLv3es/u9Z8DzH8bifxe8mk/3//oEfr//+MyInv7+HtfzcuYJSQY3byZ9K3YGDGSGkMDCgaGP7+Zfj3l4GVlYGBEVMpNg1EAABdi4Sb+zSY9AAAAABJRU5ErkJggg==">조류인플루엔자 확산…달걀값 또다시 들썩 | 중앙일보</a>, <a add_date="1671427570" href="https://www.joongang.co.kr/article/25126621">[팩플] “더 오래, 더 젊게”…달라지는 카톡, 카카오의 노림수는 | 중앙일보</a>, <a add_date="1671427573" href="https://www.joongang.co.kr/article/25126583" icon="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAA4klEQVQ4jWP8//8/AymABV3gz2+Gv38YGBkZGBgYIEaxsUO5WDX8Pb7777r5/398Y2BgYGBkZFLVYUmpYOQVQKj4jwa+fv6za813B6lvusw/qxL+Pb77/99fZHkmdCdx8TDpmTPw8jMwMTGp6jDKKDEwoqjB0EAIDFcNf/8y/P3LwMjIwMFJlIZ/ty79f/eKUVCUSccUi3HIkfLv3es/u9Z8DzH8bifxe8mk/3//oEfr//+MyInv7+HtfzcuYJSQY3byZ9K3YGDGSGkMDCgaGP7+Zfj3l4GVlYGBEVMpNg1EAABdi4Sb+zSY9AAAAABJRU5ErkJggg==">1년 4번 그들은 점을 찍는다…美 Fed, 과학인가 예언인가 | 중앙일보</a>, <a add_date="1671427576" href="https://www.joongang.co.kr/article/25126570" icon="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAA4klEQVQ4jWP8//8/AymABV3gz2+Gv38YGBkZGBgYIEaxsUO5WDX8Pb7777r5/398Y2BgYGBkZFLVYUmpYOQVQKj4jwa+fv6za813B6lvusw/qxL+Pb77/99fZHkmdCdx8TDpmTPw8jMwMTGp6jDKKDEwoqjB0EAIDFcNf/8y/P3LwMjIwMFJlIZ/ty79f/eKUVCUSccUi3HIkfLv3es/u9Z8DzH8bifxe8mk/3//oEfr//+MyInv7+HtfzcuYJSQY3byZ9K3YGDGSGkMDCgaGP7+Zfj3l4GVlYGBEVMpNg1EAABdi4Sb+zSY9AAAAABJRU5ErkJggg==">“신곡 설령 좋아도 관심없다”…3000억 저작권 굴리는 29살 | 중앙일보</a>, <a add_date="1671427581" href="https://www.joongang.co.kr/article/25126732">[VIEW] 퍼펙트스톰 긴박감 사라졌지만, 수출ㆍ고용ㆍ부동산 위기 계속 | 중앙일보</a>, <a add_date="1671427591" href="https://www.joongang.co.kr/article/25126615">밤 7시 '구찌 매장' 불꺼졌다…쇼핑몰은 거대한 미술관이 됐다 | 중앙일보</a>, <a add_date="1671427597" href="https://www.joongang.co.kr/article/25126830">한국공대, 지역산업 연계 오픈랩 혁신기술 설명회 개최 | 중앙일보</a>, <a add_date="1671427600" href="https://www.joongang.co.kr/article/25126823">전문대교육협의회, ‘2022년 전문대학인상’ 시상 | 중앙일보</a>, <a add_date="1671427603" href="https://www.joongang.co.kr/article/25126793">'공무원' 인기는 뚝…중·고교생 희망직업으로 뜬 이 계열 | 중앙일보</a>, <a add_date="1671427606" href="https://www.joongang.co.kr/article/25126784">제주 항공기·여객선 운항 정상화…"서해안엔 계속 많은 눈 예상" | 중앙일보</a>, <a add_date="1671427609" href="https://www.joongang.co.kr/article/25126756">[THINK ENGLISH] 손흥민, 마스크 벗고 런던에서 훈련 재개 | 중앙일보</a>, <a add_date="1671427613" href="https://www.joongang.co.kr/article/25126745">[소년중앙] 어디 가야 먹을 수 있나요, 찬바람 불면 돌아오는 제철 붕어빵 | 중앙일보</a>, <a add_date="1671427617" href="https://www.joongang.co.kr/article/25126790">"다시 만나자" 안받아준 前연인…준비한 흉기로 살해한 50대男 | 중앙일보</a>, <a add_date="1671427620" href="https://www.joongang.co.kr/article/25126769">말다툼하다 아내 살해한 남편, 범행 후 그 옆에서 술 마셨다 | 중앙일보</a>, <a add_date="1671427628" href="https://www.joongang.co.kr/article/25126656">[오늘의 운세] 12월 19일 | 중앙일보</a>, <a add_date="1671427632" href="https://www.joongang.co.kr/article/25126787">용산역에만 17분 멈췄다…전장연 출근길 기습에 "좀 그만하라" | 중앙일보</a>, <a add_date="1671427640" href="https://www.joongang.co.kr/article/25126666">[우리말 바루기] ‘결제’를 할까, ‘결재’를 할까? | 중앙일보</a>, <a add_date="1671427644" href="https://www.joongang.co.kr/article/25126629">여객기 109편 결항…19일까지 많은 눈, 아침 -15도 강추위 | 중앙일보</a>, <a add_date="1671427651" href="https://www.joongang.co.kr/article/25126542">"돈 뺏길지 모른다""집주인 연락 안돼" 매일 고성 터지는 이곳 | 중앙일보</a>, <a add_date="1671427658" href="https://www.joongang.co.kr/article/25126818">미얀마 최대도시 양곤서 또 폭발 사고…페리 승객 11명 부상 | 중앙일보</a>, <a add_date="1671427662" href="https://www.joongang.co.kr/article/25126753">"메시 우승 자격 있다"…'축구의 신'에 축하 건넨 '축구 황제' | 중앙일보</a>, <a add_date="1671427665" href="https://www.joongang.co.kr/article/25126649">히잡시위 지지한 ‘이란 국민배우’ 체포 | 중앙일보</a>, <a add_date="1671427670" href="https://www.joongang.co.kr/article/25126545">"우크라, 미국 만류에도 러시아軍 최고 지휘관 암살 시도" | 중앙일보</a>, <a add_date="1671427673" href="https://www.joongang.co.kr/article/25126510">'건강이상설' 푸틴 군 사령관들 소집…"우크라 작전 제안하라" | 중앙일보</a>, <a add_date="1671427676" href="https://www.joongang.co.kr/article/25126489">코로나 지원금 100억 빼돌린 美목사…호화주택 사려다 덜미 | 중앙일보</a>, <a add_date="1671427680" href="https://www.joongang.co.kr/article/25126482">한 달 전 인니 지진 사망자, 300명대 아닌 602명…재집계 수치 발표 | 중앙일보</a>, <a add_date="1671427684" href="https://www.joongang.co.kr/article/25126355">美, 中 36개 기업 수출통제 추가…AI 등 첨단산업 육성 견제 | 중앙일보</a>, <a add_date="1671427689" href="https://www.joongang.co.kr/article/25126270">백지시위 물어뜯은 中대사 "외부사주 받은 색깔혁명 냄새난다" | 중앙일보</a>, <a add_date="1671427693" href="https://www.joongang.co.kr/article/25126320" icon="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAA4klEQVQ4jWP8//8/AymABV3gz2+Gv38YGBkZGBgYIEaxsUO5WDX8Pb7777r5/398Y2BgYGBkZFLVYUmpYOQVQKj4jwa+fv6za813B6lvusw/qxL+Pb77/99fZHkmdCdx8TDpmTPw8jMwMTGp6jDKKDEwoqjB0EAIDFcNf/8y/P3LwMjIwMFJlIZ/ty79f/eKUVCUSccUi3HIkfLv3es/u9Z8DzH8bifxe8mk/3//oEfr//+MyInv7+HtfzcuYJSQY3byZ9K3YGDGSGkMDCgaGP7+Zfj3l4GVlYGBEVMpNg1EAABdi4Sb+zSY9AAAAABJRU5ErkJggg==">“워뇽이 보면 기분 좋잖아”…‘여덕몰이상’은 따로 있다 | 중앙일보</a>, <a add_date="1671427697" href="https://www.joongang.co.kr/article/25126659">저렇게 어리숙해서야, 쯧쯧…근데 왜 부럽지? | 중앙일보</a>, <a add_date="1671427700" href="https://www.joongang.co.kr/article/25126693">“혐오·거짓에 돈·권력 주는 SNS, 민주주의 위기 낳았다” | 중앙일보</a>, <a add_date="1671427703" href="https://www.joongang.co.kr/article/25126572">“유리 위 걷듯 지나온 길…정형의 그릇에 무한한 이야기 담겠다” | 중앙일보</a>, <a add_date="1671427706" href="https://www.joongang.co.kr/article/25126367">우주선 닮은 동대문 DDP에서 열리는 연말 ‘빛의 예술’ 미디어 아트쇼… 무료관람 | 중앙일보</a>, <a add_date="1671427709" href="https://www.joongang.co.kr/article/25126323" icon="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAA4klEQVQ4jWP8//8/AymABV3gz2+Gv38YGBkZGBgYIEaxsUO5WDX8Pb7777r5/398Y2BgYGBkZFLVYUmpYOQVQKj4jwa+fv6za813B6lvusw/qxL+Pb77/99fZHkmdCdx8TDpmTPw8jMwMTGp6jDKKDEwoqjB0EAIDFcNf/8y/P3LwMjIwMFJlIZ/ty79f/eKUVCUSccUi3HIkfLv3es/u9Z8DzH8bifxe8mk/3//oEfr//+MyInv7+HtfzcuYJSQY3byZ9K3YGDGSGkMDCgaGP7+Zfj3l4GVlYGBEVMpNg1EAABdi4Sb+zSY9AAAAABJRU5ErkJggg==">‘예약률 100%’ 색다른 미식…창밖으로 나온 컬리의 초대 | 중앙일보</a>, <a add_date="1671427711" href="https://www.joongang.co.kr/article/25126314">근육이 말을 듣지 않는다, 로봇공학자의 대담한 도전은[BOOK 연말연시 읽을만한 책] | 중앙일보</a>, <a add_date="1671427716" href="https://www.joongang.co.kr/article/25126310">좌우명은 '더 멀리', 이 집안을 알면 유럽사·세계사가 보인다[BOOK 연말연시 읽을만한 책] | 중앙일보</a>, <a add_date="1671427720" href="https://www.joongang.co.kr/article/25126254" icon="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAA4klEQVQ4jWP8//8/AymABV3gz2+Gv38YGBkZGBgYIEaxsUO5WDX8Pb7777r5/398Y2BgYGBkZFLVYUmpYOQVQKj4jwa+fv6za813B6lvusw/qxL+Pb77/99fZHkmdCdx8TDpmTPw8jMwMTGp6jDKKDEwoqjB0EAIDFcNf/8y/P3LwMjIwMFJlIZ/ty79f/eKUVCUSccUi3HIkfLv3es/u9Z8DzH8bifxe8mk/3//oEfr//+MyInv7+HtfzcuYJSQY3byZ9K3YGDGSGkMDCgaGP7+Zfj3l4GVlYGBEVMpNg1EAABdi4Sb+zSY9AAAAABJRU5ErkJggg==">후크 "이승기에 54억 전액 지급…합의 못했지만 분쟁 종결" | 중앙일보</a>, <a add_date="1671427749" href="https://www.joongang.co.kr/article/25126086">BTS RM, 알고보니 기름 수저?…방송중 "SK에너지!" 외친 사연 | 중앙일보</a>, <a add_date="1671427753" href="https://www.joongang.co.kr/article/25126012" icon="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAA4klEQVQ4jWP8//8/AymABV3gz2+Gv38YGBkZGBgYIEaxsUO5WDX8Pb7777r5/398Y2BgYGBkZFLVYUmpYOQVQKj4jwa+fv6za813B6lvusw/qxL+Pb77/99fZHkmdCdx8TDpmTPw8jMwMTGp6jDKKDEwoqjB0EAIDFcNf/8y/P3LwMjIwMFJlIZ/ty79f/eKUVCUSccUi3HIkfLv3es/u9Z8DzH8bifxe8mk/3//oEfr//+MyInv7+HtfzcuYJSQY3byZ9K3YGDGSGkMDCgaGP7+Zfj3l4GVlYGBEVMpNg1EAABdi4Sb+zSY9AAAAABJRU5ErkJggg==">31만1200 vs 7만5000…‘걸그룹 천하’ 증명한 숫자 | 중앙일보</a>, <a add_date="1671427764" href="https://www.joongang.co.kr/article/25126726">메시, 골든볼 MVP에도 뽑혔다…월드컵 역대 첫 2회 수상 | 중앙일보</a>, <a add_date="1671427768" href="https://www.joongang.co.kr/article/25126667">[삼성화재배 AI와 함께하는 바둑 해설] 조용히 완성된 철갑 공격군 | 중앙일보</a>, <a add_date="1671427771" href="https://www.joongang.co.kr/article/25126624">황선우, 1분40초 벽 깼다…쇼트코스 자유형 200m '2연패' | 중앙일보</a>, <a add_date="1671427775" href="https://www.joongang.co.kr/article/25126483">메르스? 코로나? 결승 이틀 앞둔 프랑스 발칵, 5명이 아프다 | 중앙일보</a>, <a add_date="1671427779" href="https://www.joongang.co.kr/article/25126246">벤투 감독, 폴란드 사령탑 거론…레반도프스키 지휘하나 | 중앙일보</a>, <a add_date="1671427783" href="https://www.joongang.co.kr/article/25126081">벤투 사단 '비트박스 코치', 한국 이웃에 남긴 마지막 선물 | 중앙일보</a>, <a add_date="1671427787" href="https://www.joongang.co.kr/article/25125831">역시, 메시 | 중앙일보</a>, <a add_date="1671427791" href="https://www.joongang.co.kr/article/25125786">여자배구 조송화 계약해지 무효소송 패소… 잔여연봉 4억원 수령 어려워져 | 중앙일보</a>, <a add_date="1671427795" href="https://www.joongang.co.kr/article/25126376">황선우 앞세운 쇼트코스 계영 800ｍ 韓신기록...역대최고 4위 | 중앙일보</a>, <a add_date="1671427799" href="https://www.joongang.co.kr/article/25126500">난임, 이것도 원인이었어? 여성 70%가 걸리는 '은밀한 질환' [건강한 가족] | 중앙일보</a>, <a add_date="1671427802" href="https://www.joongang.co.kr/article/25126300">삼겹살·목살이면 충분해, 연말 홈파티 모두가 좋아할 요리 추천 [쿠킹] | 중앙일보</a>, <a add_date="1671427805" href="https://www.joongang.co.kr/article/25125708">"약혼반지 대신 이 팔찌를" 60년대에 젠더리스 컨셉, 뉴요커 홀린 까르띠에 [더 하이엔드] | 중앙일보</a>, <a add_date="1671427808" href="https://www.joongang.co.kr/article/25124732">"답 안하는 걸로 할게요"…조규성 3초 고민하게 만든 질문 | 중앙일보</a>, <a add_date="1671427812" href="https://www.joongang.co.kr/article/25124162" icon="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAA4klEQVQ4jWP8//8/AymABV3gz2+Gv38YGBkZGBgYIEaxsUO5WDX8Pb7777r5/398Y2BgYGBkZFLVYUmpYOQVQKj4jwa+fv6za813B6lvusw/qxL+Pb77/99fZHkmdCdx8TDpmTPw8jMwMTGp6jDKKDEwoqjB0EAIDFcNf/8y/P3LwMjIwMFJlIZ/ty79f/eKUVCUSccUi3HIkfLv3es/u9Z8DzH8bifxe8mk/3//oEfr//+MyInv7+HtfzcuYJSQY3byZ9K3YGDGSGkMDCgaGP7+Zfj3l4GVlYGBEVMpNg1EAABdi4Sb+zSY9AAAAABJRU5ErkJggg==">“이 사회 전체가 의자였구나” 샤워실 목욕의자의 재발견 | 중앙일보</a>, <a add_date="1671427816" href="https://www.joongang.co.kr/article/25122912">카페인 있어도 고혈압과 무관? 커피가 '두얼굴 헐크'인 까닭 [건강한 가족] | 중앙일보</a>, <a add_date="1671427819" href="https://www.joongang.co.kr/article/25122804">올가을 가뭄에 당도↑, 최근 비에 산도↓…역대급 당산비 감귤 [e즐펀한 토크] | 중앙일보</a>, <a add_date="1671427825" href="https://www.joongang.co.kr/article/25121510">예거 르쿨트르와 레터링 아티스트 알렉스 트로슈가 만드는 새로운 의미 [더 하이엔드] | 중앙일보</a>, <a add_date="1671427829" href="https://www.joongang.co.kr/article/25126648">[사랑방] 보훈처, 가수 이미자에 감사패 수여 | 중앙일보</a>, <a add_date="1671427837" href="https://www.joongang.co.kr/article/25125827">[사랑방] SK그룹, 이웃사랑성금 120억 | 중앙일보</a>, <a add_date="1671427840" href="https://www.joongang.co.kr/article/25125508">[사랑방] 권모세 회장 등 4명 ‘서울대AMP대상’ | 중앙일보</a>, <a add_date="1671427843" href="https://www.joongang.co.kr/article/25125189">[인사] 에스원 外 | 중앙일보</a>, <a add_date="1671427849" href="https://www.joongang.co.kr/article/25124850">[사진] 서울국제포럼-나카소네 세계평화연 ‘한·일 신시대를 향해’ 포럼 | 중앙일보</a>, <a add_date="1671427852" href="https://www.joongang.co.kr/article/25124840">[사랑방] 이덕로 한국행정학회장 취임 | 중앙일보</a>, <a add_date="1671427858" href="https://www.joongang.co.kr/article/25123714" icon="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAA4klEQVQ4jWP8//8/AymABV3gz2+Gv38YGBkZGBgYIEaxsUO5WDX8Pb7777r5/398Y2BgYGBkZFLVYUmpYOQVQKj4jwa+fv6za813B6lvusw/qxL+Pb77/99fZHkmdCdx8TDpmTPw8jMwMTGp6jDKKDEwoqjB0EAIDFcNf/8y/P3LwMjIwMFJlIZ/ty79f/eKUVCUSccUi3HIkfLv3es/u9Z8DzH8bifxe8mk/3//oEfr//+MyInv7+HtfzcuYJSQY3byZ9K3YGDGSGkMDCgaGP7+Zfj3l4GVlYGBEVMpNg1EAABdi4Sb+zSY9AAAAABJRU5ErkJggg==">[사진] 중앙광고대상 시상식 | 중앙일보</a>]



```python
index = list(np.arange(len(tag_li)))
df_tag = pd.DataFrame({'index':index, 'tag' : tag_li})
df_tag
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
      <th>index</th>
      <th>tag</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>[통계는 과연 거짓말을 할까? - 매일경제]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>[故마라도나에 우승컵 바쳤다…‘메신’의 짜릿한 ‘카타르’시스 | 중앙일보]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>[文정부 ‘통계조작’ 공방 격화…與 “통계조작성장” 野 “모욕주기” | 중앙일보]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>[정청래, 박지원 복당 공개반대 "한 번 배신하면 또 배신" | 중앙일보]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>[尹지지율, 중도·20대가 쌍끌이로 올렸다…6월 이후 첫 40%대 [리얼미터] | ...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>79</th>
      <td>79</td>
      <td>[[사랑방] 권모세 회장 등 4명 ‘서울대AMP대상’ | 중앙일보]</td>
    </tr>
    <tr>
      <th>80</th>
      <td>80</td>
      <td>[[인사] 에스원 外 | 중앙일보]</td>
    </tr>
    <tr>
      <th>81</th>
      <td>81</td>
      <td>[[사진] 서울국제포럼-나카소네 세계평화연 ‘한·일 신시대를 향해’ 포럼 | 중앙일보]</td>
    </tr>
    <tr>
      <th>82</th>
      <td>82</td>
      <td>[[사랑방] 이덕로 한국행정학회장 취임 | 중앙일보]</td>
    </tr>
    <tr>
      <th>83</th>
      <td>83</td>
      <td>[[사진] 중앙광고대상 시상식 | 중앙일보]</td>
    </tr>
  </tbody>
</table>
<p>84 rows × 2 columns</p>
</div>



## title dataframe

- 기사 제목을 column값으로 가짐


```python
class MyHTMLParser(HTMLParser):

    def __init__(self):
        HTMLParser.__init__(self)
        self.is_a = False
        self.li = []
    def handle_starttag(self, tag, attrs):
        if tag == 'a':  # <a> 태그 시작
            self.is_a = True

    def handle_endtag(self, tag):
        if tag == 'a':  # <a> 태그 닫힘
            self.is_a = False

    def handle_data(self, data):
        if self.is_a:  # <a>~<a> 구간인 경우
            self.li.append(data)     # 데이터를 출력
```


```python
with open('./open/bookmarks_84.html') as f:
    parser = MyHTMLParser()
    parser.feed(f.read())

print(parser.li)
```

    ['통계는 과연 거짓말을 할까? - 매일경제', '故마라도나에 우승컵 바쳤다…‘메신’의 짜릿한 ‘카타르’시스 | 중앙일보', '文정부 ‘통계조작’ 공방 격화…與 “통계조작성장” 野 “모욕주기” | 중앙일보', '정청래, 박지원 복당 공개반대 "한 번 배신하면 또 배신" | 중앙일보', '尹지지율, 중도·20대가 쌍끌이로 올렸다…6월 이후 첫 40%대 [리얼미터] | 중앙일보', '[사진] 서초동 사저 찾아 작별 인사 | 중앙일보', "법인세보다 장관 '빅2'가 핵심...野 반대한 한동훈·이상민 예산은 | 중앙일보", "'게임체인저' 신입당원은 20만...국힘, 당원 포섭전쟁은 끝났다 | 중앙일보", '‘당심 100% 룰’ 김용태 “윤핵관 당헌·당규 손바닥 뒤집듯 바꿔…한심” | 중앙일보', '尹 \'조용한 생일\'…"직언 들어줘 감사하다" 참모들 축하 메시지 | 중앙일보', '[속보] 북, 탄도미사일 2발 발사…“일본 EEZ 밖 낙하 추정” | 중앙일보', '이정현 "당대표 나가든 자문을 하든 소극적으로 있지 않겠다" | 중앙일보', '유승민 여론조사 보니…친윤 "이게 당원투표 100% 이유" | 중앙일보', "정부가 눌러 5% 안넘은 근원물가…'공공요금 인상' 내년엔 | 중앙일보", 'LG엔솔, 오창공장 배터리 라인 신·증설에 4조 투자…1800명 채용 | 중앙일보', '퇴직연금 중도인출 “집 사려고, 전세 보증금 때문에” 81.6% | 중앙일보', '기아, 카타르 월드컵에 차량 297대 투입…브랜드 홍보 | 중앙일보', "[알림] '펫(pet) 톡톡': 평생 간직하세요, 인생 사진 2탄 | 중앙일보", '금리↑ 집값↓…“2% 청약통장 깨서 7% 대출이자 갚는다” | 중앙일보', '[사진] 청약경쟁률 8년 만에 한 자릿수로 | 중앙일보', 'SK그룹, CES 2023 무대로 넷제로 역량 선보인다 | 중앙일보', '조류인플루엔자 확산…달걀값 또다시 들썩 | 중앙일보', '[팩플] “더 오래, 더 젊게”…달라지는 카톡, 카카오의 노림수는 | 중앙일보', '1년 4번 그들은 점을 찍는다…美 Fed, 과학인가 예언인가 | 중앙일보', '“신곡 설령 좋아도 관심없다”…3000억 저작권 굴리는 29살 | 중앙일보', '[VIEW] 퍼펙트스톰 긴박감 사라졌지만, 수출ㆍ고용ㆍ부동산 위기 계속 | 중앙일보', "밤 7시 '구찌 매장' 불꺼졌다…쇼핑몰은 거대한 미술관이 됐다 | 중앙일보", '한국공대, 지역산업 연계 오픈랩 혁신기술 설명회 개최 | 중앙일보', '전문대교육협의회, ‘2022년 전문대학인상’ 시상 | 중앙일보', "'공무원' 인기는 뚝…중·고교생 희망직업으로 뜬 이 계열 | 중앙일보", '제주 항공기·여객선 운항 정상화…"서해안엔 계속 많은 눈 예상" | 중앙일보', '[THINK ENGLISH] 손흥민, 마스크 벗고 런던에서 훈련 재개 | 중앙일보', '[소년중앙] 어디 가야 먹을 수 있나요, 찬바람 불면 돌아오는 제철 붕어빵 | 중앙일보', '"다시 만나자" 안받아준 前연인…준비한 흉기로 살해한 50대男 | 중앙일보', '말다툼하다 아내 살해한 남편, 범행 후 그 옆에서 술 마셨다 | 중앙일보', '[오늘의 운세] 12월 19일 | 중앙일보', '용산역에만 17분 멈췄다…전장연 출근길 기습에 "좀 그만하라" | 중앙일보', '[우리말 바루기] ‘결제’를 할까, ‘결재’를 할까? | 중앙일보', '여객기 109편 결항…19일까지 많은 눈, 아침 -15도 강추위 | 중앙일보', '"돈 뺏길지 모른다""집주인 연락 안돼" 매일 고성 터지는 이곳 | 중앙일보', '미얀마 최대도시 양곤서 또 폭발 사고…페리 승객 11명 부상 | 중앙일보', '"메시 우승 자격 있다"…\'축구의 신\'에 축하 건넨 \'축구 황제\' | 중앙일보', '히잡시위 지지한 ‘이란 국민배우’ 체포 | 중앙일보', '"우크라, 미국 만류에도 러시아軍 최고 지휘관 암살 시도" | 중앙일보', '\'건강이상설\' 푸틴 군 사령관들 소집…"우크라 작전 제안하라" | 중앙일보', '코로나 지원금 100억 빼돌린 美목사…호화주택 사려다 덜미 | 중앙일보', '한 달 전 인니 지진 사망자, 300명대 아닌 602명…재집계 수치 발표 | 중앙일보', '美, 中 36개 기업 수출통제 추가…AI 등 첨단산업 육성 견제 | 중앙일보', '백지시위 물어뜯은 中대사 "외부사주 받은 색깔혁명 냄새난다" | 중앙일보', '“워뇽이 보면 기분 좋잖아”…‘여덕몰이상’은 따로 있다 | 중앙일보', '저렇게 어리숙해서야, 쯧쯧…근데 왜 부럽지? | 중앙일보', '“혐오·거짓에 돈·권력 주는 SNS, 민주주의 위기 낳았다” | 중앙일보', '“유리 위 걷듯 지나온 길…정형의 그릇에 무한한 이야기 담겠다” | 중앙일보', '우주선 닮은 동대문 DDP에서 열리는 연말 ‘빛의 예술’ 미디어 아트쇼… 무료관람 | 중앙일보', '‘예약률 100%’ 색다른 미식…창밖으로 나온 컬리의 초대 | 중앙일보', '근육이 말을 듣지 않는다, 로봇공학자의 대담한 도전은[BOOK 연말연시 읽을만한 책] | 중앙일보', "좌우명은 '더 멀리', 이 집안을 알면 유럽사·세계사가 보인다[BOOK 연말연시 읽을만한 책] | 중앙일보", '후크 "이승기에 54억 전액 지급…합의 못했지만 분쟁 종결" | 중앙일보', 'BTS RM, 알고보니 기름 수저?…방송중 "SK에너지!" 외친 사연 | 중앙일보', '31만1200 vs 7만5000…‘걸그룹 천하’ 증명한 숫자 | 중앙일보', '메시, 골든볼 MVP에도 뽑혔다…월드컵 역대 첫 2회 수상 | 중앙일보', '[삼성화재배 AI와 함께하는 바둑 해설] 조용히 완성된 철갑 공격군 | 중앙일보', "황선우, 1분40초 벽 깼다…쇼트코스 자유형 200m '2연패' | 중앙일보", '메르스? 코로나? 결승 이틀 앞둔 프랑스 발칵, 5명이 아프다 | 중앙일보', '벤투 감독, 폴란드 사령탑 거론…레반도프스키 지휘하나 | 중앙일보', "벤투 사단 '비트박스 코치', 한국 이웃에 남긴 마지막 선물 | 중앙일보", '역시, 메시 | 중앙일보', '여자배구 조송화 계약해지 무효소송 패소… 잔여연봉 4억원 수령 어려워져 | 중앙일보', '황선우 앞세운 쇼트코스 계영 800ｍ 韓신기록...역대최고 4위 | 중앙일보', "난임, 이것도 원인이었어? 여성 70%가 걸리는 '은밀한 질환' [건강한 가족] | 중앙일보", '삼겹살·목살이면 충분해, 연말 홈파티 모두가 좋아할 요리 추천 [쿠킹] | 중앙일보', '"약혼반지 대신 이 팔찌를" 60년대에 젠더리스 컨셉, 뉴요커 홀린 까르띠에 [더 하이엔드] | 중앙일보', '"답 안하는 걸로 할게요"…조규성 3초 고민하게 만든 질문 | 중앙일보', '“이 사회 전체가 의자였구나” 샤워실 목욕의자의 재발견 | 중앙일보', "카페인 있어도 고혈압과 무관? 커피가 '두얼굴 헐크'인 까닭 [건강한 가족] | 중앙일보", '올가을 가뭄에 당도↑, 최근 비에 산도↓…역대급 당산비 감귤 [e즐펀한 토크] | 중앙일보', '예거 르쿨트르와 레터링 아티스트 알렉스 트로슈가 만드는 새로운 의미 [더 하이엔드] | 중앙일보', '[사랑방] 보훈처, 가수 이미자에 감사패 수여 | 중앙일보', '[사랑방] SK그룹, 이웃사랑성금 120억 | 중앙일보', '[사랑방] 권모세 회장 등 4명 ‘서울대AMP대상’ | 중앙일보', '[인사] 에스원 外 | 중앙일보', '[사진] 서울국제포럼-나카소네 세계평화연 ‘한·일 신시대를 향해’ 포럼 | 중앙일보', '[사랑방] 이덕로 한국행정학회장 취임 | 중앙일보', '[사진] 중앙광고대상 시상식 | 중앙일보']



```python
index = list(np.arange(len(parser.li)))
df = pd.DataFrame({'index':index, 'title' : parser.li})
df
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
      <th>index</th>
      <th>title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>통계는 과연 거짓말을 할까? - 매일경제</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>故마라도나에 우승컵 바쳤다…‘메신’의 짜릿한 ‘카타르’시스 | 중앙일보</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>文정부 ‘통계조작’ 공방 격화…與 “통계조작성장” 野 “모욕주기” | 중앙일보</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>정청래, 박지원 복당 공개반대 "한 번 배신하면 또 배신" | 중앙일보</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>尹지지율, 중도·20대가 쌍끌이로 올렸다…6월 이후 첫 40%대 [리얼미터] | 중앙일보</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>79</th>
      <td>79</td>
      <td>[사랑방] 권모세 회장 등 4명 ‘서울대AMP대상’ | 중앙일보</td>
    </tr>
    <tr>
      <th>80</th>
      <td>80</td>
      <td>[인사] 에스원 外 | 중앙일보</td>
    </tr>
    <tr>
      <th>81</th>
      <td>81</td>
      <td>[사진] 서울국제포럼-나카소네 세계평화연 ‘한·일 신시대를 향해’ 포럼 | 중앙일보</td>
    </tr>
    <tr>
      <th>82</th>
      <td>82</td>
      <td>[사랑방] 이덕로 한국행정학회장 취임 | 중앙일보</td>
    </tr>
    <tr>
      <th>83</th>
      <td>83</td>
      <td>[사진] 중앙광고대상 시상식 | 중앙일보</td>
    </tr>
  </tbody>
</table>
<p>84 rows × 2 columns</p>
</div>



## sub dataframe

- 예측값을 입력할 답안지


```python
df_sub = pd.DataFrame({'index':index, 'topic_idx' : [0 for i in range(len(df))]})
df_sub
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
      <th>index</th>
      <th>topic_idx</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>79</th>
      <td>79</td>
      <td>0</td>
    </tr>
    <tr>
      <th>80</th>
      <td>80</td>
      <td>0</td>
    </tr>
    <tr>
      <th>81</th>
      <td>81</td>
      <td>0</td>
    </tr>
    <tr>
      <th>82</th>
      <td>82</td>
      <td>0</td>
    </tr>
    <tr>
      <th>83</th>
      <td>83</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>84 rows × 2 columns</p>
</div>



## dataframe to csv


```python
df.to_csv("./open/df.csv", index = False)
df_sub.to_csv("./open/df_sub.csv", index = False)
df_tag.to_csv("./open/df_tag.csv", index = False)
```

# load Data


```python
train = pd.read_csv('./open/train_data.csv')
test = pd.read_csv('./open/test_data.csv')
topic_dict = pd.read_csv('./open/topic_dict.csv')
sample_submission = pd.read_csv('./open/sample_submission.csv')

# 우리 데이터셋
bookmarks = pd.read_csv('./open/df.csv')
our_sub = pd.read_csv('./open/df_sub.csv')
our_tag = pd.read_csv('./open/df_tag.csv')

# df_c : 직접 기사에 라벨링을 한 데이터프레임
our_cset = pd.read_csv('./open/df_c.csv', encoding = 'CP949')
```


```python
train
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
      <th>index</th>
      <th>title</th>
      <th>topic_idx</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>인천→핀란드 항공기 결항…휴가철 여행객 분통</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>실리콘밸리 넘어서겠다…구글 15조원 들여 美전역 거점화</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>이란 외무 긴장완화 해결책은 미국이 경제전쟁 멈추는 것</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>NYT 클린턴 측근韓기업 특수관계 조명…공과 사 맞물려종합</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>시진핑 트럼프에 중미 무역협상 조속 타결 희망</td>
      <td>4</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>45649</th>
      <td>45649</td>
      <td>KB금융 미국 IB 스티펠과 제휴…선진국 시장 공략</td>
      <td>1</td>
    </tr>
    <tr>
      <th>45650</th>
      <td>45650</td>
      <td>1보 서울시교육청 신종코로나 확산에 개학 연기·휴업 검토</td>
      <td>2</td>
    </tr>
    <tr>
      <th>45651</th>
      <td>45651</td>
      <td>게시판 키움증권 2020 키움 영웅전 실전투자대회</td>
      <td>1</td>
    </tr>
    <tr>
      <th>45652</th>
      <td>45652</td>
      <td>답변하는 배기동 국립중앙박물관장</td>
      <td>2</td>
    </tr>
    <tr>
      <th>45653</th>
      <td>45653</td>
      <td>2020 한국인터넷기자상 시상식 내달 1일 개최…특별상 김성후</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>45654 rows × 3 columns</p>
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
      <th>index</th>
      <th>title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>45654</td>
      <td>유튜브 내달 2일까지 크리에이터 지원 공간 운영</td>
    </tr>
    <tr>
      <th>1</th>
      <td>45655</td>
      <td>어버이날 맑다가 흐려져…남부지방 옅은 황사</td>
    </tr>
    <tr>
      <th>2</th>
      <td>45656</td>
      <td>내년부터 국가RD 평가 때 논문건수는 반영 않는다</td>
    </tr>
    <tr>
      <th>3</th>
      <td>45657</td>
      <td>김명자 신임 과총 회장 원로와 젊은 과학자 지혜 모을 것</td>
    </tr>
    <tr>
      <th>4</th>
      <td>45658</td>
      <td>회색인간 작가 김동식 양심고백 등 새 소설집 2권 출간</td>
    </tr>
  </tbody>
</table>
</div>




```python
topic_dict
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
      <th>topic</th>
      <th>topic_idx</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>IT과학</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>경제</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>사회</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>생활문화</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>세계</td>
      <td>4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>스포츠</td>
      <td>5</td>
    </tr>
    <tr>
      <th>6</th>
      <td>정치</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>




```python
bookmarks.head(3)
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
      <th>index</th>
      <th>title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>통계는 과연 거짓말을 할까? - 매일경제</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>故마라도나에 우승컵 바쳤다…‘메신’의 짜릿한 ‘카타르’시스 | 중앙일보</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>文정부 ‘통계조작’ 공방 격화…與 “통계조작성장” 野 “모욕주기” | 중앙일보</td>
    </tr>
  </tbody>
</table>
</div>




```python
our_sub.head(3)
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
      <th>index</th>
      <th>topic_idx</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
our_tag.head(3)
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
      <th>index</th>
      <th>tag</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>&lt;a add_date="1584968656" href="https://www.mk....</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>&lt;a add_date="1671427337" href="https://www.joo...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>&lt;a add_date="1671427356" href="https://www.joo...</td>
    </tr>
  </tbody>
</table>
</div>




```python
our_cset.head(3)
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
      <th>index</th>
      <th>title</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>통계는 과연 거짓말을 할까? - 매일경제</td>
      <td>사회</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>故마라도나에 우승컵 바쳤다…‘메신’의 짜릿한 ‘카타르’시스 | 중앙일보</td>
      <td>스포츠</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>文정부 ‘통계조작’ 공방 격화…與 “통계조작성장” 野 “모욕주기” | 중앙일보</td>
      <td>정치</td>
    </tr>
  </tbody>
</table>
</div>



# labeling : text to num


```python
answer = pd.read_csv('./open/topic_dict.csv')
answer_dict = {}
for value, key in enumerate(answer['topic']):
    answer_dict[key] = value
    
answer_dict
```




    {'IT과학': 0, '경제': 1, '사회': 2, '생활문화': 3, '세계': 4, '스포츠': 5, '정치': 6}




```python
an_li = []
for topic in our_cset['label']:
    an_li.append(answer_dict[topic])
print(an_li)
```

    [2, 5, 6, 6, 6, 6, 6, 6, 6, 6, 4, 6, 6, 1, 1, 1, 1, 3, 1, 1, 1, 1, 3, 3, 3, 1, 3, 0, 2, 2, 3, 5, 3, 2, 2, 3, 2, 3, 3, 2, 4, 5, 4, 4, 4, 4, 4, 4, 4, 3, 3, 2, 3, 3, 3, 0, 3, 2, 3, 3, 5, 5, 5, 5, 5, 5, 5, 5, 5, 3, 3, 3, 5, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2]



```python
our_cset['label'] = an_li
our_cset
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
      <th>index</th>
      <th>title</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>통계는 과연 거짓말을 할까? - 매일경제</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>故마라도나에 우승컵 바쳤다…‘메신’의 짜릿한 ‘카타르’시스 | 중앙일보</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>文정부 ‘통계조작’ 공방 격화…與 “통계조작성장” 野 “모욕주기” | 중앙일보</td>
      <td>6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>정청래, 박지원 복당 공개반대 "한 번 배신하면 또 배신" | 중앙일보</td>
      <td>6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>尹지지율, 중도·20대가 쌍끌이로 올렸다…6월 이후 첫 40%대 [리얼미터] | 중앙일보</td>
      <td>6</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>79</th>
      <td>79</td>
      <td>[사랑방] 권모세 회장 등 4명 ‘서울대AMP대상’ | 중앙일보</td>
      <td>2</td>
    </tr>
    <tr>
      <th>80</th>
      <td>80</td>
      <td>[인사] 에스원 外 | 중앙일보</td>
      <td>2</td>
    </tr>
    <tr>
      <th>81</th>
      <td>81</td>
      <td>[사진] 서울국제포럼-나카소네 세계평화연 ‘한·일 신시대를 향해’ 포럼 | 중앙일보</td>
      <td>2</td>
    </tr>
    <tr>
      <th>82</th>
      <td>82</td>
      <td>[사랑방] 이덕로 한국행정학회장 취임 | 중앙일보</td>
      <td>2</td>
    </tr>
    <tr>
      <th>83</th>
      <td>83</td>
      <td>[사진] 중앙광고대상 시상식 | 중앙일보</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>84 rows × 3 columns</p>
</div>



# Dataset


```python
train, val = train_test_split(train, test_size=0.2, random_state=2021)
```


```python
class NTDataset(Dataset):
    def __init__(self, csv_file):
        self.dataset = csv_file
        self.tokenizer = AutoTokenizer.from_pretrained("klue/roberta-large")

        print(self.dataset.describe())
  
    def __len__(self):
        return len(self.dataset)

  
    def __getitem__(self, idx):
        row = self.dataset.iloc[idx, 1:3].values
        text = row[0]
        y = row[1]
        inputs = self.tokenizer(
            text, 
            return_tensors='pt',
            truncation=True,
            max_length=15,
            pad_to_max_length=True,
            add_special_tokens=True
            )

        input_ids = inputs['input_ids'][0]
        attention_mask = inputs['attention_mask'][0]

        return input_ids, attention_mask, y
```


```python
class NTDataset_test(Dataset):
  
    def __init__(self, csv_file):
        self.dataset = csv_file
        self.tokenizer = AutoTokenizer.from_pretrained("klue/roberta-large")

        print(self.dataset.describe())
  
    def __len__(self):
        return len(self.dataset)
  
    def __getitem__(self, idx):
        row = self.dataset.iloc[idx, 1:2].values
        text = row[0]
        inputs = self.tokenizer(
            text, 
            return_tensors='pt',
            truncation=True,
            max_length=15,
            pad_to_max_length=True,
            add_special_tokens=True
            )

        input_ids = inputs['input_ids'][0]
        attention_mask = inputs['attention_mask'][0]

        return input_ids, attention_mask
```


```python
train_dataset = NTDataset(train)
val_dataset = NTDataset(train)
test_dataset = NTDataset_test(test)

# 우리 데이터셋
our_dataset = NTDataset_test(bookmarks)
```

                  index     topic_idx
    count  36523.000000  36523.000000
    mean   22786.501629      3.163732
    std    13174.530040      1.932000
    min        0.000000      0.000000
    25%    11371.000000      2.000000
    50%    22784.000000      3.000000
    75%    34188.500000      5.000000
    max    45653.000000      6.000000
                  index     topic_idx
    count  36523.000000  36523.000000
    mean   22786.501629      3.163732
    std    13174.530040      1.932000
    min        0.000000      0.000000
    25%    11371.000000      2.000000
    50%    22784.000000      3.000000
    75%    34188.500000      5.000000
    max    45653.000000      6.000000
                  index
    count   9131.000000
    mean   50219.000000
    std     2636.036988
    min    45654.000000
    25%    47936.500000
    50%    50219.000000
    75%    52501.500000
    max    54784.000000
               index
    count  84.000000
    mean   41.500000
    std    24.392622
    min     0.000000
    25%    20.750000
    50%    41.500000
    75%    62.250000
    max    83.000000


# Model : Roberta-large


```python
device = torch.device("cuda")
model = RobertaForSequenceClassification.from_pretrained("klue/roberta-large", num_labels=7).to(device)
```

    Some weights of the model checkpoint at klue/roberta-large were not used when initializing RobertaForSequenceClassification: ['lm_head.dense.weight', 'lm_head.decoder.bias', 'lm_head.layer_norm.bias', 'lm_head.decoder.weight', 'lm_head.bias', 'lm_head.layer_norm.weight', 'lm_head.dense.bias']
    - This IS expected if you are initializing RobertaForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
    - This IS NOT expected if you are initializing RobertaForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
    Some weights of RobertaForSequenceClassification were not initialized from the model checkpoint at klue/roberta-large and are newly initialized: ['classifier.out_proj.weight', 'classifier.dense.weight', 'classifier.out_proj.bias', 'classifier.dense.bias']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.


# Train


```python
epochs = 5
batch_size = 128

optimizer = AdamW(model.parameters(), lr=1e-5)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 우리 데이터셋
our_loader = DataLoader(our_dataset, batch_size=batch_size, shuffle=False)
```

    /home/smcho1201/.conda/envs/torch_sm/lib/python3.10/site-packages/transformers/optimization.py:306: FutureWarning: This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this warning
      warnings.warn(



```python
# train
losses = []
accuracies = []
total_loss = 0.0
correct = 0
total = 0

for i in range(epochs):

    model.train()

    for input_ids_batch, attention_masks_batch, y_batch in tqdm(train_loader):
        optimizer.zero_grad()
        y_batch = y_batch.to(device)
        y_pred = model(input_ids_batch.to(device), attention_mask=attention_masks_batch.to(device))[0]
        loss = F.cross_entropy(y_pred, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, predicted = torch.max(y_pred, 1)
        correct += (predicted == y_batch).sum()
        total += len(y_batch)

    losses.append(total_loss)
    accuracies.append(correct.float() / total)
    print("Train Loss:", total_loss / total, "Accuracy:", correct.float() / total)
    

PATH = './weights/'

# torch.save(model, PATH + 'model.pt')  # 전체 모델 저장
# torch.save(model.state_dict(), PATH + 'model_state_dict.pt')  # 모델 객체의 state_dict 저장
# torch.save({
#     'model': model.state_dict(),
#     'optimizer': optimizer.state_dict()
# }, PATH + 'all.tar')  # 여러 가지 값 저장, 학습 중 진행 상황 저장을 위해 epoch, loss 값 등 일반 scalar값 저장 가능

```


      0%|          | 0/286 [00:00<?, ?it/s]


    /home/smcho1201/.conda/envs/torch_sm/lib/python3.10/site-packages/transformers/tokenization_utils_base.py:2336: FutureWarning: The `pad_to_max_length` argument is deprecated and will be removed in a future version, use `padding=True` or `padding='longest'` to pad to the longest sequence in the batch, or use `padding='max_length'` to pad to a max length. In this case, you can give a specific length with `max_length` (e.g. `max_length=45`) or leave max_length to None to pad to the maximal input size of the model (e.g. 512 for Bert).
      warnings.warn(


    Train Loss: 0.0040584215916016145 Accuracy: tensor(0.8326, device='cuda:0')



      0%|          | 0/286 [00:00<?, ?it/s]


    Train Loss: 0.0032016056550589326 Accuracy: tensor(0.8658, device='cuda:0')



      0%|          | 0/286 [00:00<?, ?it/s]


    Train Loss: 0.0027588900289475227 Accuracy: tensor(0.8838, device='cuda:0')



      0%|          | 0/286 [00:00<?, ?it/s]


    Train Loss: 0.002440005875581273 Accuracy: tensor(0.8969, device='cuda:0')



      0%|          | 0/286 [00:00<?, ?it/s]


    Train Loss: 0.002179187908676855 Accuracy: tensor(0.9081, device='cuda:0')


# validation


```python
PATH = './weights/'
model = torch.load(PATH + 'model.pt')  # 전체 모델을 통째로 불러옴, 클래스 선언 필수
model.load_state_dict(torch.load(PATH + 'model_state_dict.pt'))  # state_dict를 불러 온 후, 모델에 저장

# validation
model.eval()

pred = []
correct = 0
total = 0

for input_ids_batch, attention_masks_batch, y_batch in tqdm(val_loader):
    y_batch = y_batch.to(device)
    y_pred = model(input_ids_batch.to(device), attention_mask=attention_masks_batch.to(device))[0]
    _, predicted = torch.max(y_pred, 1)
    pred.append(predicted)
    correct += (predicted == y_batch).sum()
    total += len(y_batch)
        
print("val accuracy:", correct.float() / total)
```


      0%|          | 0/286 [00:00<?, ?it/s]


    /home/smcho1201/.conda/envs/torch_sm/lib/python3.10/site-packages/transformers/tokenization_utils_base.py:2336: FutureWarning: The `pad_to_max_length` argument is deprecated and will be removed in a future version, use `padding=True` or `padding='longest'` to pad to the longest sequence in the batch, or use `padding='max_length'` to pad to a max length. In this case, you can give a specific length with `max_length` (e.g. `max_length=45`) or leave max_length to None to pad to the maximal input size of the model (e.g. 512 for Bert).
      warnings.warn(


    val accuracy: tensor(0.9770, device='cuda:0')


# Test for our dataset


```python
PATH = './weights/'
model = torch.load(PATH + 'model.pt')  # 전체 모델을 통째로 불러옴, 클래스 선언 필수
model.load_state_dict(torch.load(PATH + 'model_state_dict.pt'))  # state_dict를 불러 온 후, 모델에 저장

# test
model.eval()

pred = []

for input_ids_batch, attention_masks_batch in tqdm(our_loader):
    y_pred = model(input_ids_batch.to(device), attention_mask=attention_masks_batch.to(device))[0]
    _, predicted = torch.max(y_pred, 1)
    pred.extend(predicted.tolist())
```


      0%|          | 0/1 [00:00<?, ?it/s]


# Metric

## Accuracy


```python
acc_li = []
for i in range(len(pred)):
    acc_li.append(pred[i] == an_li[i])
    
sum(acc_li) / len(acc_li)
```




    0.7976190476190477




```python

```

## f1-score - macro


```python
f1_score(an_li, pred, average='macro')
```




    0.7684738197879625



# making sub dataframe


```python
our_sub['topic_idx'] = pred
our_sub
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
      <th>index</th>
      <th>topic_idx</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>6</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>79</th>
      <td>79</td>
      <td>2</td>
    </tr>
    <tr>
      <th>80</th>
      <td>80</td>
      <td>2</td>
    </tr>
    <tr>
      <th>81</th>
      <td>81</td>
      <td>6</td>
    </tr>
    <tr>
      <th>82</th>
      <td>82</td>
      <td>2</td>
    </tr>
    <tr>
      <th>83</th>
      <td>83</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
<p>84 rows × 2 columns</p>
</div>



# answer dataframe


```python
topic_dict
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
      <th>topic</th>
      <th>topic_idx</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>IT과학</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>경제</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>사회</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>생활문화</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>세계</td>
      <td>4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>스포츠</td>
      <td>5</td>
    </tr>
    <tr>
      <th>6</th>
      <td>정치</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>




```python
answer_li = []
for i in range(len(our_sub)):
    answer_li.append(answer["topic"][our_sub["topic_idx"][i]])
```


```python
print(answer_li)
```

    ['경제', '스포츠', '정치', '정치', '정치', '정치', '정치', 'IT과학', '정치', '스포츠', '정치', '정치', '정치', '경제', '경제', '경제', '경제', 'IT과학', '경제', '경제', '경제', '경제', '생활문화', 'IT과학', '생활문화', '생활문화', '생활문화', 'IT과학', '사회', '사회', '생활문화', '스포츠', '생활문화', '사회', '사회', '경제', '사회', '생활문화', '생활문화', '경제', '세계', '스포츠', '세계', '세계', '세계', '세계', '세계', '세계', '세계', '생활문화', '생활문화', '사회', '생활문화', '생활문화', '생활문화', 'IT과학', '생활문화', '스포츠', '생활문화', '생활문화', '스포츠', 'IT과학', '스포츠', '스포츠', '스포츠', '스포츠', '스포츠', '스포츠', '스포츠', 'IT과학', '생활문화', '생활문화', '스포츠', '사회', 'IT과학', '생활문화', '생활문화', '생활문화', '사회', '사회', '사회', '정치', '사회', '생활문화']



```python
df_answer = pd.DataFrame(answer_li)
df_answer['tag'] = our_tag['tag']
df_answer.columns = ['answer', 'tag']
df_answer
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
      <th>answer</th>
      <th>tag</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>경제</td>
      <td>&lt;a add_date="1584968656" href="https://www.mk....</td>
    </tr>
    <tr>
      <th>1</th>
      <td>스포츠</td>
      <td>&lt;a add_date="1671427337" href="https://www.joo...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>정치</td>
      <td>&lt;a add_date="1671427356" href="https://www.joo...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>정치</td>
      <td>&lt;a add_date="1671427361" href="https://www.joo...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>정치</td>
      <td>&lt;a add_date="1671427364" href="https://www.joo...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>79</th>
      <td>사회</td>
      <td>&lt;a add_date="1671427840" href="https://www.joo...</td>
    </tr>
    <tr>
      <th>80</th>
      <td>사회</td>
      <td>&lt;a add_date="1671427843" href="https://www.joo...</td>
    </tr>
    <tr>
      <th>81</th>
      <td>정치</td>
      <td>&lt;a add_date="1671427849" href="https://www.joo...</td>
    </tr>
    <tr>
      <th>82</th>
      <td>사회</td>
      <td>&lt;a add_date="1671427852" href="https://www.joo...</td>
    </tr>
    <tr>
      <th>83</th>
      <td>생활문화</td>
      <td>&lt;a add_date="1671427858" href="https://www.joo...</td>
    </tr>
  </tbody>
</table>
<p>84 rows × 2 columns</p>
</div>




```python
df_answer.to_csv('./open/roberta_large_epoch5.csv', index=False, encoding = 'UTF-8')
```

# Bar graph of result


```python
topic_dict
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
      <th>topic</th>
      <th>topic_idx</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>IT과학</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>경제</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>사회</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>생활문화</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>세계</td>
      <td>4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>스포츠</td>
      <td>5</td>
    </tr>
    <tr>
      <th>6</th>
      <td>정치</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>




```python
idx_li = list(topic_dict['topic'])

idx_dict = {}

for idx, topic in enumerate(idx_li):
    idx_dict[f'{topic}'] = len(our_sub[our_sub["topic_idx"] == idx]['index'].values)
```


```python
fig = plt.figure(figsize = (10,10), dpi = 100)
axs = fig.subplots()

X = list(idx_dict.keys())
Y = list(idx_dict.values())

bar=axs.bar(X,Y, color = (0.8, 0.4, 0.5), alpha = 0.7)
```

    findfont: Font family 'Malgun Gothic' not found.
    findfont: Font family 'Malgun Gothic' not found.
    /home/smcho1201/.conda/envs/torch_sm/lib/python3.10/site-packages/IPython/core/events.py:89: UserWarning: Glyph 44284 (\N{HANGUL SYLLABLE GWA}) missing from current font.
      func(*args, **kwargs)
    /home/smcho1201/.conda/envs/torch_sm/lib/python3.10/site-packages/IPython/core/events.py:89: UserWarning: Glyph 54617 (\N{HANGUL SYLLABLE HAG}) missing from current font.
      func(*args, **kwargs)
    findfont: Font family 'Malgun Gothic' not found.
    /home/smcho1201/.conda/envs/torch_sm/lib/python3.10/site-packages/IPython/core/events.py:89: UserWarning: Glyph 44221 (\N{HANGUL SYLLABLE GYEONG}) missing from current font.
      func(*args, **kwargs)
    /home/smcho1201/.conda/envs/torch_sm/lib/python3.10/site-packages/IPython/core/events.py:89: UserWarning: Glyph 51228 (\N{HANGUL SYLLABLE JE}) missing from current font.
      func(*args, **kwargs)
    findfont: Font family 'Malgun Gothic' not found.
    /home/smcho1201/.conda/envs/torch_sm/lib/python3.10/site-packages/IPython/core/events.py:89: UserWarning: Glyph 49324 (\N{HANGUL SYLLABLE SA}) missing from current font.
      func(*args, **kwargs)
    /home/smcho1201/.conda/envs/torch_sm/lib/python3.10/site-packages/IPython/core/events.py:89: UserWarning: Glyph 54924 (\N{HANGUL SYLLABLE HOE}) missing from current font.
      func(*args, **kwargs)
    findfont: Font family 'Malgun Gothic' not found.
    /home/smcho1201/.conda/envs/torch_sm/lib/python3.10/site-packages/IPython/core/events.py:89: UserWarning: Glyph 49373 (\N{HANGUL SYLLABLE SAENG}) missing from current font.
      func(*args, **kwargs)
    /home/smcho1201/.conda/envs/torch_sm/lib/python3.10/site-packages/IPython/core/events.py:89: UserWarning: Glyph 54876 (\N{HANGUL SYLLABLE HWAL}) missing from current font.
      func(*args, **kwargs)
    /home/smcho1201/.conda/envs/torch_sm/lib/python3.10/site-packages/IPython/core/events.py:89: UserWarning: Glyph 47928 (\N{HANGUL SYLLABLE MUN}) missing from current font.
      func(*args, **kwargs)
    /home/smcho1201/.conda/envs/torch_sm/lib/python3.10/site-packages/IPython/core/events.py:89: UserWarning: Glyph 54868 (\N{HANGUL SYLLABLE HWA}) missing from current font.
      func(*args, **kwargs)
    findfont: Font family 'Malgun Gothic' not found.
    /home/smcho1201/.conda/envs/torch_sm/lib/python3.10/site-packages/IPython/core/events.py:89: UserWarning: Glyph 49464 (\N{HANGUL SYLLABLE SE}) missing from current font.
      func(*args, **kwargs)
    /home/smcho1201/.conda/envs/torch_sm/lib/python3.10/site-packages/IPython/core/events.py:89: UserWarning: Glyph 44228 (\N{HANGUL SYLLABLE GYE}) missing from current font.
      func(*args, **kwargs)
    findfont: Font family 'Malgun Gothic' not found.
    /home/smcho1201/.conda/envs/torch_sm/lib/python3.10/site-packages/IPython/core/events.py:89: UserWarning: Glyph 49828 (\N{HANGUL SYLLABLE SEU}) missing from current font.
      func(*args, **kwargs)
    /home/smcho1201/.conda/envs/torch_sm/lib/python3.10/site-packages/IPython/core/events.py:89: UserWarning: Glyph 54252 (\N{HANGUL SYLLABLE PO}) missing from current font.
      func(*args, **kwargs)
    /home/smcho1201/.conda/envs/torch_sm/lib/python3.10/site-packages/IPython/core/events.py:89: UserWarning: Glyph 52768 (\N{HANGUL SYLLABLE CEU}) missing from current font.
      func(*args, **kwargs)
    findfont: Font family 'Malgun Gothic' not found.
    /home/smcho1201/.conda/envs/torch_sm/lib/python3.10/site-packages/IPython/core/events.py:89: UserWarning: Glyph 51221 (\N{HANGUL SYLLABLE JEONG}) missing from current font.
      func(*args, **kwargs)
    /home/smcho1201/.conda/envs/torch_sm/lib/python3.10/site-packages/IPython/core/events.py:89: UserWarning: Glyph 52824 (\N{HANGUL SYLLABLE CI}) missing from current font.
      func(*args, **kwargs)
    findfont: Font family 'Malgun Gothic' not found.
    findfont: Font family 'Malgun Gothic' not found.
    findfont: Font family 'Malgun Gothic' not found.
    findfont: Font family 'Malgun Gothic' not found.
    findfont: Font family 'Malgun Gothic' not found.
    findfont: Font family 'Malgun Gothic' not found.
    findfont: Font family 'Malgun Gothic' not found.
    findfont: Font family 'Malgun Gothic' not found.
    findfont: Font family 'Malgun Gothic' not found.
    findfont: Font family 'Malgun Gothic' not found.
    findfont: Font family 'Malgun Gothic' not found.
    findfont: Font family 'Malgun Gothic' not found.
    findfont: Font family 'Malgun Gothic' not found.
    findfont: Font family 'Malgun Gothic' not found.
    findfont: Font family 'Malgun Gothic' not found.
    findfont: Font family 'Malgun Gothic' not found.
    findfont: Font family 'Malgun Gothic' not found.
    findfont: Font family 'Malgun Gothic' not found.
    findfont: Font family 'Malgun Gothic' not found.
    /home/smcho1201/.conda/envs/torch_sm/lib/python3.10/site-packages/IPython/core/pylabtools.py:151: UserWarning: Glyph 44284 (\N{HANGUL SYLLABLE GWA}) missing from current font.
      fig.canvas.print_figure(bytes_io, **kw)
    /home/smcho1201/.conda/envs/torch_sm/lib/python3.10/site-packages/IPython/core/pylabtools.py:151: UserWarning: Glyph 54617 (\N{HANGUL SYLLABLE HAG}) missing from current font.
      fig.canvas.print_figure(bytes_io, **kw)
    findfont: Font family 'Malgun Gothic' not found.
    /home/smcho1201/.conda/envs/torch_sm/lib/python3.10/site-packages/IPython/core/pylabtools.py:151: UserWarning: Glyph 44221 (\N{HANGUL SYLLABLE GYEONG}) missing from current font.
      fig.canvas.print_figure(bytes_io, **kw)
    /home/smcho1201/.conda/envs/torch_sm/lib/python3.10/site-packages/IPython/core/pylabtools.py:151: UserWarning: Glyph 51228 (\N{HANGUL SYLLABLE JE}) missing from current font.
      fig.canvas.print_figure(bytes_io, **kw)
    findfont: Font family 'Malgun Gothic' not found.
    /home/smcho1201/.conda/envs/torch_sm/lib/python3.10/site-packages/IPython/core/pylabtools.py:151: UserWarning: Glyph 49324 (\N{HANGUL SYLLABLE SA}) missing from current font.
      fig.canvas.print_figure(bytes_io, **kw)
    /home/smcho1201/.conda/envs/torch_sm/lib/python3.10/site-packages/IPython/core/pylabtools.py:151: UserWarning: Glyph 54924 (\N{HANGUL SYLLABLE HOE}) missing from current font.
      fig.canvas.print_figure(bytes_io, **kw)
    findfont: Font family 'Malgun Gothic' not found.
    /home/smcho1201/.conda/envs/torch_sm/lib/python3.10/site-packages/IPython/core/pylabtools.py:151: UserWarning: Glyph 49373 (\N{HANGUL SYLLABLE SAENG}) missing from current font.
      fig.canvas.print_figure(bytes_io, **kw)
    /home/smcho1201/.conda/envs/torch_sm/lib/python3.10/site-packages/IPython/core/pylabtools.py:151: UserWarning: Glyph 54876 (\N{HANGUL SYLLABLE HWAL}) missing from current font.
      fig.canvas.print_figure(bytes_io, **kw)
    /home/smcho1201/.conda/envs/torch_sm/lib/python3.10/site-packages/IPython/core/pylabtools.py:151: UserWarning: Glyph 47928 (\N{HANGUL SYLLABLE MUN}) missing from current font.
      fig.canvas.print_figure(bytes_io, **kw)
    /home/smcho1201/.conda/envs/torch_sm/lib/python3.10/site-packages/IPython/core/pylabtools.py:151: UserWarning: Glyph 54868 (\N{HANGUL SYLLABLE HWA}) missing from current font.
      fig.canvas.print_figure(bytes_io, **kw)
    findfont: Font family 'Malgun Gothic' not found.
    /home/smcho1201/.conda/envs/torch_sm/lib/python3.10/site-packages/IPython/core/pylabtools.py:151: UserWarning: Glyph 49464 (\N{HANGUL SYLLABLE SE}) missing from current font.
      fig.canvas.print_figure(bytes_io, **kw)
    /home/smcho1201/.conda/envs/torch_sm/lib/python3.10/site-packages/IPython/core/pylabtools.py:151: UserWarning: Glyph 44228 (\N{HANGUL SYLLABLE GYE}) missing from current font.
      fig.canvas.print_figure(bytes_io, **kw)
    findfont: Font family 'Malgun Gothic' not found.
    /home/smcho1201/.conda/envs/torch_sm/lib/python3.10/site-packages/IPython/core/pylabtools.py:151: UserWarning: Glyph 49828 (\N{HANGUL SYLLABLE SEU}) missing from current font.
      fig.canvas.print_figure(bytes_io, **kw)
    /home/smcho1201/.conda/envs/torch_sm/lib/python3.10/site-packages/IPython/core/pylabtools.py:151: UserWarning: Glyph 54252 (\N{HANGUL SYLLABLE PO}) missing from current font.
      fig.canvas.print_figure(bytes_io, **kw)
    /home/smcho1201/.conda/envs/torch_sm/lib/python3.10/site-packages/IPython/core/pylabtools.py:151: UserWarning: Glyph 52768 (\N{HANGUL SYLLABLE CEU}) missing from current font.
      fig.canvas.print_figure(bytes_io, **kw)
    findfont: Font family 'Malgun Gothic' not found.
    /home/smcho1201/.conda/envs/torch_sm/lib/python3.10/site-packages/IPython/core/pylabtools.py:151: UserWarning: Glyph 51221 (\N{HANGUL SYLLABLE JEONG}) missing from current font.
      fig.canvas.print_figure(bytes_io, **kw)
    /home/smcho1201/.conda/envs/torch_sm/lib/python3.10/site-packages/IPython/core/pylabtools.py:151: UserWarning: Glyph 52824 (\N{HANGUL SYLLABLE CI}) missing from current font.
      fig.canvas.print_figure(bytes_io, **kw)
    findfont: Font family 'Malgun Gothic' not found.
    findfont: Font family 'Malgun Gothic' not found.
    findfont: Font family 'Malgun Gothic' not found.
    findfont: Font family 'Malgun Gothic' not found.
    findfont: Font family 'Malgun Gothic' not found.
    findfont: Font family 'Malgun Gothic' not found.
    findfont: Font family 'Malgun Gothic' not found.
    findfont: Font family 'Malgun Gothic' not found.
    findfont: Font family 'Malgun Gothic' not found.
    findfont: Font family 'Malgun Gothic' not found.
    findfont: Font family 'Malgun Gothic' not found.
    findfont: Font family 'Malgun Gothic' not found.
    findfont: Font family 'Malgun Gothic' not found.
    findfont: Font family 'Malgun Gothic' not found.
    findfont: Font family 'Malgun Gothic' not found.
    findfont: Font family 'Malgun Gothic' not found.
    findfont: Font family 'Malgun Gothic' not found.



    
![png](output_64_1.png)
    


# Devide 7 Topics

- 북마크를 총 7개의 주제로 나눈다.


```python
f=codecs.open("./open/bookmarks_84.html", 'r')
soup = BeautifulSoup(f.read(), 'html.parser')
df_answer = pd.read_csv('./open/roberta_large_epoch3.csv')
```


```python
soup.dt
```




    <dt><h3 add_date="1629940719" last_modified="1671427858">신문 스크랩</h3>
    <dl><p>
    <dt><a add_date="1584968656" href="https://www.mk.co.kr/opinion/contributors/view/2016/08/618807/" icon="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAABPUlEQVQ4jaWTsUoDQRCGvz0OJWLEgEXSCCo2wVcQX8PGQsHCSgt9DBFJoYKp7QURH8FGJNgaFQsLEUXBC0d21+J273bXSxTdZmZ25v93ZnZGALxvzmj+cCb274T4EawBZWTf6NLoNYi8YDGEQBppdZm5Yze2utfNMG9PfLaWUc+PoCCammZs6wRRawDwsTKbE/gZ2EQmG1TWj2FkHDFapbLRzsEwJAP3RPV5KmstICKqz/lO24tBGciHTsbeXCJuLmZ3t9dFgO3DIIL04oD+1Xnx4OUZ6elRaQkFgfuZStNrbyPvb5DdDsnhDihdShDnYOUSgE4Skt1VkBqS3je/tUsJtAnQry/Fa65fgvCaqPwAocg6LR0Z+o2djXLq1BUCre5OoB3tBZtBCJa/ABOWIPEXRQZ3FhisnoD/rfMXtEaiNPk3uHYAAAAASUVORK5CYII=">통계는 과연 거짓말을 할까? - 매일경제</a>
    <dt><a add_date="1671427337" href="https://www.joongang.co.kr/article/25126744" icon="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAA4klEQVQ4jWP8//8/AymABV3gz2+Gv38YGBkZGBgYIEaxsUO5WDX8Pb7777r5/398Y2BgYGBkZFLVYUmpYOQVQKj4jwa+fv6za813B6lvusw/qxL+Pb77/99fZHkmdCdx8TDpmTPw8jMwMTGp6jDKKDEwoqjB0EAIDFcNf/8y/P3LwMjIwMFJlIZ/ty79f/eKUVCUSccUi3HIkfLv3es/u9Z8DzH8bifxe8mk/3//oEfr//+MyInv7+HtfzcuYJSQY3byZ9K3YGDGSGkMDCgaGP7+Zfj3l4GVlYGBEVMpNg1EAABdi4Sb+zSY9AAAAABJRU5ErkJggg==">故마라도나에 우승컵 바쳤다…‘메신’의 짜릿한 ‘카타르’시스 | 중앙일보</a>
    <dt><a add_date="1671427356" href="https://www.joongang.co.kr/article/25126791">文정부 ‘통계조작’ 공방 격화…與 “통계조작성장” 野 “모욕주기” | 중앙일보</a>
    <dt><a add_date="1671427361" href="https://www.joongang.co.kr/article/25126774">정청래, 박지원 복당 공개반대 "한 번 배신하면 또 배신" | 중앙일보</a>
    <dt><a add_date="1671427364" href="https://www.joongang.co.kr/article/25126755">尹지지율, 중도·20대가 쌍끌이로 올렸다…6월 이후 첫 40%대 [리얼미터] | 중앙일보</a>
    <dt><a add_date="1671427367" href="https://www.joongang.co.kr/article/25126692">[사진] 서초동 사저 찾아 작별 인사 | 중앙일보</a>
    <dt><a add_date="1671427371" href="https://www.joongang.co.kr/article/25126618">법인세보다 장관 '빅2'가 핵심...野 반대한 한동훈·이상민 예산은 | 중앙일보</a>
    <dt><a add_date="1671427376" href="https://www.joongang.co.kr/article/25126593">'게임체인저' 신입당원은 20만...국힘, 당원 포섭전쟁은 끝났다 | 중앙일보</a>
    <dt><a add_date="1671427381" href="https://www.joongang.co.kr/article/25126580" icon="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAA4klEQVQ4jWP8//8/AymABV3gz2+Gv38YGBkZGBgYIEaxsUO5WDX8Pb7777r5/398Y2BgYGBkZFLVYUmpYOQVQKj4jwa+fv6za813B6lvusw/qxL+Pb77/99fZHkmdCdx8TDpmTPw8jMwMTGp6jDKKDEwoqjB0EAIDFcNf/8y/P3LwMjIwMFJlIZ/ty79f/eKUVCUSccUi3HIkfLv3es/u9Z8DzH8bifxe8mk/3//oEfr//+MyInv7+HtfzcuYJSQY3byZ9K3YGDGSGkMDCgaGP7+Zfj3l4GVlYGBEVMpNg1EAABdi4Sb+zSY9AAAAABJRU5ErkJggg==">‘당심 100% 룰’ 김용태 “윤핵관 당헌·당규 손바닥 뒤집듯 바꿔…한심” | 중앙일보</a>
    <dt><a add_date="1671427389" href="https://www.joongang.co.kr/article/25126571">尹 '조용한 생일'…"직언 들어줘 감사하다" 참모들 축하 메시지 | 중앙일보</a>
    <dt><a add_date="1671427395" href="https://www.joongang.co.kr/article/25126550">[속보] 북, 탄도미사일 2발 발사…“일본 EEZ 밖 낙하 추정” | 중앙일보</a>
    <dt><a add_date="1671427399" href="https://www.joongang.co.kr/article/25126364" icon="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAA4klEQVQ4jWP8//8/AymABV3gz2+Gv38YGBkZGBgYIEaxsUO5WDX8Pb7777r5/398Y2BgYGBkZFLVYUmpYOQVQKj4jwa+fv6za813B6lvusw/qxL+Pb77/99fZHkmdCdx8TDpmTPw8jMwMTGp6jDKKDEwoqjB0EAIDFcNf/8y/P3LwMjIwMFJlIZ/ty79f/eKUVCUSccUi3HIkfLv3es/u9Z8DzH8bifxe8mk/3//oEfr//+MyInv7+HtfzcuYJSQY3byZ9K3YGDGSGkMDCgaGP7+Zfj3l4GVlYGBEVMpNg1EAABdi4Sb+zSY9AAAAABJRU5ErkJggg==">이정현 "당대표 나가든 자문을 하든 소극적으로 있지 않겠다" | 중앙일보</a>
    <dt><a add_date="1671427511" href="https://www.joongang.co.kr/article/25126336" icon="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAA4klEQVQ4jWP8//8/AymABV3gz2+Gv38YGBkZGBgYIEaxsUO5WDX8Pb7777r5/398Y2BgYGBkZFLVYUmpYOQVQKj4jwa+fv6za813B6lvusw/qxL+Pb77/99fZHkmdCdx8TDpmTPw8jMwMTGp6jDKKDEwoqjB0EAIDFcNf/8y/P3LwMjIwMFJlIZ/ty79f/eKUVCUSccUi3HIkfLv3es/u9Z8DzH8bifxe8mk/3//oEfr//+MyInv7+HtfzcuYJSQY3byZ9K3YGDGSGkMDCgaGP7+Zfj3l4GVlYGBEVMpNg1EAABdi4Sb+zSY9AAAAABJRU5ErkJggg==">유승민 여론조사 보니…친윤 "이게 당원투표 100% 이유" | 중앙일보</a>
    <dt><a add_date="1671427539" href="https://www.joongang.co.kr/article/25126817">정부가 눌러 5% 안넘은 근원물가…'공공요금 인상' 내년엔 | 중앙일보</a>
    <dt><a add_date="1671427543" href="https://www.joongang.co.kr/article/25126811">LG엔솔, 오창공장 배터리 라인 신·증설에 4조 투자…1800명 채용 | 중앙일보</a>
    <dt><a add_date="1671427547" href="https://www.joongang.co.kr/article/25126800">퇴직연금 중도인출 “집 사려고, 전세 보증금 때문에” 81.6% | 중앙일보</a>
    <dt><a add_date="1671427550" href="https://www.joongang.co.kr/article/25126770">기아, 카타르 월드컵에 차량 297대 투입…브랜드 홍보 | 중앙일보</a>
    <dt><a add_date="1671427553" href="https://www.joongang.co.kr/article/25126728">[알림] '펫(pet) 톡톡': 평생 간직하세요, 인생 사진 2탄 | 중앙일보</a>
    <dt><a add_date="1671427556" href="https://www.joongang.co.kr/article/25126677">금리↑ 집값↓…“2% 청약통장 깨서 7% 대출이자 갚는다” | 중앙일보</a>
    <dt><a add_date="1671427559" href="https://www.joongang.co.kr/article/25126674">[사진] 청약경쟁률 8년 만에 한 자릿수로 | 중앙일보</a>
    <dt><a add_date="1671427562" href="https://www.joongang.co.kr/article/25126670">SK그룹, CES 2023 무대로 넷제로 역량 선보인다 | 중앙일보</a>
    <dt><a add_date="1671427566" href="https://www.joongang.co.kr/article/25126664" icon="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAA4klEQVQ4jWP8//8/AymABV3gz2+Gv38YGBkZGBgYIEaxsUO5WDX8Pb7777r5/398Y2BgYGBkZFLVYUmpYOQVQKj4jwa+fv6za813B6lvusw/qxL+Pb77/99fZHkmdCdx8TDpmTPw8jMwMTGp6jDKKDEwoqjB0EAIDFcNf/8y/P3LwMjIwMFJlIZ/ty79f/eKUVCUSccUi3HIkfLv3es/u9Z8DzH8bifxe8mk/3//oEfr//+MyInv7+HtfzcuYJSQY3byZ9K3YGDGSGkMDCgaGP7+Zfj3l4GVlYGBEVMpNg1EAABdi4Sb+zSY9AAAAABJRU5ErkJggg==">조류인플루엔자 확산…달걀값 또다시 들썩 | 중앙일보</a>
    <dt><a add_date="1671427570" href="https://www.joongang.co.kr/article/25126621">[팩플] “더 오래, 더 젊게”…달라지는 카톡, 카카오의 노림수는 | 중앙일보</a>
    <dt><a add_date="1671427573" href="https://www.joongang.co.kr/article/25126583" icon="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAA4klEQVQ4jWP8//8/AymABV3gz2+Gv38YGBkZGBgYIEaxsUO5WDX8Pb7777r5/398Y2BgYGBkZFLVYUmpYOQVQKj4jwa+fv6za813B6lvusw/qxL+Pb77/99fZHkmdCdx8TDpmTPw8jMwMTGp6jDKKDEwoqjB0EAIDFcNf/8y/P3LwMjIwMFJlIZ/ty79f/eKUVCUSccUi3HIkfLv3es/u9Z8DzH8bifxe8mk/3//oEfr//+MyInv7+HtfzcuYJSQY3byZ9K3YGDGSGkMDCgaGP7+Zfj3l4GVlYGBEVMpNg1EAABdi4Sb+zSY9AAAAABJRU5ErkJggg==">1년 4번 그들은 점을 찍는다…美 Fed, 과학인가 예언인가 | 중앙일보</a>
    <dt><a add_date="1671427576" href="https://www.joongang.co.kr/article/25126570" icon="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAA4klEQVQ4jWP8//8/AymABV3gz2+Gv38YGBkZGBgYIEaxsUO5WDX8Pb7777r5/398Y2BgYGBkZFLVYUmpYOQVQKj4jwa+fv6za813B6lvusw/qxL+Pb77/99fZHkmdCdx8TDpmTPw8jMwMTGp6jDKKDEwoqjB0EAIDFcNf/8y/P3LwMjIwMFJlIZ/ty79f/eKUVCUSccUi3HIkfLv3es/u9Z8DzH8bifxe8mk/3//oEfr//+MyInv7+HtfzcuYJSQY3byZ9K3YGDGSGkMDCgaGP7+Zfj3l4GVlYGBEVMpNg1EAABdi4Sb+zSY9AAAAABJRU5ErkJggg==">“신곡 설령 좋아도 관심없다”…3000억 저작권 굴리는 29살 | 중앙일보</a>
    <dt><a add_date="1671427581" href="https://www.joongang.co.kr/article/25126732">[VIEW] 퍼펙트스톰 긴박감 사라졌지만, 수출ㆍ고용ㆍ부동산 위기 계속 | 중앙일보</a>
    <dt><a add_date="1671427591" href="https://www.joongang.co.kr/article/25126615">밤 7시 '구찌 매장' 불꺼졌다…쇼핑몰은 거대한 미술관이 됐다 | 중앙일보</a>
    <dt><a add_date="1671427597" href="https://www.joongang.co.kr/article/25126830">한국공대, 지역산업 연계 오픈랩 혁신기술 설명회 개최 | 중앙일보</a>
    <dt><a add_date="1671427600" href="https://www.joongang.co.kr/article/25126823">전문대교육협의회, ‘2022년 전문대학인상’ 시상 | 중앙일보</a>
    <dt><a add_date="1671427603" href="https://www.joongang.co.kr/article/25126793">'공무원' 인기는 뚝…중·고교생 희망직업으로 뜬 이 계열 | 중앙일보</a>
    <dt><a add_date="1671427606" href="https://www.joongang.co.kr/article/25126784">제주 항공기·여객선 운항 정상화…"서해안엔 계속 많은 눈 예상" | 중앙일보</a>
    <dt><a add_date="1671427609" href="https://www.joongang.co.kr/article/25126756">[THINK ENGLISH] 손흥민, 마스크 벗고 런던에서 훈련 재개 | 중앙일보</a>
    <dt><a add_date="1671427613" href="https://www.joongang.co.kr/article/25126745">[소년중앙] 어디 가야 먹을 수 있나요, 찬바람 불면 돌아오는 제철 붕어빵 | 중앙일보</a>
    <dt><a add_date="1671427617" href="https://www.joongang.co.kr/article/25126790">"다시 만나자" 안받아준 前연인…준비한 흉기로 살해한 50대男 | 중앙일보</a>
    <dt><a add_date="1671427620" href="https://www.joongang.co.kr/article/25126769">말다툼하다 아내 살해한 남편, 범행 후 그 옆에서 술 마셨다 | 중앙일보</a>
    <dt><a add_date="1671427628" href="https://www.joongang.co.kr/article/25126656">[오늘의 운세] 12월 19일 | 중앙일보</a>
    <dt><a add_date="1671427632" href="https://www.joongang.co.kr/article/25126787">용산역에만 17분 멈췄다…전장연 출근길 기습에 "좀 그만하라" | 중앙일보</a>
    <dt><a add_date="1671427640" href="https://www.joongang.co.kr/article/25126666">[우리말 바루기] ‘결제’를 할까, ‘결재’를 할까? | 중앙일보</a>
    <dt><a add_date="1671427644" href="https://www.joongang.co.kr/article/25126629">여객기 109편 결항…19일까지 많은 눈, 아침 -15도 강추위 | 중앙일보</a>
    <dt><a add_date="1671427651" href="https://www.joongang.co.kr/article/25126542">"돈 뺏길지 모른다""집주인 연락 안돼" 매일 고성 터지는 이곳 | 중앙일보</a>
    <dt><a add_date="1671427658" href="https://www.joongang.co.kr/article/25126818">미얀마 최대도시 양곤서 또 폭발 사고…페리 승객 11명 부상 | 중앙일보</a>
    <dt><a add_date="1671427662" href="https://www.joongang.co.kr/article/25126753">"메시 우승 자격 있다"…'축구의 신'에 축하 건넨 '축구 황제' | 중앙일보</a>
    <dt><a add_date="1671427665" href="https://www.joongang.co.kr/article/25126649">히잡시위 지지한 ‘이란 국민배우’ 체포 | 중앙일보</a>
    <dt><a add_date="1671427670" href="https://www.joongang.co.kr/article/25126545">"우크라, 미국 만류에도 러시아軍 최고 지휘관 암살 시도" | 중앙일보</a>
    <dt><a add_date="1671427673" href="https://www.joongang.co.kr/article/25126510">'건강이상설' 푸틴 군 사령관들 소집…"우크라 작전 제안하라" | 중앙일보</a>
    <dt><a add_date="1671427676" href="https://www.joongang.co.kr/article/25126489">코로나 지원금 100억 빼돌린 美목사…호화주택 사려다 덜미 | 중앙일보</a>
    <dt><a add_date="1671427680" href="https://www.joongang.co.kr/article/25126482">한 달 전 인니 지진 사망자, 300명대 아닌 602명…재집계 수치 발표 | 중앙일보</a>
    <dt><a add_date="1671427684" href="https://www.joongang.co.kr/article/25126355">美, 中 36개 기업 수출통제 추가…AI 등 첨단산업 육성 견제 | 중앙일보</a>
    <dt><a add_date="1671427689" href="https://www.joongang.co.kr/article/25126270">백지시위 물어뜯은 中대사 "외부사주 받은 색깔혁명 냄새난다" | 중앙일보</a>
    <dt><a add_date="1671427693" href="https://www.joongang.co.kr/article/25126320" icon="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAA4klEQVQ4jWP8//8/AymABV3gz2+Gv38YGBkZGBgYIEaxsUO5WDX8Pb7777r5/398Y2BgYGBkZFLVYUmpYOQVQKj4jwa+fv6za813B6lvusw/qxL+Pb77/99fZHkmdCdx8TDpmTPw8jMwMTGp6jDKKDEwoqjB0EAIDFcNf/8y/P3LwMjIwMFJlIZ/ty79f/eKUVCUSccUi3HIkfLv3es/u9Z8DzH8bifxe8mk/3//oEfr//+MyInv7+HtfzcuYJSQY3byZ9K3YGDGSGkMDCgaGP7+Zfj3l4GVlYGBEVMpNg1EAABdi4Sb+zSY9AAAAABJRU5ErkJggg==">“워뇽이 보면 기분 좋잖아”…‘여덕몰이상’은 따로 있다 | 중앙일보</a>
    <dt><a add_date="1671427697" href="https://www.joongang.co.kr/article/25126659">저렇게 어리숙해서야, 쯧쯧…근데 왜 부럽지? | 중앙일보</a>
    <dt><a add_date="1671427700" href="https://www.joongang.co.kr/article/25126693">“혐오·거짓에 돈·권력 주는 SNS, 민주주의 위기 낳았다” | 중앙일보</a>
    <dt><a add_date="1671427703" href="https://www.joongang.co.kr/article/25126572">“유리 위 걷듯 지나온 길…정형의 그릇에 무한한 이야기 담겠다” | 중앙일보</a>
    <dt><a add_date="1671427706" href="https://www.joongang.co.kr/article/25126367">우주선 닮은 동대문 DDP에서 열리는 연말 ‘빛의 예술’ 미디어 아트쇼… 무료관람 | 중앙일보</a>
    <dt><a add_date="1671427709" href="https://www.joongang.co.kr/article/25126323" icon="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAA4klEQVQ4jWP8//8/AymABV3gz2+Gv38YGBkZGBgYIEaxsUO5WDX8Pb7777r5/398Y2BgYGBkZFLVYUmpYOQVQKj4jwa+fv6za813B6lvusw/qxL+Pb77/99fZHkmdCdx8TDpmTPw8jMwMTGp6jDKKDEwoqjB0EAIDFcNf/8y/P3LwMjIwMFJlIZ/ty79f/eKUVCUSccUi3HIkfLv3es/u9Z8DzH8bifxe8mk/3//oEfr//+MyInv7+HtfzcuYJSQY3byZ9K3YGDGSGkMDCgaGP7+Zfj3l4GVlYGBEVMpNg1EAABdi4Sb+zSY9AAAAABJRU5ErkJggg==">‘예약률 100%’ 색다른 미식…창밖으로 나온 컬리의 초대 | 중앙일보</a>
    <dt><a add_date="1671427711" href="https://www.joongang.co.kr/article/25126314">근육이 말을 듣지 않는다, 로봇공학자의 대담한 도전은[BOOK 연말연시 읽을만한 책] | 중앙일보</a>
    <dt><a add_date="1671427716" href="https://www.joongang.co.kr/article/25126310">좌우명은 '더 멀리', 이 집안을 알면 유럽사·세계사가 보인다[BOOK 연말연시 읽을만한 책] | 중앙일보</a>
    <dt><a add_date="1671427720" href="https://www.joongang.co.kr/article/25126254" icon="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAA4klEQVQ4jWP8//8/AymABV3gz2+Gv38YGBkZGBgYIEaxsUO5WDX8Pb7777r5/398Y2BgYGBkZFLVYUmpYOQVQKj4jwa+fv6za813B6lvusw/qxL+Pb77/99fZHkmdCdx8TDpmTPw8jMwMTGp6jDKKDEwoqjB0EAIDFcNf/8y/P3LwMjIwMFJlIZ/ty79f/eKUVCUSccUi3HIkfLv3es/u9Z8DzH8bifxe8mk/3//oEfr//+MyInv7+HtfzcuYJSQY3byZ9K3YGDGSGkMDCgaGP7+Zfj3l4GVlYGBEVMpNg1EAABdi4Sb+zSY9AAAAABJRU5ErkJggg==">후크 "이승기에 54억 전액 지급…합의 못했지만 분쟁 종결" | 중앙일보</a>
    <dt><a add_date="1671427749" href="https://www.joongang.co.kr/article/25126086">BTS RM, 알고보니 기름 수저?…방송중 "SK에너지!" 외친 사연 | 중앙일보</a>
    <dt><a add_date="1671427753" href="https://www.joongang.co.kr/article/25126012" icon="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAA4klEQVQ4jWP8//8/AymABV3gz2+Gv38YGBkZGBgYIEaxsUO5WDX8Pb7777r5/398Y2BgYGBkZFLVYUmpYOQVQKj4jwa+fv6za813B6lvusw/qxL+Pb77/99fZHkmdCdx8TDpmTPw8jMwMTGp6jDKKDEwoqjB0EAIDFcNf/8y/P3LwMjIwMFJlIZ/ty79f/eKUVCUSccUi3HIkfLv3es/u9Z8DzH8bifxe8mk/3//oEfr//+MyInv7+HtfzcuYJSQY3byZ9K3YGDGSGkMDCgaGP7+Zfj3l4GVlYGBEVMpNg1EAABdi4Sb+zSY9AAAAABJRU5ErkJggg==">31만1200 vs 7만5000…‘걸그룹 천하’ 증명한 숫자 | 중앙일보</a>
    <dt><a add_date="1671427764" href="https://www.joongang.co.kr/article/25126726">메시, 골든볼 MVP에도 뽑혔다…월드컵 역대 첫 2회 수상 | 중앙일보</a>
    <dt><a add_date="1671427768" href="https://www.joongang.co.kr/article/25126667">[삼성화재배 AI와 함께하는 바둑 해설] 조용히 완성된 철갑 공격군 | 중앙일보</a>
    <dt><a add_date="1671427771" href="https://www.joongang.co.kr/article/25126624">황선우, 1분40초 벽 깼다…쇼트코스 자유형 200m '2연패' | 중앙일보</a>
    <dt><a add_date="1671427775" href="https://www.joongang.co.kr/article/25126483">메르스? 코로나? 결승 이틀 앞둔 프랑스 발칵, 5명이 아프다 | 중앙일보</a>
    <dt><a add_date="1671427779" href="https://www.joongang.co.kr/article/25126246">벤투 감독, 폴란드 사령탑 거론…레반도프스키 지휘하나 | 중앙일보</a>
    <dt><a add_date="1671427783" href="https://www.joongang.co.kr/article/25126081">벤투 사단 '비트박스 코치', 한국 이웃에 남긴 마지막 선물 | 중앙일보</a>
    <dt><a add_date="1671427787" href="https://www.joongang.co.kr/article/25125831">역시, 메시 | 중앙일보</a>
    <dt><a add_date="1671427791" href="https://www.joongang.co.kr/article/25125786">여자배구 조송화 계약해지 무효소송 패소… 잔여연봉 4억원 수령 어려워져 | 중앙일보</a>
    <dt><a add_date="1671427795" href="https://www.joongang.co.kr/article/25126376">황선우 앞세운 쇼트코스 계영 800ｍ 韓신기록...역대최고 4위 | 중앙일보</a>
    <dt><a add_date="1671427799" href="https://www.joongang.co.kr/article/25126500">난임, 이것도 원인이었어? 여성 70%가 걸리는 '은밀한 질환' [건강한 가족] | 중앙일보</a>
    <dt><a add_date="1671427802" href="https://www.joongang.co.kr/article/25126300">삼겹살·목살이면 충분해, 연말 홈파티 모두가 좋아할 요리 추천 [쿠킹] | 중앙일보</a>
    <dt><a add_date="1671427805" href="https://www.joongang.co.kr/article/25125708">"약혼반지 대신 이 팔찌를" 60년대에 젠더리스 컨셉, 뉴요커 홀린 까르띠에 [더 하이엔드] | 중앙일보</a>
    <dt><a add_date="1671427808" href="https://www.joongang.co.kr/article/25124732">"답 안하는 걸로 할게요"…조규성 3초 고민하게 만든 질문 | 중앙일보</a>
    <dt><a add_date="1671427812" href="https://www.joongang.co.kr/article/25124162" icon="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAA4klEQVQ4jWP8//8/AymABV3gz2+Gv38YGBkZGBgYIEaxsUO5WDX8Pb7777r5/398Y2BgYGBkZFLVYUmpYOQVQKj4jwa+fv6za813B6lvusw/qxL+Pb77/99fZHkmdCdx8TDpmTPw8jMwMTGp6jDKKDEwoqjB0EAIDFcNf/8y/P3LwMjIwMFJlIZ/ty79f/eKUVCUSccUi3HIkfLv3es/u9Z8DzH8bifxe8mk/3//oEfr//+MyInv7+HtfzcuYJSQY3byZ9K3YGDGSGkMDCgaGP7+Zfj3l4GVlYGBEVMpNg1EAABdi4Sb+zSY9AAAAABJRU5ErkJggg==">“이 사회 전체가 의자였구나” 샤워실 목욕의자의 재발견 | 중앙일보</a>
    <dt><a add_date="1671427816" href="https://www.joongang.co.kr/article/25122912">카페인 있어도 고혈압과 무관? 커피가 '두얼굴 헐크'인 까닭 [건강한 가족] | 중앙일보</a>
    <dt><a add_date="1671427819" href="https://www.joongang.co.kr/article/25122804">올가을 가뭄에 당도↑, 최근 비에 산도↓…역대급 당산비 감귤 [e즐펀한 토크] | 중앙일보</a>
    <dt><a add_date="1671427825" href="https://www.joongang.co.kr/article/25121510">예거 르쿨트르와 레터링 아티스트 알렉스 트로슈가 만드는 새로운 의미 [더 하이엔드] | 중앙일보</a>
    <dt><a add_date="1671427829" href="https://www.joongang.co.kr/article/25126648">[사랑방] 보훈처, 가수 이미자에 감사패 수여 | 중앙일보</a>
    <dt><a add_date="1671427837" href="https://www.joongang.co.kr/article/25125827">[사랑방] SK그룹, 이웃사랑성금 120억 | 중앙일보</a>
    <dt><a add_date="1671427840" href="https://www.joongang.co.kr/article/25125508">[사랑방] 권모세 회장 등 4명 ‘서울대AMP대상’ | 중앙일보</a>
    <dt><a add_date="1671427843" href="https://www.joongang.co.kr/article/25125189">[인사] 에스원 外 | 중앙일보</a>
    <dt><a add_date="1671427849" href="https://www.joongang.co.kr/article/25124850">[사진] 서울국제포럼-나카소네 세계평화연 ‘한·일 신시대를 향해’ 포럼 | 중앙일보</a>
    <dt><a add_date="1671427852" href="https://www.joongang.co.kr/article/25124840">[사랑방] 이덕로 한국행정학회장 취임 | 중앙일보</a>
    <dt><a add_date="1671427858" href="https://www.joongang.co.kr/article/25123714" icon="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAA4klEQVQ4jWP8//8/AymABV3gz2+Gv38YGBkZGBgYIEaxsUO5WDX8Pb7777r5/398Y2BgYGBkZFLVYUmpYOQVQKj4jwa+fv6za813B6lvusw/qxL+Pb77/99fZHkmdCdx8TDpmTPw8jMwMTGp6jDKKDEwoqjB0EAIDFcNf/8y/P3LwMjIwMFJlIZ/ty79f/eKUVCUSccUi3HIkfLv3es/u9Z8DzH8bifxe8mk/3//oEfr//+MyInv7+HtfzcuYJSQY3byZ9K3YGDGSGkMDCgaGP7+Zfj3l4GVlYGBEVMpNg1EAABdi4Sb+zSY9AAAAABJRU5ErkJggg==">[사진] 중앙광고대상 시상식 | 중앙일보</a>
    </dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></dt></p></dl><p>
    <p>
    </p></p></dt>




```python
class dev_topic:
    def __init__(self, soup, df_answer):
        f=codecs.open("./open/bookmarks_84.html", 'r')
        soup = BeautifulSoup(f.read(), 'html.parser')
        df_answer = pd.read_csv('./open/roberta_large_epoch3.csv')
        
        self.soup = soup
        self.df_answer = df_answer
        
    def separate_tag(self, topic):
        main_soup = soup.dt
        main_soup.h3.string.replace_with(f"{topic}")
        
        
        for i in range(len(main_soup.find_all('a'))):
            main_soup.find('a').decompose()

        
        head = '<DT>' + str(main_soup.find('h3')).replace('h3','H3') + '\n' + '<dl><p>'
    
        tail = '</dl><p>'
        # tail = '<p>'
        
        num = len(self.df_answer[self.df_answer['answer'] == f'{topic}'])
        mid = str(main_soup.find_all('dt')[len(df_answer) - num])
        mid = mid.split('\n')[:-1]
        
        IT_li = list(df_answer[df_answer['answer'] == topic]['tag'])
        
        mid_fin = ''
        for i in range(len(IT_li)):
            mid_fin += mid[i]
            mid_fin += IT_li[i]
            mid_fin += '\n'

        
        fin_text = head + '\n' + mid_fin + tail
        return fin_text
        
```

# Result


```python
topic_li = list(topic_dict['topic'])
topic_li
```




    ['IT과학', '경제', '사회', '생활문화', '세계', '스포츠', '정치']




```python
result = ''
for topic in topic_li:
    result += dev_topic(soup, df_answer).separate_tag(topic)
    result += '\n'
    
print(result)
```

    <DT><H3 add_date="1629940719" last_modified="1671427858">IT과학</H3>
    <dl><p>
    <dt><a add_date="1671427376" href="https://www.joongang.co.kr/article/25126593">'게임체인저' 신입당원은 20만...국힘, 당원 포섭전쟁은 끝났다 | 중앙일보</a>
    <dt><a add_date="1671427543" href="https://www.joongang.co.kr/article/25126811">LG엔솔, 오창공장 배터리 라인 신·증설에 4조 투자…1800명 채용 | 중앙일보</a>
    <dt><a add_date="1671427553" href="https://www.joongang.co.kr/article/25126728">[알림] '펫(pet) 톡톡': 평생 간직하세요, 인생 사진 2탄 | 중앙일보</a>
    <dt><a add_date="1671427562" href="https://www.joongang.co.kr/article/25126670">SK그룹, CES 2023 무대로 넷제로 역량 선보인다 | 중앙일보</a>
    <dt><a add_date="1671427573" href="https://www.joongang.co.kr/article/25126583" icon="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAA4klEQVQ4jWP8//8/AymABV3gz2+Gv38YGBkZGBgYIEaxsUO5WDX8Pb7777r5/398Y2BgYGBkZFLVYUmpYOQVQKj4jwa+fv6za813B6lvusw/qxL+Pb77/99fZHkmdCdx8TDpmTPw8jMwMTGp6jDKKDEwoqjB0EAIDFcNf/8y/P3LwMjIwMFJlIZ/ty79f/eKUVCUSccUi3HIkfLv3es/u9Z8DzH8bifxe8mk/3//oEfr//+MyInv7+HtfzcuYJSQY3byZ9K3YGDGSGkMDCgaGP7+Zfj3l4GVlYGBEVMpNg1EAABdi4Sb+zSY9AAAAABJRU5ErkJggg==">1년 4번 그들은 점을 찍는다…美 Fed, 과학인가 예언인가 | 중앙일보</a>
    <dt><a add_date="1671427597" href="https://www.joongang.co.kr/article/25126830">한국공대, 지역산업 연계 오픈랩 혁신기술 설명회 개최 | 중앙일보</a>
    <dt><a add_date="1671427711" href="https://www.joongang.co.kr/article/25126314">근육이 말을 듣지 않는다, 로봇공학자의 대담한 도전은[BOOK 연말연시 읽을만한 책] | 중앙일보</a>
    <dt><a add_date="1671427768" href="https://www.joongang.co.kr/article/25126667">[삼성화재배 AI와 함께하는 바둑 해설] 조용히 완성된 철갑 공격군 | 중앙일보</a>
    </dl><p>
    <DT><H3 add_date="1629940719" last_modified="1671427858">경제</H3>
    <dl><p>
    <dt><a add_date="1671427539" href="https://www.joongang.co.kr/article/25126817">정부가 눌러 5% 안넘은 근원물가…'공공요금 인상' 내년엔 | 중앙일보</a>
    <dt><a add_date="1671427547" href="https://www.joongang.co.kr/article/25126800">퇴직연금 중도인출 “집 사려고, 전세 보증금 때문에” 81.6% | 중앙일보</a>
    <dt><a add_date="1671427556" href="https://www.joongang.co.kr/article/25126677">금리↑ 집값↓…“2% 청약통장 깨서 7% 대출이자 갚는다” | 중앙일보</a>
    <dt><a add_date="1671427559" href="https://www.joongang.co.kr/article/25126674">[사진] 청약경쟁률 8년 만에 한 자릿수로 | 중앙일보</a>
    </dl><p>
    <DT><H3 add_date="1629940719" last_modified="1671427858">사회</H3>
    <dl><p>
    <dt><a add_date="1584968656" href="https://www.mk.co.kr/opinion/contributors/view/2016/08/618807/" icon="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAABPUlEQVQ4jaWTsUoDQRCGvz0OJWLEgEXSCCo2wVcQX8PGQsHCSgt9DBFJoYKp7QURH8FGJNgaFQsLEUXBC0d21+J273bXSxTdZmZ25v93ZnZGALxvzmj+cCb274T4EawBZWTf6NLoNYi8YDGEQBppdZm5Yze2utfNMG9PfLaWUc+PoCCammZs6wRRawDwsTKbE/gZ2EQmG1TWj2FkHDFapbLRzsEwJAP3RPV5KmstICKqz/lO24tBGciHTsbeXCJuLmZ3t9dFgO3DIIL04oD+1Xnx4OUZ6elRaQkFgfuZStNrbyPvb5DdDsnhDihdShDnYOUSgE4Skt1VkBqS3je/tUsJtAnQry/Fa65fgvCaqPwAocg6LR0Z+o2djXLq1BUCre5OoB3tBZtBCJa/ABOWIPEXRQZ3FhisnoD/rfMXtEaiNPk3uHYAAAAASUVORK5CYII=">통계는 과연 거짓말을 할까? - 매일경제</a>
    <dt><a add_date="1671427367" href="https://www.joongang.co.kr/article/25126692">[사진] 서초동 사저 찾아 작별 인사 | 중앙일보</a>
    <dt><a add_date="1671427566" href="https://www.joongang.co.kr/article/25126664" icon="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAA4klEQVQ4jWP8//8/AymABV3gz2+Gv38YGBkZGBgYIEaxsUO5WDX8Pb7777r5/398Y2BgYGBkZFLVYUmpYOQVQKj4jwa+fv6za813B6lvusw/qxL+Pb77/99fZHkmdCdx8TDpmTPw8jMwMTGp6jDKKDEwoqjB0EAIDFcNf/8y/P3LwMjIwMFJlIZ/ty79f/eKUVCUSccUi3HIkfLv3es/u9Z8DzH8bifxe8mk/3//oEfr//+MyInv7+HtfzcuYJSQY3byZ9K3YGDGSGkMDCgaGP7+Zfj3l4GVlYGBEVMpNg1EAABdi4Sb+zSY9AAAAABJRU5ErkJggg==">조류인플루엔자 확산…달걀값 또다시 들썩 | 중앙일보</a>
    <dt><a add_date="1671427600" href="https://www.joongang.co.kr/article/25126823">전문대교육협의회, ‘2022년 전문대학인상’ 시상 | 중앙일보</a>
    <dt><a add_date="1671427603" href="https://www.joongang.co.kr/article/25126793">'공무원' 인기는 뚝…중·고교생 희망직업으로 뜬 이 계열 | 중앙일보</a>
    <dt><a add_date="1671427617" href="https://www.joongang.co.kr/article/25126790">"다시 만나자" 안받아준 前연인…준비한 흉기로 살해한 50대男 | 중앙일보</a>
    <dt><a add_date="1671427620" href="https://www.joongang.co.kr/article/25126769">말다툼하다 아내 살해한 남편, 범행 후 그 옆에서 술 마셨다 | 중앙일보</a>
    <dt><a add_date="1671427632" href="https://www.joongang.co.kr/article/25126787">용산역에만 17분 멈췄다…전장연 출근길 기습에 "좀 그만하라" | 중앙일보</a>
    <dt><a add_date="1671427651" href="https://www.joongang.co.kr/article/25126542">"돈 뺏길지 모른다""집주인 연락 안돼" 매일 고성 터지는 이곳 | 중앙일보</a>
    <dt><a add_date="1671427700" href="https://www.joongang.co.kr/article/25126693">“혐오·거짓에 돈·권력 주는 SNS, 민주주의 위기 낳았다” | 중앙일보</a>
    <dt><a add_date="1671427799" href="https://www.joongang.co.kr/article/25126500">난임, 이것도 원인이었어? 여성 70%가 걸리는 '은밀한 질환' [건강한 가족] | 중앙일보</a>
    <dt><a add_date="1671427829" href="https://www.joongang.co.kr/article/25126648">[사랑방] 보훈처, 가수 이미자에 감사패 수여 | 중앙일보</a>
    <dt><a add_date="1671427837" href="https://www.joongang.co.kr/article/25125827">[사랑방] SK그룹, 이웃사랑성금 120억 | 중앙일보</a>
    <dt><a add_date="1671427840" href="https://www.joongang.co.kr/article/25125508">[사랑방] 권모세 회장 등 4명 ‘서울대AMP대상’ | 중앙일보</a>
    <dt><a add_date="1671427843" href="https://www.joongang.co.kr/article/25125189">[인사] 에스원 外 | 중앙일보</a>
    <dt><a add_date="1671427849" href="https://www.joongang.co.kr/article/25124850">[사진] 서울국제포럼-나카소네 세계평화연 ‘한·일 신시대를 향해’ 포럼 | 중앙일보</a>
    <dt><a add_date="1671427852" href="https://www.joongang.co.kr/article/25124840">[사랑방] 이덕로 한국행정학회장 취임 | 중앙일보</a>
    </dl><p>
    <DT><H3 add_date="1629940719" last_modified="1671427858">생활문화</H3>
    <dl><p>
    <dt><a add_date="1671427570" href="https://www.joongang.co.kr/article/25126621">[팩플] “더 오래, 더 젊게”…달라지는 카톡, 카카오의 노림수는 | 중앙일보</a>
    <dt><a add_date="1671427576" href="https://www.joongang.co.kr/article/25126570" icon="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAA4klEQVQ4jWP8//8/AymABV3gz2+Gv38YGBkZGBgYIEaxsUO5WDX8Pb7777r5/398Y2BgYGBkZFLVYUmpYOQVQKj4jwa+fv6za813B6lvusw/qxL+Pb77/99fZHkmdCdx8TDpmTPw8jMwMTGp6jDKKDEwoqjB0EAIDFcNf/8y/P3LwMjIwMFJlIZ/ty79f/eKUVCUSccUi3HIkfLv3es/u9Z8DzH8bifxe8mk/3//oEfr//+MyInv7+HtfzcuYJSQY3byZ9K3YGDGSGkMDCgaGP7+Zfj3l4GVlYGBEVMpNg1EAABdi4Sb+zSY9AAAAABJRU5ErkJggg==">“신곡 설령 좋아도 관심없다”…3000억 저작권 굴리는 29살 | 중앙일보</a>
    <dt><a add_date="1671427581" href="https://www.joongang.co.kr/article/25126732">[VIEW] 퍼펙트스톰 긴박감 사라졌지만, 수출ㆍ고용ㆍ부동산 위기 계속 | 중앙일보</a>
    <dt><a add_date="1671427591" href="https://www.joongang.co.kr/article/25126615">밤 7시 '구찌 매장' 불꺼졌다…쇼핑몰은 거대한 미술관이 됐다 | 중앙일보</a>
    <dt><a add_date="1671427606" href="https://www.joongang.co.kr/article/25126784">제주 항공기·여객선 운항 정상화…"서해안엔 계속 많은 눈 예상" | 중앙일보</a>
    <dt><a add_date="1671427613" href="https://www.joongang.co.kr/article/25126745">[소년중앙] 어디 가야 먹을 수 있나요, 찬바람 불면 돌아오는 제철 붕어빵 | 중앙일보</a>
    <dt><a add_date="1671427628" href="https://www.joongang.co.kr/article/25126656">[오늘의 운세] 12월 19일 | 중앙일보</a>
    <dt><a add_date="1671427640" href="https://www.joongang.co.kr/article/25126666">[우리말 바루기] ‘결제’를 할까, ‘결재’를 할까? | 중앙일보</a>
    <dt><a add_date="1671427644" href="https://www.joongang.co.kr/article/25126629">여객기 109편 결항…19일까지 많은 눈, 아침 -15도 강추위 | 중앙일보</a>
    <dt><a add_date="1671427693" href="https://www.joongang.co.kr/article/25126320" icon="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAA4klEQVQ4jWP8//8/AymABV3gz2+Gv38YGBkZGBgYIEaxsUO5WDX8Pb7777r5/398Y2BgYGBkZFLVYUmpYOQVQKj4jwa+fv6za813B6lvusw/qxL+Pb77/99fZHkmdCdx8TDpmTPw8jMwMTGp6jDKKDEwoqjB0EAIDFcNf/8y/P3LwMjIwMFJlIZ/ty79f/eKUVCUSccUi3HIkfLv3es/u9Z8DzH8bifxe8mk/3//oEfr//+MyInv7+HtfzcuYJSQY3byZ9K3YGDGSGkMDCgaGP7+Zfj3l4GVlYGBEVMpNg1EAABdi4Sb+zSY9AAAAABJRU5ErkJggg==">“워뇽이 보면 기분 좋잖아”…‘여덕몰이상’은 따로 있다 | 중앙일보</a>
    <dt><a add_date="1671427697" href="https://www.joongang.co.kr/article/25126659">저렇게 어리숙해서야, 쯧쯧…근데 왜 부럽지? | 중앙일보</a>
    <dt><a add_date="1671427703" href="https://www.joongang.co.kr/article/25126572">“유리 위 걷듯 지나온 길…정형의 그릇에 무한한 이야기 담겠다” | 중앙일보</a>
    <dt><a add_date="1671427706" href="https://www.joongang.co.kr/article/25126367">우주선 닮은 동대문 DDP에서 열리는 연말 ‘빛의 예술’ 미디어 아트쇼… 무료관람 | 중앙일보</a>
    <dt><a add_date="1671427709" href="https://www.joongang.co.kr/article/25126323" icon="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAA4klEQVQ4jWP8//8/AymABV3gz2+Gv38YGBkZGBgYIEaxsUO5WDX8Pb7777r5/398Y2BgYGBkZFLVYUmpYOQVQKj4jwa+fv6za813B6lvusw/qxL+Pb77/99fZHkmdCdx8TDpmTPw8jMwMTGp6jDKKDEwoqjB0EAIDFcNf/8y/P3LwMjIwMFJlIZ/ty79f/eKUVCUSccUi3HIkfLv3es/u9Z8DzH8bifxe8mk/3//oEfr//+MyInv7+HtfzcuYJSQY3byZ9K3YGDGSGkMDCgaGP7+Zfj3l4GVlYGBEVMpNg1EAABdi4Sb+zSY9AAAAABJRU5ErkJggg==">‘예약률 100%’ 색다른 미식…창밖으로 나온 컬리의 초대 | 중앙일보</a>
    <dt><a add_date="1671427716" href="https://www.joongang.co.kr/article/25126310">좌우명은 '더 멀리', 이 집안을 알면 유럽사·세계사가 보인다[BOOK 연말연시 읽을만한 책] | 중앙일보</a>
    <dt><a add_date="1671427749" href="https://www.joongang.co.kr/article/25126086">BTS RM, 알고보니 기름 수저?…방송중 "SK에너지!" 외친 사연 | 중앙일보</a>
    <dt><a add_date="1671427753" href="https://www.joongang.co.kr/article/25126012" icon="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAA4klEQVQ4jWP8//8/AymABV3gz2+Gv38YGBkZGBgYIEaxsUO5WDX8Pb7777r5/398Y2BgYGBkZFLVYUmpYOQVQKj4jwa+fv6za813B6lvusw/qxL+Pb77/99fZHkmdCdx8TDpmTPw8jMwMTGp6jDKKDEwoqjB0EAIDFcNf/8y/P3LwMjIwMFJlIZ/ty79f/eKUVCUSccUi3HIkfLv3es/u9Z8DzH8bifxe8mk/3//oEfr//+MyInv7+HtfzcuYJSQY3byZ9K3YGDGSGkMDCgaGP7+Zfj3l4GVlYGBEVMpNg1EAABdi4Sb+zSY9AAAAABJRU5ErkJggg==">31만1200 vs 7만5000…‘걸그룹 천하’ 증명한 숫자 | 중앙일보</a>
    <dt><a add_date="1671427802" href="https://www.joongang.co.kr/article/25126300">삼겹살·목살이면 충분해, 연말 홈파티 모두가 좋아할 요리 추천 [쿠킹] | 중앙일보</a>
    <dt><a add_date="1671427805" href="https://www.joongang.co.kr/article/25125708">"약혼반지 대신 이 팔찌를" 60년대에 젠더리스 컨셉, 뉴요커 홀린 까르띠에 [더 하이엔드] | 중앙일보</a>
    <dt><a add_date="1671427812" href="https://www.joongang.co.kr/article/25124162" icon="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAA4klEQVQ4jWP8//8/AymABV3gz2+Gv38YGBkZGBgYIEaxsUO5WDX8Pb7777r5/398Y2BgYGBkZFLVYUmpYOQVQKj4jwa+fv6za813B6lvusw/qxL+Pb77/99fZHkmdCdx8TDpmTPw8jMwMTGp6jDKKDEwoqjB0EAIDFcNf/8y/P3LwMjIwMFJlIZ/ty79f/eKUVCUSccUi3HIkfLv3es/u9Z8DzH8bifxe8mk/3//oEfr//+MyInv7+HtfzcuYJSQY3byZ9K3YGDGSGkMDCgaGP7+Zfj3l4GVlYGBEVMpNg1EAABdi4Sb+zSY9AAAAABJRU5ErkJggg==">“이 사회 전체가 의자였구나” 샤워실 목욕의자의 재발견 | 중앙일보</a>
    <dt><a add_date="1671427816" href="https://www.joongang.co.kr/article/25122912">카페인 있어도 고혈압과 무관? 커피가 '두얼굴 헐크'인 까닭 [건강한 가족] | 중앙일보</a>
    <dt><a add_date="1671427819" href="https://www.joongang.co.kr/article/25122804">올가을 가뭄에 당도↑, 최근 비에 산도↓…역대급 당산비 감귤 [e즐펀한 토크] | 중앙일보</a>
    <dt><a add_date="1671427825" href="https://www.joongang.co.kr/article/25121510">예거 르쿨트르와 레터링 아티스트 알렉스 트로슈가 만드는 새로운 의미 [더 하이엔드] | 중앙일보</a>
    <dt><a add_date="1671427858" href="https://www.joongang.co.kr/article/25123714" icon="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAA4klEQVQ4jWP8//8/AymABV3gz2+Gv38YGBkZGBgYIEaxsUO5WDX8Pb7777r5/398Y2BgYGBkZFLVYUmpYOQVQKj4jwa+fv6za813B6lvusw/qxL+Pb77/99fZHkmdCdx8TDpmTPw8jMwMTGp6jDKKDEwoqjB0EAIDFcNf/8y/P3LwMjIwMFJlIZ/ty79f/eKUVCUSccUi3HIkfLv3es/u9Z8DzH8bifxe8mk/3//oEfr//+MyInv7+HtfzcuYJSQY3byZ9K3YGDGSGkMDCgaGP7+Zfj3l4GVlYGBEVMpNg1EAABdi4Sb+zSY9AAAAABJRU5ErkJggg==">[사진] 중앙광고대상 시상식 | 중앙일보</a>
    </dl><p>
    <DT><H3 add_date="1629940719" last_modified="1671427858">세계</H3>
    <dl><p>
    <dt><a add_date="1671427658" href="https://www.joongang.co.kr/article/25126818">미얀마 최대도시 양곤서 또 폭발 사고…페리 승객 11명 부상 | 중앙일보</a>
    <dt><a add_date="1671427665" href="https://www.joongang.co.kr/article/25126649">히잡시위 지지한 ‘이란 국민배우’ 체포 | 중앙일보</a>
    <dt><a add_date="1671427670" href="https://www.joongang.co.kr/article/25126545">"우크라, 미국 만류에도 러시아軍 최고 지휘관 암살 시도" | 중앙일보</a>
    <dt><a add_date="1671427673" href="https://www.joongang.co.kr/article/25126510">'건강이상설' 푸틴 군 사령관들 소집…"우크라 작전 제안하라" | 중앙일보</a>
    <dt><a add_date="1671427676" href="https://www.joongang.co.kr/article/25126489">코로나 지원금 100억 빼돌린 美목사…호화주택 사려다 덜미 | 중앙일보</a>
    <dt><a add_date="1671427680" href="https://www.joongang.co.kr/article/25126482">한 달 전 인니 지진 사망자, 300명대 아닌 602명…재집계 수치 발표 | 중앙일보</a>
    <dt><a add_date="1671427684" href="https://www.joongang.co.kr/article/25126355">美, 中 36개 기업 수출통제 추가…AI 등 첨단산업 육성 견제 | 중앙일보</a>
    <dt><a add_date="1671427689" href="https://www.joongang.co.kr/article/25126270">백지시위 물어뜯은 中대사 "외부사주 받은 색깔혁명 냄새난다" | 중앙일보</a>
    </dl><p>
    <DT><H3 add_date="1629940719" last_modified="1671427858">스포츠</H3>
    <dl><p>
    <dt><a add_date="1671427337" href="https://www.joongang.co.kr/article/25126744" icon="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAA4klEQVQ4jWP8//8/AymABV3gz2+Gv38YGBkZGBgYIEaxsUO5WDX8Pb7777r5/398Y2BgYGBkZFLVYUmpYOQVQKj4jwa+fv6za813B6lvusw/qxL+Pb77/99fZHkmdCdx8TDpmTPw8jMwMTGp6jDKKDEwoqjB0EAIDFcNf/8y/P3LwMjIwMFJlIZ/ty79f/eKUVCUSccUi3HIkfLv3es/u9Z8DzH8bifxe8mk/3//oEfr//+MyInv7+HtfzcuYJSQY3byZ9K3YGDGSGkMDCgaGP7+Zfj3l4GVlYGBEVMpNg1EAABdi4Sb+zSY9AAAAABJRU5ErkJggg==">故마라도나에 우승컵 바쳤다…‘메신’의 짜릿한 ‘카타르’시스 | 중앙일보</a>
    <dt><a add_date="1671427389" href="https://www.joongang.co.kr/article/25126571">尹 '조용한 생일'…"직언 들어줘 감사하다" 참모들 축하 메시지 | 중앙일보</a>
    <dt><a add_date="1671427550" href="https://www.joongang.co.kr/article/25126770">기아, 카타르 월드컵에 차량 297대 투입…브랜드 홍보 | 중앙일보</a>
    <dt><a add_date="1671427609" href="https://www.joongang.co.kr/article/25126756">[THINK ENGLISH] 손흥민, 마스크 벗고 런던에서 훈련 재개 | 중앙일보</a>
    <dt><a add_date="1671427662" href="https://www.joongang.co.kr/article/25126753">"메시 우승 자격 있다"…'축구의 신'에 축하 건넨 '축구 황제' | 중앙일보</a>
    <dt><a add_date="1671427720" href="https://www.joongang.co.kr/article/25126254" icon="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAA4klEQVQ4jWP8//8/AymABV3gz2+Gv38YGBkZGBgYIEaxsUO5WDX8Pb7777r5/398Y2BgYGBkZFLVYUmpYOQVQKj4jwa+fv6za813B6lvusw/qxL+Pb77/99fZHkmdCdx8TDpmTPw8jMwMTGp6jDKKDEwoqjB0EAIDFcNf/8y/P3LwMjIwMFJlIZ/ty79f/eKUVCUSccUi3HIkfLv3es/u9Z8DzH8bifxe8mk/3//oEfr//+MyInv7+HtfzcuYJSQY3byZ9K3YGDGSGkMDCgaGP7+Zfj3l4GVlYGBEVMpNg1EAABdi4Sb+zSY9AAAAABJRU5ErkJggg==">후크 "이승기에 54억 전액 지급…합의 못했지만 분쟁 종결" | 중앙일보</a>
    <dt><a add_date="1671427764" href="https://www.joongang.co.kr/article/25126726">메시, 골든볼 MVP에도 뽑혔다…월드컵 역대 첫 2회 수상 | 중앙일보</a>
    <dt><a add_date="1671427771" href="https://www.joongang.co.kr/article/25126624">황선우, 1분40초 벽 깼다…쇼트코스 자유형 200m '2연패' | 중앙일보</a>
    <dt><a add_date="1671427775" href="https://www.joongang.co.kr/article/25126483">메르스? 코로나? 결승 이틀 앞둔 프랑스 발칵, 5명이 아프다 | 중앙일보</a>
    <dt><a add_date="1671427779" href="https://www.joongang.co.kr/article/25126246">벤투 감독, 폴란드 사령탑 거론…레반도프스키 지휘하나 | 중앙일보</a>
    <dt><a add_date="1671427783" href="https://www.joongang.co.kr/article/25126081">벤투 사단 '비트박스 코치', 한국 이웃에 남긴 마지막 선물 | 중앙일보</a>
    <dt><a add_date="1671427787" href="https://www.joongang.co.kr/article/25125831">역시, 메시 | 중앙일보</a>
    <dt><a add_date="1671427791" href="https://www.joongang.co.kr/article/25125786">여자배구 조송화 계약해지 무효소송 패소… 잔여연봉 4억원 수령 어려워져 | 중앙일보</a>
    <dt><a add_date="1671427795" href="https://www.joongang.co.kr/article/25126376">황선우 앞세운 쇼트코스 계영 800ｍ 韓신기록...역대최고 4위 | 중앙일보</a>
    <dt><a add_date="1671427808" href="https://www.joongang.co.kr/article/25124732">"답 안하는 걸로 할게요"…조규성 3초 고민하게 만든 질문 | 중앙일보</a>
    </dl><p>
    <DT><H3 add_date="1629940719" last_modified="1671427858">정치</H3>
    <dl><p>
    <dt><a add_date="1671427356" href="https://www.joongang.co.kr/article/25126791">文정부 ‘통계조작’ 공방 격화…與 “통계조작성장” 野 “모욕주기” | 중앙일보</a>
    <dt><a add_date="1671427361" href="https://www.joongang.co.kr/article/25126774">정청래, 박지원 복당 공개반대 "한 번 배신하면 또 배신" | 중앙일보</a>
    <dt><a add_date="1671427364" href="https://www.joongang.co.kr/article/25126755">尹지지율, 중도·20대가 쌍끌이로 올렸다…6월 이후 첫 40%대 [리얼미터] | 중앙일보</a>
    <dt><a add_date="1671427371" href="https://www.joongang.co.kr/article/25126618">법인세보다 장관 '빅2'가 핵심...野 반대한 한동훈·이상민 예산은 | 중앙일보</a>
    <dt><a add_date="1671427381" href="https://www.joongang.co.kr/article/25126580" icon="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAA4klEQVQ4jWP8//8/AymABV3gz2+Gv38YGBkZGBgYIEaxsUO5WDX8Pb7777r5/398Y2BgYGBkZFLVYUmpYOQVQKj4jwa+fv6za813B6lvusw/qxL+Pb77/99fZHkmdCdx8TDpmTPw8jMwMTGp6jDKKDEwoqjB0EAIDFcNf/8y/P3LwMjIwMFJlIZ/ty79f/eKUVCUSccUi3HIkfLv3es/u9Z8DzH8bifxe8mk/3//oEfr//+MyInv7+HtfzcuYJSQY3byZ9K3YGDGSGkMDCgaGP7+Zfj3l4GVlYGBEVMpNg1EAABdi4Sb+zSY9AAAAABJRU5ErkJggg==">‘당심 100% 룰’ 김용태 “윤핵관 당헌·당규 손바닥 뒤집듯 바꿔…한심” | 중앙일보</a>
    <dt><a add_date="1671427395" href="https://www.joongang.co.kr/article/25126550">[속보] 북, 탄도미사일 2발 발사…“일본 EEZ 밖 낙하 추정” | 중앙일보</a>
    <dt><a add_date="1671427399" href="https://www.joongang.co.kr/article/25126364" icon="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAA4klEQVQ4jWP8//8/AymABV3gz2+Gv38YGBkZGBgYIEaxsUO5WDX8Pb7777r5/398Y2BgYGBkZFLVYUmpYOQVQKj4jwa+fv6za813B6lvusw/qxL+Pb77/99fZHkmdCdx8TDpmTPw8jMwMTGp6jDKKDEwoqjB0EAIDFcNf/8y/P3LwMjIwMFJlIZ/ty79f/eKUVCUSccUi3HIkfLv3es/u9Z8DzH8bifxe8mk/3//oEfr//+MyInv7+HtfzcuYJSQY3byZ9K3YGDGSGkMDCgaGP7+Zfj3l4GVlYGBEVMpNg1EAABdi4Sb+zSY9AAAAABJRU5ErkJggg==">이정현 "당대표 나가든 자문을 하든 소극적으로 있지 않겠다" | 중앙일보</a>
    <dt><a add_date="1671427511" href="https://www.joongang.co.kr/article/25126336" icon="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAA4klEQVQ4jWP8//8/AymABV3gz2+Gv38YGBkZGBgYIEaxsUO5WDX8Pb7777r5/398Y2BgYGBkZFLVYUmpYOQVQKj4jwa+fv6za813B6lvusw/qxL+Pb77/99fZHkmdCdx8TDpmTPw8jMwMTGp6jDKKDEwoqjB0EAIDFcNf/8y/P3LwMjIwMFJlIZ/ty79f/eKUVCUSccUi3HIkfLv3es/u9Z8DzH8bifxe8mk/3//oEfr//+MyInv7+HtfzcuYJSQY3byZ9K3YGDGSGkMDCgaGP7+Zfj3l4GVlYGBEVMpNg1EAABdi4Sb+zSY9AAAAABJRU5ErkJggg==">유승민 여론조사 보니…친윤 "이게 당원투표 100% 이유" | 중앙일보</a>
    </dl><p>
    



```python
add = '''<!DOCTYPE NETSCAPE-Bookmark-file-1>
<!-- This is an automatically generated file.
     It will be read and overwritten.
     DO NOT EDIT! -->
<META HTTP-EQUIV="Content-Type" CONTENT="text/html; charset=UTF-8">
<TITLE>Bookmarks</TITLE>
<H1>Bookmarks</H1>'''
```


```python
add_result = add + '\n' + result
print(add_result)
```

    <!DOCTYPE NETSCAPE-Bookmark-file-1>
    <!-- This is an automatically generated file.
         It will be read and overwritten.
         DO NOT EDIT! -->
    <META HTTP-EQUIV="Content-Type" CONTENT="text/html; charset=UTF-8">
    <TITLE>Bookmarks</TITLE>
    <H1>Bookmarks</H1>
    <DT><H3 add_date="1629940719" last_modified="1671427858">IT과학</H3>
    <dl><p>
    <dt><a add_date="1671427376" href="https://www.joongang.co.kr/article/25126593">'게임체인저' 신입당원은 20만...국힘, 당원 포섭전쟁은 끝났다 | 중앙일보</a>
    <dt><a add_date="1671427543" href="https://www.joongang.co.kr/article/25126811">LG엔솔, 오창공장 배터리 라인 신·증설에 4조 투자…1800명 채용 | 중앙일보</a>
    <dt><a add_date="1671427553" href="https://www.joongang.co.kr/article/25126728">[알림] '펫(pet) 톡톡': 평생 간직하세요, 인생 사진 2탄 | 중앙일보</a>
    <dt><a add_date="1671427562" href="https://www.joongang.co.kr/article/25126670">SK그룹, CES 2023 무대로 넷제로 역량 선보인다 | 중앙일보</a>
    <dt><a add_date="1671427573" href="https://www.joongang.co.kr/article/25126583" icon="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAA4klEQVQ4jWP8//8/AymABV3gz2+Gv38YGBkZGBgYIEaxsUO5WDX8Pb7777r5/398Y2BgYGBkZFLVYUmpYOQVQKj4jwa+fv6za813B6lvusw/qxL+Pb77/99fZHkmdCdx8TDpmTPw8jMwMTGp6jDKKDEwoqjB0EAIDFcNf/8y/P3LwMjIwMFJlIZ/ty79f/eKUVCUSccUi3HIkfLv3es/u9Z8DzH8bifxe8mk/3//oEfr//+MyInv7+HtfzcuYJSQY3byZ9K3YGDGSGkMDCgaGP7+Zfj3l4GVlYGBEVMpNg1EAABdi4Sb+zSY9AAAAABJRU5ErkJggg==">1년 4번 그들은 점을 찍는다…美 Fed, 과학인가 예언인가 | 중앙일보</a>
    <dt><a add_date="1671427597" href="https://www.joongang.co.kr/article/25126830">한국공대, 지역산업 연계 오픈랩 혁신기술 설명회 개최 | 중앙일보</a>
    <dt><a add_date="1671427711" href="https://www.joongang.co.kr/article/25126314">근육이 말을 듣지 않는다, 로봇공학자의 대담한 도전은[BOOK 연말연시 읽을만한 책] | 중앙일보</a>
    <dt><a add_date="1671427768" href="https://www.joongang.co.kr/article/25126667">[삼성화재배 AI와 함께하는 바둑 해설] 조용히 완성된 철갑 공격군 | 중앙일보</a>
    </dl><p>
    <DT><H3 add_date="1629940719" last_modified="1671427858">경제</H3>
    <dl><p>
    <dt><a add_date="1671427539" href="https://www.joongang.co.kr/article/25126817">정부가 눌러 5% 안넘은 근원물가…'공공요금 인상' 내년엔 | 중앙일보</a>
    <dt><a add_date="1671427547" href="https://www.joongang.co.kr/article/25126800">퇴직연금 중도인출 “집 사려고, 전세 보증금 때문에” 81.6% | 중앙일보</a>
    <dt><a add_date="1671427556" href="https://www.joongang.co.kr/article/25126677">금리↑ 집값↓…“2% 청약통장 깨서 7% 대출이자 갚는다” | 중앙일보</a>
    <dt><a add_date="1671427559" href="https://www.joongang.co.kr/article/25126674">[사진] 청약경쟁률 8년 만에 한 자릿수로 | 중앙일보</a>
    </dl><p>
    <DT><H3 add_date="1629940719" last_modified="1671427858">사회</H3>
    <dl><p>
    <dt><a add_date="1584968656" href="https://www.mk.co.kr/opinion/contributors/view/2016/08/618807/" icon="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAABPUlEQVQ4jaWTsUoDQRCGvz0OJWLEgEXSCCo2wVcQX8PGQsHCSgt9DBFJoYKp7QURH8FGJNgaFQsLEUXBC0d21+J273bXSxTdZmZ25v93ZnZGALxvzmj+cCb274T4EawBZWTf6NLoNYi8YDGEQBppdZm5Yze2utfNMG9PfLaWUc+PoCCammZs6wRRawDwsTKbE/gZ2EQmG1TWj2FkHDFapbLRzsEwJAP3RPV5KmstICKqz/lO24tBGciHTsbeXCJuLmZ3t9dFgO3DIIL04oD+1Xnx4OUZ6elRaQkFgfuZStNrbyPvb5DdDsnhDihdShDnYOUSgE4Skt1VkBqS3je/tUsJtAnQry/Fa65fgvCaqPwAocg6LR0Z+o2djXLq1BUCre5OoB3tBZtBCJa/ABOWIPEXRQZ3FhisnoD/rfMXtEaiNPk3uHYAAAAASUVORK5CYII=">통계는 과연 거짓말을 할까? - 매일경제</a>
    <dt><a add_date="1671427367" href="https://www.joongang.co.kr/article/25126692">[사진] 서초동 사저 찾아 작별 인사 | 중앙일보</a>
    <dt><a add_date="1671427566" href="https://www.joongang.co.kr/article/25126664" icon="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAA4klEQVQ4jWP8//8/AymABV3gz2+Gv38YGBkZGBgYIEaxsUO5WDX8Pb7777r5/398Y2BgYGBkZFLVYUmpYOQVQKj4jwa+fv6za813B6lvusw/qxL+Pb77/99fZHkmdCdx8TDpmTPw8jMwMTGp6jDKKDEwoqjB0EAIDFcNf/8y/P3LwMjIwMFJlIZ/ty79f/eKUVCUSccUi3HIkfLv3es/u9Z8DzH8bifxe8mk/3//oEfr//+MyInv7+HtfzcuYJSQY3byZ9K3YGDGSGkMDCgaGP7+Zfj3l4GVlYGBEVMpNg1EAABdi4Sb+zSY9AAAAABJRU5ErkJggg==">조류인플루엔자 확산…달걀값 또다시 들썩 | 중앙일보</a>
    <dt><a add_date="1671427600" href="https://www.joongang.co.kr/article/25126823">전문대교육협의회, ‘2022년 전문대학인상’ 시상 | 중앙일보</a>
    <dt><a add_date="1671427603" href="https://www.joongang.co.kr/article/25126793">'공무원' 인기는 뚝…중·고교생 희망직업으로 뜬 이 계열 | 중앙일보</a>
    <dt><a add_date="1671427617" href="https://www.joongang.co.kr/article/25126790">"다시 만나자" 안받아준 前연인…준비한 흉기로 살해한 50대男 | 중앙일보</a>
    <dt><a add_date="1671427620" href="https://www.joongang.co.kr/article/25126769">말다툼하다 아내 살해한 남편, 범행 후 그 옆에서 술 마셨다 | 중앙일보</a>
    <dt><a add_date="1671427632" href="https://www.joongang.co.kr/article/25126787">용산역에만 17분 멈췄다…전장연 출근길 기습에 "좀 그만하라" | 중앙일보</a>
    <dt><a add_date="1671427651" href="https://www.joongang.co.kr/article/25126542">"돈 뺏길지 모른다""집주인 연락 안돼" 매일 고성 터지는 이곳 | 중앙일보</a>
    <dt><a add_date="1671427700" href="https://www.joongang.co.kr/article/25126693">“혐오·거짓에 돈·권력 주는 SNS, 민주주의 위기 낳았다” | 중앙일보</a>
    <dt><a add_date="1671427799" href="https://www.joongang.co.kr/article/25126500">난임, 이것도 원인이었어? 여성 70%가 걸리는 '은밀한 질환' [건강한 가족] | 중앙일보</a>
    <dt><a add_date="1671427829" href="https://www.joongang.co.kr/article/25126648">[사랑방] 보훈처, 가수 이미자에 감사패 수여 | 중앙일보</a>
    <dt><a add_date="1671427837" href="https://www.joongang.co.kr/article/25125827">[사랑방] SK그룹, 이웃사랑성금 120억 | 중앙일보</a>
    <dt><a add_date="1671427840" href="https://www.joongang.co.kr/article/25125508">[사랑방] 권모세 회장 등 4명 ‘서울대AMP대상’ | 중앙일보</a>
    <dt><a add_date="1671427843" href="https://www.joongang.co.kr/article/25125189">[인사] 에스원 外 | 중앙일보</a>
    <dt><a add_date="1671427849" href="https://www.joongang.co.kr/article/25124850">[사진] 서울국제포럼-나카소네 세계평화연 ‘한·일 신시대를 향해’ 포럼 | 중앙일보</a>
    <dt><a add_date="1671427852" href="https://www.joongang.co.kr/article/25124840">[사랑방] 이덕로 한국행정학회장 취임 | 중앙일보</a>
    </dl><p>
    <DT><H3 add_date="1629940719" last_modified="1671427858">생활문화</H3>
    <dl><p>
    <dt><a add_date="1671427570" href="https://www.joongang.co.kr/article/25126621">[팩플] “더 오래, 더 젊게”…달라지는 카톡, 카카오의 노림수는 | 중앙일보</a>
    <dt><a add_date="1671427576" href="https://www.joongang.co.kr/article/25126570" icon="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAA4klEQVQ4jWP8//8/AymABV3gz2+Gv38YGBkZGBgYIEaxsUO5WDX8Pb7777r5/398Y2BgYGBkZFLVYUmpYOQVQKj4jwa+fv6za813B6lvusw/qxL+Pb77/99fZHkmdCdx8TDpmTPw8jMwMTGp6jDKKDEwoqjB0EAIDFcNf/8y/P3LwMjIwMFJlIZ/ty79f/eKUVCUSccUi3HIkfLv3es/u9Z8DzH8bifxe8mk/3//oEfr//+MyInv7+HtfzcuYJSQY3byZ9K3YGDGSGkMDCgaGP7+Zfj3l4GVlYGBEVMpNg1EAABdi4Sb+zSY9AAAAABJRU5ErkJggg==">“신곡 설령 좋아도 관심없다”…3000억 저작권 굴리는 29살 | 중앙일보</a>
    <dt><a add_date="1671427581" href="https://www.joongang.co.kr/article/25126732">[VIEW] 퍼펙트스톰 긴박감 사라졌지만, 수출ㆍ고용ㆍ부동산 위기 계속 | 중앙일보</a>
    <dt><a add_date="1671427591" href="https://www.joongang.co.kr/article/25126615">밤 7시 '구찌 매장' 불꺼졌다…쇼핑몰은 거대한 미술관이 됐다 | 중앙일보</a>
    <dt><a add_date="1671427606" href="https://www.joongang.co.kr/article/25126784">제주 항공기·여객선 운항 정상화…"서해안엔 계속 많은 눈 예상" | 중앙일보</a>
    <dt><a add_date="1671427613" href="https://www.joongang.co.kr/article/25126745">[소년중앙] 어디 가야 먹을 수 있나요, 찬바람 불면 돌아오는 제철 붕어빵 | 중앙일보</a>
    <dt><a add_date="1671427628" href="https://www.joongang.co.kr/article/25126656">[오늘의 운세] 12월 19일 | 중앙일보</a>
    <dt><a add_date="1671427640" href="https://www.joongang.co.kr/article/25126666">[우리말 바루기] ‘결제’를 할까, ‘결재’를 할까? | 중앙일보</a>
    <dt><a add_date="1671427644" href="https://www.joongang.co.kr/article/25126629">여객기 109편 결항…19일까지 많은 눈, 아침 -15도 강추위 | 중앙일보</a>
    <dt><a add_date="1671427693" href="https://www.joongang.co.kr/article/25126320" icon="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAA4klEQVQ4jWP8//8/AymABV3gz2+Gv38YGBkZGBgYIEaxsUO5WDX8Pb7777r5/398Y2BgYGBkZFLVYUmpYOQVQKj4jwa+fv6za813B6lvusw/qxL+Pb77/99fZHkmdCdx8TDpmTPw8jMwMTGp6jDKKDEwoqjB0EAIDFcNf/8y/P3LwMjIwMFJlIZ/ty79f/eKUVCUSccUi3HIkfLv3es/u9Z8DzH8bifxe8mk/3//oEfr//+MyInv7+HtfzcuYJSQY3byZ9K3YGDGSGkMDCgaGP7+Zfj3l4GVlYGBEVMpNg1EAABdi4Sb+zSY9AAAAABJRU5ErkJggg==">“워뇽이 보면 기분 좋잖아”…‘여덕몰이상’은 따로 있다 | 중앙일보</a>
    <dt><a add_date="1671427697" href="https://www.joongang.co.kr/article/25126659">저렇게 어리숙해서야, 쯧쯧…근데 왜 부럽지? | 중앙일보</a>
    <dt><a add_date="1671427703" href="https://www.joongang.co.kr/article/25126572">“유리 위 걷듯 지나온 길…정형의 그릇에 무한한 이야기 담겠다” | 중앙일보</a>
    <dt><a add_date="1671427706" href="https://www.joongang.co.kr/article/25126367">우주선 닮은 동대문 DDP에서 열리는 연말 ‘빛의 예술’ 미디어 아트쇼… 무료관람 | 중앙일보</a>
    <dt><a add_date="1671427709" href="https://www.joongang.co.kr/article/25126323" icon="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAA4klEQVQ4jWP8//8/AymABV3gz2+Gv38YGBkZGBgYIEaxsUO5WDX8Pb7777r5/398Y2BgYGBkZFLVYUmpYOQVQKj4jwa+fv6za813B6lvusw/qxL+Pb77/99fZHkmdCdx8TDpmTPw8jMwMTGp6jDKKDEwoqjB0EAIDFcNf/8y/P3LwMjIwMFJlIZ/ty79f/eKUVCUSccUi3HIkfLv3es/u9Z8DzH8bifxe8mk/3//oEfr//+MyInv7+HtfzcuYJSQY3byZ9K3YGDGSGkMDCgaGP7+Zfj3l4GVlYGBEVMpNg1EAABdi4Sb+zSY9AAAAABJRU5ErkJggg==">‘예약률 100%’ 색다른 미식…창밖으로 나온 컬리의 초대 | 중앙일보</a>
    <dt><a add_date="1671427716" href="https://www.joongang.co.kr/article/25126310">좌우명은 '더 멀리', 이 집안을 알면 유럽사·세계사가 보인다[BOOK 연말연시 읽을만한 책] | 중앙일보</a>
    <dt><a add_date="1671427749" href="https://www.joongang.co.kr/article/25126086">BTS RM, 알고보니 기름 수저?…방송중 "SK에너지!" 외친 사연 | 중앙일보</a>
    <dt><a add_date="1671427753" href="https://www.joongang.co.kr/article/25126012" icon="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAA4klEQVQ4jWP8//8/AymABV3gz2+Gv38YGBkZGBgYIEaxsUO5WDX8Pb7777r5/398Y2BgYGBkZFLVYUmpYOQVQKj4jwa+fv6za813B6lvusw/qxL+Pb77/99fZHkmdCdx8TDpmTPw8jMwMTGp6jDKKDEwoqjB0EAIDFcNf/8y/P3LwMjIwMFJlIZ/ty79f/eKUVCUSccUi3HIkfLv3es/u9Z8DzH8bifxe8mk/3//oEfr//+MyInv7+HtfzcuYJSQY3byZ9K3YGDGSGkMDCgaGP7+Zfj3l4GVlYGBEVMpNg1EAABdi4Sb+zSY9AAAAABJRU5ErkJggg==">31만1200 vs 7만5000…‘걸그룹 천하’ 증명한 숫자 | 중앙일보</a>
    <dt><a add_date="1671427802" href="https://www.joongang.co.kr/article/25126300">삼겹살·목살이면 충분해, 연말 홈파티 모두가 좋아할 요리 추천 [쿠킹] | 중앙일보</a>
    <dt><a add_date="1671427805" href="https://www.joongang.co.kr/article/25125708">"약혼반지 대신 이 팔찌를" 60년대에 젠더리스 컨셉, 뉴요커 홀린 까르띠에 [더 하이엔드] | 중앙일보</a>
    <dt><a add_date="1671427812" href="https://www.joongang.co.kr/article/25124162" icon="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAA4klEQVQ4jWP8//8/AymABV3gz2+Gv38YGBkZGBgYIEaxsUO5WDX8Pb7777r5/398Y2BgYGBkZFLVYUmpYOQVQKj4jwa+fv6za813B6lvusw/qxL+Pb77/99fZHkmdCdx8TDpmTPw8jMwMTGp6jDKKDEwoqjB0EAIDFcNf/8y/P3LwMjIwMFJlIZ/ty79f/eKUVCUSccUi3HIkfLv3es/u9Z8DzH8bifxe8mk/3//oEfr//+MyInv7+HtfzcuYJSQY3byZ9K3YGDGSGkMDCgaGP7+Zfj3l4GVlYGBEVMpNg1EAABdi4Sb+zSY9AAAAABJRU5ErkJggg==">“이 사회 전체가 의자였구나” 샤워실 목욕의자의 재발견 | 중앙일보</a>
    <dt><a add_date="1671427816" href="https://www.joongang.co.kr/article/25122912">카페인 있어도 고혈압과 무관? 커피가 '두얼굴 헐크'인 까닭 [건강한 가족] | 중앙일보</a>
    <dt><a add_date="1671427819" href="https://www.joongang.co.kr/article/25122804">올가을 가뭄에 당도↑, 최근 비에 산도↓…역대급 당산비 감귤 [e즐펀한 토크] | 중앙일보</a>
    <dt><a add_date="1671427825" href="https://www.joongang.co.kr/article/25121510">예거 르쿨트르와 레터링 아티스트 알렉스 트로슈가 만드는 새로운 의미 [더 하이엔드] | 중앙일보</a>
    <dt><a add_date="1671427858" href="https://www.joongang.co.kr/article/25123714" icon="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAA4klEQVQ4jWP8//8/AymABV3gz2+Gv38YGBkZGBgYIEaxsUO5WDX8Pb7777r5/398Y2BgYGBkZFLVYUmpYOQVQKj4jwa+fv6za813B6lvusw/qxL+Pb77/99fZHkmdCdx8TDpmTPw8jMwMTGp6jDKKDEwoqjB0EAIDFcNf/8y/P3LwMjIwMFJlIZ/ty79f/eKUVCUSccUi3HIkfLv3es/u9Z8DzH8bifxe8mk/3//oEfr//+MyInv7+HtfzcuYJSQY3byZ9K3YGDGSGkMDCgaGP7+Zfj3l4GVlYGBEVMpNg1EAABdi4Sb+zSY9AAAAABJRU5ErkJggg==">[사진] 중앙광고대상 시상식 | 중앙일보</a>
    </dl><p>
    <DT><H3 add_date="1629940719" last_modified="1671427858">세계</H3>
    <dl><p>
    <dt><a add_date="1671427658" href="https://www.joongang.co.kr/article/25126818">미얀마 최대도시 양곤서 또 폭발 사고…페리 승객 11명 부상 | 중앙일보</a>
    <dt><a add_date="1671427665" href="https://www.joongang.co.kr/article/25126649">히잡시위 지지한 ‘이란 국민배우’ 체포 | 중앙일보</a>
    <dt><a add_date="1671427670" href="https://www.joongang.co.kr/article/25126545">"우크라, 미국 만류에도 러시아軍 최고 지휘관 암살 시도" | 중앙일보</a>
    <dt><a add_date="1671427673" href="https://www.joongang.co.kr/article/25126510">'건강이상설' 푸틴 군 사령관들 소집…"우크라 작전 제안하라" | 중앙일보</a>
    <dt><a add_date="1671427676" href="https://www.joongang.co.kr/article/25126489">코로나 지원금 100억 빼돌린 美목사…호화주택 사려다 덜미 | 중앙일보</a>
    <dt><a add_date="1671427680" href="https://www.joongang.co.kr/article/25126482">한 달 전 인니 지진 사망자, 300명대 아닌 602명…재집계 수치 발표 | 중앙일보</a>
    <dt><a add_date="1671427684" href="https://www.joongang.co.kr/article/25126355">美, 中 36개 기업 수출통제 추가…AI 등 첨단산업 육성 견제 | 중앙일보</a>
    <dt><a add_date="1671427689" href="https://www.joongang.co.kr/article/25126270">백지시위 물어뜯은 中대사 "외부사주 받은 색깔혁명 냄새난다" | 중앙일보</a>
    </dl><p>
    <DT><H3 add_date="1629940719" last_modified="1671427858">스포츠</H3>
    <dl><p>
    <dt><a add_date="1671427337" href="https://www.joongang.co.kr/article/25126744" icon="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAA4klEQVQ4jWP8//8/AymABV3gz2+Gv38YGBkZGBgYIEaxsUO5WDX8Pb7777r5/398Y2BgYGBkZFLVYUmpYOQVQKj4jwa+fv6za813B6lvusw/qxL+Pb77/99fZHkmdCdx8TDpmTPw8jMwMTGp6jDKKDEwoqjB0EAIDFcNf/8y/P3LwMjIwMFJlIZ/ty79f/eKUVCUSccUi3HIkfLv3es/u9Z8DzH8bifxe8mk/3//oEfr//+MyInv7+HtfzcuYJSQY3byZ9K3YGDGSGkMDCgaGP7+Zfj3l4GVlYGBEVMpNg1EAABdi4Sb+zSY9AAAAABJRU5ErkJggg==">故마라도나에 우승컵 바쳤다…‘메신’의 짜릿한 ‘카타르’시스 | 중앙일보</a>
    <dt><a add_date="1671427389" href="https://www.joongang.co.kr/article/25126571">尹 '조용한 생일'…"직언 들어줘 감사하다" 참모들 축하 메시지 | 중앙일보</a>
    <dt><a add_date="1671427550" href="https://www.joongang.co.kr/article/25126770">기아, 카타르 월드컵에 차량 297대 투입…브랜드 홍보 | 중앙일보</a>
    <dt><a add_date="1671427609" href="https://www.joongang.co.kr/article/25126756">[THINK ENGLISH] 손흥민, 마스크 벗고 런던에서 훈련 재개 | 중앙일보</a>
    <dt><a add_date="1671427662" href="https://www.joongang.co.kr/article/25126753">"메시 우승 자격 있다"…'축구의 신'에 축하 건넨 '축구 황제' | 중앙일보</a>
    <dt><a add_date="1671427720" href="https://www.joongang.co.kr/article/25126254" icon="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAA4klEQVQ4jWP8//8/AymABV3gz2+Gv38YGBkZGBgYIEaxsUO5WDX8Pb7777r5/398Y2BgYGBkZFLVYUmpYOQVQKj4jwa+fv6za813B6lvusw/qxL+Pb77/99fZHkmdCdx8TDpmTPw8jMwMTGp6jDKKDEwoqjB0EAIDFcNf/8y/P3LwMjIwMFJlIZ/ty79f/eKUVCUSccUi3HIkfLv3es/u9Z8DzH8bifxe8mk/3//oEfr//+MyInv7+HtfzcuYJSQY3byZ9K3YGDGSGkMDCgaGP7+Zfj3l4GVlYGBEVMpNg1EAABdi4Sb+zSY9AAAAABJRU5ErkJggg==">후크 "이승기에 54억 전액 지급…합의 못했지만 분쟁 종결" | 중앙일보</a>
    <dt><a add_date="1671427764" href="https://www.joongang.co.kr/article/25126726">메시, 골든볼 MVP에도 뽑혔다…월드컵 역대 첫 2회 수상 | 중앙일보</a>
    <dt><a add_date="1671427771" href="https://www.joongang.co.kr/article/25126624">황선우, 1분40초 벽 깼다…쇼트코스 자유형 200m '2연패' | 중앙일보</a>
    <dt><a add_date="1671427775" href="https://www.joongang.co.kr/article/25126483">메르스? 코로나? 결승 이틀 앞둔 프랑스 발칵, 5명이 아프다 | 중앙일보</a>
    <dt><a add_date="1671427779" href="https://www.joongang.co.kr/article/25126246">벤투 감독, 폴란드 사령탑 거론…레반도프스키 지휘하나 | 중앙일보</a>
    <dt><a add_date="1671427783" href="https://www.joongang.co.kr/article/25126081">벤투 사단 '비트박스 코치', 한국 이웃에 남긴 마지막 선물 | 중앙일보</a>
    <dt><a add_date="1671427787" href="https://www.joongang.co.kr/article/25125831">역시, 메시 | 중앙일보</a>
    <dt><a add_date="1671427791" href="https://www.joongang.co.kr/article/25125786">여자배구 조송화 계약해지 무효소송 패소… 잔여연봉 4억원 수령 어려워져 | 중앙일보</a>
    <dt><a add_date="1671427795" href="https://www.joongang.co.kr/article/25126376">황선우 앞세운 쇼트코스 계영 800ｍ 韓신기록...역대최고 4위 | 중앙일보</a>
    <dt><a add_date="1671427808" href="https://www.joongang.co.kr/article/25124732">"답 안하는 걸로 할게요"…조규성 3초 고민하게 만든 질문 | 중앙일보</a>
    </dl><p>
    <DT><H3 add_date="1629940719" last_modified="1671427858">정치</H3>
    <dl><p>
    <dt><a add_date="1671427356" href="https://www.joongang.co.kr/article/25126791">文정부 ‘통계조작’ 공방 격화…與 “통계조작성장” 野 “모욕주기” | 중앙일보</a>
    <dt><a add_date="1671427361" href="https://www.joongang.co.kr/article/25126774">정청래, 박지원 복당 공개반대 "한 번 배신하면 또 배신" | 중앙일보</a>
    <dt><a add_date="1671427364" href="https://www.joongang.co.kr/article/25126755">尹지지율, 중도·20대가 쌍끌이로 올렸다…6월 이후 첫 40%대 [리얼미터] | 중앙일보</a>
    <dt><a add_date="1671427371" href="https://www.joongang.co.kr/article/25126618">법인세보다 장관 '빅2'가 핵심...野 반대한 한동훈·이상민 예산은 | 중앙일보</a>
    <dt><a add_date="1671427381" href="https://www.joongang.co.kr/article/25126580" icon="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAA4klEQVQ4jWP8//8/AymABV3gz2+Gv38YGBkZGBgYIEaxsUO5WDX8Pb7777r5/398Y2BgYGBkZFLVYUmpYOQVQKj4jwa+fv6za813B6lvusw/qxL+Pb77/99fZHkmdCdx8TDpmTPw8jMwMTGp6jDKKDEwoqjB0EAIDFcNf/8y/P3LwMjIwMFJlIZ/ty79f/eKUVCUSccUi3HIkfLv3es/u9Z8DzH8bifxe8mk/3//oEfr//+MyInv7+HtfzcuYJSQY3byZ9K3YGDGSGkMDCgaGP7+Zfj3l4GVlYGBEVMpNg1EAABdi4Sb+zSY9AAAAABJRU5ErkJggg==">‘당심 100% 룰’ 김용태 “윤핵관 당헌·당규 손바닥 뒤집듯 바꿔…한심” | 중앙일보</a>
    <dt><a add_date="1671427395" href="https://www.joongang.co.kr/article/25126550">[속보] 북, 탄도미사일 2발 발사…“일본 EEZ 밖 낙하 추정” | 중앙일보</a>
    <dt><a add_date="1671427399" href="https://www.joongang.co.kr/article/25126364" icon="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAA4klEQVQ4jWP8//8/AymABV3gz2+Gv38YGBkZGBgYIEaxsUO5WDX8Pb7777r5/398Y2BgYGBkZFLVYUmpYOQVQKj4jwa+fv6za813B6lvusw/qxL+Pb77/99fZHkmdCdx8TDpmTPw8jMwMTGp6jDKKDEwoqjB0EAIDFcNf/8y/P3LwMjIwMFJlIZ/ty79f/eKUVCUSccUi3HIkfLv3es/u9Z8DzH8bifxe8mk/3//oEfr//+MyInv7+HtfzcuYJSQY3byZ9K3YGDGSGkMDCgaGP7+Zfj3l4GVlYGBEVMpNg1EAABdi4Sb+zSY9AAAAABJRU5ErkJggg==">이정현 "당대표 나가든 자문을 하든 소극적으로 있지 않겠다" | 중앙일보</a>
    <dt><a add_date="1671427511" href="https://www.joongang.co.kr/article/25126336" icon="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAA4klEQVQ4jWP8//8/AymABV3gz2+Gv38YGBkZGBgYIEaxsUO5WDX8Pb7777r5/398Y2BgYGBkZFLVYUmpYOQVQKj4jwa+fv6za813B6lvusw/qxL+Pb77/99fZHkmdCdx8TDpmTPw8jMwMTGp6jDKKDEwoqjB0EAIDFcNf/8y/P3LwMjIwMFJlIZ/ty79f/eKUVCUSccUi3HIkfLv3es/u9Z8DzH8bifxe8mk/3//oEfr//+MyInv7+HtfzcuYJSQY3byZ9K3YGDGSGkMDCgaGP7+Zfj3l4GVlYGBEVMpNg1EAABdi4Sb+zSY9AAAAABJRU5ErkJggg==">유승민 여론조사 보니…친윤 "이게 당원투표 100% 이유" | 중앙일보</a>
    </dl><p>
    



```python
import os
print(os.path.isfile("html_test_add.html"))

file = open('./open/classified_bookmarks.html','w',encoding='UTF-8')

file.write(add_result)
file.close()
```

    False





    17202




```python
f=codecs.open("./open/classified_bookmarks.html", 'r', encoding = 'UTF-8')
soup = BeautifulSoup(f.read(), 'html.parser')
print(soup.prettify())
```

    <!DOCTYPE NETSCAPE-Bookmark-file-1>
    <!-- This is an automatically generated file.
         It will be read and overwritten.
         DO NOT EDIT! -->
    <meta content="text/html; charset=utf-8" http-equiv="Content-Type"/>
    <title>
     Bookmarks
    </title>
    <h1>
     Bookmarks
    </h1>
    <dt>
     <h3 add_date="1629940719" last_modified="1671427858">
      IT과학
     </h3>
     <dl>
      <p>
       <dt>
        <a add_date="1671427376" href="https://www.joongang.co.kr/article/25126593">
         '게임체인저' 신입당원은 20만...국힘, 당원 포섭전쟁은 끝났다 | 중앙일보
        </a>
        <dt>
         <a add_date="1671427543" href="https://www.joongang.co.kr/article/25126811">
          LG엔솔, 오창공장 배터리 라인 신·증설에 4조 투자…1800명 채용 | 중앙일보
         </a>
         <dt>
          <a add_date="1671427553" href="https://www.joongang.co.kr/article/25126728">
           [알림] '펫(pet) 톡톡': 평생 간직하세요, 인생 사진 2탄 | 중앙일보
          </a>
          <dt>
           <a add_date="1671427562" href="https://www.joongang.co.kr/article/25126670">
            SK그룹, CES 2023 무대로 넷제로 역량 선보인다 | 중앙일보
           </a>
           <dt>
            <a add_date="1671427573" href="https://www.joongang.co.kr/article/25126583" icon="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAA4klEQVQ4jWP8//8/AymABV3gz2+Gv38YGBkZGBgYIEaxsUO5WDX8Pb7777r5/398Y2BgYGBkZFLVYUmpYOQVQKj4jwa+fv6za813B6lvusw/qxL+Pb77/99fZHkmdCdx8TDpmTPw8jMwMTGp6jDKKDEwoqjB0EAIDFcNf/8y/P3LwMjIwMFJlIZ/ty79f/eKUVCUSccUi3HIkfLv3es/u9Z8DzH8bifxe8mk/3//oEfr//+MyInv7+HtfzcuYJSQY3byZ9K3YGDGSGkMDCgaGP7+Zfj3l4GVlYGBEVMpNg1EAABdi4Sb+zSY9AAAAABJRU5ErkJggg==">
             1년 4번 그들은 점을 찍는다…美 Fed, 과학인가 예언인가 | 중앙일보
            </a>
            <dt>
             <a add_date="1671427597" href="https://www.joongang.co.kr/article/25126830">
              한국공대, 지역산업 연계 오픈랩 혁신기술 설명회 개최 | 중앙일보
             </a>
             <dt>
              <a add_date="1671427711" href="https://www.joongang.co.kr/article/25126314">
               근육이 말을 듣지 않는다, 로봇공학자의 대담한 도전은[BOOK 연말연시 읽을만한 책] | 중앙일보
              </a>
              <dt>
               <a add_date="1671427768" href="https://www.joongang.co.kr/article/25126667">
                [삼성화재배 AI와 함께하는 바둑 해설] 조용히 완성된 철갑 공격군 | 중앙일보
               </a>
              </dt>
             </dt>
            </dt>
           </dt>
          </dt>
         </dt>
        </dt>
       </dt>
      </p>
     </dl>
     <p>
      <dt>
       <h3 add_date="1629940719" last_modified="1671427858">
        경제
       </h3>
       <dl>
        <p>
         <dt>
          <a add_date="1671427539" href="https://www.joongang.co.kr/article/25126817">
           정부가 눌러 5% 안넘은 근원물가…'공공요금 인상' 내년엔 | 중앙일보
          </a>
          <dt>
           <a add_date="1671427547" href="https://www.joongang.co.kr/article/25126800">
            퇴직연금 중도인출 “집 사려고, 전세 보증금 때문에” 81.6% | 중앙일보
           </a>
           <dt>
            <a add_date="1671427556" href="https://www.joongang.co.kr/article/25126677">
             금리↑ 집값↓…“2% 청약통장 깨서 7% 대출이자 갚는다” | 중앙일보
            </a>
            <dt>
             <a add_date="1671427559" href="https://www.joongang.co.kr/article/25126674">
              [사진] 청약경쟁률 8년 만에 한 자릿수로 | 중앙일보
             </a>
            </dt>
           </dt>
          </dt>
         </dt>
        </p>
       </dl>
       <p>
        <dt>
         <h3 add_date="1629940719" last_modified="1671427858">
          사회
         </h3>
         <dl>
          <p>
           <dt>
            <a add_date="1584968656" href="https://www.mk.co.kr/opinion/contributors/view/2016/08/618807/" icon="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAABPUlEQVQ4jaWTsUoDQRCGvz0OJWLEgEXSCCo2wVcQX8PGQsHCSgt9DBFJoYKp7QURH8FGJNgaFQsLEUXBC0d21+J273bXSxTdZmZ25v93ZnZGALxvzmj+cCb274T4EawBZWTf6NLoNYi8YDGEQBppdZm5Yze2utfNMG9PfLaWUc+PoCCammZs6wRRawDwsTKbE/gZ2EQmG1TWj2FkHDFapbLRzsEwJAP3RPV5KmstICKqz/lO24tBGciHTsbeXCJuLmZ3t9dFgO3DIIL04oD+1Xnx4OUZ6elRaQkFgfuZStNrbyPvb5DdDsnhDihdShDnYOUSgE4Skt1VkBqS3je/tUsJtAnQry/Fa65fgvCaqPwAocg6LR0Z+o2djXLq1BUCre5OoB3tBZtBCJa/ABOWIPEXRQZ3FhisnoD/rfMXtEaiNPk3uHYAAAAASUVORK5CYII=">
             통계는 과연 거짓말을 할까? - 매일경제
            </a>
            <dt>
             <a add_date="1671427367" href="https://www.joongang.co.kr/article/25126692">
              [사진] 서초동 사저 찾아 작별 인사 | 중앙일보
             </a>
             <dt>
              <a add_date="1671427566" href="https://www.joongang.co.kr/article/25126664" icon="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAA4klEQVQ4jWP8//8/AymABV3gz2+Gv38YGBkZGBgYIEaxsUO5WDX8Pb7777r5/398Y2BgYGBkZFLVYUmpYOQVQKj4jwa+fv6za813B6lvusw/qxL+Pb77/99fZHkmdCdx8TDpmTPw8jMwMTGp6jDKKDEwoqjB0EAIDFcNf/8y/P3LwMjIwMFJlIZ/ty79f/eKUVCUSccUi3HIkfLv3es/u9Z8DzH8bifxe8mk/3//oEfr//+MyInv7+HtfzcuYJSQY3byZ9K3YGDGSGkMDCgaGP7+Zfj3l4GVlYGBEVMpNg1EAABdi4Sb+zSY9AAAAABJRU5ErkJggg==">
               조류인플루엔자 확산…달걀값 또다시 들썩 | 중앙일보
              </a>
              <dt>
               <a add_date="1671427600" href="https://www.joongang.co.kr/article/25126823">
                전문대교육협의회, ‘2022년 전문대학인상’ 시상 | 중앙일보
               </a>
               <dt>
                <a add_date="1671427603" href="https://www.joongang.co.kr/article/25126793">
                 '공무원' 인기는 뚝…중·고교생 희망직업으로 뜬 이 계열 | 중앙일보
                </a>
                <dt>
                 <a add_date="1671427617" href="https://www.joongang.co.kr/article/25126790">
                  "다시 만나자" 안받아준 前연인…준비한 흉기로 살해한 50대男 | 중앙일보
                 </a>
                 <dt>
                  <a add_date="1671427620" href="https://www.joongang.co.kr/article/25126769">
                   말다툼하다 아내 살해한 남편, 범행 후 그 옆에서 술 마셨다 | 중앙일보
                  </a>
                  <dt>
                   <a add_date="1671427632" href="https://www.joongang.co.kr/article/25126787">
                    용산역에만 17분 멈췄다…전장연 출근길 기습에 "좀 그만하라" | 중앙일보
                   </a>
                   <dt>
                    <a add_date="1671427651" href="https://www.joongang.co.kr/article/25126542">
                     "돈 뺏길지 모른다""집주인 연락 안돼" 매일 고성 터지는 이곳 | 중앙일보
                    </a>
                    <dt>
                     <a add_date="1671427700" href="https://www.joongang.co.kr/article/25126693">
                      “혐오·거짓에 돈·권력 주는 SNS, 민주주의 위기 낳았다” | 중앙일보
                     </a>
                     <dt>
                      <a add_date="1671427799" href="https://www.joongang.co.kr/article/25126500">
                       난임, 이것도 원인이었어? 여성 70%가 걸리는 '은밀한 질환' [건강한 가족] | 중앙일보
                      </a>
                      <dt>
                       <a add_date="1671427829" href="https://www.joongang.co.kr/article/25126648">
                        [사랑방] 보훈처, 가수 이미자에 감사패 수여 | 중앙일보
                       </a>
                       <dt>
                        <a add_date="1671427837" href="https://www.joongang.co.kr/article/25125827">
                         [사랑방] SK그룹, 이웃사랑성금 120억 | 중앙일보
                        </a>
                        <dt>
                         <a add_date="1671427840" href="https://www.joongang.co.kr/article/25125508">
                          [사랑방] 권모세 회장 등 4명 ‘서울대AMP대상’ | 중앙일보
                         </a>
                         <dt>
                          <a add_date="1671427843" href="https://www.joongang.co.kr/article/25125189">
                           [인사] 에스원 外 | 중앙일보
                          </a>
                          <dt>
                           <a add_date="1671427849" href="https://www.joongang.co.kr/article/25124850">
                            [사진] 서울국제포럼-나카소네 세계평화연 ‘한·일 신시대를 향해’ 포럼 | 중앙일보
                           </a>
                           <dt>
                            <a add_date="1671427852" href="https://www.joongang.co.kr/article/25124840">
                             [사랑방] 이덕로 한국행정학회장 취임 | 중앙일보
                            </a>
                           </dt>
                          </dt>
                         </dt>
                        </dt>
                       </dt>
                      </dt>
                     </dt>
                    </dt>
                   </dt>
                  </dt>
                 </dt>
                </dt>
               </dt>
              </dt>
             </dt>
            </dt>
           </dt>
          </p>
         </dl>
         <p>
          <dt>
           <h3 add_date="1629940719" last_modified="1671427858">
            생활문화
           </h3>
           <dl>
            <p>
             <dt>
              <a add_date="1671427570" href="https://www.joongang.co.kr/article/25126621">
               [팩플] “더 오래, 더 젊게”…달라지는 카톡, 카카오의 노림수는 | 중앙일보
              </a>
              <dt>
               <a add_date="1671427576" href="https://www.joongang.co.kr/article/25126570" icon="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAA4klEQVQ4jWP8//8/AymABV3gz2+Gv38YGBkZGBgYIEaxsUO5WDX8Pb7777r5/398Y2BgYGBkZFLVYUmpYOQVQKj4jwa+fv6za813B6lvusw/qxL+Pb77/99fZHkmdCdx8TDpmTPw8jMwMTGp6jDKKDEwoqjB0EAIDFcNf/8y/P3LwMjIwMFJlIZ/ty79f/eKUVCUSccUi3HIkfLv3es/u9Z8DzH8bifxe8mk/3//oEfr//+MyInv7+HtfzcuYJSQY3byZ9K3YGDGSGkMDCgaGP7+Zfj3l4GVlYGBEVMpNg1EAABdi4Sb+zSY9AAAAABJRU5ErkJggg==">
                “신곡 설령 좋아도 관심없다”…3000억 저작권 굴리는 29살 | 중앙일보
               </a>
               <dt>
                <a add_date="1671427581" href="https://www.joongang.co.kr/article/25126732">
                 [VIEW] 퍼펙트스톰 긴박감 사라졌지만, 수출ㆍ고용ㆍ부동산 위기 계속 | 중앙일보
                </a>
                <dt>
                 <a add_date="1671427591" href="https://www.joongang.co.kr/article/25126615">
                  밤 7시 '구찌 매장' 불꺼졌다…쇼핑몰은 거대한 미술관이 됐다 | 중앙일보
                 </a>
                 <dt>
                  <a add_date="1671427606" href="https://www.joongang.co.kr/article/25126784">
                   제주 항공기·여객선 운항 정상화…"서해안엔 계속 많은 눈 예상" | 중앙일보
                  </a>
                  <dt>
                   <a add_date="1671427613" href="https://www.joongang.co.kr/article/25126745">
                    [소년중앙] 어디 가야 먹을 수 있나요, 찬바람 불면 돌아오는 제철 붕어빵 | 중앙일보
                   </a>
                   <dt>
                    <a add_date="1671427628" href="https://www.joongang.co.kr/article/25126656">
                     [오늘의 운세] 12월 19일 | 중앙일보
                    </a>
                    <dt>
                     <a add_date="1671427640" href="https://www.joongang.co.kr/article/25126666">
                      [우리말 바루기] ‘결제’를 할까, ‘결재’를 할까? | 중앙일보
                     </a>
                     <dt>
                      <a add_date="1671427644" href="https://www.joongang.co.kr/article/25126629">
                       여객기 109편 결항…19일까지 많은 눈, 아침 -15도 강추위 | 중앙일보
                      </a>
                      <dt>
                       <a add_date="1671427693" href="https://www.joongang.co.kr/article/25126320" icon="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAA4klEQVQ4jWP8//8/AymABV3gz2+Gv38YGBkZGBgYIEaxsUO5WDX8Pb7777r5/398Y2BgYGBkZFLVYUmpYOQVQKj4jwa+fv6za813B6lvusw/qxL+Pb77/99fZHkmdCdx8TDpmTPw8jMwMTGp6jDKKDEwoqjB0EAIDFcNf/8y/P3LwMjIwMFJlIZ/ty79f/eKUVCUSccUi3HIkfLv3es/u9Z8DzH8bifxe8mk/3//oEfr//+MyInv7+HtfzcuYJSQY3byZ9K3YGDGSGkMDCgaGP7+Zfj3l4GVlYGBEVMpNg1EAABdi4Sb+zSY9AAAAABJRU5ErkJggg==">
                        “워뇽이 보면 기분 좋잖아”…‘여덕몰이상’은 따로 있다 | 중앙일보
                       </a>
                       <dt>
                        <a add_date="1671427697" href="https://www.joongang.co.kr/article/25126659">
                         저렇게 어리숙해서야, 쯧쯧…근데 왜 부럽지? | 중앙일보
                        </a>
                        <dt>
                         <a add_date="1671427703" href="https://www.joongang.co.kr/article/25126572">
                          “유리 위 걷듯 지나온 길…정형의 그릇에 무한한 이야기 담겠다” | 중앙일보
                         </a>
                         <dt>
                          <a add_date="1671427706" href="https://www.joongang.co.kr/article/25126367">
                           우주선 닮은 동대문 DDP에서 열리는 연말 ‘빛의 예술’ 미디어 아트쇼… 무료관람 | 중앙일보
                          </a>
                          <dt>
                           <a add_date="1671427709" href="https://www.joongang.co.kr/article/25126323" icon="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAA4klEQVQ4jWP8//8/AymABV3gz2+Gv38YGBkZGBgYIEaxsUO5WDX8Pb7777r5/398Y2BgYGBkZFLVYUmpYOQVQKj4jwa+fv6za813B6lvusw/qxL+Pb77/99fZHkmdCdx8TDpmTPw8jMwMTGp6jDKKDEwoqjB0EAIDFcNf/8y/P3LwMjIwMFJlIZ/ty79f/eKUVCUSccUi3HIkfLv3es/u9Z8DzH8bifxe8mk/3//oEfr//+MyInv7+HtfzcuYJSQY3byZ9K3YGDGSGkMDCgaGP7+Zfj3l4GVlYGBEVMpNg1EAABdi4Sb+zSY9AAAAABJRU5ErkJggg==">
                            ‘예약률 100%’ 색다른 미식…창밖으로 나온 컬리의 초대 | 중앙일보
                           </a>
                           <dt>
                            <a add_date="1671427716" href="https://www.joongang.co.kr/article/25126310">
                             좌우명은 '더 멀리', 이 집안을 알면 유럽사·세계사가 보인다[BOOK 연말연시 읽을만한 책] | 중앙일보
                            </a>
                            <dt>
                             <a add_date="1671427749" href="https://www.joongang.co.kr/article/25126086">
                              BTS RM, 알고보니 기름 수저?…방송중 "SK에너지!" 외친 사연 | 중앙일보
                             </a>
                             <dt>
                              <a add_date="1671427753" href="https://www.joongang.co.kr/article/25126012" icon="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAA4klEQVQ4jWP8//8/AymABV3gz2+Gv38YGBkZGBgYIEaxsUO5WDX8Pb7777r5/398Y2BgYGBkZFLVYUmpYOQVQKj4jwa+fv6za813B6lvusw/qxL+Pb77/99fZHkmdCdx8TDpmTPw8jMwMTGp6jDKKDEwoqjB0EAIDFcNf/8y/P3LwMjIwMFJlIZ/ty79f/eKUVCUSccUi3HIkfLv3es/u9Z8DzH8bifxe8mk/3//oEfr//+MyInv7+HtfzcuYJSQY3byZ9K3YGDGSGkMDCgaGP7+Zfj3l4GVlYGBEVMpNg1EAABdi4Sb+zSY9AAAAABJRU5ErkJggg==">
                               31만1200 vs 7만5000…‘걸그룹 천하’ 증명한 숫자 | 중앙일보
                              </a>
                              <dt>
                               <a add_date="1671427802" href="https://www.joongang.co.kr/article/25126300">
                                삼겹살·목살이면 충분해, 연말 홈파티 모두가 좋아할 요리 추천 [쿠킹] | 중앙일보
                               </a>
                               <dt>
                                <a add_date="1671427805" href="https://www.joongang.co.kr/article/25125708">
                                 "약혼반지 대신 이 팔찌를" 60년대에 젠더리스 컨셉, 뉴요커 홀린 까르띠에 [더 하이엔드] | 중앙일보
                                </a>
                                <dt>
                                 <a add_date="1671427812" href="https://www.joongang.co.kr/article/25124162" icon="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAA4klEQVQ4jWP8//8/AymABV3gz2+Gv38YGBkZGBgYIEaxsUO5WDX8Pb7777r5/398Y2BgYGBkZFLVYUmpYOQVQKj4jwa+fv6za813B6lvusw/qxL+Pb77/99fZHkmdCdx8TDpmTPw8jMwMTGp6jDKKDEwoqjB0EAIDFcNf/8y/P3LwMjIwMFJlIZ/ty79f/eKUVCUSccUi3HIkfLv3es/u9Z8DzH8bifxe8mk/3//oEfr//+MyInv7+HtfzcuYJSQY3byZ9K3YGDGSGkMDCgaGP7+Zfj3l4GVlYGBEVMpNg1EAABdi4Sb+zSY9AAAAABJRU5ErkJggg==">
                                  “이 사회 전체가 의자였구나” 샤워실 목욕의자의 재발견 | 중앙일보
                                 </a>
                                 <dt>
                                  <a add_date="1671427816" href="https://www.joongang.co.kr/article/25122912">
                                   카페인 있어도 고혈압과 무관? 커피가 '두얼굴 헐크'인 까닭 [건강한 가족] | 중앙일보
                                  </a>
                                  <dt>
                                   <a add_date="1671427819" href="https://www.joongang.co.kr/article/25122804">
                                    올가을 가뭄에 당도↑, 최근 비에 산도↓…역대급 당산비 감귤 [e즐펀한 토크] | 중앙일보
                                   </a>
                                   <dt>
                                    <a add_date="1671427825" href="https://www.joongang.co.kr/article/25121510">
                                     예거 르쿨트르와 레터링 아티스트 알렉스 트로슈가 만드는 새로운 의미 [더 하이엔드] | 중앙일보
                                    </a>
                                    <dt>
                                     <a add_date="1671427858" href="https://www.joongang.co.kr/article/25123714" icon="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAA4klEQVQ4jWP8//8/AymABV3gz2+Gv38YGBkZGBgYIEaxsUO5WDX8Pb7777r5/398Y2BgYGBkZFLVYUmpYOQVQKj4jwa+fv6za813B6lvusw/qxL+Pb77/99fZHkmdCdx8TDpmTPw8jMwMTGp6jDKKDEwoqjB0EAIDFcNf/8y/P3LwMjIwMFJlIZ/ty79f/eKUVCUSccUi3HIkfLv3es/u9Z8DzH8bifxe8mk/3//oEfr//+MyInv7+HtfzcuYJSQY3byZ9K3YGDGSGkMDCgaGP7+Zfj3l4GVlYGBEVMpNg1EAABdi4Sb+zSY9AAAAABJRU5ErkJggg==">
                                      [사진] 중앙광고대상 시상식 | 중앙일보
                                     </a>
                                    </dt>
                                   </dt>
                                  </dt>
                                 </dt>
                                </dt>
                               </dt>
                              </dt>
                             </dt>
                            </dt>
                           </dt>
                          </dt>
                         </dt>
                        </dt>
                       </dt>
                      </dt>
                     </dt>
                    </dt>
                   </dt>
                  </dt>
                 </dt>
                </dt>
               </dt>
              </dt>
             </dt>
            </p>
           </dl>
           <p>
            <dt>
             <h3 add_date="1629940719" last_modified="1671427858">
              세계
             </h3>
             <dl>
              <p>
               <dt>
                <a add_date="1671427658" href="https://www.joongang.co.kr/article/25126818">
                 미얀마 최대도시 양곤서 또 폭발 사고…페리 승객 11명 부상 | 중앙일보
                </a>
                <dt>
                 <a add_date="1671427665" href="https://www.joongang.co.kr/article/25126649">
                  히잡시위 지지한 ‘이란 국민배우’ 체포 | 중앙일보
                 </a>
                 <dt>
                  <a add_date="1671427670" href="https://www.joongang.co.kr/article/25126545">
                   "우크라, 미국 만류에도 러시아軍 최고 지휘관 암살 시도" | 중앙일보
                  </a>
                  <dt>
                   <a add_date="1671427673" href="https://www.joongang.co.kr/article/25126510">
                    '건강이상설' 푸틴 군 사령관들 소집…"우크라 작전 제안하라" | 중앙일보
                   </a>
                   <dt>
                    <a add_date="1671427676" href="https://www.joongang.co.kr/article/25126489">
                     코로나 지원금 100억 빼돌린 美목사…호화주택 사려다 덜미 | 중앙일보
                    </a>
                    <dt>
                     <a add_date="1671427680" href="https://www.joongang.co.kr/article/25126482">
                      한 달 전 인니 지진 사망자, 300명대 아닌 602명…재집계 수치 발표 | 중앙일보
                     </a>
                     <dt>
                      <a add_date="1671427684" href="https://www.joongang.co.kr/article/25126355">
                       美, 中 36개 기업 수출통제 추가…AI 등 첨단산업 육성 견제 | 중앙일보
                      </a>
                      <dt>
                       <a add_date="1671427689" href="https://www.joongang.co.kr/article/25126270">
                        백지시위 물어뜯은 中대사 "외부사주 받은 색깔혁명 냄새난다" | 중앙일보
                       </a>
                      </dt>
                     </dt>
                    </dt>
                   </dt>
                  </dt>
                 </dt>
                </dt>
               </dt>
              </p>
             </dl>
             <p>
              <dt>
               <h3 add_date="1629940719" last_modified="1671427858">
                스포츠
               </h3>
               <dl>
                <p>
                 <dt>
                  <a add_date="1671427337" href="https://www.joongang.co.kr/article/25126744" icon="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAA4klEQVQ4jWP8//8/AymABV3gz2+Gv38YGBkZGBgYIEaxsUO5WDX8Pb7777r5/398Y2BgYGBkZFLVYUmpYOQVQKj4jwa+fv6za813B6lvusw/qxL+Pb77/99fZHkmdCdx8TDpmTPw8jMwMTGp6jDKKDEwoqjB0EAIDFcNf/8y/P3LwMjIwMFJlIZ/ty79f/eKUVCUSccUi3HIkfLv3es/u9Z8DzH8bifxe8mk/3//oEfr//+MyInv7+HtfzcuYJSQY3byZ9K3YGDGSGkMDCgaGP7+Zfj3l4GVlYGBEVMpNg1EAABdi4Sb+zSY9AAAAABJRU5ErkJggg==">
                   故마라도나에 우승컵 바쳤다…‘메신’의 짜릿한 ‘카타르’시스 | 중앙일보
                  </a>
                  <dt>
                   <a add_date="1671427389" href="https://www.joongang.co.kr/article/25126571">
                    尹 '조용한 생일'…"직언 들어줘 감사하다" 참모들 축하 메시지 | 중앙일보
                   </a>
                   <dt>
                    <a add_date="1671427550" href="https://www.joongang.co.kr/article/25126770">
                     기아, 카타르 월드컵에 차량 297대 투입…브랜드 홍보 | 중앙일보
                    </a>
                    <dt>
                     <a add_date="1671427609" href="https://www.joongang.co.kr/article/25126756">
                      [THINK ENGLISH] 손흥민, 마스크 벗고 런던에서 훈련 재개 | 중앙일보
                     </a>
                     <dt>
                      <a add_date="1671427662" href="https://www.joongang.co.kr/article/25126753">
                       "메시 우승 자격 있다"…'축구의 신'에 축하 건넨 '축구 황제' | 중앙일보
                      </a>
                      <dt>
                       <a add_date="1671427720" href="https://www.joongang.co.kr/article/25126254" icon="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAA4klEQVQ4jWP8//8/AymABV3gz2+Gv38YGBkZGBgYIEaxsUO5WDX8Pb7777r5/398Y2BgYGBkZFLVYUmpYOQVQKj4jwa+fv6za813B6lvusw/qxL+Pb77/99fZHkmdCdx8TDpmTPw8jMwMTGp6jDKKDEwoqjB0EAIDFcNf/8y/P3LwMjIwMFJlIZ/ty79f/eKUVCUSccUi3HIkfLv3es/u9Z8DzH8bifxe8mk/3//oEfr//+MyInv7+HtfzcuYJSQY3byZ9K3YGDGSGkMDCgaGP7+Zfj3l4GVlYGBEVMpNg1EAABdi4Sb+zSY9AAAAABJRU5ErkJggg==">
                        후크 "이승기에 54억 전액 지급…합의 못했지만 분쟁 종결" | 중앙일보
                       </a>
                       <dt>
                        <a add_date="1671427764" href="https://www.joongang.co.kr/article/25126726">
                         메시, 골든볼 MVP에도 뽑혔다…월드컵 역대 첫 2회 수상 | 중앙일보
                        </a>
                        <dt>
                         <a add_date="1671427771" href="https://www.joongang.co.kr/article/25126624">
                          황선우, 1분40초 벽 깼다…쇼트코스 자유형 200m '2연패' | 중앙일보
                         </a>
                         <dt>
                          <a add_date="1671427775" href="https://www.joongang.co.kr/article/25126483">
                           메르스? 코로나? 결승 이틀 앞둔 프랑스 발칵, 5명이 아프다 | 중앙일보
                          </a>
                          <dt>
                           <a add_date="1671427779" href="https://www.joongang.co.kr/article/25126246">
                            벤투 감독, 폴란드 사령탑 거론…레반도프스키 지휘하나 | 중앙일보
                           </a>
                           <dt>
                            <a add_date="1671427783" href="https://www.joongang.co.kr/article/25126081">
                             벤투 사단 '비트박스 코치', 한국 이웃에 남긴 마지막 선물 | 중앙일보
                            </a>
                            <dt>
                             <a add_date="1671427787" href="https://www.joongang.co.kr/article/25125831">
                              역시, 메시 | 중앙일보
                             </a>
                             <dt>
                              <a add_date="1671427791" href="https://www.joongang.co.kr/article/25125786">
                               여자배구 조송화 계약해지 무효소송 패소… 잔여연봉 4억원 수령 어려워져 | 중앙일보
                              </a>
                              <dt>
                               <a add_date="1671427795" href="https://www.joongang.co.kr/article/25126376">
                                황선우 앞세운 쇼트코스 계영 800ｍ 韓신기록...역대최고 4위 | 중앙일보
                               </a>
                               <dt>
                                <a add_date="1671427808" href="https://www.joongang.co.kr/article/25124732">
                                 "답 안하는 걸로 할게요"…조규성 3초 고민하게 만든 질문 | 중앙일보
                                </a>
                               </dt>
                              </dt>
                             </dt>
                            </dt>
                           </dt>
                          </dt>
                         </dt>
                        </dt>
                       </dt>
                      </dt>
                     </dt>
                    </dt>
                   </dt>
                  </dt>
                 </dt>
                </p>
               </dl>
               <p>
                <dt>
                 <h3 add_date="1629940719" last_modified="1671427858">
                  정치
                 </h3>
                 <dl>
                  <p>
                   <dt>
                    <a add_date="1671427356" href="https://www.joongang.co.kr/article/25126791">
                     文정부 ‘통계조작’ 공방 격화…與 “통계조작성장” 野 “모욕주기” | 중앙일보
                    </a>
                    <dt>
                     <a add_date="1671427361" href="https://www.joongang.co.kr/article/25126774">
                      정청래, 박지원 복당 공개반대 "한 번 배신하면 또 배신" | 중앙일보
                     </a>
                     <dt>
                      <a add_date="1671427364" href="https://www.joongang.co.kr/article/25126755">
                       尹지지율, 중도·20대가 쌍끌이로 올렸다…6월 이후 첫 40%대 [리얼미터] | 중앙일보
                      </a>
                      <dt>
                       <a add_date="1671427371" href="https://www.joongang.co.kr/article/25126618">
                        법인세보다 장관 '빅2'가 핵심...野 반대한 한동훈·이상민 예산은 | 중앙일보
                       </a>
                       <dt>
                        <a add_date="1671427381" href="https://www.joongang.co.kr/article/25126580" icon="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAA4klEQVQ4jWP8//8/AymABV3gz2+Gv38YGBkZGBgYIEaxsUO5WDX8Pb7777r5/398Y2BgYGBkZFLVYUmpYOQVQKj4jwa+fv6za813B6lvusw/qxL+Pb77/99fZHkmdCdx8TDpmTPw8jMwMTGp6jDKKDEwoqjB0EAIDFcNf/8y/P3LwMjIwMFJlIZ/ty79f/eKUVCUSccUi3HIkfLv3es/u9Z8DzH8bifxe8mk/3//oEfr//+MyInv7+HtfzcuYJSQY3byZ9K3YGDGSGkMDCgaGP7+Zfj3l4GVlYGBEVMpNg1EAABdi4Sb+zSY9AAAAABJRU5ErkJggg==">
                         ‘당심 100% 룰’ 김용태 “윤핵관 당헌·당규 손바닥 뒤집듯 바꿔…한심” | 중앙일보
                        </a>
                        <dt>
                         <a add_date="1671427395" href="https://www.joongang.co.kr/article/25126550">
                          [속보] 북, 탄도미사일 2발 발사…“일본 EEZ 밖 낙하 추정” | 중앙일보
                         </a>
                         <dt>
                          <a add_date="1671427399" href="https://www.joongang.co.kr/article/25126364" icon="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAA4klEQVQ4jWP8//8/AymABV3gz2+Gv38YGBkZGBgYIEaxsUO5WDX8Pb7777r5/398Y2BgYGBkZFLVYUmpYOQVQKj4jwa+fv6za813B6lvusw/qxL+Pb77/99fZHkmdCdx8TDpmTPw8jMwMTGp6jDKKDEwoqjB0EAIDFcNf/8y/P3LwMjIwMFJlIZ/ty79f/eKUVCUSccUi3HIkfLv3es/u9Z8DzH8bifxe8mk/3//oEfr//+MyInv7+HtfzcuYJSQY3byZ9K3YGDGSGkMDCgaGP7+Zfj3l4GVlYGBEVMpNg1EAABdi4Sb+zSY9AAAAABJRU5ErkJggg==">
                           이정현 "당대표 나가든 자문을 하든 소극적으로 있지 않겠다" | 중앙일보
                          </a>
                          <dt>
                           <a add_date="1671427511" href="https://www.joongang.co.kr/article/25126336" icon="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAA4klEQVQ4jWP8//8/AymABV3gz2+Gv38YGBkZGBgYIEaxsUO5WDX8Pb7777r5/398Y2BgYGBkZFLVYUmpYOQVQKj4jwa+fv6za813B6lvusw/qxL+Pb77/99fZHkmdCdx8TDpmTPw8jMwMTGp6jDKKDEwoqjB0EAIDFcNf/8y/P3LwMjIwMFJlIZ/ty79f/eKUVCUSccUi3HIkfLv3es/u9Z8DzH8bifxe8mk/3//oEfr//+MyInv7+HtfzcuYJSQY3byZ9K3YGDGSGkMDCgaGP7+Zfj3l4GVlYGBEVMpNg1EAABdi4Sb+zSY9AAAAABJRU5ErkJggg==">
                            유승민 여론조사 보니…친윤 "이게 당원투표 100% 이유" | 중앙일보
                           </a>
                          </dt>
                         </dt>
                        </dt>
                       </dt>
                      </dt>
                     </dt>
                    </dt>
                   </dt>
                  </p>
                 </dl>
                 <p>
                 </p>
                </dt>
               </p>
              </dt>
             </p>
            </dt>
           </p>
          </dt>
         </p>
        </dt>
       </p>
      </dt>
     </p>
    </dt>



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
