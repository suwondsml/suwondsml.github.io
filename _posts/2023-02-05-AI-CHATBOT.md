---
layout: post
title: AI CHATBOT[Tourist Attractions in Seoul]
authors: [Subin Sung]
categories: [1기 AI/SW developers(개인 프로젝트)]
---


# **AI(인공지능) 기반 챗봇 서비스**

## **프로젝트 소개**

주제 : 챗봇을 이용한 서울 관광지 추천

서울의 행정구별 관광지에 대한 정보를 제공하여 사용자들에게 신선한 체험의 기회를 제공하기 위해 챗봇을 제작하였다. 이를 통해 많은 사람들이 알지 못했던 장소를 알릴 수 있는 효과를 기대할 수 있다. 사용자 별로 특색에 맞는 관광지를 추천해 주며, 관광지를 찾아야 하는 불필요한 노동과 수고, 시간낭비를 덜 수 있다.



## **DATA**

### **오픈API 및 웹크롤링을 이용해 데이터 제작**

공공데이터포털에서 제공하는 '한국관광공사 국문 관광정보 서비스_GW' 오픈 API를 이용해 사용할 데이터 제작하며 추가적으로 필요한 데이터는 웹크롤링을 이용한다.

![image-20230206144321659](https://ifh.cc/g/5CCn0F.png)



![img](https://ifh.cc/g/Roy5no.jpg) <br/>
여러 데이터를 종합해 만든 데이터 모음



## **챗봇 소개**

- 만들어진 데이터를 활용하여 카테고리별 챗봇을 제작.

- 여러 카테고리의 챗봇을 하나의 앱에서 사용할 수 있도록 앱 제작.

- 카테고리는 대분류 중분류 소분류 로 구분한다.

- 대분류에는 문화예술/자연명소/레포츠/쇼핑/음식점/숙박시설이 있다. 

- 문화예술은 다시 건축.조형물/산업관광지/역사관광지/체험관광지/휴양관광지/문화시설로 나뉜다.

- 자연명소는 강.계곡.폭포/공원 으로 나뉜다.

  

## **작품 구상도**

### SW 구성도

- Dialog Flow를 이용 : 구글의 머신러닝기술을 활용하여 사용자의 니즈를 예측하여 그에 맞는 질문과 대답들을 업데이트한다.

![img](https://ifh.cc/g/q8WZg8.png)

- ngrok 이용 페이스북 메시지 등과 같이 외부 서버와 연결하여 외부 사용자가 이용가능하게 한다.

![img](https://th.bing.com/th/id/OIP.4pRFlMrovybUI_RxbXjj1AHaCs?w=348&h=127&c=7&r=0&o=5&dpr=1.5&pid=1.7)

![img](https://ifh.cc/g/czNTqk.png) <br/>
Dialog Flow와 ngrok 연결




```python
from flask import Flask, request
import requests
app = Flask(__name__)
FB_API_URL = 
VERIFY_TOKEN=
PAGE_ACCESS_TOKEN=
def send_message(recipient_id, text):
    """Send a response to Facebook"""
    payload = {
        'message': {
            'text': text
        },
        'recipient': {
            'id': recipient_id
        },
        'notification_type': 'regular'
    }

    auth = {
        'access_token': PAGE_ACCESS_TOKEN
    }

    response = requests.post(
        FB_API_URL,
        params=auth,
        json=payload
    )

    return response.json()

def get_bot_response(message):
    """This is just a dummy function, returning a variation of what
    the user said. Replace this function with one connected to chatbot."""
    return "This is a dummy response to '{}'".format(message)


def verify_webhook(req):
    if req.args.get("hub.verify_token") == VERIFY_TOKEN:
        return req.args.get("hub.challenge")
    else:
        return "incorrect"

def respond(sender, message):
    """Formulate a response to the user and
    pass it on to a function that sends it."""
    response = get_bot_response(message)
    send_message(sender, response)


def is_user_message(message):
    """Check if the message is a message from the user"""
    return (message.get('message') and
            message['message'].get('text') and
            not message['message'].get("is_echo"))


@app.route("/webhook", methods=['GET'])
def listen():
    """This is the main function flask uses to
    listen at the `/webhook` endpoint"""
    if request.method == 'GET':
        return verify_webhook(request)

@app.route("/webhook", methods=['POST'])
def talk():
    payload = request.get_json()
    event = payload['entry'][0]['messaging']
    for x in event:
        if is_user_message(x):
            text = x['message']['text']
            sender_id = x['sender']['id']
            respond(sender_id, text)

    return "ok"

@app.route('/')
def hello():
    return 'hello'

if __name__ == '__main__':
    app.run(threaded=True, port=5000)

```

ngrok과 페이스북 메세지 연결 코드



- Open Api를 이용하여 정보를 수집하고 이를 Dialog Flow와 연결하여 챗봇이 사용자의 니즈에 맞는 정보를 제공한다. 본 챗봇의 경우 관광지 데이터를 공공데이터포털 에서 제공받아 이를 Dialog Flow와 연결하여 사용자에게 정보를 제공한다.

  

- 공공데이터포털에서 제공하지 않는 정보들이나 인터넷에 있는 정보를 활용하기 위하여 Web crawling 기술을 활용해 더 많은 정보를 제공한다.

- Dialog Flow 대화문 작성


```python
# -*- coding: utf-8 -*-



# 파일로 출력하기

i = 1

# 출력, 입력 값 JSON 파일을 생성합니다.

prev = str(conversations[0].contentName) + str(conversations[0].contentType)

f = open(prev + '.json', 'w', encoding='UTF-8')

f.write('{ "id": "10d3155d-4468-4118-8f5d-15009af446d0", "name": "' + prev + '", "auto": true, "contexts": [], "responses": [ { "resetContexts": false, "affectedContexts": [], "parameters": [], "messages": [ { "type": 0, "lang": "en", "speech": "' + conversations[0].answer + '" } ], "defaultResponsePlatforms": {}, "speech": [] } ], "priority": 500000, "webhookUsed": false, "webhookForSlotFilling": false, "fallbackIntent": false, "events": [] }')

f.close()

f = open(prev + '_usersays_en.json', 'w', encoding='UTF-8')

f.write("[")

f.write('{ "id": "3330d5a3-f38e-48fd-a3e6-000000000001", "data": [ { "text": "' + conversations[0].question + '", "userDefined": false } ], "isTemplate": false, "count": 0 },')



while True:

    if i >= len(conversations):

        f.write("]")

        f.close()

        break;

    c = conversations[i]

    if prev == str(c.contentName) + str(c.contentType):

        f.write('{ "id": "3330d5a3-f38e-48fd-a3e6-000000000001", "data": [ { "text": "' + c.question + '", "userDefined": false } ], "isTemplate": false, "count": 0 }')

    else:

        f.write("]")

        f.close()

        # 출력, 입력 값 JSON 파일을 생성합니다.

        prev = str(c.contentName) + str(c.contentType)

        f = open(prev + '.json', 'w', encoding='UTF-8')

        f.write('{ "id": "10d3155d-4468-4118-8f5d-15009af446d0", "name": "' + prev + '", "auto": true, "contexts": [], "responses": [ { "resetContexts": false, "affectedContexts": [], "parameters": [], "messages": [ { "type": 0, "lang": "en", "speech": "' + c.answer + '" } ], "defaultResponsePlatforms": {}, "speech": [] } ], "priority": 500000, "webhookUsed": false, "webhookForSlotFilling": false, "fallbackIntent": false, "events": [] }')

        f.close()

        f = open(prev + '_usersays_en.json', 'w', encoding='UTF-8')

        f.write("[")

        f.write('{ "id": "3330d5a3-f38e-48fd-a3e6-000000000001", "data": [ { "text": "' + c.question + '", "userDefined": false } ], "isTemplate": false, "count": 0 }')

    i = i + 1
```

코드 실행 결과
![img](https://ifh.cc/g/Gsd3c1.png)

### HW 구성도

- UX/UI : Facebook Messenger

![img](https://ifh.cc/g/yRXy20.png)

- Database : MongoDB

![img](https://ifh.cc/g/kpzoTc.png)

- 챗봇은 데이터를 보관하고 분석하는 AWS 서버를 활용하여 사용자에게 데이터를 제공한다.

- 또한, Amazon EC2를 이용하여 원하는 수의 가상 서버를 구축하고 보안 및 네트워킹 을 구성하며 스토리지를 관리한다.


![img](https://ifh.cc/g/hrS7hL.png)




## **서비스 흐름도**

- 페이스북이 제공하는 기능(카테고리 등)을 이용해 사용자가 이용하기 편한 화면으로 구성한다.
- NLP Engine으로 Dialog Flov를 활용한다. nerok을 이용해 서버를 만들어 DialogFlow의 Webhook을 연결한다.
- 이후 페이스북 메신저와 DialogFlow를 연결해
  UX/UI를 구성하고, 백엔드 서비스를 띄워 답변 제공이 가능하게 한다.

![img](https://ifh.cc/g/FnF4tb.png)

## **메뉴 구성도**


![img](https://ifh.cc/g/aw6YO0.png)

## **엔티티 관계도**

![img](https://ifh.cc/g/vn4Xqc.png)

## **알고리즘 시나리오**

![img](https://ifh.cc/g/FHpxr6.png)

ⅰ. 사용자로부터 원하는 관광지에 관련된 키워드를 입력 받습니다.

ⅱ. 관련된 키워드가 Database와 일치하는 지 확인합니다.

ⅲ. 일치할 경우 챗봇이 수집한 데이터를 기반으로 사용자의 요구에 알맞은 관광정보를 추천합니다.

ⅳ. 일치하지 않을 경우 대화를 종료합니다.

## **결과물**

![img](https://ifh.cc/g/NTYW8D.png)
![img](https://ifh.cc/g/MCn3JJ.png)

[한이음 결과보고서 제출용 작품시연 영상](https://www.youtube.com/watch?v=pdcLrldZBpM)



![img](https://ifh.cc/g/Hcct0g.png)
