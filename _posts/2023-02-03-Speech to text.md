---
layout: post
title: Speech to text
authors: [Taekyung Kim]
categories: [1기 AI/SW developers(개인프로젝트)]
---


```python
from google.colab import drive
drive.mount('/content/drive')
```

    Mounted at /content/drive


# Install library


```python
pip install speechrecognition
```

    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Collecting speechrecognition
      Downloading SpeechRecognition-3.9.0-py2.py3-none-any.whl (32.8 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m32.8/32.8 MB[0m [31m35.6 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting requests>=2.26.0
      Downloading requests-2.28.2-py3-none-any.whl (62 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m62.8/62.8 KB[0m [31m6.9 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.8/dist-packages (from requests>=2.26.0->speechrecognition) (2.1.1)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.8/dist-packages (from requests>=2.26.0->speechrecognition) (2022.12.7)
    Requirement already satisfied: urllib3<1.27,>=1.21.1 in /usr/local/lib/python3.8/dist-packages (from requests>=2.26.0->speechrecognition) (1.24.3)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.8/dist-packages (from requests>=2.26.0->speechrecognition) (2.10)
    Installing collected packages: requests, speechrecognition
      Attempting uninstall: requests
        Found existing installation: requests 2.25.1
        Uninstalling requests-2.25.1:
          Successfully uninstalled requests-2.25.1
    Successfully installed requests-2.28.2 speechrecognition-3.9.0


# Import library


```python
import speech_recognition as sr
```


```python
import librosa.display
import IPython.display as ipd
```


```python
r = sr.Recognizer()
r
```




    <speech_recognition.Recognizer at 0x7f04c757c4f0>



# Import audio file


```python
audio_file = sr.AudioFile('/content/drive/MyDrive/test_kor.wav')
```


```python
from IPython.display import Audio
file_name = '/content/drive/MyDrive/test_kor.wav'
Audio(file_name)
```





<audio  controls="controls" >
    Your browser does not support the audio element.
</audio>





```python
with audio_file as source:
  audio = r.record(source)
```


```python
print(r.recognize_google(audio, language='ko-KR'))
```

    result2:
    {   'alternative': [   {   'confidence': 0.86883628,
                               'transcript': '만약 귀하의 신용카드가 분실 되었다면 승인되지 않은 사용을 '
                                             '방지하기 위해 은행에 최대한 빨리 알려 주십시오 그렇지 않으면 '
                                             '귀하는 발생 시키지 않은 대금을 지불하도록 요구 받을 수 '
                                             '있습니다'},
                           {   'transcript': '만약 귀하의 신용카드가 분실 되었다면 승인되지 않은 사용을 '
                                             '방지하기 위해 은행에 최대한 빨리 알려 주십시오 그렇지 않으면 '
                                             '귀하는 발생시키지 않은 대금을 지불하도록 요구받을 수 있습니다'},
                           {   'transcript': '만약 귀하의 신용카드가 분실 되었다면 승인되지 않은 사용을 '
                                             '방지하기 위해 은행에 최대한 빨리 알려 주십시오 그렇지 않으면 '
                                             '귀하는 발생시키지 않은 대금을 지불하도록 요구 받을 수 있습니다'},
                           {   'transcript': '만약 귀하의 신용카드가 분실 되었다면 승인되지 않은 사용을 '
                                             '방지하기 위해 은행에 최대한 빨리 알려 주십시오 그렇지 않으면 '
                                             '귀하는 발생시키지 아는 대금을 지불하도록 요구 받을 수 있습니다'},
                           {   'transcript': '만약 귀하의 신용카드가 분실 되었다면 승인되지 않은 사용을 '
                                             '방지하기 위해 은행에 최대한 빨리 알려 주십시오 그렇지 않으면 '
                                             '귀하는 발생시키지 아는 대금을 지불하도록 요구받을 수 있습니다'}],
        'final': True}
    만약 귀하의 신용카드가 분실 되었다면 승인되지 않은 사용을 방지하기 위해 은행에 최대한 빨리 알려 주십시오 그렇지 않으면 귀하는 발생 시키지 않은 대금을 지불하도록 요구 받을 수 있습니다



```python
#마크다운 파일로 변경하는 방법
!jupyter nbconvert --to markdown "/content/drive/MyDrive/notebook_tes
```