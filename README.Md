# 화재 탐지 시스템


## 💻프로젝트 소개
화재 , 연기를 탐지하여 WebSocket 서버를 이용하여 알림을 보내는 화재(+연기) 탐지 시스템입니다.
위 시스템을 SmartFactory에 추가

## ⏲개발기간
 
 > 22.11.13 - 22.11.20


## 🛠 개발 환경
torch 버전: 1.90 +cpu

``` C
import torch
print(torch.__version__)
```

torchvision 버전: '0.10.0+cpu'
``` C
import torchvision
print(torchvision.__version__)
```

### 🕹실행 방법

<p> torchvision 만 설치하면 자동으로 torch는 설치됩니다.
 
```C
pip install torchvision == 0.10.0
```

```C
 
 1. WebSocket( https://github.com/choisunghwan/WebSocketserver ) 다운 > server.js 실행 ( 터미널 명령어 : node.server)
 
 2. detection.py 실행
 
```
 
### Screenshot
 
 #### 1. 화재 탐지 한다.

<img src="https://user-images.githubusercontent.com/33335762/202965703-d2c9a1b3-4585-411f-ab48-bb7fca92be8f.png" width="480" />

 #### 2-1 화재 탐지 전 페이지 화면
 
<img src="https://user-images.githubusercontent.com/33335762/202965890-29b9308f-b27a-47c8-9eef-9396bd1a7ebb.png" width="480" />

#### 2-2 화재 탐지시 페이지 화면
 
 <img src="https://user-images.githubusercontent.com/33335762/202965833-644469cf-7070-4cdc-8e1e-2f3110ddf9c6.png" width="480" />

