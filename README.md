# 제주어, 표준어 양방향 음성 번역 애플리케이션 

- **2024년 1학기 자연어처리 최종 프로젝트**
- 팀 우분투 : 김한영, 서가연, 정아영, 허채연



### Method 
---
- **STT**
  - Faster Whisper 모델
  
  
- **Translation**
  - 제주어 -> 표준어 : 
  - 표준어 -> 제주어 : 


- **TTS**
  - couqui-dfjslkd 파인튜닝 : [가중치 다운로드](https://drive.google.com/file/d/11axqmuhOsRRkD2UpvJIpDlqi1Kfyw7-f/view?usp=sharing)
  

- **Dataset**
  - 한국어 방언 발화 데이터 : https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=121
  - JIT 제주 방언 데이터 : https://github.com/kakaobrain/jejueo



### 디렉터리 구조
---
원활한 데모 실행을 위해 아래와 같은 디렉터리 구조를 확인해주세요.
```bash
├── data/
│   └── bpe_tokenizer.json
├── model/
│   ├── bartbase_model2
│   └── kobart2_sort
│   └── transformer 
├── TTS/
│   ├── data/10.wav
│   ├── best_model.pth
│   ├── config.json
│   └── vocab.json
├── Interface
│   ├── app.py
│   ├── tokenizer.py
│   ├── utils.py
│   ├── example.html
└── ├── templates/index.html 

``` 

### 데모 실행 
---
```
python Interface/app.py 
```

