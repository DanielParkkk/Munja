# 문자 (Munja) — 옛한글 OCR 번역 서비스

> 고문서 이미지를 업로드하면 옛한글을 자동으로 판독하고, 현대 한국어·영어·일본어·중국어로 번역해주는 웹 서비스입니다.

🔗 **Live Demo:** https://munja-production.up.railway.app

---

## Overview

조선시대 고문서, 훈민정음 해례본 등 옛한글로 기록된 문헌은 현대인이 읽기 어렵습니다. **Munja**는 YOLOv5 기반 커스텀 OCR 파이프라인과 GPT-5.2 Vision 번역을 결합해 이 문제를 해결합니다.

박물관·문화유산 설명문처럼 일반 사용자가 읽기 어려운 옛한글 텍스트를 누구나 쉽게 이해할 수 있도록 만드는 것이 목표입니다.

이미지 한 장을 올리면:
1. YOLOv5가 문자 영역을 탐지하고
2. CNN 분류 모델이 각 문자를 식별해 텍스트로 조합한 뒤
3. GPT-5.2가 현대어로 번역합니다

---

## Tech Stack

| 구분 | 기술 |
|------|------|
| Frontend | Vanilla HTML / CSS / JavaScript (SPA) |
| Backend | Python 3.9, Flask, Flask-CORS |
| OCR | YOLOv5 (Detection) + Custom CNN Classifier |
| Translation | OpenAI GPT-5.2 Vision |
| Database | PostgreSQL (Railway) |
| Deployment | Railway (Hobby Plan) |
| Model Storage | Google Drive + gdown 자동 다운로드 |

---

## Architecture

```
사용자 브라우저
      │
      │  ① 이미지 업로드
      ▼
web/templates/translator-C.html (Frontend SPA)
      │
      │  POST /ocr              POST /translate
      ▼                               ▼
         web/app.py (Flask Backend)
              │                       │
              ▼                       ▼
  src/munja/ocr/detect.py    src/munja/translate/translate_ocr.py
  (YOLOv5 + CNN)             (GPT-5.2 Vision)
              │                       │
              ▼                       ▼
       옛한글 텍스트 추출       현대어 번역 결과
                                       │
                                       ▼
                               PostgreSQL (Railway)
                               번역 히스토리 저장
```

---

## 동작 방식

1. 사용자가 웹 UI에서 이미지를 업로드합니다.
2. `/ocr` API가 `src/munja/ocr/detect.py`를 통해 글자 영역 검출과 글자 분류를 수행합니다.
3. OCR 결과를 세로쓰기 우→좌 순서에 맞게 재정렬하여 옛한글 문자열을 만듭니다.
4. `/translate` API가 `src/munja/translate/translate_ocr.py`를 사용해 GPT-5.2로 다음 정보를 생성합니다.
   - OCR 보정 결과
   - 현대 한국어 번역
   - 영어 번역
   - 애매한 부분에 대한 메모
5. 요청 언어가 `ja`, `zh`이면 현대 한국어 번역을 다시 일본어/중국어로 번역합니다.
6. 결과를 DB에 저장하고 최근 번역 이력을 `/history`로 조회합니다.

---

## Project Structure

```
Munja/
├── app.py                        # 진입점 (호환용)
├── run.py                        # 로컬 테스트 진입점 (호환용)
├── Procfile                      # Railway 서버 시작 커맨드
├── runtime.txt                   # Python 버전 고정 (3.9.25)
├── requirements.txt              # Python 의존성 패키지
│
├── web/                          # Flask 웹 서버
│   ├── app.py                    # 라우트 정의, DB 연동, OCR/번역 엔드포인트
│   └── templates/
│       └── translator-C.html     # 프론트엔드 SPA
│
├── src/munja/
│   ├── ocr/                      # OCR 파이프라인
│   │   ├── detect.py             # YOLOv5 문자 탐지 + 텍스트 조합
│   │   ├── classifier.py         # CNN 문자 분류 모델
│   │   ├── iou_cal.py            # 바운딩 박스 IoU 계산
│   │   ├── img_crop.py           # 문자 영역 크롭
│   │   └── ema.py                # 모델 EMA 유틸
│   └── translate/
│       └── translate_ocr.py      # GPT-5.2 Vision 번역 엔진
│
├── assets/
│   ├── models/                   # 모델 가중치 (auto-download)
│   │   ├── weights_detector.pt   # YOLOv5 탐지 모델 (153MB)
│   │   └── weights_classifier.pth# CNN 분류 모델 (97MB)
│   ├── data/
│   │   └── class.csv             # 옛한글 문자 클래스 레이블 (7148개)
│   └── fonts/
│       └── NanumBarunGothic-YetHangul.ttf
│
├── vendor/yolov5/                # 서드파티 YOLOv5 코드
│   ├── models/
│   └── utils/
│
├── samples/images/               # 테스트용 고문서 샘플 이미지
├── scripts/run_sample.py         # 샘플 실행 스크립트
└── docs/                         # 기획 문서
```

---

## API Endpoints

### `POST /ocr`
업로드한 이미지에서 옛한글 텍스트를 추출합니다.

요청: `multipart/form-data` — `image`

```json
{ "text": "옛한글 OCR 결과" }
```

### `POST /translate`
OCR 결과를 번역하고 DB에 저장합니다.

요청 필드: `image`, `text`, `lang` (ko/en/ja/zh), `username`

```json
{
  "text": "최종 번역문",
  "ancient_text_corrected": "보정된 옛한글",
  "modern_korean": "현대 한국어 번역",
  "english_translation": "영어 번역",
  "notes": ""
}
```

### `GET /history`
최근 번역 6개를 반환합니다.

### `POST /seed`
샘플 번역 데이터를 DB에 추가합니다. (개발용)

---

## Getting Started

### Prerequisites
- Python 3.9
- OpenAI API Key
- PostgreSQL (또는 Railway 배포 환경)

### Installation

```bash
git clone https://github.com/DanielParkkk/Munja.git
cd Munja
git checkout develop
pip install -r requirements.txt
```

### Environment Variables

프로젝트 루트에 `.env` 파일을 생성하세요:

```env
OPENAI_API_KEY=your_openai_api_key
DATABASE_URL=your_postgresql_connection_string
PORT=5000
```

### Run

```bash
python app.py
```

서버 시작 시 Google Drive에서 모델 파일을 자동으로 다운로드합니다 (최초 1회, 약 250MB).  
브라우저에서 `http://localhost:5000` 으로 접속하면 UI가 열립니다.

---

## Key Design Decisions

**모델 사전 로딩**  
Flask 서버 시작 시점에 YOLOv5 모델을 한 번만 로딩합니다. 요청마다 모델을 로딩하면 매 OCR 요청에 수십 초가 소요되기 때문입니다.

**Google Drive 자동 다운로드**  
250MB에 달하는 모델 파일을 Git에 직접 커밋하지 않고, 서버 시작 시 `gdown`으로 자동 다운로드합니다. Git LFS 포인터 파일 오염을 방지하기 위해 10MB 이하 파일은 재다운로드 로직을 포함합니다.

**이미지 base64 DB 저장**  
번역 히스토리 상세 모달에서 원본 이미지를 보여주기 위해 이미지를 base64로 인코딩해 PostgreSQL에 저장합니다.

**세로쓰기 RTL 정렬**  
고문서는 세로쓰기, 오른쪽→왼쪽 방향입니다. `detect.py`의 `reorder_for_vertical_rtl()` 함수가 탐지된 문자 바운딩 박스를 열 단위로 묶고 우→좌, 위→아래 순서로 정렬합니다.

**커스텀 OCR 파이프라인**  
일반 OCR 엔진을 그대로 쓰는 구조가 아니라 "탐지 + 글자 분류" 방식의 커스텀 파이프라인입니다. 옛한글 특성상 기존 OCR 엔진으로는 인식이 불가능하기 때문입니다.

**관심사 분리 (Separation of Concerns)**  
`web/`, `src/munja/`, `vendor/`, `assets/` 로 역할을 명확히 분리해 유지보수성을 높였습니다.

---

## Deployment

Railway Hobby Plan으로 배포되어 있습니다.

| 항목 | 값 |
|------|-----|
| Branch | `develop` |
| Start Command | `python app.py` (Procfile) |
| Python Version | 3.9.25 (runtime.txt) |
| Environment Variables | `OPENAI_API_KEY`, `DATABASE_URL`, `PORT` |

---

## 주의사항

- 모델 가중치 파일은 저장소에 포함되어 있지 않으며, 최초 실행 시 자동 다운로드됩니다.
- DB가 없으면 `/history`, `/translate` 저장 기능이 정상 동작하지 않습니다.
- `vendor/yolov5/`는 서드파티 코드이며 직접 수정하지 않습니다.

---

## License

This project is for academic purposes.
