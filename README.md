# Munja

옛한글 문서 이미지를 OCR로 읽고 현대 한국어와 외국어로 번역하는 Flask 기반 웹 애플리케이션입니다.

## 현재 폴더 구조

```text
Munja/
├─ app.py
├─ run.py
├─ Procfile
├─ runtime.txt
├─ requirements.txt
├─ README.md
├─ assets/
│  ├─ data/
│  │  └─ class.csv
│  ├─ fonts/
│  │  └─ NanumBarunGothic-YetHangul.ttf
│  └─ models/
│     ├─ weights_classifier.pth
│     └─ weights_detector.pt
├─ docs/
│  └─ ᄒᆞᆫ글번역기_기획안_문자.docx
├─ samples/
│  └─ images/
├─ scripts/
│  └─ run_sample.py
├─ src/
│  └─ munja/
│     ├─ ocr/
│     │  ├─ classifier.py
│     │  ├─ detect.py
│     │  ├─ ema.py
│     │  ├─ img_crop.py
│     │  └─ iou_cal.py
│     └─ translate/
│        └─ translate_ocr.py
├─ vendor/
│  └─ yolov5/
│     ├─ export.py
│     ├─ models/
│     └─ utils/
└─ web/
   ├─ app.py
   └─ templates/
      └─ translator-C.html
```

## 구조 분리 원칙

- `web/`: Flask 서버와 HTML 템플릿
- `src/munja/`: 서비스 로직
- `src/munja/ocr/`: OCR 파이프라인
- `src/munja/translate/`: 번역 로직
- `assets/`: 모델 가중치, 문자 매핑, 폰트
- `samples/`: 테스트용 이미지
- `docs/`: 기획 문서
- `vendor/yolov5/`: 서드파티 YOLOv5 코드

## 실행 경로

기존처럼 아래 명령으로 실행할 수 있습니다.

```powershell
python app.py
```

샘플 실행:

```powershell
python run.py
```

루트의 `app.py`, `run.py`는 호환용 진입점이고 실제 로직은 각각 `web/app.py`, `scripts/run_sample.py`에 있습니다.

## 주요 코드 위치

- 웹 서버: `web/app.py`
- OCR: `src/munja/ocr/detect.py`
- 글자 분류기: `src/munja/ocr/classifier.py`
- 번역: `src/munja/translate/translate_ocr.py`
- UI: `web/templates/translator-C.html`

## 환경 변수

```env
OPENAI_API_KEY=your_openai_api_key
DATABASE_URL=your_postgres_connection_string
PORT=5000
```

## 참고

- OCR 모델 가중치는 `assets/models/`에 둡니다.
- 문자 클래스 매핑은 `assets/data/class.csv`를 사용합니다.
- YOLOv5 관련 코드는 `vendor/yolov5/` 아래로 격리했습니다.
