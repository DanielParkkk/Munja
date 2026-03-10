from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
import os
import tempfile
import traceback

load_dotenv()

app = Flask(__name__, static_folder='.')
CORS(app)

# ── 모델 자동 다운로드 ──────────────────────────────
def download_model(file_id, dest_path):
    if os.path.exists(dest_path):
        size = os.path.getsize(dest_path)
        if size > 10 * 1024 * 1024:  # 10MB 이상이면 정상 파일
            print(f"✅ 모델 이미 존재: {dest_path} ({size//1024//1024}MB)")
            return
        else:
            print(f"⚠️ 파일이 너무 작음({size}bytes), 재다운로드...")
            os.remove(dest_path)

    print(f"📥 모델 다운로드 중: {dest_path}")
    import gdown
    gdown.download(id=file_id, output=dest_path, quiet=False, fuzzy=True)
    size = os.path.getsize(dest_path)
    print(f"✅ 다운로드 완료: {dest_path} ({size//1024//1024}MB)")

# 서버 시작 시 모델 다운로드
print("🔄 모델 파일 확인 중...")
download_model('1LhQ6AKWzhhG-w880Fs_G2HJagtXvQ-YK', 'weights_detector.pt')
download_model('1XK8-NGDEHxvopSaxElXvdYI1CXtgZgN3', 'weights_classifier.pth')
print("✅ 모델 준비 완료!")

# ── OCR 엔드포인트 ──────────────────────────────
@app.route('/ocr', methods=['POST'])
def ocr():
    if 'image' not in request.files:
        return jsonify({'error': '이미지가 없습니다.'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': '파일명이 비어 있습니다.'}), 400

    suffix = os.path.splitext(file.filename)[-1] or '.png'
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        from detect_v5 import extract_ancient_text
        result = extract_ancient_text(tmp_path)
        return jsonify({'text': result})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ── 번역 엔드포인트 ──────────────────────────────
@app.route('/translate', methods=['POST'])
def translate():
    if 'image' not in request.files:
        return jsonify({'error': '이미지가 없습니다.'}), 400

    file = request.files['image']
    ocr_text = request.form.get('text', '').strip()
    target_lang = request.form.get('lang', 'ko')

    if not ocr_text:
        return jsonify({'error': '번역할 텍스트가 없습니다.'}), 400

    suffix = os.path.splitext(file.filename)[-1] or '.png'
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        from translate_ocr import translate_image
        result = translate_image(
            image_path=tmp_path,
            ocr_text=ocr_text,
            api_key=os.environ.get('OPENAI_API_KEY'),
        )

        # 언어 선택에 따라 결과 반환
        lang_map = {
            'ko': result.get('modern_korean', ''),
            'en': result.get('english_translation', ''),
            'ja': result.get('modern_korean', ''),  # ja/zh는 modern_korean 기반으로 추가 번역 가능
            'zh': result.get('modern_korean', ''),
        }

        return jsonify({
            'text': lang_map.get(target_lang, result.get('modern_korean', '')),
            'ancient_text_corrected': result.get('ancient_text_corrected', ocr_text),
            'modern_korean': result.get('modern_korean', ''),
            'english_translation': result.get('english_translation', ''),
            'notes': result.get('notes', ''),
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ── 프론트엔드 서빙 ──────────────────────────────
@app.route('/')
def index():
    return send_from_directory('.', 'translator-C.html')


if __name__ == '__main__':
    print("=" * 50)
    print("  옛한글 번역기 서버 시작")
    print("=" * 50)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
