from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import tempfile
import traceback

app = Flask(__name__, static_folder='.')
CORS(app)

# ── OCR 엔드포인트 ──────────────────────────────
@app.route('/ocr', methods=['POST'])
def ocr():
    if 'image' not in request.files:
        return jsonify({'error': '이미지가 없습니다.'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': '파일명이 비어 있습니다.'}), 400

    # 임시 파일로 저장
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
