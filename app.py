from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
import os
import tempfile
import traceback

load_dotenv()

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
    data = request.json
    text = data.get('text', '').strip()
    target_lang = data.get('lang', 'ko')

    if not text:
        return jsonify({'error': '번역할 텍스트가 없습니다.'}), 400

    lang_map = {
        'ko': '현대 한국어',
        'en': 'English',
        'ja': '日本語',
        'zh': '中文'
    }

    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

        response = client.chat.completions.create(
            model='gpt-4o-mini',
            messages=[
                {'role': 'system', 'content': f'당신은 옛한글 전문 번역가입니다. 입력된 옛한글 원문을 {lang_map.get(target_lang, "현대 한국어")}로 자연스럽게 번역해주세요. 번역문만 출력하세요.'},
                {'role': 'user', 'content': text}
            ],
            max_tokens=2048
        )

        result = response.choices[0].message.content
        return jsonify({'text': result})

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


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
