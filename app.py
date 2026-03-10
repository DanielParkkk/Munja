from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
import os
import tempfile
import traceback
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime

load_dotenv()

app = Flask(__name__, static_folder='.')
CORS(app)

# ── DB 연결 ──────────────────────────────
def get_db():
    return psycopg2.connect(os.environ.get('DATABASE_URL'), sslmode='require')

def init_db():
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute('''
            CREATE TABLE IF NOT EXISTS translations (
                id SERIAL PRIMARY KEY,
                username TEXT NOT NULL DEFAULT '익명',
                ocr_text TEXT NOT NULL,
                translated_text TEXT NOT NULL,
                lang TEXT NOT NULL DEFAULT 'ko',
                created_at TIMESTAMP DEFAULT NOW()
            )
        ''')
        conn.commit()
        cur.close()
        conn.close()
        print("✅ DB 테이블 준비 완료!")
    except Exception as e:
        print(f"⚠️ DB 초기화 오류: {e}")

# ── 모델 자동 다운로드 ──────────────────────────────
def download_model(file_id, dest_path):
    if os.path.exists(dest_path):
        size = os.path.getsize(dest_path)
        if size > 10 * 1024 * 1024:
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

# 서버 시작 시 모델 다운로드 + DB 초기화
print("🔄 모델 파일 확인 중...")
download_model('1LhQ6AKWzhhG-w880Fs_G2HJagtXvQ-YK', 'weights_detector.pt')
download_model('1XK8-NGDEHxvopSaxElXvdYI1CXtgZgN3', 'weights_classifier.pth')
print("✅ 모델 준비 완료!")
init_db()

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
    username = request.form.get('username', '익명').strip() or '익명'

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

        lang_map = {
            'ko': result.get('modern_korean', ''),
            'en': result.get('english_translation', ''),
            'ja': result.get('modern_korean', ''),
            'zh': result.get('modern_korean', ''),
        }
        translated_text = lang_map.get(target_lang, result.get('modern_korean', ''))

        # DB에 저장
        try:
            conn = get_db()
            cur = conn.cursor()
            cur.execute(
                'INSERT INTO translations (username, ocr_text, translated_text, lang) VALUES (%s, %s, %s, %s)',
                (username, ocr_text[:500], translated_text[:1000], target_lang)
            )
            conn.commit()
            cur.close()
            conn.close()
        except Exception as db_err:
            print(f"⚠️ DB 저장 오류: {db_err}")

        return jsonify({
            'text': translated_text,
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


# ── 히스토리 조회 ──────────────────────────────
@app.route('/history', methods=['GET'])
def history():
    try:
        conn = get_db()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute('''
            SELECT id, username, ocr_text, translated_text, lang, created_at
            FROM translations
            ORDER BY created_at DESC
            LIMIT 50
        ''')
        rows = cur.fetchall()
        cur.close()
        conn.close()

        result = []
        for row in rows:
            result.append({
                'id': row['id'],
                'user': row['username'],
                'ocr': row['ocr_text'],
                'tr': row['translated_text'],
                'lang': row['lang'],
                'time': row['created_at'].strftime('%Y-%m-%d %H:%M'),
            })
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify([]), 500


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

