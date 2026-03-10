import base64
import os
import sys
import tempfile
import traceback
from pathlib import Path

import psycopg2
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from psycopg2.extras import RealDictCursor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
MODELS_DIR = PROJECT_ROOT / "assets" / "models"
TEMPLATES_DIR = PROJECT_ROOT / "web" / "templates"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

app = Flask(__name__, template_folder=str(TEMPLATES_DIR))
CORS(app)


def get_db():
    return psycopg2.connect(os.environ.get("DATABASE_URL"), sslmode="require")


def init_db():
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS translations (
                id SERIAL PRIMARY KEY,
                username TEXT NOT NULL DEFAULT '익명',
                ocr_text TEXT NOT NULL,
                translated_text TEXT NOT NULL,
                lang TEXT NOT NULL DEFAULT 'ko',
                image_data TEXT,
                created_at TIMESTAMP DEFAULT NOW()
            )
            """
        )
        cur.execute(
            """
            ALTER TABLE translations ADD COLUMN IF NOT EXISTS image_data TEXT
            """
        )
        conn.commit()
        cur.close()
        conn.close()
        print("DB table ready")
    except Exception as exc:
        print(f"DB init failed: {exc}")


def download_model(file_id, dest_path):
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if dest_path.exists():
        size = dest_path.stat().st_size
        if size > 10 * 1024 * 1024:
            print(f"Model already present: {dest_path} ({size // 1024 // 1024}MB)")
            return
        print(f"Model file too small ({size} bytes), redownloading: {dest_path}")
        dest_path.unlink()

    print(f"Downloading model: {dest_path}")
    import gdown

    gdown.download(id=file_id, output=str(dest_path), quiet=False, fuzzy=True)
    size = dest_path.stat().st_size
    print(f"Download complete: {dest_path} ({size // 1024 // 1024}MB)")


print("Checking model files...")
download_model("1LhQ6AKWzhhG-w880Fs_G2HJagtXvQ-YK", MODELS_DIR / "weights_detector.pt")
download_model("1XK8-NGDEHxvopSaxElXvdYI1CXtgZgN3", MODELS_DIR / "weights_classifier.pth")
print("Models ready")
init_db()

print("Loading OCR models...")
from src.munja.ocr.detect import extract_ancient_text_with_models, load_models

OCR_MODELS = load_models()
print("OCR models loaded")


def run_ocr(image_path):
    return extract_ancient_text_with_models(image_path, OCR_MODELS)


@app.route("/ocr", methods=["POST"])
def ocr():
    if "image" not in request.files:
        return jsonify({"error": "이미지가 없습니다."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "파일명이 비어 있습니다."}), 400

    suffix = os.path.splitext(file.filename)[-1] or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        result = run_ocr(tmp_path)
        return jsonify({"text": result})
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.route("/translate", methods=["POST"])
def translate():
    if "image" not in request.files:
        return jsonify({"error": "이미지가 없습니다."}), 400

    file = request.files["image"]
    ocr_text = request.form.get("text", "").strip()
    target_lang = request.form.get("lang", "ko")
    username = request.form.get("username", "익명").strip() or "익명"

    if not ocr_text:
        return jsonify({"error": "번역할 텍스트가 없습니다."}), 400

    suffix = os.path.splitext(file.filename)[-1] or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    with open(tmp_path, "rb") as handle:
        img_bytes = handle.read()
    img_data = base64.b64encode(img_bytes).decode("utf-8")
    mime = "image/png" if suffix.lower() == ".png" else "image/jpeg"
    image_data_url = f"data:{mime};base64,{img_data}"

    try:
        from openai import OpenAI
        from src.munja.translate.translate_ocr import translate_image

        result = translate_image(
            image_path=tmp_path,
            ocr_text=ocr_text,
            api_key=os.environ.get("OPENAI_API_KEY"),
        )

        lang_map = {
            "ko": result.get("modern_korean", ""),
            "en": result.get("english_translation", ""),
        }
        translated_text = lang_map.get(target_lang, "")

        if target_lang in ("ja", "zh"):
            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            lang_name = "Japanese" if target_lang == "ja" else "Chinese"
            base_text = result.get("modern_korean", "")
            resp = client.chat.completions.create(
                model="gpt-5.2",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"Translate the following modern Korean text into natural {lang_name}. "
                            "Output only the translation."
                        ),
                    },
                    {"role": "user", "content": base_text},
                ],
                max_completion_tokens=1024,
            )
            translated_text = resp.choices[0].message.content.strip()

        try:
            conn = get_db()
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO translations (username, ocr_text, translated_text, lang, image_data)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (username, ocr_text[:500], translated_text[:1000], target_lang, image_data_url),
            )
            conn.commit()
            cur.close()
            conn.close()
        except Exception as db_err:
            print(f"DB insert failed: {db_err}")

        return jsonify(
            {
                "text": translated_text,
                "ancient_text_corrected": result.get("ancient_text_corrected", ocr_text),
                "modern_korean": result.get("modern_korean", ""),
                "english_translation": result.get("english_translation", ""),
                "notes": result.get("notes", ""),
            }
        )
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.route("/history", methods=["GET"])
def history():
    try:
        conn = get_db()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            """
            SELECT id, username, ocr_text, translated_text, lang, image_data, created_at
            FROM translations
            ORDER BY created_at DESC
            LIMIT 6
            """
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()

        result = []
        for row in rows:
            result.append(
                {
                    "id": row["id"],
                    "user": row["username"],
                    "ocr": row["ocr_text"],
                    "tr": row["translated_text"],
                    "lang": row["lang"],
                    "image": row["image_data"] or "",
                    "time": row["created_at"].strftime("%Y-%m-%d %H:%M"),
                }
            )
        return jsonify(result)
    except Exception:
        traceback.print_exc()
        return jsonify([]), 500


@app.route("/seed", methods=["POST"])
def seed():
    samples = [
        {
            "username": "김민서",
            "ocr_text": "샘플 옛한글 원문 1",
            "translated_text": "샘플 현대 한국어 번역 1",
            "lang": "ko",
        },
        {
            "username": "이서윤",
            "ocr_text": "샘플 옛한글 원문 2",
            "translated_text": "샘플 현대 한국어 번역 2",
            "lang": "ko",
        },
        {
            "username": "Daniel",
            "ocr_text": "샘플 옛한글 원문 3",
            "translated_text": "Sample English translation 3",
            "lang": "en",
        },
    ]

    try:
        conn = get_db()
        cur = conn.cursor()
        for sample in samples:
            cur.execute(
                """
                INSERT INTO translations (username, ocr_text, translated_text, lang)
                VALUES (%s, %s, %s, %s)
                """,
                (
                    sample["username"],
                    sample["ocr_text"],
                    sample["translated_text"],
                    sample["lang"],
                ),
            )
        conn.commit()
        cur.close()
        conn.close()
        return jsonify({"message": "샘플 데이터 3건이 추가되었습니다."})
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


@app.route("/")
def index():
    return render_template("translator-C.html")


if __name__ == "__main__":
    print("=" * 50)
    print("  Munja server start")
    print("=" * 50)
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
