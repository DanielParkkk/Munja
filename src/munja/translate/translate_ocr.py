import argparse
import base64
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


DEFAULT_MODEL = "gpt-5.2"
API_URL = "https://api.openai.com/v1/chat/completions"

SYSTEM_PROMPT = """You are a translation assistant for historical Korean documents.
You will receive:
1. An OCR transcription in old Hangul / historical Korean.
2. The source document image.

Tasks:
- Correct obvious OCR mistakes by checking the image when possible.
- Translate the old Korean into natural modern standard Korean.
- Then translate the modern Korean into natural English.
- Preserve uncertainty honestly when the OCR is ambiguous.

Return only valid JSON with this exact schema:
{
  "ancient_text_corrected": "string",
  "modern_korean": "string",
  "english_translation": "string",
  "notes": "short string with ambiguity notes, or empty string"
}
"""


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue

        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        os.environ.setdefault(key, value)


def image_to_data_url(image_path: Path) -> str:
    suffix = image_path.suffix.lower()
    mime_type = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }.get(suffix)
    if mime_type is None:
        raise ValueError(f"Unsupported image format: {image_path.suffix}")

    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def parse_json_content(raw_content: str) -> Dict[str, Any]:
    text = raw_content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return json.loads(text)


def translate_with_gpt(
    image_path: Path,
    ancient_text: str,
    api_key: str,
    model: str = DEFAULT_MODEL,
    timeout: int = 120,
) -> Dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    user_prompt = (
        "Use the OCR transcription as the primary clue, and the image as visual evidence.\n"
        "Translate in two steps: old Korean -> modern Korean -> English.\n"
        "Return JSON only.\n\n"
        f"OCR transcription:\n{ancient_text}"
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_to_data_url(image_path)},
                    },
                ],
            },
        ],
        "max_completion_tokens": 1200,
        "temperature": 0.2,
    }

    response = requests.post(API_URL, headers=headers, json=payload, timeout=timeout)
    if not response.ok:
        raise RuntimeError(f"OpenAI API error {response.status_code}: {response.text}")
    data = response.json()

    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise RuntimeError(f"Unexpected OpenAI response: {json.dumps(data, ensure_ascii=False)}") from exc

    result = parse_json_content(content)
    result.setdefault("ancient_text_corrected", ancient_text)
    result.setdefault("modern_korean", "")
    result.setdefault("english_translation", "")
    result.setdefault("notes", "")
    return result


def get_ancient_text(image_path: Path, supplied_text: Optional[str]) -> str:
    if supplied_text:
        return supplied_text.strip()

    from src.munja.ocr.detect import extract_ancient_text

    return extract_ancient_text(str(image_path))


def build_output_payload(image_path: Path, ancient_text: str, translation: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "image_path": str(image_path),
        "ancient_text_raw": ancient_text,
        "ancient_text_corrected": translation.get("ancient_text_corrected", ancient_text),
        "modern_korean": translation.get("modern_korean", ""),
        "english_translation": translation.get("english_translation", ""),
        "notes": translation.get("notes", ""),
    }


def translate_image(
    image_path: str,
    ocr_text: Optional[str] = None,
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
) -> Dict[str, Any]:
    load_env_file(Path(__file__).resolve().parents[3] / ".env")

    resolved_image_path = Path(image_path).resolve()
    if not resolved_image_path.exists():
        raise FileNotFoundError(f"Image file not found: {resolved_image_path}")

    resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not resolved_api_key:
        raise EnvironmentError("OPENAI_API_KEY was not found in .env or api_key was not provided.")

    ancient_text = get_ancient_text(resolved_image_path, ocr_text)
    translation = translate_with_gpt(
        image_path=resolved_image_path,
        ancient_text=ancient_text,
        api_key=resolved_api_key,
        model=model,
    )
    return build_output_payload(resolved_image_path, ancient_text, translation)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run old-Hangul OCR and translate it to modern Korean and English with GPT."
    )
    parser.add_argument("--image", required=True, help="Path to the source image")
    parser.add_argument(
        "--ocr-text",
        default=None,
        help="Optional old Korean transcription. If omitted, the OCR pipeline runs first.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenAI model name. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON output path. Default: <image_stem>.translation.json",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="OpenAI API key. If omitted, OPENAI_API_KEY is used.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = translate_image(
        image_path=args.image,
        ocr_text=args.ocr_text,
        api_key=args.api_key,
        model=args.model,
    )

    image_path = Path(output["image_path"])
    output_path = Path(args.output) if args.output else image_path.with_suffix(".translation.json")
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"\nSaved translation JSON to: {output_path}")


if __name__ == "__main__":
    main()
