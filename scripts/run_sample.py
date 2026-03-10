import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from src.munja.ocr.detect import extract_ancient_text
from src.munja.translate.translate_ocr import translate_image


def main() -> None:
    sample_path = PROJECT_ROOT / "samples" / "images" / "옛한글.png"

    ancient_text = extract_ancient_text(str(sample_path))
    result = translate_image(str(sample_path), ocr_text=ancient_text)
    print(result)


if __name__ == "__main__":
    main()
