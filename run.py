from detect_v5 import extract_ancient_text
from translate_ocr import translate_image

ancien_text = extract_ancient_text('sample_images\옛한글.png')

result = translate_image(
      "sample_images/옛한글.png",
      ocr_text= ancien_text
  )

print(result)