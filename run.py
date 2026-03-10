from detect_v5 import extract_ancient_text
from translate_ocr import translate_image

ancien_text = extract_ancient_text('sample_images\옛한글.png')

result = translate_image(
      "sample_images/옛한글.png",
      ocr_text= ancien_text
  )

print('엣한글 원문 :',  result['ancient_text_raw'])
print('현대어 번역 :',  result['modern_korean'])
print('영어 번역 :',  result['english_translation'])
print('일본어 번역 :', result['japanese_translation'])
print('중국어 번역 :', result['chinese_translation'])
