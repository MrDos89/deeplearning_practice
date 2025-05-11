from googletrans import Translator
# 번역기 초기화
translator = Translator()
# 번역할 영어 텍스트
english_text = """
Artificial Intelligence (AI) is rapidly changing our world. 
It helps us in many areas like healthcare, education, and business. 
Machine learning and deep learning are important parts of AI technology.
"""
# 번역 수행
result = translator.translate(english_text, dest='ko')
print("\n원본 텍스트:")
print(english_text)
print("\n번역 결과:")
print(result.text)