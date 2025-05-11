from transformers import pipeline

# klue/bert-base NER 라벨 매핑 (예시)
ENTITY_LABELS = {
    'LABEL_0': '인물(PS)',
    'LABEL_1': '기관(ORG)',
    'LABEL_2': '지명(LOC)',
    'LABEL_3': '작품(WORK)',
    'LABEL_4': '날짜(DAT)',
    'LABEL_5': '시간(TIM)',
    'LABEL_6': '수량(NUM)',
    'LABEL_7': '이벤트(EVT)',
    'LABEL_8': '용어(TERM)'
}

text = input("개체명 인식을 할 문장을 입력하세요: ")

classifier = pipeline("ner", model="klue/bert-base")
result = classifier(text)

print("\n[개체명 인식 결과]")
if not result:
    print("추출된 개체명이 없습니다.")
else:
    for entity in result:
        label = ENTITY_LABELS.get(entity['entity'], entity['entity'])
        print(f"- 단어: '{entity['word']}' | 엔티티: {label} | 점수: {entity['score']:.4f}")