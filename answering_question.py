import os
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

# 허깅페이스 토큰 입력 (선택)
hf_token = input("허깅페이스 토큰을 입력하세요(엔터시 생략): ").strip()
if hf_token:
    os.environ["HF_TOKEN"] = hf_token
    token_arg = {"token": hf_token}
else:
    token_arg = {}

# 토큰 없이 사용 가능한 공개 QA 모델
MODEL_NAME = "deepset/xlm-roberta-base-squad2"

question = input("질문을 입력하세요: ")
context = input("문맥(지문)을 입력하세요: ")

model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME, **token_arg)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, **token_arg)
inputs = tokenizer(question, context, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1
    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end])
    )

print("\n[질문응답 결과]")
print(f"질문: {question}")
print(f"문맥: {context}")
print(f"답변: {answer.strip()}")