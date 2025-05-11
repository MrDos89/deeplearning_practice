from fastapi import APIRouter
from pydantic import BaseModel
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch

router = APIRouter()

MODEL_NAME = "deepset/xlm-roberta-base-squad2"
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class QARequest(BaseModel):
    question: str
    context: str

@router.post("/answer")
def answer_question(request: QARequest):
    inputs = tokenizer(request.question, request.context, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        answer_start = torch.argmax(outputs.start_logits)
        answer_end = torch.argmax(outputs.end_logits) + 1
        answer = tokenizer.convert_tokens_to_string(
            tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end])
        )
    return {
        "question": request.question,
        "context": request.context,
        "answer": answer.strip()
    } 