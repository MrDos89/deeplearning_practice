from fastapi import APIRouter
from pydantic import BaseModel
from transformers import pipeline

router = APIRouter()

# 분류 파이프라인 (한글 토픽 분류, 예시로 zero-shot 사용)
classifier = pipeline("zero-shot-classification", model="joeddav/xlm-roberta-large-xnli")
CANDIDATE_LABELS = ["스포츠", "정치", "경제", "사회", "문화", "기술", "연예"]

class TextRequest(BaseModel):
    text: str

@router.post("/classify")
def classify_text(request: TextRequest):
    result = classifier(request.text, candidate_labels=CANDIDATE_LABELS)
    # 가장 높은 score의 label만 반환
    best_label = result["labels"][0]
    best_score = result["scores"][0]
    return {
        "input": request.text,
        "topic": best_label,
        "score": best_score,
        "all_scores": dict(zip(result["labels"], result["scores"]))
    } 