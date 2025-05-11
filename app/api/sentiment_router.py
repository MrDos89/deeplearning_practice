from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from text_classification import analyze_sentiment

router = APIRouter()

class TextRequest(BaseModel):
    text: str
    model: str = "beomi/KcELECTRA-base-v2022"

@router.post("/sentiment")
async def analyze_sentiment_api(request: TextRequest):
    try:
        result = analyze_sentiment(request.text, request.model)
        return {
            "status": "success",
            "input_text": request.text,
            "model": request.model,
            "result": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 