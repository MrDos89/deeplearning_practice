from fastapi import APIRouter
from pydantic import BaseModel
from transformers import pipeline

router = APIRouter()

translator = pipeline("translation_en_to_ko", model="Helsinki-NLP/opus-mt-en-ROMANCE")

class TranslationRequest(BaseModel):
    text: str

@router.post("/translate")
def translate_text(request: TranslationRequest):
    result = translator(request.text, max_length=512)
    translation = result[0]['translation_text']
    return {
        "input": request.text,
        "translation": translation
    } 