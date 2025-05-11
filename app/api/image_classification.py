from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
from typing import List

router = APIRouter()

class ClassificationResult(BaseModel):
    category_name: str
    score: float

class ClassificationResponse(BaseModel):
    status: str
    results: List[ClassificationResult]

@router.post("/classify-image", response_model=ClassificationResponse)
async def classify_image(file: UploadFile = File(...)):
    try:
        # 이미지 파일 읽기
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="이미지를 읽을 수 없습니다.")
        
        # MediaPipe 이미지로 변환
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
        
        # 모델 로드 및 분류
        base_options = python.BaseOptions(model_asset_path='app/models/efficientnet_lite0.tflite')
        options = vision.ImageClassifierOptions(
            base_options=base_options, max_results=2)
        classifier = vision.ImageClassifier.create_from_options(options)
        
        # 이미지 분류
        classification_result = classifier.classify(mp_image)
        
        # 결과 처리
        results = []
        if classification_result.classifications:
            for classification in classification_result.classifications:
                for category in classification.categories:
                    results.append(ClassificationResult(
                        category_name=category.category_name,
                        score=float(category.score)
                    ))
        
        return ClassificationResponse(
            status="success",
            results=results
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 