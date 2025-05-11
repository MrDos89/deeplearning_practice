from fastapi import FastAPI
from app.routers import example
from app.api.test_classification import router as test_classification_router
from app.api.translation import router as translation_router
from app.api.answering_question import router as answering_question_router
from app.api.stt import router as stt_router
from app.api.sentiment_router import router as sentiment_router
from app.api.image_classification import router as image_classification_router
from app.api.object_detection import router as object_detection_router
from app.api.face_landmarks import router as face_landmarks_router
from app.api.face_detection import router as face_detection_router
from app.api.hand_landmarks import router as hand_landmarks_router
from app.api.pose_landmarks import router as pose_landmarks_router

app = FastAPI()

app.include_router(example.router)
app.include_router(test_classification_router, prefix="/api", tags=["Text Classification"])
app.include_router(translation_router, prefix="/api", tags=["Translation"])
app.include_router(answering_question_router, prefix="/api", tags=["Question Answering"])
app.include_router(stt_router, prefix="/api", tags=["Speech To Text"])
app.include_router(sentiment_router, prefix="/api", tags=["Sentiment Analysis"])
app.include_router(image_classification_router, prefix="/api", tags=["Image Classification"])
app.include_router(object_detection_router, prefix="/api", tags=["Object Detection"])
app.include_router(face_landmarks_router, prefix="/api", tags=["Face Landmarks"])
app.include_router(face_detection_router, prefix="/api", tags=["Face Detection"])
app.include_router(hand_landmarks_router, prefix="/api", tags=["Hand Landmarks"])
app.include_router(pose_landmarks_router, prefix="/api", tags=["Pose Landmarks"])

@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI!"} 