from fastapi import APIRouter

router = APIRouter()

@router.get("/")
def root():
    return {"message": "Hello, FastAPI!"}

@router.get("/ping")
def ping():
    return {"message": "pong"} 