from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
import io

router = APIRouter()

def visualize(image, detection_result):
    if not detection_result.detections:
        # 얼굴이 감지되지 않은 경우 메시지 표시
        cv2.putText(image, "No faces detected", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return image
        
    MARGIN = 10
    ROW_SIZE = 10
    FONT_SIZE = 1
    FONT_THICKNESS = 1
    TEXT_COLOR = (255, 0, 0)
    
    for detection in detection_result.detections:
        # 바운딩 박스 그리기
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 3)

        # 키포인트 그리기
        if hasattr(detection, 'keypoints'):
            for keypoint in detection.keypoints:
                keypoint_px = (int(keypoint.x * image.shape[1]), int(keypoint.y * image.shape[0]))
                cv2.circle(image, keypoint_px, 2, (0, 255, 0), -1)

        # 라벨과 점수 그리기
        if detection.categories and len(detection.categories) > 0:
            category = detection.categories[0]
            category_name = category.category_name
            probability = round(category.score, 2)
            result_text = f"{category_name} ({probability})"
            text_location = (MARGIN + bbox.origin_x, MARGIN + ROW_SIZE + bbox.origin_y)
            cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                        FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)
    
    return image

@router.post("/detect-faces")
async def detect_faces(file: UploadFile = File(...)):
    try:
        # 이미지 파일 읽기
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="이미지를 읽을 수 없습니다.")
        
        # MediaPipe 이미지로 변환
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
        
        # 모델 로드 및 얼굴 감지
        base_options = python.BaseOptions(model_asset_path='app/models/blaze_face_short_range.tflite')
        options = vision.FaceDetectorOptions(base_options=base_options)
        detector = vision.FaceDetector.create_from_options(options)
        
        # 얼굴 감지
        detection_result = detector.detect(mp_image)
        
        # 결과 시각화
        img_copy = img.copy()
        annotated_image = visualize(img_copy, detection_result)
        
        # 이미지를 바이트로 변환
        _, img_encoded = cv2.imencode('.jpg', annotated_image)
        img_bytes = img_encoded.tobytes()
        
        return StreamingResponse(io.BytesIO(img_bytes), media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 