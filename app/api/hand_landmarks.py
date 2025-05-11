from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import cv2
import numpy as np
import io

router = APIRouter()

def draw_landmarks_on_image(rgb_image, detection_result):
    if not detection_result.hand_landmarks:
        # 손이 감지되지 않은 경우 메시지 표시
        cv2.putText(rgb_image, "No hands detected", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return rgb_image
        
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # 손 랜드마크 그리기
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style())

        # 손 방향 표시
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - 10
        cv2.putText(annotated_image, f"{handedness[0].category_name}",
                    (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    1, (88, 205, 54), 1, cv2.LINE_AA)

    return annotated_image

@router.post("/detect-hand-landmarks")
async def detect_hand_landmarks(file: UploadFile = File(...)):
    try:
        # 이미지 파일 읽기
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="이미지를 읽을 수 없습니다.")
        
        # MediaPipe 이미지로 변환
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
        
        # 모델 로드 및 손 랜드마크 감지
        base_options = python.BaseOptions(model_asset_path='app/models/hand_landmarker.task')
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2)
        detector = vision.HandLandmarker.create_from_options(options)
        
        # 손 랜드마크 감지
        detection_result = detector.detect(mp_image)
        
        # 결과 시각화
        annotated_image = draw_landmarks_on_image(img, detection_result)
        
        # 이미지를 바이트로 변환
        _, img_encoded = cv2.imencode('.jpg', annotated_image)
        img_bytes = img_encoded.tobytes()
        
        return StreamingResponse(io.BytesIO(img_bytes), media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 