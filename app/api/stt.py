from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
from transformers import pipeline
import os
import tempfile
import subprocess

router = APIRouter()

stt = pipeline("automatic-speech-recognition", model="openai/whisper-base", device=-1)

def convert_to_wav(input_path):
    temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_wav.close()
    output_path = temp_wav.name
    command = [
        "ffmpeg", "-y", "-i", input_path,
        "-ar", "16000", "-ac", "1", output_path
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return output_path
    except Exception as e:
        return None

@router.post("/stt")
def speech_to_text(file: UploadFile = File(...)):
    # 업로드 파일을 임시 파일로 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[-1]) as temp_audio:
        temp_audio.write(file.file.read())
        temp_audio_path = temp_audio.name

    # wav로 변환
    wav_path = convert_to_wav(temp_audio_path)
    if wav_path is None:
        os.remove(temp_audio_path)
        return JSONResponse(status_code=400, content={"error": "오디오 파일 변환에 실패했습니다. ffmpeg 설치 및 파일 포맷을 확인하세요."})

    # 음성 인식
    result = stt(wav_path, generate_kwargs={"language": "korean"})
    text = result["text"]

    # 임시 파일 삭제
    os.remove(temp_audio_path)
    os.remove(wav_path)

    return {"text": text} 