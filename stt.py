import os
import subprocess
import tempfile
from transformers import pipeline

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
        print("ffmpeg 변환 중 오류 발생:", e)
        return None

# 한국어 음성 인식 파이프라인 (Whisper-base, 토큰 없이 사용 가능)
stt = pipeline("automatic-speech-recognition", model="openai/whisper-base", device=-1)

file_path = input("한국어 음성 파일 경로를 입력하세요 (예: sample.wav): ")

wav_path = convert_to_wav(file_path)
if wav_path is None:
    print("오디오 파일 변환에 실패했습니다. ffmpeg 설치 및 파일 경로를 확인하세요.")
else:
    result = stt(wav_path, generate_kwargs={"language": "korean"})
    text = result["text"]

    print("\n[음성 인식 결과]")
    print(f"파일: {file_path}")
    print(f"텍스트: {text}")

    os.remove(wav_path)
