import argparse
from transformers import pipeline


def get_classifier(model_name: str):
    return pipeline("sentiment-analysis", model=model_name)


def analyze_sentiment(text: str, model_name: str):
    classifier = get_classifier(model_name)
    return classifier(text)


def main():
    parser = argparse.ArgumentParser(description="한글 감정분석 실행기")
    parser.add_argument("--text", type=str, required=True, help="분석할 문장")
    parser.add_argument(
        "--model",
        type=str,
        default="beomi/KcELECTRA-base-v2022",
        help="감정분석에 사용할 허깅페이스 모델명 (기본값: beomi/KcELECTRA-base-v2022)",
    )
    args = parser.parse_args()

    result = analyze_sentiment(args.text, args.model)
    print("\n[감정분석 결과]")
    print(f"입력 문장: {args.text}")
    print(f"모델명: {args.model}")
    print(f"분석 결과: {result}\n")


if __name__ == "__main__":
    main()