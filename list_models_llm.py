import os
from dotenv import load_dotenv
load_dotenv()
# pyrefly: ignore [missing-import]
import google.generativeai as genai

def main():
    api_key = os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=api_key)
    print("Danh sách các model LLM khả dụng cho API Key của bạn:")
    try:
        models = genai.list_models()
        for m in models:
            if 'generateContent' in m.supported_generation_methods:
                print(f" - {m.name}")
    except Exception as e:
        print(f"Lỗi: {e}")

if __name__ == "__main__":
    main()
