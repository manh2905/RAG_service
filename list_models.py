import os
from dotenv import load_dotenv
load_dotenv()
# pyrefly: ignore [missing-import]
import google.generativeai as genai

def main():
    api_key = os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=api_key)
    print("Danh sách các model khả dụng cho API Key của bạn:")
    try:
        models = genai.list_models()
        found_embedding = False
        for m in models:
            if 'embedContent' in m.supported_generation_methods:
                print(f" - {m.name}")
                found_embedding = True
                
        if not found_embedding:
            print("Không tìm thấy model nào hỗ trợ embedding!")
    except Exception as e:
        print(f"Lỗi: {e}")

if __name__ == "__main__":
    main()
