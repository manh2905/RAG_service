import asyncio
import os
from dotenv import load_dotenv
load_dotenv()

from llama_index.embeddings.gemini import GeminiEmbedding

async def main():
    api_key = os.getenv("GOOGLE_API_KEY")
    print(f"Sử dụng Key: {api_key[:5]}...{api_key[-5:]}")
    
    try:
        # Khởi tạo y hệt trong hệ thống
        embed_model = GeminiEmbedding(
            model_name="models/embedding-001",
            api_key=api_key
        )
        print("Khởi tạo model thành công, đang thử tạo embedding...")
        
        # Test lấy embedding
        result = await embed_model.aget_text_embedding("Xin chào Việt Nam")
        print(f"✅ THÀNH CÔNG! Độ dài vector: {len(result)}")
        
    except Exception as e:
        import traceback
        print("❌ LỖI RỒI:")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
