import asyncio
from services.rag_engine import query_rag
from models.schemas import QueryRequest

async def main():
    print("--------------------------------------------------")
    print("🤖 ĐANG KHỞI ĐỘNG HỆ THỐNG TÌM KIẾM RAG...")
    
    # Giả lập câu hỏi từ Node.js
    req = QueryRequest(
        question="Nội dung chính của tài liệu này là gì? Trả lời chi tiết giúp tôi.",
        conversation_id="conv_test_123",
        history=[]
    )
    
    print(f"👤 Câu hỏi: {req.question}")
    print("⏳ Đang tìm kiếm trong Qdrant và nhờ Gemini suy luận...")
    print("--------------------------------------------------\n")
    
    try:
        response = await query_rag(req)
        
        print("✅ TRẢ LỜI TỪ AI:\n")
        print(response.answer)
        print("\n--------------------------------------------------")
        
        if response.citations:
            print(f"📚 NGUỒN TRÍCH DẪN (Tìm thấy {len(response.citations)} đoạn tài liệu liên quan):")
            for idx, cit in enumerate(response.citations):
                print(f"  [{idx+1}] Trang {cit.page_number} | Doc ID: {cit.doc_id}")
                print(f"      Trích đoạn: {cit.snippet[:100]}...\n")
        else:
            print("❌ Không tìm thấy thông tin liên quan trong tài liệu (RAG không hoạt động hoặc hỏi ngoài lề).")
            
        print("📊 THỐNG KÊ TOKEN:")
        print(response.usage)
        
    except Exception as e:
        print(f"❌ LỖI: {e}")

if __name__ == "__main__":
    asyncio.run(main())
