import asyncio
from services.rag_engine import process_query
from services.doc_manager import hide_document_background, delete_document_background
from models.schemas import QueryRequest, VisibilityRequest, DeleteRequest
import logging

logging.basicConfig(level=logging.INFO)

async def test_search(desc=""):
    print(f"\n==============================================")
    print(f"🤖 {desc}")
    req = QueryRequest(
        question="Tài liệu này nói về chủ đề gì?",
        conversation_id="conv_1",
        history=[]
    )
    try:
        response = await process_query(req)
        print("✅ TRẢ LỜI TỪ AI:\n" + response.answer)
        if response.citations:
            print(f"\n📚 NGUỒN TRÍCH DẪN (Tìm thấy {len(response.citations)} đoạn tài liệu liên quan):")
            for cit in response.citations:
                print(f"  - Doc ID: {cit.doc_id} | Đoạn văn: {cit.snippet[:100]}...")
        else:
            print("\n❌ Không tìm thấy thông tin.")
    except Exception as e:
        print(f"❌ LỖI RAG: {e}")

async def main():
    print("🚀 BẮT ĐẦU KIỂM THỬ TOÀN BỘ CÁC CHỨC NĂNG RAG...")
    
    # 1. Test Search (Khi tài liệu đang hiển thị)
    await test_search("BƯỚC 1: TEST TÌM KIẾM RAG LẦN 1")
    
    # 2. Test Hide (Ẩn tài liệu)
    print("\n==============================================")
    print("🔒 BƯỚC 2: ẨN TÀI LIỆU (doc_real_test)")
    from services.doc_manager import unhide_document_background
    await hide_document_background("doc_real_test", "job_hide", "http://localhost/cb")
    print("✅ Đã ẩn tài liệu thành công!")
    
    # 3. Test Search (Sau khi ẩn)
    await test_search("BƯỚC 3: TEST TÌM KIẾM RAG (KHI TÀI LIỆU BỊ ẨN)")
    
    # 4. Test Unhide (Hiện lại tài liệu)
    print("\n==============================================")
    print("🔓 BƯỚC 4: HIỆN LẠI TÀI LIỆU (doc_real_test)")
    await unhide_document_background("doc_real_test", "job_unhide", "http://localhost/cb")
    print("✅ Đã hiện lại tài liệu thành công!")
    
    # 5. Test Delete (Xóa tài liệu)
    print("\n==============================================")
    print("🗑️ BƯỚC 5: XÓA VĨNH VIỄN TÀI LIỆU (doc_real_test)")
    await delete_document_background("doc_real_test", "job_delete", "http://localhost/cb")
    print("✅ Đã xóa tài liệu khỏi Qdrant!")
    
    # 6. Test Search (Sau khi xóa)
    await test_search("BƯỚC 6: TEST TÌM KIẾM RAG (KHI TÀI LIỆU ĐÃ BỊ XÓA)")
    print("\n🎉 KIỂM THỬ HOÀN TẤT!")

if __name__ == "__main__":
    asyncio.run(main())
