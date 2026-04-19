"""
Công cụ tự động cào dữ liệu HÀNG LOẠT từ Wikipedia.
Bạn chỉ cần thêm các từ khoá vào mảng KEYWORDS bên dưới.
"""

from pathlib import Path
from src.scraper_service import ScraperService

# ========================================================
# NHẬP DANH SÁCH CÁC TỪ KHOÁ CẦN CÀO TỪ WIKI VÀO MẢNG NÀY:
# Thêm các từ khoá khác vào đây, mỗi từ nằm trong dấu ngoặc kép và cách nhau bằng dấu phẩy
# ========================================================
KEYWORDS = [
    "Hội An",
    "Thành cổ Quảng Trị"
]

def main():
    print("=" * 60)
    print(f"CÔNG CỤ CÀO WIKIPEDIA — Đang xử lý {len(KEYWORDS)} từ khoá")
    print("=" * 60)

    base_dir = Path(__file__).resolve().parent
    scraper = ScraperService(base_dir=base_dir)

    success_count = 0
    fail_count = 0

    for i, keyword in enumerate(KEYWORDS, 1):
        if not keyword.strip():
            continue
            
        print(f"\n[{i}/{len(KEYWORDS)}] Đang cào: '{keyword}'...")
        saved_path = scraper.scrape_wikipedia(keyword=keyword, lang="vi")
        
        if saved_path:
            print(f"   ✅ Đã lưu: {saved_path}")
            success_count += 1
        else:
            print(f"   ❌ Thất bại hoặc không có nội dung.")
            fail_count += 1

    print("\n" + "=" * 60)
    print(f"TỔNG KẾT: Thành công {success_count} | Thất bại {fail_count}")
    print("Chạy lệnh 'python main.py ingest' để nạp toàn bộ vào hệ thống AI.")

if __name__ == "__main__":
    main()
