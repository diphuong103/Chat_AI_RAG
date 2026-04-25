"""
Công cụ tự động cào dữ liệu HÀNG LOẠT từ các đường link Website.
Bạn chỉ cần thêm các đường link (URL) vào mảng URLS bên dưới.
"""

from pathlib import Path
from src.scraper_service import ScraperService

# ========================================================
# NHẬP DANH SÁCH CÁC ĐƯỜNG LINK (URL) CẦN CÀO VÀO MẢNG NÀY:
# Thêm các link URL khác vào đây, mỗi link nằm trong dấu ngoặc kép và cách nhau bằng dấu phẩy
# ========================================================
URLS = [
    "https://vinpearl.com/vi/40-dia-diem-du-lich-viet-nam-noi-tieng-nhat-dinh-nen-den-mot-lan",

]

def main():
    print("=" * 60)
    print(f"CÔNG CỤ CÀO WEBSITE — Đang xử lý {len(URLS)} đường link")
    print("=" * 60)

    base_dir = Path(__file__).resolve().parent
    scraper = ScraperService(base_dir=base_dir)

    success_count = 0
    fail_count = 0

    for i, url in enumerate(URLS, 1):
        if not url.strip():
            continue
            
        print(f"\n[{i}/{len(URLS)}] Đang cào URL: '{url}'...")
        saved_path = scraper.scrape_url(url=url)
        
        if saved_path:
            print(f"   ✅ Đã lưu: {saved_path}")
            success_count += 1
        else:
            print(f"   ❌ Thất bại hoặc website chặn lấy nội dung.")
            fail_count += 1

    print("\n" + "=" * 60)
    print(f"TỔNG KẾT: Thành công {success_count} | Thất bại {fail_count}")
    print("Chạy lệnh 'python main.py ingest' để nạp toàn bộ vào hệ thống AI.")

if __name__ == "__main__":
    main()
