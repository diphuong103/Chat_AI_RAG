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
    # ==========================================================
    # 1. ĐỊA ĐIỂM & CẨM NANG DU LỊCH
    # ==========================================================
    "https://vinpearl.com/vi/40-dia-diem-du-lich-viet-nam-noi-tieng-nhat-dinh-nen-den-mot-lan",
    "https://vinpearl.com/vi/du-lich-mien-bac",
    "https://vinpearl.com/vi/du-lich-mien-trung",
    "https://vinpearl.com/vi/du-lich-mien-nam",

    # ==========================================================
    # 2. LỄ HỘI
    # ==========================================================
    "https://toplist.vn/top-list/le-hoi-truyen-thong-noi-tieng-nhat-viet-nam-1842.html",
    "https://mytour.vn/vi/blog/bai-viet/top-14-le-hoi-truyen-thong-loi-cuon-nhat-viet-nam.html",
    "https://vinpearl.com/vi/tong-hop-cac-le-hoi-o-viet-nam-noi-tieng-dac-sac-nhat-3-mien",
    "https://vietsensetravel.com/22-le-hoi-noi-tieng-viet-nam-n.html",
    "https://dulichfree.com/cac-le-hoi-o-viet-nam/",

    # ==========================================================
    # 3. VISA & THỰC TIỄN KHÁCH QUỐC TẾ
    # ==========================================================
    "https://vietnam.travel/plan-your-trip/visa-requirements",
    "https://vietnam.travel/plan-your-trip",

    # ==========================================================
    # 4. DI CHUYỂN & SÂN BAY
    # ==========================================================
    "https://noibaiairport.vn",
    "https://vinpearl.com/vi/san-bay-quoc-te-noi-bai",
    "https://vinpearl.com/vi/san-bay-quoc-te-tan-son-nhat",
    "https://vinpearl.com/vi/san-bay-quoc-te-da-nang",

    # ==========================================================
    # 5. GIÁ VÉ THAM QUAN
    # ==========================================================
    "https://dulichkhatvongviet.com/gia-ve-tham-quan-du-lich-viet-nam/",

    # ==========================================================
    # 6. THỜI TIẾT & MÙA DU LỊCH
    # ==========================================================
    "https://vinpearl.com/vi/thoi-tiet-viet-nam",
    "https://vinpearl.com/vi/thoi-diem-du-lich-viet-nam",

    # ==========================================================
    # 7. LƯU TRÚ & KHÁCH SẠN
    # ==========================================================
    "https://vinpearl.com/vi/cac-loai-hinh-luu-tru-du-lich",

    # ==========================================================
    # 8. ẨM THỰC (bổ sung thực tế, quán ăn, giá cả)
    # ==========================================================
    "https://vinpearl.com/vi/am-thuc-viet-nam",
    "https://vinpearl.com/vi/mon-an-viet-nam-noi-tieng",

    # ==========================================================
    # 9. VĂN HÓA & PHONG TỤC
    # ==========================================================
    "https://vinpearl.com/vi/van-hoa-viet-nam",
    "https://vietnam.travel/things-to-do",
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
