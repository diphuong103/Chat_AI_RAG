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
    # ==========================================================
    # 1. ĐỊA ĐIỂM & ĐIỂM ĐẾN — Toàn Việt Nam
    # ==========================================================
    # --- Tổng quan ---
    "Du lịch Việt Nam",
    "Danh sách Di sản thế giới tại Việt Nam",
    # --- Miền Bắc ---
    "Hà Nội",
    "Vịnh Hạ Long",
    "Sa Pa",
    "Ninh Bình",
    "Tràng An",
    "Tam Cốc – Bích Động",
    "Hải Phòng",
    "Đảo Cát Bà",
    "Lạng Sơn",
    "Hà Giang",
    "Cao Bằng",
    "Mai Châu",
    "Mộc Châu",
    "Tuyên Quang",
    "Yên Bái",
    # --- Miền Trung ---
    "Huế",
    "Đà Nẵng",
    "Hội An",
    "Thánh địa Mỹ Sơn",
    "Phong Nha – Kẻ Bàng",
    "Nha Trang",
    "Quy Nhơn",
    "Đà Lạt",
    "Bà Nà Hills",
    "Phan Thiết",
    "Mũi Né",
    "Quảng Bình",
    "Quảng Ngãi",
    "Kon Tum",
    "Pleiku",
    "Buôn Ma Thuột",
    # --- Miền Nam ---
    "Thành phố Hồ Chí Minh",
    "Phú Quốc",
    "Cần Thơ",
    "Vũng Tàu",
    "Côn Đảo",
    "Châu Đốc",
    "Đồng bằng sông Cửu Long",
    "Mỹ Tho",
    "Bến Tre",
    "Long An",

    # ==========================================================
    # 2. ẨM THỰC & ĐẶC SẢN
    # ==========================================================
    "Ẩm thực Việt Nam",
    "Phở",
    "Bún chả",
    "Bánh mì Việt Nam",
    "Bún bò Huế",
    "Cao lầu",
    "Mì Quảng",
    "Cơm tấm",
    "Bánh xèo",
    "Gỏi cuốn",
    "Chả giò",
    "Bánh cuốn",
    "Cà phê Việt Nam",
    "Cà phê sữa đá",
    "Nước mắm",

    # ==========================================================
    # 3. DI CHUYỂN & PHƯƠNG TIỆN
    # ==========================================================
    "Sân bay quốc tế Nội Bài",
    "Sân bay quốc tế Tân Sơn Nhất",
    "Sân bay quốc tế Đà Nẵng",
    "Sân bay quốc tế Cam Ranh",
    "Sân bay quốc tế Phú Quốc",
    "Danh sách sân bay tại Việt Nam",
    "Đường sắt Việt Nam",
    "Tàu thống nhất",
    "Hàng không Việt Nam",
    "Vietnam Airlines",
    "Vietjet Air",
    "Bamboo Airways",
    "Grab (ứng dụng)",

    # ==========================================================
    # 4. THỰC TIỄN CHO KHÁCH QUỐC TẾ
    # ==========================================================
    "Chính sách thị thực của Việt Nam",
    "Đồng (tiền Việt Nam)",
    "Ngày nghỉ lễ ở Việt Nam",

    # ==========================================================
    # 5. THỜI TIẾT & MÙA DU LỊCH
    # ==========================================================
    "Khí hậu Việt Nam",
    "Lễ hội Việt Nam",
    "Tết Nguyên Đán",

    # ==========================================================
    # 6. HOẠT ĐỘNG & TRẢI NGHIỆM
    # ==========================================================
    "Vườn quốc gia Việt Nam",
    "Vườn quốc gia Phong Nha – Kẻ Bàng",
    "Vườn quốc gia Cát Tiên",
    "Vườn quốc gia Cúc Phương",
    "Chợ Bến Thành",
    "Chợ Đông Xuân",
    "Phố đi bộ Hồ Hoàn Kiếm",
    "Làng nghề truyền thống Việt Nam",

    # ==========================================================
    # 7. VĂN HÓA & NGÔN NGỮ
    # ==========================================================
    "Văn hóa Việt Nam",
    "Tiếng Việt",
    "Tôn giáo tại Việt Nam",
    "Phong tục Việt Nam",
    "Áo dài",
    "Múa rối nước",
]

# ========================================================
# CẤU HÌNH CÀO SÂU (DEEP SCRAPING)
# ========================================================
DEEP_SCRAPE = True  # Nếu True, sẽ tự động tìm và cào thêm các bài liên quan
MAX_RELATED = 10    # Số lượng bài liên quan tối đa cho mỗi từ khóa (nếu DEEP_SCRAPE=True)

def main():
    print("=" * 60)
    print(f"CÔNG CỤ CÀO WIKIPEDIA — Đang xử lý {len(KEYWORDS)} từ khoá")
    mode_str = f"Chế độ cào sâu (Tối đa {MAX_RELATED} bài liên quan/từ khóa)" if DEEP_SCRAPE else "Chế độ cào thông thường (1 bài/từ khóa)"
    print(f"[{mode_str}]")
    print("=" * 60)

    base_dir = Path(__file__).resolve().parent
    scraper = ScraperService(base_dir=base_dir)

    success_count = 0
    fail_count = 0

    for i, keyword in enumerate(KEYWORDS, 1):
        if not keyword.strip():
            continue
            
        print(f"\n[{i}/{len(KEYWORDS)}] Đang xử lý: '{keyword}'...")
        
        if DEEP_SCRAPE:
            saved_paths = scraper.scrape_wikipedia_deep(keyword=keyword, lang="vi", max_related=MAX_RELATED)
            if saved_paths:
                print(f"   ✅ Đã lưu {len(saved_paths)} bài viết liên quan đến '{keyword}'")
                success_count += len(saved_paths)
            else:
                print(f"   ❌ Thất bại hoặc không có nội dung cho '{keyword}'.")
                fail_count += 1
        else:
            saved_path = scraper.scrape_wikipedia(keyword=keyword, lang="vi")
            if saved_path:
                print(f"   ✅ Đã lưu: {saved_path}")
                success_count += 1
            else:
                print(f"   ❌ Thất bại hoặc không có nội dung.")
                fail_count += 1

    print("\n" + "=" * 60)
    print(f"TỔNG KẾT: Cào thành công {success_count} bài viết | Thất bại {fail_count} từ khóa")
    print("Chạy lệnh 'python ingest_data.py' để nạp toàn bộ vào hệ thống AI.")

if __name__ == "__main__":
    main()
