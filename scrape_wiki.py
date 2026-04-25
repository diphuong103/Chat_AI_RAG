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
    # ── Đã có sẵn ──
    "Hội An",
    "Thành cổ Quảng Trị",
    "Nghệ An",
    "Huế",
    "Quảng Ninh",
    "Hà Nội",

    # ── Miền Bắc ──
    "Hồ Hoàn Kiếm",
    "Văn Miếu - Quốc Tử Giám",
    "Vịnh Hạ Long",
    "Đảo Cát Bà",
    "Ninh Bình",
    "Tràng An",
    "Sapa",
    "Fansipan",
    "Hà Giang",
    "Cao nguyên đá Đồng Văn",
    "Điện Biên Phủ",
    "Mai Châu",
    "Mộc Châu",

    # ── Miền Trung ──
    "Đại Nội Huế",
    "Lăng Khải Định",
    "Chùa Thiên Mụ",
    "Phố cổ Hội An",
    "Mỹ Sơn",
    "Đà Nẵng",
    "Ngũ Hành Sơn",
    "Bà Nà Hills",
    "Quảng Nam",
    "Quảng Ngãi",
    "Phong Nha-Kẻ Bàng",
    "Động Phong Nha",
    "Đèo Hải Vân",
    "Bình Định",

    # ── Tây Nguyên ──
    "Đà Lạt",
    "Vườn quốc gia Bidoup Núi Bà",
    "Buôn Ma Thuột",
    "Vườn quốc gia Yok Đôn",
    "Kon Tum",
    "Gia Lai",
    "Pleiku",
    "Đắk Nông",

    # ── Miền Nam & Biển đảo ──
    "Thành phố Hồ Chí Minh",
    "Địa đạo Củ Chi",
    "Dinh Độc Lập",
    "Đồng bằng sông Cửu Long",
    "Cần Thơ",
    "Chợ nổi Cái Răng",
    "Phú Quốc",
    "Côn Đảo",
    "Vũng Tàu",
    "Mũi Né",
    "Phan Thiết",
    "Nha Trang",
    "Vịnh Nha Trang",
    "Đảo Lý Sơn",
    "Quy Nhơn",
    "Bình Thuận",
    "Kiên Giang",
    "Cà Mau",

    # ── Vườn quốc gia & Thiên nhiên ──
    "Vườn quốc gia Cúc Phương",
    "Vườn quốc gia Ba Vì",
    "Vườn quốc gia Bạch Mã",
    "Vườn quốc gia Cát Tiên",
    "Vườn quốc gia Tràm Chim",
    "Khu dự trữ sinh quyển Cần Giờ",
    "Ruộng bậc thang Mù Cang Chải",
    "Hồ Ba Bể",

    # ── Di sản & Lịch sử ──
    "Di sản thế giới UNESCO Việt Nam",
    "Hoàng thành Thăng Long",
    "Kinh thành Huế",
    "Thánh địa Mỹ Sơn",
    "Quần thể danh thắng Tràng An",
    "Thành nhà Hồ",
    "Đường Trường Sơn",
    "Bảo tàng lịch sử quốc gia Việt Nam",
    "Cố đô Hoa Lư",

    # ── Văn hóa & Ẩm thực ──
    "Ẩm thực Việt Nam",
    "Phở",
    "Bánh mì Việt Nam",
    "Cà phê Việt Nam",
    "Lễ hội Việt Nam",
    "Tết Nguyên Đán",
    "Làng nghề truyền thống Việt Nam",
    "Nhã nhạc cung đình Huế",
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
