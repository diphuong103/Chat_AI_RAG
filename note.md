# Danh Sách Cải Tiến & Roadmap Phát Triển Chatbot

## 📌 1. Các vấn đề cần cải tiến hiện hành (Theo feedback)
- **Truy xuất thông tin chi tiết**: Nghiên cứu kỹ thuật để AI lấy và nhấn mạnh chính xác vào các "thực thể" (Entities) như: các địa điểm cụ thể, thời gian lịch trình, giá vé, danh sách các món ăn ngon ở một địa điểm du lịch thay vì trả lời chung chung. 
- **Tích hợp thời tiết (Real-time data API)**: Làm sao xử lý các câu hỏi về dự báo thời tiết tại một địa điểm cụ thể trong một mốc thời gian trong tương lai. (*Lưu ý: RAG đọc tài liệu tĩnh không làm được việc này, bắt buộc phải dùng kiến trúc Agent Function Calling để gọi API thời tiết bên ngoài*).
- **Trò chuyện thông minh & ngữ cảnh hơn**: Tối ưu thêm hệ thống Prompt, nhiệt độ (Temperature) của mô hình và thuật toán Re-ranker để câu trả lời mang âm điệu tự nhiên, thông minh và xâu chuỗi ngữ cảnh mượt mà hơn.
- **Trả lời đa ngôn ngữ (Multilingual)**: Đã thiết lập Rule ngôn ngữ, nhưng cần lập quy trình kiểm thử (test) thực tế với bộ câu hỏi trộn lẫn tiếng Anh, Trung, Hàn, Nhật để xem LLM dịch và xử lý ngữ cảnh có mượt không.

---

## 🚀 2. Đề xuất thêm các Case / Tính năng nâng cao (Advanced RAG & Agent)

Dưới đây là một số hướng mở rộng cực kỳ mạnh mẽ để nâng tầm dự án:

### 2.1. Cấu trúc hóa thẻ Metadata (Metadata Filtering)
- **Vấn đề**: Khi dữ liệu lớn lên đến hàng vạn bài, AI dễ bị "ảo giác" hoặc lấy lộn xộn. 
- **Giải pháp**: Xử lý Scraping kỹ hơn. Mỗi tài liệu cần gắn thẻ từ khóa (VD: `location="Đà Nẵng"`, `topic="Ẩm thực"`). Khi người dùng hỏi *"Có món nào ngon ở Đà Nẵng"*, DataBase sẽ tự động thu hẹp vùng tìm kiếm dựa trên thẻ trước khi đưa cho AI đọc.

### 2.2. Trích dẫn nguồn tài liệu (Citations & Reference)
- Khi AI trả lời xong một thông tin, yêu cầu AI cung cấp cả link (hoặc tên tài liệu) mà nó vừa tham khảo. VD: *"Bạn có thể ăn Bún Chả (Nguồn tham khảo: Cẩm nang ăn uống VNExpress)"*. Tăng uy tín tuyệt đối cho câu trả lời.

### 2.3. Agentic RAG: Trợ lý có khả năng hành động (Function Calling)
- Thay vì chỉ là bot tư vấn, nâng cấp nó thành **Agent AI**:
  - Gắn API Thời tiết (OpenWeatherMap) để xem thời tiết.
  - Gắn API tỷ giá ngoại tệ, giá vàng hiện thời (Vietcombank, SJC).
  - Gắn API đặt phòng, tìm vé máy bay. 
  - (Luồng xử lý: `Phân tích ý user` ➡️ `Agent ra lệnh gọi API` ➡️ `Lấy kết quả API gộp chung vào RAG` ➡️ `Trả lời user`).

### 2.4. Khả năng hỗ trợ Hình ảnh, Bản đồ (Multimodal)
- Người dùng hỏi: *"Chỉ cho tôi đường đi đến Tháp Bà Ponagar"* -> Bot sẽ trả về đoạn văn giới thiệu kèm theo tọa độ Google Maps hoặc ảnh thumbnail đính kèm ngay trên khung chat. Thậm chí tích hợp Vision AI để gởi ảnh cho Bot chuẩn đoán.

### 2.5. Xử lý Dữ liệu Bảng biểu phức tạp (Table QA/GraphRAG)
- **Vấn đề**: RAG dạng chunk chữ rất dở trong việc so sánh các cột trong bảng (ví dụ: Bảng vé máy bay của 5 hãng, hoặc bảng lãi suất của 20 ngân hàng). 
- **Giải pháp**: Ứng dụng kỹ thuật `GraphRAG` hoặc bóc tách riêng Bảng lưu vào SQL DB. Khi user hỏi so sánh lãi suất, Bot sẽ viết lệnh Text-to-SQL thay vì tìm kiếm vector.

### 2.6. Cá nhân hóa Trải nghiệm (Personalized Recommendation)
- Xây dựng một History Database dài hạn (ví dụ lưu SQLite thông tin cá nhân user). 
- Nếu nó biết User thích ăn cay và du lịch bụi, khi hỏi đi Đã Nẵng AI sẽ nghiêng về dọn ra các Option quán ăn bình dân, món nêm nếm cay nồng thay vì giới thiệu Resort nghỉ dưỡng cao cấp.

### 2.7. Gửi hình ảnh để tìm kiếm vị trí điểm du lịch
- Người dùng gửi hình ảnh một địa điểm, AI sẽ nhận diện và trả về thông tin chi tiết về địa điểm đó.