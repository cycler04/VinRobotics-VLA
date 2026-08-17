# Workflow: tạo research draft ngắn trước báo cáo đầy đủ

Mục tiêu là tạo một outline Markdown ngắn, chỉ giữ các điểm quyết định hướng đọc paper và
cấu trúc báo cáo sau này. Draft không phải báo cáo sơ bộ và không được chứa phần giải thích
chi tiết.

Viết bằng tiếng Việt, giữ thuật ngữ English khi cần. Mặc định không vượt quá **600 từ hoặc
80 dòng**.

## 1. Xác định file và câu hỏi chính

- Lưu trong `docs/research_draft/`, theo cùng cấu trúc thư mục với
  `docs/research_finding/`. Đặt tên file mô tả dạng `snake_case`, không dùng tiền tố đánh số (ví dụ: `vifailback.md`).
- Ghi paper/source, một câu hỏi nghiên cứu và phạm vi chính trong vài dòng.
- Cập nhật draft cùng chủ đề nếu đã tồn tại; không tạo bản trùng.

## 2. Chọn key points

Quét abstract, contributions, method, experiments và limitations. Chỉ giữ **3–5 key
points** có thể thay đổi kết luận của báo cáo, ưu tiên:

- vấn đề/gap paper thực sự xử lý;
- contribution hoặc cơ chế cốt lõi;
- evidence mạnh nhất;
- claim cần nghi ngờ hoặc ablation còn thiếu;
- limitation quan trọng nhất.

Không tóm tắt tuần tự từng section, không chép nhiều số liệu và không mở rộng mỗi câu hỏi
thành một tiểu luận. Với PDF dài, dùng [research reader](../agents/02_research_reader.md)
để lấy evidence map ngắn.

## 3. Viết draft

Dùng cấu trúc mặc định:

```markdown
# Research draft: <paper/chủ đề>

## 1. Mục tiêu
## 2. Key points cần làm rõ
## 3. Câu hỏi và evidence cần tìm
## 4. Outline báo cáo đầy đủ
## 5. Unknown và bước tiếp theo
```

Phần câu hỏi dùng một bảng tối đa **5 dòng** với các cột `Priority`, `Question`,
`Evidence/source cần kiểm tra`, `Status`. Chỉ dùng `P0` và `P1` trừ khi người dùng yêu cầu
roadmap rộng hơn.

Outline chỉ gồm các heading chính và tối đa một câu gợi ý cho mỗi heading. Giữ `Why` trước
`How`; nếu paper có nhiều artifact ngang hàng như dataset, benchmark, model và system, có
thể dùng chúng làm heading riêng.

## 4. Ranh giới

- Không viết executive summary, diễn giải method đầy đủ hoặc bảng claim→evidence hoàn chỉnh.
- Không lặp cùng một ý ở key points, question table và outline.
- Không biến inference thành fact; dùng `Unknown` khi chưa đủ nguồn.
- Draft chỉ là bản đồ cho bước nghiên cứu tiếp theo, không phải nguồn bằng chứng.

## 5. Tự kiểm

- Có đọc trong dưới hai phút và thấy ngay 3–5 điểm quan trọng nhất không?
- Bỏ một mục có làm mất hướng nghiên cứu không? Nếu không, bỏ mục đó.
- Outline có đủ để bắt đầu báo cáo nhưng chưa chứa nội dung của báo cáo không?
