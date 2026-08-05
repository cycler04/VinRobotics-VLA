# Đọc và đánh giá paper

## Khi dùng

Dùng khi cần đọc paper/PDF, giải thích phương pháp, đối chiếu nhiều paper, tìm
liên hệ paper-code hoặc tạo research note.

## Quy trình

1. Chốt câu hỏi nghiên cứu và output mong muốn trước khi đọc sâu. Viết câu hỏi
   dưới dạng: vấn đề nào đang tồn tại, vì sao đáng xử lý và paper tuyên bố cải
   thiện điều gì.
2. Xác minh đúng phiên bản paper, nguồn chính thức và code/dataset đi kèm nếu có.
3. Với PDF dài, tìm abstract, contributions, method, experiments, limitations
   và appendix liên quan trước; không nạp toàn bộ artifact vào context nếu có
   thể trích đúng trang/section.
4. Lập problem map trước khi tóm tắt phương pháp:
   - pain point hoặc failure mode quan sát được;
   - nguyên nhân mà tác giả giả định;
   - giới hạn của prior work;
   - tiêu chí nào cho thấy vấn đề đã được xử lý.
5. Ghi riêng:
   - claim của tác giả;
   - bằng chứng/metric/dataset hỗ trợ;
   - diễn giải của người đọc;
   - giới hạn, missing baseline và threat to validity.
6. Với từng thành phần modeling, training và benchmark, trả lời rõ:
   - nó xử lý vấn đề con hoặc failure mode nào trong problem map;
   - cơ chế kỳ vọng tạo ra cải thiện là gì;
   - evidence hoặc ablation nào kiểm tra đúng cơ chế đó;
   - vấn đề nào vẫn chưa được xử lý.
7. Nếu có code, ánh xạ claim quan trọng tới module/config/checkpoint thực tế.
8. Kết thúc bằng kết luận trả lời câu hỏi ban đầu, mức tin cậy và 1–3 thử nghiệm
   tiếp theo có khả năng bác bỏ hoặc củng cố kết luận.

## Cấu trúc output mặc định

Sau citation/identity và câu hỏi nghiên cứu, luôn trình bày `Why` trước `How`.

### 1. Why — xử lý vấn đề gì?

- Bối cảnh và đối tượng chịu ảnh hưởng.
- Pain point/failure mode cụ thể; tránh mô tả mục tiêu chung chung.
- Vì sao cách trước đó chưa đủ và gap paper chọn giải quyết.
- Success criteria: metric, hành vi hoặc điều kiện nào phải thay đổi để claim
  được xem là có căn cứ.

### 2. How — xử lý như thế nào?

Mở đầu bằng một ánh xạ ngắn `vấn đề con -> cơ chế -> bằng chứng`. Sau đó giữ các
phần kỹ thuật phù hợp với paper:

- **Modeling/architecture:** mô tả input, representation, module và data flow,
  đồng thời nói mỗi lựa chọn sửa failure mode nào. Không liệt kê block mà không
  giải thích vai trò đối với vấn đề.
- **Training/data/objective:** mô tả dữ liệu, target, loss, curriculum,
  post-training hoặc optimization; chỉ ra chúng cung cấp signal gì còn thiếu và
  vì sao signal đó có thể sửa vấn đề.
- **Inference/system behavior:** nếu có, giải thích policy, decoding, retry,
  memory, controller hoặc runtime mechanism kích hoạt cải thiện ở đâu.
- **Benchmark/evidence:** nêu dataset, split, baseline, metric và protocol; với
  mỗi kết quả, nói nó kiểm tra claim/problem nào và có phân lập đúng cơ chế bằng
  ablation hay không. Phân biệt reported result với independent reproduction.
- **Residual problems:** nêu failure mode, phạm vi hoặc giả định vẫn còn sau giải
  pháp.

Khi paper có nhiều thành phần, ưu tiên bảng truy vết:

| Vấn đề/failure mode | Thành phần xử lý | Cơ chế kỳ vọng | Evidence/ablation | Phần chưa giải quyết |
|---|---|---|---|---|

Không ép đủ mọi heading nếu nguồn không có thông tin. Gắn nhãn `Unknown` thay vì
suy diễn modeling, training hoặc benchmark chưa được công bố.

## Thành phần tối thiểu

- Citation/identity của nguồn.
- Câu hỏi nghiên cứu.
- `Why`: vấn đề, prior limitation, gap và success criteria.
- `How`: modeling/training/benchmark liên kết với vấn đề mà từng phần xử lý.
- Evidence table hoặc danh sách claim → evidence.
- Limitations và điểm chưa rõ.
- Liên hệ với workspace.
- Đề xuất thử nghiệm tiếp theo.

Không chép dài nguyên văn. Quote chỉ dùng khi wording chính xác là đối tượng cần
phân tích.
