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

Trình bày phần này dưới heading `Method`, theo data flow hoặc thứ tự vận hành của
hệ thống giống cấu trúc method của paper. Mỗi module/stage là một subsection văn
xuôi riêng và phải mở đầu bằng câu **"Failure mode được xử lý:"** (hoặc một câu
tương đương rõ nghĩa), rồi lần lượt giải thích input, xử lý, output và evidence
kiểm tra cơ chế. Không dùng bảng truy vết làm cấu trúc chính của `Method`.

Sau `## 4. Method`, giữ riêng `## 5. Training` nếu paper có huấn luyện hoặc
post-training. Phần này mô tả stage, dữ liệu, target/loss, module được freeze hay
update, hyperparameter và compute. Trong `Method` chỉ giữ cơ chế/modeling/data
flow/inference; không gộp training config vào một method subsection.

Đây là cấu trúc mặc định, không phải khuôn cứng. Nếu paper có nhiều artifact là
đóng góp ngang hàng — đặc biệt `Dataset`, `Benchmark`, `Model` hoặc `System` —
đưa mỗi artifact thành một main section và đặt collection/protocol/training bên
trong section sở hữu nó. Không ép một dataset/benchmark paper vào hai mục chung
`Method` và `Training`, vì cách đó làm mờ contract và evidence riêng của từng
artifact.

Giữ các phần kỹ thuật phù hợp với paper:

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

Khi paper có nhiều thành phần, đi theo pipeline bằng các subsection đánh số,
ví dụ `4.1 Data`, `4.2 Module A`, `4.3 Inference`; phần training tách thành mục 5.
Bảng chỉ dùng
cho claim → evidence, benchmark, ablation hoặc so sánh nhiều cấu hình; không thay
thế giải thích method bằng prose.

Không ép đủ mọi heading nếu nguồn không có thông tin. Gắn nhãn `Unknown` thay vì
suy diễn modeling, training hoặc benchmark chưa được công bố.

## Thành phần tối thiểu

- Citation/identity của nguồn.
- Câu hỏi nghiên cứu.
- `Why`: vấn đề, prior limitation, gap và success criteria.
- `How`/`Method`: narrative theo data flow; mỗi module liên kết rõ với failure
  mode mà nó xử lý.
- `Training`: mục 5 riêng trong cấu trúc mặc định; với paper nhiều artifact,
  training nằm trong main section của model mà nó huấn luyện.
- Evidence table hoặc danh sách claim → evidence.
- Limitations và điểm chưa rõ.
- Liên hệ với workspace.
- Đề xuất thử nghiệm tiếp theo.

Không chép dài nguyên văn. Quote chỉ dùng khi wording chính xác là đối tượng cần
phân tích.
