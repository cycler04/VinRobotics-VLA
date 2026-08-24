# VLA Paper Tracker — Ghi chú nghiên cứu & Mức độ quan trọng

Link Tracker: [app.notion.com/p/e6fe219965064b1c9707012dbefcb2bc?v=3aefb0e63a018143a3c7000c349fd79f&amp;source=copy_link](https://app.notion.com/p/e6fe219965064b1c9707012dbefcb2bc?v=3aefb0e63a018143a3c7000c349fd79f&source=copy_link)

Sử dụng tracker như một chỉ mục trung tâm để quản lý bài báo (paper), báo cáo (report), mã nguồn (source code), trạng thái (status), số lượng trích dẫn (citation count), độ uy tín (credibility), và phân loại nghiên cứu (research category).


## Ý nghĩa của từng Tracker View

Tracker có các chế độ xem (view) khác nhau phục vụ cho các nhiệm vụ nghiên cứu khác nhau. Các view này cùng hiển thị một cơ sở dữ liệu bài báo nền tảng, nhưng mỗi view lại tập trung vào một cách tiếp cận làm việc khác nhau với các bài báo.

### Table — Cơ sở dữ liệu bài báo chi tiết

Chế độ xem **Table** (Bảng) là nguồn thông tin chính xác duy nhất (main source of truth) cho thông tin bài báo. Sử dụng view này khi bạn cần kiểm tra hoặc cập nhật thông tin chi tiết và các liên kết của từng bài báo.

- Hiển thị toàn bộ bản ghi của bài báo: Bài báo (Paper), Tác giả (Author), Trạng thái (Status), Hội thảo (Conference), Năm (Year), Loại (Type), Số lượng trích dẫn (Citation count), Độ uy tín (Credibility), URL Bài báo (Paper URL), Mã nguồn GitHub (GitHub source), Báo cáo (Report), và Sơ đồ (Diagram).
- Phù hợp nhất cho việc **kiểm tra chi tiết bài báo, duy trì metadata, và tìm chính xác báo cáo hoặc bản triển khai**.
- Đây nên là view chính được sử dụng khi thêm mới hoặc cập nhật một bài báo.

### Board — Tiến độ nghiên cứu theo Trạng thái

Chế độ xem **Board** (Bảng tiến độ) sắp xếp các bài báo theo **Trạng thái (Status)** hiện tại như Not Started, Inspecting, Refining, Complete, Rejected, hoặc Not Relevant.

- Phù hợp nhất để **nắm bắt tiến độ nghiên cứu chỉ qua một cái nhìn tổng quan**.
- Hữu ích để quyết định bài báo nào vẫn cần khảo sát, báo cáo nào cần chỉnh sửa thêm, và bài báo nào đã hoàn tất.
- Xem đây như một chế độ xem quy trình làm việc (workflow view) thay vì nơi tra cứu thông tin chi tiết của bài báo.

### Summary — Tổng quan nghiên cứu cấp cao

Chế độ xem **Summary** (Tóm tắt) dùng để hiểu cơ cấu tổng thể và hướng đi của paper tracker thay vì kiểm tra từng bản ghi riêng lẻ.

- Sử dụng view này để xem **những lĩnh vực nghiên cứu VLA nào đang được bao phủ và các bài báo quan trọng đang tập trung ở đâu**.
- Hữu ích cho việc xác định các danh mục chính như Long Horizon, Memory, Real-time Chunking, Retry / Recovery, Baseline, và Spatial research.
- Phù hợp nhất cho **việc lập kế hoạch nghiên cứu, xác định thứ tự ưu tiên và nhanh chóng truyền đạt trạng thái tổng quan tài liệu (literature review)**.

### Cách kết hợp các View hiệu quả

**Table → Board → Summary** thể hiện 3 cấp độ theo dõi nghiên cứu:

1. **Table:** Bài báo này là gì và chúng ta có những thông tin nào về nó?
2. **Board:** Trạng thái nghiên cứu hiện tại của bài báo này là gì?
3. **Summary:** Bộ sưu tập bài báo tổng thể cho chúng ta biết điều gì về hướng đi nghiên cứu?

Đối với các bài báo quan trọng như **MemoryVLA** và **ReMem-VLA**, sử dụng Table để duy trì metadata chuẩn xác và các liên kết báo cáo, sử dụng Board để theo dõi tiến độ hoàn thành việc khảo sát, và sử dụng Summary để đánh giá tầm quan trọng của chúng trong bức tranh tổng thể về nghiên cứu VLA long-horizon.

## Báo cáo quan trọng (Important Report)

Các bài báo dưới đây được ưu tiên dựa trên mức độ ảnh hưởng trực tiếp của chúng đối với hướng nghiên cứu hiện tại về **long-horizon VLA + memory**. Đánh giá mức độ quan trọng dựa trên sự liên quan đến nghiên cứu và giá trị triển khai, chứ không chỉ dựa vào số lượng trích dẫn.

| Bài báo (Paper) | Mức độ quan trọng (Importance) | Lý do quan trọng (Why it matters)                                                                                                                                                                                                                                                                                           | Trạng thái hiện tại (Current status) |
| ----------------- | ---------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------- |
| MemoryVLA         | Critical                           | Trực tiếp giải quyết vấn đề bộ nhớ nhận thức-tư duy (perceptual-cognitive memory) cho thao tác robot và là tài liệu tham khảo gần nhất cho việc nghiên cứu module memory hiện tại. Chi tiết thiết kế và huấn luyện của nó cung cấp một baseline triển khai cụ thể.                      | Complete                                 |
| ReMem-VLA         | Critical                           | Trực tiếp nghiên cứu các truy vấn bộ nhớ đệ quy (recurrent memory queries) ở nhiều cấp độ cho VLA. Bài báo này quan trọng để so sánh các cơ chế bộ nhớ thay thế và hiểu cách các truy vấn đệ quy duy trì lịch sử có ích cho việc điều khiển long-horizon.                         | Complete                                 |
| Seer              | High                               | Cung cấp một hướng đi long-horizon bổ trợ thông qua động lực học ngược dự đoán (predictive inverse dynamics) và dự đoán tương lai thay vì dùng bộ nhớ tường minh. Hữu ích như một phương án thay thế về mặt khái niệm và kiến trúc so với các phương pháp dựa trên memory. | Complete                                 |
| Hi Robot          | High                               | Liên quan đến VLA phân cấp (hierarchical VLA) và việc tuân theo chỉ dẫn long-horizon. Bài báo cung cấp một cách phân rã khác cho các tác vụ dài và hữu ích cho việc tách biệt các vấn đề về bộ nhớ với các vấn đề về lập kế hoạch (planning) và điều khiển phân cấp.        | Complete                                 |
| LingBot-VA        | High                               | Giới thiệu mô hình thế giới nhân quả (causal world modeling) cho điều khiển robot. Quan trọng như một con đường khác dẫn đến hành vi long-horizon thông qua việc hiểu thế giới theo dạng dự đoán thay vì sử dụng một module memory chuyên dụng.                                          | Complete                                 |
| ThinkAct          | Medium–High                       | Liên quan đến lập kế hoạch tiềm ẩn thị giác (visual latent planning) và phục hồi sự cố (recovery). Hữu ích cho việc đánh giá liệu các thất bại trong tác vụ long-horizon nên được xử lý bằng cách lập kế hoạch/suy luận tốt hơn hay bằng cách tăng dung lượng memory của VLA.  | Complete                                 |

## Thứ tự ưu tiên cho nghiên cứu hiện tại

1. **MemoryVLA** — tài liệu tham chiếu triển khai chính cho VLA dựa trên bộ nhớ (memory-based VLA).
2. **ReMem-VLA** — đối tượng so sánh chính cho bộ nhớ truy vấn đệ quy (recurrent-query memory) và thiết kế bộ nhớ thay thế.
3. **Seer / LingBot-VA** — các phương án thay thế quan trọng cho việc dự đoán tương lai và hành vi long-horizon dựa trên mô hình thế giới (world-model-based).
4. **Hi Robot** — quan trọng cho việc phân rã phân cấp (hierarchical decomposition) các tác vụ long-horizon.
5. **ThinkAct** — hữu ích cho việc so sánh khả năng lập kế hoạch (planning), suy luận (reasoning) và phục hồi sự cố (recovery).

## Quy tắc theo dõi nghiên cứu (Research Tracking Rule)

Đối với mỗi bài báo quan trọng, hãy luôn đồng bộ 4 thông tin sau trong tracker chính:

- **Paper URL** — đường dẫn chính thức của bài báo/preprint.
- **Report** — báo cáo phân tích chi tiết nội bộ và các phát hiện khi triển khai.
- **Source GitHub** — mã nguồn công khai (nếu có).
- **Status / Type / Credibility / Cite** — các metadata hiện tại được dùng để ưu tiên cho việc nghiên cứu sâu hơn.

Tracker chính hiện tại chứa các mục dành riêng cho **MemoryVLA** và **ReMem-VLA**, cả hai đều được phân loại thuộc nhóm **Long Horizon** và đánh dấu **Complete**. MemoryVLA đã có báo cáo chi tiết nội bộ; ReMem-VLA nên được gắn liên kết báo cáo ngay khi phân tích của nó được hoàn tất.
