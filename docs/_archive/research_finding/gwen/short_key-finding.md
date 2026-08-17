# Những mô-đun hiện đại khác với LLM nguyên bản

- RoPE (Rotary Position Embedding): mã hóa vị trí.
- Attention tối ưu bộ nhớ:
  - Multi-Query Attention (MQA);
  - Grouped-Query Attention (GQA);
  - FlashAttention.
- Lớp feed-forward: ReLU → SwiGLU.
- Chuẩn hóa lớp: LayerNorm → RMSNorm.
- MoE (Mixture-of-Experts): dùng router để đưa mỗi token đến một số FFN chuyên biệt,
  nhờ đó tăng tổng dung lượng tham số mà không kích hoạt toàn bộ FFN cho mọi token.

## Kiến trúc đặc biệt ở Qwen là gì?

- **Các lớp chuỗi lai:** Qwen3.5/3.6 lặp lại ba lớp Gated DeltaNet
  (attention tuyến tính/hồi quy) và một lớp gated full attention. Thiết kế này
  giảm chi phí ngữ cảnh dài, đồng thời định kỳ phục hồi khả năng trộn token toàn cục.
- **Attention lai + sparse MoE + MTP:** ví dụ Qwen3.6-35B-A3B có tổng 35B tham số
  nhưng chỉ kích hoạt khoảng 3B cho mỗi token, đồng thời thêm dự đoán nhiều token
  để giải mã nhanh hơn.
- **Thiết kế đa phương thức rõ ràng:** Qwen3-VL bổ sung Interleaved-MRoPE, đặc trưng
  thị giác đa cấp DeepStack và timestamp dạng văn bản để grounding ảnh/video theo
  không gian-thời gian.

Các thành phần riêng lẻ **không chỉ có ở Qwen**. Điểm khác biệt nằm ở cách kết hợp
được công bố rõ ràng. GQA, RoPE, RMSNorm, SwiGLU và MoE cũng phổ biến trong các LLM
khác; chế độ thinking/non-thinking chủ yếu là hành vi hậu huấn luyện và suy luận,
không phải một khối nơ-ron mới.

Nguồn: [model card Qwen3.6](https://huggingface.co/Qwen/Qwen3.6-35B-A3B),
[báo cáo kỹ thuật Qwen3-VL](https://arxiv.org/abs/2511.21631) và
[báo cáo kỹ thuật Qwen3](https://arxiv.org/abs/2505.09388). Truy cập 2026-07-20.
