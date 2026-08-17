# Mối liên hệ giữa paper và code Real-Time Chunking

## Kết luận ngắn

Repository `third_party/01_real-time-chunking-kinetix` là mã thí nghiệm cho **phần mô phỏng
Kinetix dùng chung của hai paper**, không phải toàn bộ hệ thống robot:

- Paper **Real-Time Execution of Action Chunking Flow Policies** dùng repo để xây benchmark
  12 tác vụ Kinetix, train flow policy và đánh giá RTC tại inference dưới nhiều độ trễ.
- Paper **Training-Time Action Conditioning for Efficient Real-Time Chunking** tái sử dụng
  cùng dữ liệu, model và evaluator, nhưng fine-tune policy để model học cách hoàn thành action
  chunk từ một prefix đã cam kết.
- Repo không chứa policy `π0.5`/`π0.6`, dữ liệu camera, robot driver hay runtime bất đồng bộ
  dùng trong các thí nghiệm robot thật.

```text
Kinetix worlds
    ↓
train expert → tạo demonstration → train flow policy
                                      ↓
                 ┌────────────────────┴────────────────────┐
                 │                                         │
Paper 1: guidance khi inference              Paper 2: prefix conditioning khi train
                 │                                         │
                 └──────── eval delay/horizon trên Kinetix ┘
```

## Paper dùng code để làm gì?

| Phần của paper | Code tương ứng | Vai trò trong nghiên cứu |
| --- | --- | --- |
| Tạo 12 môi trường điều khiển động | [`worlds/l/`](../../../third_party/01_real-time-chunking-kinetix/worlds/l), [`train_expert.py`](../../../third_party/01_real-time-chunking-kinetix/src/train_expert.py) | Cung cấp benchmark có quán tính và lực, nơi lỗi chuyển tiếp giữa hai chunk ảnh hưởng rõ đến kết quả |
| Train expert và thu demonstration | [`train_expert.py`](../../../third_party/01_real-time-chunking-kinetix/src/train_expert.py), [`generate_data.py`](../../../third_party/01_real-time-chunking-kinetix/src/generate_data.py) | Tạo dữ liệu imitation learning cho từng tác vụ Kinetix |
| Train action-chunk flow policy | [`train_flow.py`](../../../third_party/01_real-time-chunking-kinetix/src/train_flow.py), [`model.py`](../../../third_party/01_real-time-chunking-kinetix/src/model.py) | Học policy MLP-Mixer sinh chunk dài `H=8` bằng conditional flow matching |
| Thuật toán RTC tại inference của paper 1 | [`FlowPolicy.realtime_action`](../../../third_party/01_real-time-chunking-kinetix/src/model.py#L219) | Dùng VJP/pseudoinverse guidance để chunk mới khớp chunk cũ; đây là phần hiện thực gần trực tiếp các Phương trình 2–4 và Algorithm 1 |
| Soft masking của paper 1 | [`get_prefix_weights`](../../../third_party/01_real-time-chunking-kinetix/src/model.py#L40) | Giữ cứng các action đã cam kết, giảm dần ràng buộc trên vùng overlap và để phần tương lai được sinh tự do |
| RTC tại training của paper 2 | [`FlowPolicy.loss`](../../../third_party/01_real-time-chunking-kinetix/src/model.py#L267) | Lấy mẫu độ trễ, đưa prefix sạch vào model với flow time bằng `1`, rồi chỉ tính loss trên postfix |
| Sampling của policy đã condition ở training | [`FlowPolicy.realtime_action`](../../../third_party/01_real-time-chunking-kinetix/src/model.py#L253) | Khi `simulated_delay` được bật, ghi đè prefix bằng action cũ và sinh phần còn lại bằng forward sampling thông thường |
| Đo ảnh hưởng của delay và execution horizon | [`eval_flow.py`](../../../third_party/01_real-time-chunking-kinetix/src/eval_flow.py) | So sánh naive, RTC, BID và hard masking; chạy 2.048 rollout cho mỗi cấu hình mặc định |

## Kết nối riêng với từng paper

### Paper 1: RTC tại inference

Paper đầu tiên đề xuất sinh chunk mới trong lúc robot vẫn thực thi chunk cũ. Những action sẽ
được thực thi trong khoảng inference trở thành prefix không được phép thay đổi. Trong
`model.py`, quan hệ này được hiện thực như sau:

1. `get_prefix_weights` tạo hard prefix và vùng soft overlap theo Phương trình 5.
2. `realtime_action` ước lượng action sạch, tính sai lệch với chunk cũ và dùng `jax.vjp` để
   hiệu chỉnh velocity của flow theo các Phương trình 2–4.
3. `eval_flow.py` lấy `d` action đầu từ chunk cũ, lấy phần còn lại từ chunk mới, rồi dịch chunk
   để giữ đúng mốc thời gian ở vòng kế tiếp.

Vì vậy, paper dùng code chủ yếu để kiểm tra câu hỏi: **RTC có giữ hiệu năng khi inference chậm
hơn controller hay không?**

### Paper 2: RTC tại training

Paper thứ hai bỏ bước backpropagation guidance lúc inference. Thay vào đó, model được fine-tune
để học phân phối “postfix hợp lệ khi đã biết prefix”. Branch `simulated_delay` trong
`FlowPolicy.loss` hiện thực ba thay đổi của paper:

1. lấy mẫu số action prefix tương ứng với delay;
2. giữ prefix ở trạng thái sạch và đặt flow time riêng của prefix thành `1`;
3. loại prefix khỏi loss, chỉ tối ưu velocity của postfix.

Theo hướng dẫn của repo, thí nghiệm này dùng checkpoint epoch 24 của policy thường, đặt
`simulated_delay=5`, rồi fine-tune thêm 8 epoch
([README](../../../third_party/01_real-time-chunking-kinetix/README.md#L37)).
Như vậy paper hai không có một pipeline độc lập; nó dùng lại benchmark của paper một để so sánh
**chi phí guidance tại inference** với **chi phí fine-tuning trước khi triển khai**.

## Những gì repo không tái tạo

Các giới hạn sau đã được xác minh từ paper và code:

- `eval_flow.py` mô phỏng phép căn chỉnh chunk bằng `jax.lax.scan`; nó không hiện thực thread
  inference nền, mutex, condition variable hay bộ ước lượng delay động của runtime thật.
- Repo chỉ dùng observation ký hiệu từ Kinetix và MLP-Mixer nhỏ. Không có image/language input
  hoặc implementation `π0.5`/`π0.6`.
- Các thí nghiệm robot thật như thao tác hai tay, xếp hộp và pha espresso không thể chạy lại từ
  repo này.
- Repo cung cấp cấu trúc để tái tạo thí nghiệm mô phỏng, nhưng workspace hiện không có
  `results.csv` hay log chứng minh rằng các số liệu trong paper đã được chạy lại cục bộ.

## Sai khác đáng chú ý giữa paper và code hiện tại

- Paper RTC đầu tiên mô tả 6 expert cho mỗi môi trường; README/code hiện mặc định train 8 seed.
- Paper dùng BID với `N=32`, `K=3` và weak policy riêng; cấu hình mặc định hiện tại là `N=16`,
  `bid_k=None`.
- Paper báo cáo Temporal Ensembling như một baseline, nhưng evaluator hiện không có nhánh
  Temporal Ensembling.

Do đó, code hiện thực đúng ý tưởng RTC và pipeline benchmark chính, nhưng chạy toàn bộ cấu hình
mặc định hiện tại chưa chắc tái tạo chính xác mọi điểm trong hình kết quả của paper.

## Nguồn

- Kevin Black, Manuel Y. Galliker và Sergey Levine, *Real-Time Execution of Action Chunking
  Flow Policies*, Mục 3–4, Algorithm 1 và Appendix A:
  [PDF cục bộ](<../../papers/realtime-chunking/Real-Time Execution of Action Chunking Flow Policies.pdf>).
- Kevin Black, Allen Z. Ren, Michael Equi và Sergey Levine, *Training-Time Action Conditioning
  for Efficient Real-Time Chunking*, Mục IV–V và Algorithm 1:
  [PDF cục bộ](<../../papers/realtime-chunking/Training-Time Action Conditioning for Efficient Real-Time Chunking.pdf>).
- Mã thí nghiệm:
  [`README.md`](../../../third_party/01_real-time-chunking-kinetix/README.md),
  [`model.py`](../../../third_party/01_real-time-chunking-kinetix/src/model.py),
  [`eval_flow.py`](../../../third_party/01_real-time-chunking-kinetix/src/eval_flow.py).

