# Sinh action tự hồi quy rời rạc

> **Phạm vi.** Các policy VLA tuần tự hóa action robot thành ký hiệu rời rạc
> và sinh chúng bằng cơ chế next-token tự hồi quy của VLM. Các hệ thống tiêu
> biểu: RT-2, OpenVLA và π0-FAST. Nguồn được kiểm tra ngày 2026-07-21.

## Ý tưởng cốt lõi

Họ này chuyển mục tiêu điều khiển liên tục thành một vocabulary hữu hạn, rồi
train bằng cùng giao diện mô hình ngôn ngữ nhân quả dùng cho văn bản:

```text
action hoặc chunk liên tục
        -> tokenizer
        -> [z1, z2, ..., zK]
        -> next-token cross-entropy

tại inference:
ngữ cảnh -> z1 -> z2 -> ... -> zK -> detokenizer -> lệnh liên tục
```

Đặc điểm quyết định không chỉ là một số nguyên xuất hiện đâu đó trong pipeline.
Các ký hiệu action được sinh **tự hồi quy**: mỗi token được dự đoán trở thành
ngữ cảnh cho token tiếp theo.

## RT-2: action dưới dạng token văn bản

RT-2 chuyển action robot thành một chuỗi tương thích với token văn bản và
co-fine-tune một VLM pretrained trên trajectory robot cùng các tác vụ
vision-language quy mô web. Mỗi chiều liên tục dùng phép rời rạc hóa 256 bin
của RT-1; các số nguyên tạo ra được biểu diễn thành token số thông thường trong
PaLI-X hoặc token action dành riêng trong PaLM-E. Khi decode action robot,
vocabulary được giới hạn ở các ký hiệu action hợp lệ.
[RT-2, §III-B](https://arxiv.org/abs/2307.15818)

Điều này làm cho policy đơn giản về mặt khái niệm:

```text
hình ảnh + chỉ dẫn -> VLM -> "1 128 91  ..." -> tâm bin -> action robot
```

Điểm hấp dẫn chính là joint training: câu trả lời web và action robot dùng chung
giao diện sequence model và tham số. Đổi lại, mỗi chiều motor đều phải decode
tuần tự và lượng tử hóa.

## OpenVLA: các mục vocabulary dành riêng

OpenVLA dùng Prismatic VLM 7B dựa trên Llama-2. Với mỗi chiều action, nó:

1. lấy khoảng percentile thứ 1 đến 99 của dữ liệu training;
2. chia khoảng đó thành 256 bin;
3. thay 256 mục ít dùng nhất của Llama tokenizer bằng token action;
4. train dự đoán next-token chuẩn, chỉ áp dụng cross-entropy lên đầu ra action.

[OpenVLA, §3.2](https://arxiv.org/abs/2406.09246)

Giới hạn percentile làm giảm độ nhạy với outlier, nhưng semantics và giới hạn
vẫn đặc thù theo dataset. Một token ID không phải action vật lý phổ quát cho đến
khi áp dụng đúng thống kê dataset và ánh xạ embodiment.

## FAST và π0-FAST: nén trước khi dự đoán

Chia bin đơn giản phát ra một token cho mỗi chiều tại mỗi timestep. Trên các
trajectory trơn, tần số cao, những giá trị liền kề tương quan mạnh; mô hình có
thể giảm token loss bằng cách sao chép token gần đây mà không học hình dạng tổng
thể của chunk. FAST thay đổi biểu diễn mục tiêu trước khi dự đoán tự hồi quy.
[FAST, §§III-V](https://arxiv.org/abs/2501.09747)

FAST tokenizer thực hiện:

```text
action chunk liên tục dài 1 giây
  -> chuẩn hóa percentile theo từng chiều
  -> biến đổi cosine rời rạc (DCT)
  -> scale và làm tròn hệ số tần số
  -> flatten các tần số thấp trước trên các chiều action
  -> byte-pair encoding (BPE)
  -> chuỗi token rời rạc có độ dài thay đổi
```

Các hệ số tần số thấp mô tả trajectory tổng thể trước. Phép làm tròn khiến ma
trận hệ số thưa; BPE gộp các pattern số nguyên lặp lại và những dải số 0. Phép
biến đổi được decode nhanh, dù việc làm tròn hệ số khiến toàn bộ pipeline từ
liên tục sang rời rạc bị mất mát. FAST+ cố định một BPE vocabulary có thể tái
sử dụng, được train trên khoảng một triệu action chunk dài một giây từ nhiều
embodiment. [FAST, §V-B-C](https://arxiv.org/abs/2501.09747)

π0-FAST là backbone π0/PaliGemma được train để phát ra các token đã nén này thay
vì dùng flow-matching action expert của π0. Vì vậy FAST thay đổi **action
tokenizer và chuỗi mục tiêu**, không thay đổi phép tính next-token nền tảng của
causal decoder. Trên hỗn hợp đánh giá trong paper, nó đạt hiệu năng ngang mô
hình flow π0 được báo cáo trong khi training nhanh hơn tới 5 lần; kết quả đó
không chứng minh hai cách tương đương trên mọi robot hoặc phân phối dữ liệu.

## Các trường hợp biên quan trọng

- **RT-1:** rời rạc và không tự hồi quy ở đầu ra action cuối. Các tác giả loại
  bỏ điều kiện hóa action tự hồi quy vì nó tăng độ trễ mà không cải thiện đáng
  kể trong ablation. RT-1 phát trực tiếp đầu ra phân loại theo từng chiều cho
  bước hiện tại. Xem [dự đoán liên tục/trực tiếp](01_continuous_regression.md).
- **π0.5:** dùng token FAST để pretraining quy mô lớn hiệu quả, sau đó thêm một
  flow-matching expert để inference cấp thấp liên tục, chi tiết. Tùy giai đoạn
  đang xét, nó vừa thuộc câu chuyện training bằng token rời rạc, vừa thuộc
  [câu chuyện flow expert](04_flow_matching_transformer_expert.md).
- **Detokenization không phải robot controller:** ánh xạ ký hiệu trở lại số đã
  chuẩn hóa vẫn đứng trước chuyển đổi embodiment, giới hạn an toàn và thực thi cấp thấp.

## Điểm mạnh và giới hạn

Điểm mạnh:

- tái sử dụng vocabulary VLM, causal Transformer, cross-entropy loss và hạ tầng
  training next-token có khả năng mở rộng;
- có thể xen kẽ văn bản, reasoning và ký hiệu action trong một không gian đầu ra;
- định nghĩa likelihood tường minh trên các chuỗi action-token;
- FAST nén chunk tần số cao mà không thêm action expert được học.

Giới hạn:

- độ trễ tự hồi quy tăng theo số token action được sinh;
- cách chia bin đơn giản làm mất độ chính xác bên trong bin và bỏ qua cấu trúc thời gian;
- một lỗi sẽ thay đổi ngữ cảnh của các token sau;
- tính hợp lệ của token, thống kê chuẩn hóa và semantics của action phải đi cùng checkpoint;
- FAST tạo thêm đánh đổi reconstruction/compression và độ dài token thay đổi;
  nó cải thiện biểu diễn mục tiêu nhưng không loại bỏ decode tuần tự.

## Nguồn

- Brohan et al. *RT-2: Vision-Language-Action Models Transfer Web Knowledge to
  Robotic Control*, §III-B, arXiv:2307.15818.
  [Paper](https://arxiv.org/abs/2307.15818)
- Kim et al. *OpenVLA: An Open-Source Vision-Language-Action Model*, §3.2,
  arXiv:2406.09246v3. [Paper](https://arxiv.org/abs/2406.09246) ·
  [Official code](https://github.com/openvla/openvla)
- Pertsch et al. *FAST: Efficient Action Tokenization for Vision-Language-Action
  Models*, arXiv:2501.09747. [Paper](https://arxiv.org/abs/2501.09747) ·
  [Official project](https://pi.website/research/fast)
- Physical Intelligence et al. *π0.5: a Vision-Language-Action Model with
  Open-World Generalization*, §IV, arXiv:2504.16054.
  [Paper](https://arxiv.org/abs/2504.16054)
