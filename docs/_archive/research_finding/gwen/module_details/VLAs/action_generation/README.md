# Các họ phương pháp sinh action của VLA hiện đại

> **Câu hỏi.** Các họ phương pháp sinh action chính của VLA hiện đại chuyển
> ngữ cảnh thị giác-ngôn ngữ thành lệnh robot như thế nào, và các mô hình tiêu
> biểu thực sự thuộc nhóm nào?
>
> **Phạm vi.** Sinh action cấp thấp, không bao gồm bộ hoàn nguyên chuẩn hóa,
> bộ lọc an toàn, controller hoặc giao diện actuator ở downstream. Nguồn được
> kiểm tra ngày 2026-07-21. Năm họ được yêu cầu được xem như một taxonomy kỹ
> thuật hữu ích, không phải các phạm trù khoa học loại trừ lẫn nhau.

## Câu trả lời ngắn

Không có một trục phân loại duy nhất gọi là “loại action decoder”. Có ít nhất
bốn lựa chọn thiết kế đang được trộn lẫn:

1. **biểu diễn:** giá trị liên tục hoặc ký hiệu rời rạc;
2. **phân rã:** một lượt song song, sinh từng token hoặc tinh chỉnh lặp từ nhiễu;
3. **mục tiêu training:** hồi quy, dự đoán next-token phân loại, dự đoán nhiễu
   diffusion hoặc flow matching;
4. **kiến trúc và quy mô:** readout nhỏ, Transformer expert định tuyến theo
   token hoặc một Diffusion Transformer riêng lấy decoder làm trung tâm.

Điều này giải thích các phần chồng lấn quan trọng. Qwen-VLA là mô hình flow
matching với action decoder DiT 1,15B tham số riêng nằm sau VLM 4B. π0.5 dùng
token FAST trong pretraining nhưng dùng continuous flow expert khi triển khai
điều khiển cấp thấp. Ngược lại, RT-1 **không** thực hiện hồi quy liên tục: nó
dự đoán một trong 256 bin cho mỗi chiều action bằng categorical cross-entropy.
[RT-1, §3.3](https://arxiv.org/abs/2212.06817)

## Bản đồ các họ

| Họ được yêu cầu | Phép tính định nghĩa | Mô hình tiêu biểu phù hợp nhất | Hiệu chỉnh hoặc phần chồng lấn quan trọng |
| --- | --- | --- | --- |
| [Hồi quy liên tục](continuous_regression.md) | Dự đoán action hoặc chunk liên tục trong một forward pass với loss kiểu L1/MSE | OpenVLA-OFT | RT-1 là policy **phân loại** song song, không phải hồi quy liên tục |
| [Action tự hồi quy rời rạc](discrete_autoregressive_actions.md) | Tuần tự hóa các ký hiệu action và sinh chúng bằng dự đoán next-token | RT-2, OpenVLA, π0-FAST | FAST thay đổi tokenizer, không thay đổi autoregressive decoder |
| [Diffusion hoặc flow decoder gọn](compact_diffusion_flow.md) | Một denoiser có điều kiện tương đối nhỏ tinh chỉnh lặp một action chunk | Các biến thể Diffusion Policy và head VLA gọn | “Gọn” là khác biệt về kiến trúc/quy mô, không phải một mục tiêu xác suất khác |
| [Flow-matching Transformer expert](flow_matching_transformer_expert.md) | Token robot dùng trọng số chuyên biệt trong một phép tính Transformer dùng chung | π0, π0.5 | π0.5 cũng pretrain bằng token FAST; đây không phải topology decoder downstream của Qwen-VLA |
| [DiT lấy decoder làm trung tâm](large_diffusion_transformer.md) | Một Transformer lớn tự đóng vai trò action decoder diffusion/flow lặp | RDT-1B; Dita; Qwen-VLA, kèm lưu ý | Dita có 334M tham số và tự gọi là lightweight; Qwen-VLA dùng một DiT flow-matching riêng sau VLM |

## Contract đầu vào/đầu ra chung

Dù decoder khác nhau, phần lớn hệ thống có thể được so sánh qua cùng một
contract trừu tượng:

```text
hình ảnh + chỉ dẫn + trạng thái robot tùy chọn
                    |
                    v
       ngữ cảnh đa phương thức / prefix
                    |
                    v
              cơ chế sinh action
                    |
                    v
     action hoặc action chunk đã chuẩn hóa
                    |
                    v
 hoàn nguyên chuẩn hóa + ánh xạ embodiment + an toàn/controller
```

Các tài liệu trong thư mục này dừng ở action chunk đã chuẩn hóa. Một mô hình
xuất ra tensor đúng vẫn cần semantics đặc thù của dataset, chẳng hạn lệnh
absolute hay delta, không gian joint hay end-effector, biểu diễn rotation,
tần số điều khiển và quy ước gripper.

## Trực giác lựa chọn

- Chọn **hồi quy song song** khi độ trễ thấp và đường thích nghi đơn giản quan
  trọng hơn việc biểu diễn tường minh nhiều mode trajectory hợp lệ.
- Chọn **tự hồi quy rời rạc** khi lợi thế cốt lõi là tái sử dụng vocabulary,
  training stack và mục tiêu next-token của VLM hiện có. FAST khiến hướng này
  khả thi hơn nhiều với các chunk tần số cao.
- Chọn **diffusion hoặc flow** khi phân phối action đa mode hoặc cần sinh đồng
  thời một trajectory nhiều chiều mạch lạc, đồng thời chấp nhận chi phí lấy mẫu lặp.
- Chọn **head nhỏ** khi compute và tính mô-đun là ưu tiên; tăng quy mô action
  Transformer khi dữ liệu embodiment/action không đồng nhất dường như cần nhiều
  capacity và điều kiện hóa chặt hơn.

Đây là các giả thuyết thiết kế, không phải xếp hạng phổ quát. Success rate được
báo cáo gắn với các dataset, robot, tần số điều khiển và recipe fine-tuning khác
nhau, nên chúng không chứng minh một họ decoder vượt trội trên toàn cục.

## Nguồn chính

- Brohan et al. *RT-1: Robotics Transformer for Real-World Control at Scale*,
  arXiv:2212.06817v2, 2023. [Paper](https://arxiv.org/abs/2212.06817)
- Brohan et al. *RT-2: Vision-Language-Action Models Transfer Web Knowledge to
  Robotic Control*, arXiv:2307.15818, 2023.
  [Paper](https://arxiv.org/abs/2307.15818)
- Kim et al. *OpenVLA: An Open-Source Vision-Language-Action Model*,
  arXiv:2406.09246v3, 2024. [Paper](https://arxiv.org/abs/2406.09246)
- Black et al. *π0: A Vision-Language-Action Flow Model for General Robot
  Control*, arXiv:2410.24164v4, 2026.
  [Paper](https://arxiv.org/abs/2410.24164)
- Pertsch et al. *FAST: Efficient Action Tokenization for Vision-Language-Action
  Models*, arXiv:2501.09747, 2025.
  [Paper](https://arxiv.org/abs/2501.09747)
- Wang et al. *Qwen-VLA: Unifying Vision-Language-Action Modeling across Tasks,
  Environments, and Robot Embodiments*, arXiv:2605.30280v2, 2026.
  [Paper](https://arxiv.org/abs/2605.30280)
