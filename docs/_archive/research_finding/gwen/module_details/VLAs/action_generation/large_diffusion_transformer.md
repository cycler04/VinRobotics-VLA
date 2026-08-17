# Bộ giải mã hành động DiT lớn hoặc lấy bộ giải mã làm trung tâm

> **Phạm vi.** Các bộ sinh hành động đặt phần lớn năng lực Transformer
> bên trong đường lặp diffusion/flow thay vì dùng một đầu đọc hành động nông.
> Các ví dụ được yêu cầu: RDT-1B, Dita và Qwen-VLA. Nguồn được kiểm tra
> ngày 2026-07-21.

## Trước hết cần sửa cách phân loại

“Diffusion Transformer độc lập cỡ lớn” không chính xác với cả ba mô hình:

| Mô hình | Quy mô bộ sinh hành động đã xác minh | Mục tiêu | Bộ giải mã có độc lập với toàn bộ VLA không? |
| -------- | ----------------------------------------------------: | ------------------------- | ---------------------------------------------------------------------------------------------- |
| RDT-1B | RDT 1,2B tham số | diffusion | Policy lấy bộ giải mã làm trung tâm, nhưng vẫn được điều kiện hóa bởi các encoder ngôn ngữ/thị giác riêng |
| Dita | 334M tham số cho policy đã công bố | DDPM diffusion | Policy tích hợp DINOv2, CLIP, Q-Former và causal DiT; các tác giả gọi nó là gọn nhẹ |
| Qwen-VLA | Bộ giải mã hành động DiT khoảng 1,15B tham số sau Qwen3.5-4B VLM | conditional flow matching | Không; DiT là bộ giải mã downstream riêng, được điều kiện hóa bằng hidden state của VLM |

Nhãn chung có thể bảo vệ được là **sinh hành động DiT lớn hoặc lấy bộ giải mã làm
trung tâm**. RDT-1B là ví dụ rõ ràng ở quy mô tỷ tham số. Dita thuộc nhóm này
vì nó đưa quá trình khử nhiễu vào chuỗi token Transformer chính, không phải
vì đạt quy mô tỷ tham số. Qwen-VLA thuộc nhóm này vì hành động được sinh bởi
bộ giải mã DiT tỷ tham số, chứ không phải các trọng số expert định tuyến theo token
bên trong backbone Qwen.

## Điều gì thay đổi so với một head nhỏ gọn?

```text
Head nhỏ gọn:
backbone đa phương thức -> một embedding hợp nhất -> bộ khử nhiễu MLP nhỏ x N

DiT lấy bộ giải mã làm trung tâm:
token ngôn ngữ/hình ảnh/trạng thái + token hành động nhiễu
        -> nhiều block Transformer với điều kiện hóa dựa trên attention x N
        -> action chunk đã tinh chỉnh
```

Lợi ích kỳ vọng là năng lực lớn hơn và khả năng điều kiện hóa chi tiết hơn ở cấp token cho
các hình ảnh, lịch sử, embodiment và action space không đồng nhất. Đổi lại, mỗi bước
diffusion hoặc flow đều gọi một Transformer đáng kể.

Ở đây, **DiT chỉ kiến trúc mạng khử nhiễu**, không phải một hàm loss bắt buộc.
RDT-1B và Dita dùng các mục tiêu diffusion bắt nguồn từ DDPM, còn Qwen-VLA dùng
DiT để dự đoán trường vận tốc flow matching. Cả ba đều áp dụng Transformer lặp lại
cho một trajectory hành động nhiễu trung gian.

## RDT-1B

RDT-1B là Robotics Diffusion Transformer được mở rộng lên 1,2B tham số cho
thao tác hai tay. Mô hình dùng các encoder thị giác SigLIP và ngôn ngữ T5-XXL
riêng, đầu vào proprioception và bộ khử nhiễu Transformer có khả năng mở rộng. Policy
được pretrain trên dữ liệu đa robot và dự đoán 64 hành động robot tiếp theo.
[Bài báo RDT-1B](https://arxiv.org/abs/2410.07864) · [model card chính thức](https://huggingface.co/robotics-diffusion-transformer/rdt-1b)

Bản thân bộ khử nhiễu có 28 layer, chiều rộng 2.048 và 32 attention head. Nó
luân phiên cross-attention với các điều kiện ngôn ngữ và hình ảnh. Khi huấn luyện,
hành động được thêm nhiễu theo lịch DDPM 1.000 bước và mô hình học ước tính hành động sạch
bằng MSE; khi triển khai, năm bước DPM-Solver++ được dùng để tạo một chunk 64 hành động.
[RDT-1B, §4.1, §5 và Phụ lục H](https://proceedings.iclr.cc/paper_files/paper/2025/file/49f80e4d2471ad4f2edf4f5f1ab62339-Paper-Conference.pdf)

Một thiết kế trung tâm là Physically Interpretable Unified Action Space. Nó
phân bổ các vị trí cho những đại lượng vật lý phổ biến—vị trí/vận tốc khớp và
end effector, bộ kẹp và chuyển động đế di động—để các robot không đồng nhất
có thể được padding/masking vào một giao diện mà không giả vờ rằng các vector thô
của chúng vốn có ý nghĩa giống nhau. Model card cảnh báo rõ rằng một
embodiment chưa từng thấy vẫn cần fine-tune trên robot đích; tensor thống nhất
không đồng nghĩa với chuyển giao embodiment zero-shot.

Các block điều kiện hóa của RDT luân phiên truy cập thông tin ngôn ngữ, hình ảnh
và trạng thái robot, cho phép bộ khử nhiễu lớn kết hợp các phương thức trong khi
tinh chỉnh action chunk. Điểm mạnh được báo cáo là năng lực mô hình hóa hành động
cho điều khiển hai tay đa phương thức, nhiều chiều; chi phí thực tế gồm một
mạng lặp lớn và độ nhạy với lựa chọn độ trễ điều khiển/action horizon.

## Dita

Dita được đề xuất để đáp lại các diffusion head nhỏ gọn. Các tác giả lập luận
rằng điều kiện hóa một bộ khử nhiễu nông bằng một embedding được hợp nhất sớm có thể che mất
những thay đổi thị giác nhỏ nhưng quan trọng đối với độ chênh hành động. Thay vào đó, Dita nối:

- token ngôn ngữ từ CLIP đã đóng băng;
- đặc trưng image patch DINOv2 được chọn bởi Q-Former có điều kiện theo chỉ dẫn;
- embedding diffusion timestep;
- token hành động 7D đã padding và thêm nhiễu.

Các token này đi vào một causal Transformer, vì vậy action chunk được khử nhiễu
trong ngữ cảnh khi attention trực tiếp đến các token thị giác lịch sử. Quá trình huấn luyện dùng
mục tiêu dự đoán nhiễu DDPM MSE. Transformer kiểu LLaMA2 gồm 12 block có
chiều rộng 768; toàn bộ policy được báo cáo có 334M tham số, trong đó 221M có thể huấn luyện.
[Dita, §3 và Phụ lục A](https://arxiv.org/abs/2503.19757)

Thiết lập cơ sở được báo cáo dùng hai observation hình ảnh lịch sử và một trajectory
dài 16. Cách diễn đạt “16 action chunk” trong bài báo còn mơ hồ; cấu hình
chính thức hiện tại dùng `traj_length=16` và `num_pred_action=15`, vì vậy báo cáo
này không nâng cách diễn đạt đó thành một khẳng định mạnh hơn rằng có 16 hành động.
Quá trình huấn luyện dùng lịch DDPM 1.000 bước; đánh giá zero-shot chính dùng DDIM
20 bước, trong khi một ablation cho thấy 10 bước mạnh nhất ở một thiết lập được báo cáo.
[Bài báo Dita](https://arxiv.org/abs/2503.19757) ·
[repository chính thức](https://github.com/RoboDita/Dita)

**Hạn chế đã xác minh của nhãn được yêu cầu.** Bài báo báo cáo 334M
tham số và mô tả rõ Dita là một baseline mã nguồn mở gọn nhẹ.
Nó là “head lớn” so với MLP ba layer và lấy bộ giải mã làm trung tâm trong cách
điều kiện hóa, nhưng không cùng lớp quy mô với RDT-1B.

Các ablation trong bài báo chỉ hỗ trợ kiến trúc trong những thiết lập đã kiểm thử:
trajectory dài hơn có ích trên ManiSkill2, hai observation frame tốt hơn một hoặc
ba trong cấu hình được báo cáo, và mười bước DDIM hoạt động tốt nhất trong số
các số bước được đánh giá cho nhiệm vụ Google Robot được trích dẫn. Đây không phải
quy luật chung của bộ giải mã. [Dita, §4.6](https://arxiv.org/abs/2503.19757)

Tên cũng phải được giữ chính xác: mô hình chính thức là **Dita**. Cả
“DiTA” lẫn “DiT-Action” đều không phải tên chuẩn trong bài báo, trang dự án hoặc
repository của mô hình.

## Qwen-VLA: bộ giải mã hành động DiT

Module Qwen-VLA quan trọng là một **bộ giải mã hành động DiT single-stream
riêng** nằm sau Qwen3.5-4B VLM. Bài báo dùng “action expert” như một
nhãn chức năng không chặt chẽ trong §2.2, nhưng kiến trúc không phải expert kiểu π0
định tuyến token robot qua các trọng số thay thế bên trong VLM Transformer.

Qwen-VLA có ranh giới nối tiếp:

```text
hình ảnh + chỉ dẫn + prompt embodiment/FPS/horizon
                         |
                         v
                  Qwen3.5-4B VLM
                         |
                  hidden state cuối
                         |
             phép chiếu tuyến tính sang chiều rộng DiT
                         |
                         +--------------------------+
                                                    |
tensor hành động nhiễu H x K -> phép chiếu hành động -> token hành động
thời gian flow tau ----------> embedding timestep -> điều khiển AdaLN
                                                    |
                                                    v
              nối [ngữ cảnh VLM ; token hành động]
                                                    |
                 16 block DiT single-stream
         self-attention chung + multi-section RoPE
                                                    |
             giữ lại và chiếu các vị trí hành động
                                                    |
                                                    v
                    trường vận tốc H x K
                                                    |
                   nhiều lần cập nhật Euler
                                                    |
                                                    v
                   action chunk liên tục
```

VLM chạy trước để tạo ngữ cảnh ngữ nghĩa. Hidden state của nó được ánh xạ
sang chiều kênh DiT bằng một layer tuyến tính. Vector hành động nhiễu tại
mỗi timestep tương lai được chiếu riêng thành một token hành động. Hai
nhóm token này được nối rồi DiT xử lý cùng nhau.

Ranh giới forward nối tiếp này không có nghĩa VLM phải giữ nguyên: quá trình
continued pretraining và supervised fine-tuning của Qwen-VLA cùng cập nhật backbone
và bộ giải mã. Nó có nghĩa hai module giữ các block và tập tham số riêng,
với hidden state VLM đóng vai trò token điều kiện hóa của DiT.

“Self-attention chung” nghĩa là bên trong DiT, các vị trí hành động có thể dùng
toàn bộ ngữ cảnh thị giác-ngôn ngữ đã chiếu và phối hợp với các timestep hành động khác.
Nó **không** có nghĩa Qwen VLM và DiT dùng chung block Transformer hoặc
expert routing. VLM đã tạo xong hidden state trước khi bộ giải mã DiT
bắt đầu. [Qwen-VLA, §2.2](https://arxiv.org/abs/2605.30280)

### Một lượt DiT tính toán gì

DiT không trực tiếp xuất action chunk cuối trong một lượt. Một lần gọi
dự đoán trường vận tốc cho tensor hành động nhiễu/trung gian hiện tại.

Gọi `Y0` là mục tiêu demonstration sạch và `Y1` là nhiễu Gaussian. Khi huấn luyện,
mô hình lấy mẫu thời gian flow `tau` và tạo:

```text
Y_tau = (1 - tau) * Y0 + tau * Y1
vận tốc mục tiêu = Y1 - Y0
```

DiT nhận `Y_tau`, `tau` và ngữ cảnh VLM đã chiếu. AdaLN đưa
embedding thời gian flow vào quá trình tính toán Transformer, còn multi-section
RoPE cung cấp cấu trúc vị trí phù hợp với backbone đa phương thức. Các
vị trí hành động đầu ra được ánh xạ trở lại tensor vận tốc `H x K` và
tối ưu bằng masked MSE. [Qwen-VLA, §§2.2 và 2.5](https://arxiv.org/abs/2605.30280)

Khi inference, quá trình sinh bắt đầu tại `tau=1` bằng nhiễu Gaussian và tích phân
về `tau=0`. Với bước Euler giảm dần `delta`, phép cập nhật về mặt khái niệm là:

```text
Y_(tau-delta) = Y_tau - delta * v_theta(Y_tau, tau, VLM_context)
```

Mỗi bước Euler chạy lại DiT 16 block trên tensor hành động đã cập nhật. Bài báo
nói “một vài” bước Euler nhưng không công bố số bước mặc định chính xác, vì vậy
báo cáo không giả định mười bước từ π0 hoặc mô hình khác. Tensor cuối tại
`tau=0`, chứ không phải vận tốc từ một lượt DiT, là action chunk được sinh ra.

### Điều gì khiến đây là DiT lớn

Khoảng 1,15B tham số bộ giải mã được phân bổ như sau:

| Thành phần DiT | Số tham số được báo cáo | Vai trò |
| -------------------------- | ----------------------------: | ----------------------------------------------------------- |
| 16 block DiT | tổng khoảng 1,13B, mỗi block 70,8M | Attention chung và biến đổi token ngữ cảnh/hành động |
| MLP chiếu hành động thô | 4,9M | Ánh xạ giữa chiều hành động thô và chiều latent DiT |
| Phép chiếu VLM-sang-DiT | 3,9M | Ánh xạ hidden state Qwen sang không gian kênh của bộ giải mã |
| Embedding timestep | 2,8M | Mã hóa thời gian flow để điều kiện hóa AdaLN |
| Điều biến AdaLN đầu ra | 4,7M | Điều kiện hóa đường đầu ra của bộ giải mã theo thời gian flow |

Quy mô này là lý do Qwen-VLA phù hợp với họ mô hình hiện tại. DiT không phải một
head nông trên một vector VLM đã pooling: nó liên tục xử lý mọi token ngữ cảnh VLM
đã chiếu cùng toàn bộ trajectory nhiễu. [Qwen-VLA, §2.2](https://arxiv.org/abs/2605.30280)

### Từ một DiT đến nhiều embodiment

Bộ giải mã luôn dự đoán tensor cố định `Y in R^(H x K)`, nhưng mỗi dataset
có thể chỉ dùng `H_task <= H` timestep và `c <= K` kênh. Các giá trị hợp lệ
nằm ở vùng đầu; phần còn lại được zero-padding. Mask nhị phân loại các kênh
và timestep đã padding khỏi flow loss, đồng thời lấy trung bình đều các kênh hoạt động
để embodiment có nhiều chiều hơn không tự động chi phối.

Một DiT được tái sử dụng mà không có output head riêng cho từng embodiment. Ý nghĩa
điều khiển đến từ:

- prompt VLM mô tả loại robot, cấu hình cánh tay, control frequency,
  quy ước hành động và horizon;
- ngữ nghĩa kênh gốc của dataset;
- phép chuẩn hóa percentile 1/99 theo từng dataset;
- validity mask chọn phần thực của `H x K`.

Vì vậy, DiT dùng chung thống nhất **giao diện tensor và tham số bộ giải mã**, chứ
không thống nhất ngữ nghĩa vật lý của chuyển động end effector dạng delta, lệnh khớp
tuyệt đối, bộ kẹp, waypoint điều hướng hoặc trajectory pose người.
[Qwen-VLA, §§2.3-2.5](https://arxiv.org/abs/2605.30280)

Kiến trúc mặc định không dùng proprioception của robot. Bài báo chỉ báo cáo
mức cải thiện nhỏ khi bổ sung state trong một ablation RoboTwin-2.0 và giữ
giao diện mặc định được điều kiện hóa bằng thị giác và prompt. Bài báo cũng liệt kê
bộ nhớ, phục hồi sau thất bại, phản hồi lực/xúc giác và đánh giá dài hạn mạnh hơn
trong số những khoảng trống còn lại.

Ranh giới chính xác vì vậy là:

```text
Qwen3.5 VLM = biểu diễn và suy luận thị giác-ngôn ngữ
Qwen-VLA DiT = bộ giải mã flow matching lặp trên trajectory liên tục
bộ điều khiển = giải chuẩn hóa, ánh xạ embodiment, an toàn và thực thi
```

**Tình trạng hiện tại.** Theo kiểm tra ngày 2026-07-21, repository Qwen-VLA
chính thức cung cấp README và asset phục vụ bài báo nhưng không có implementation,
checkpoint, package hoặc release. Kiến trúc và kết quả báo cáo đã công khai;
repository đó chưa cung cấp bằng chứng về một policy mã nguồn mở chính thức có thể chạy.
[Repository Qwen-VLA chính thức](https://github.com/QwenLM/Qwen-VLA)

## So sánh giữa các mô hình

| Thuộc tính | RDT-1B | Dita | Qwen-VLA |
| -------------------------- | ----------------------------------------------- | ------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| Bộ sinh liên tục | Diffusion Transformer | DDPM causal Transformer | Bộ giải mã DiT flow matching gồm 16 block |
| Cấu trúc điều kiện hóa | Các block điều kiện hóa đa phương thức chuyên biệt | Token đa phương thức thô trong một chuỗi in-context | Self-attention chung trên state VLM và hành động nhiễu |
| Ví dụ horizon đã công bố | 64 hành động | Trajectory dài 16; cấu hình repo hiện tại dự đoán 15 hành động | Chunk thao tác 16 hành động; chunk điều hướng 8 waypoint trong SFT/đánh giá được báo cáo |
| Cơ chế embodiment | Các vị trí hành động thống nhất, có thể diễn giải vật lý | Biểu diễn EEF 7D chung trong thiết lập cốt lõi được báo cáo | Prompt embodiment bằng văn bản + tensor padding/masking + chuẩn hóa theo dataset |
| Lưu ý chính khi phân loại | Rõ ràng là lớn, nhưng vẫn dùng encoder bên ngoài | Lấy bộ giải mã làm trung tâm, không ở quy mô tỷ tham số | DiT downstream riêng; không phải trọng số expert định tuyến theo token kiểu π0 |

## Đánh đổi

Ưu điểm tiềm năng:

- năng lực lớn hơn cho các phân phối hành động đa phương thức và đa embodiment;
- attention có thể giữ các liên kết chi tiết giữa token hình ảnh/lịch sử và
  timestep hành động;
- mô hình hành động lớn có thể mở rộng độc lập với backbone ngữ nghĩa;
- khử nhiễu trajectory chung nắm bắt tương quan thời gian.

Chi phí và điểm chưa biết:

- các lượt chạy lặp qua hàng trăm triệu hoặc hàng tỷ tham số có thể
  chi phối độ trễ điều khiển;
- quy mô mô hình, dữ liệu pretraining, phương thức đầu vào và mục tiêu thay đổi cùng lúc
  trong các so sánh đã công bố, nên lợi ích từ quy mô chưa được tách biệt rõ ràng;
- tensor thống nhất có padding không giải quyết sự không tương thích ngữ nghĩa giữa
  hệ tọa độ, đơn vị hoặc quy ước điều khiển;
- vượt trội trên một benchmark robot/trình mô phỏng không chứng minh độ bền vững
  hoặc hành vi thời gian thực tốt hơn ở nơi khác;
- không nên dùng từ “độc lập” trừ khi ranh giới loại trừ rõ các encoder và
  module điều kiện hóa.

## Nguồn

- Liu và cộng sự. *RDT-1B: a Diffusion Foundation Model for Bimanual Manipulation*,
  arXiv:2410.07864v2, 2025. [Bài báo](https://arxiv.org/abs/2410.07864) ·
  [Dự án chính thức](https://rdt-robotics.github.io/rdt-robotics/) ·
  [Model card chính thức](https://huggingface.co/robotics-diffusion-transformer/rdt-1b)
- Hou và cộng sự. *Dita: Scaling Diffusion Transformer for Generalist
  Vision-Language-Action Policy*, arXiv:2503.19757v2, ICCV 2025.
  [Bài báo](https://arxiv.org/abs/2503.19757) ·
  [Dự án chính thức](https://robodita.github.io/)
- Wang và cộng sự. *Qwen-VLA: Unifying Vision-Language-Action Modeling across Tasks,
  Environments, and Robot Embodiments*, arXiv:2605.30280v2, 2026.
  [Bài báo](https://arxiv.org/abs/2605.30280) ·
  [Repository chính thức](https://github.com/QwenLM/Qwen-VLA)
